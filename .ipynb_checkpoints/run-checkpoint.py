import datetime
import os
import torch
import logging

import gpptrans  # noqa, register custom modules
from gpptrans.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,# set_agg_dir,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from gpptrans.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from gpptrans.logger import create_logger
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.simplefilter('ignore')
def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode='custom' , eval_period=cfg.train.eval_period)


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


# 添加保存最佳模型的函数
def save_best_model(model, optimizer, scheduler, epoch, best_metric, save_path):
    """保存最佳模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric
    }, save_path)
    logging.info(f"Best model saved to {save_path} with metric: {best_metric}")


# 添加加载最佳模型的函数
def load_best_model(model, optimizer, scheduler, load_path):
    """加载最佳模型"""
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)


# 添加测试集评估函数 - 修复元组和模型输出处理问题
def evaluate_on_test_set(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # 处理元组格式的batch
            if isinstance(batch, tuple):
                # 如果是元组，假设第一个元素是数据，第二个是标签
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)
                pred = model(data)
            else:
                # 如果是单个对象
                batch = batch.to(device)
                labels = batch.y
                pred = model(batch)
            
            # 处理模型输出为元组的情况
            if isinstance(pred, tuple):
                # 通常第一个元素是主要预测结果
                pred = pred[0]
            
            # 确保pred是张量
            if not isinstance(pred, torch.Tensor):
                logging.warning(f"Unexpected prediction type: {type(pred)}")
                continue
            
            # 计算loss
            if hasattr(model, 'loss_fn'):
                loss = model.loss_fn(pred, labels)
            else:
                # 默认使用交叉熵损失
                if pred.dim() > 1 and pred.size(1) > 1:
                    loss = torch.nn.functional.cross_entropy(pred, labels)
                else:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred.squeeze(), labels.float()
                    )
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    # 根据任务类型计算指标
    if all_preds:
        preds = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        results = {'test_loss': avg_loss}
        
        # 分类任务指标
        if preds.dim() > 1 and preds.size(1) > 1:  # 多分类
            pred_labels = preds.argmax(dim=1)
            accuracy = (pred_labels == labels).float().mean().item()
            results['accuracy'] = accuracy
        else:  # 二分类或回归
            pred_labels = (preds > 0).long() if preds.dim() > 1 else (preds.squeeze() > 0).long()
            accuracy = (pred_labels == labels).float().mean().item()
            results['accuracy'] = accuracy
        
        return results
    else:
        return {'test_loss': avg_loss, 'accuracy': 0.0}


# 更通用的评估函数，处理各种输出格式
def evaluate_model_general(model, test_loader, device, task_type='classification'):
    """更通用的模型评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 将数据移到设备
            if isinstance(batch, (list, tuple)):
                # 处理元组或列表格式的batch
                batch_data = []
                for item in batch:
                    if hasattr(item, 'to'):
                        batch_data.append(item.to(device))
                    else:
                        batch_data.append(item)
                # 假设最后一个元素是标签
                labels = batch_data[-1]
                # 其余元素作为模型输入
                model_input = batch_data[0] if len(batch_data) == 2 else batch_data[:-1]
                
                if isinstance(model_input, list) and len(model_input) == 1:
                    model_input = model_input[0]
                
                pred = model(model_input)
            else:
                # 单个数据对象
                batch = batch.to(device)
                labels = batch.y
                pred = model(batch)
            
            # 处理模型输出
            if isinstance(pred, (list, tuple)):
                # 通常第一个元素是主要预测结果
                pred = pred[0]
            
            # 确保pred是张量
            if not isinstance(pred, torch.Tensor):
                logging.warning(f"Unexpected prediction type: {type(pred)}")
                continue
            
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
    
    # 计算指标
    if not all_preds:
        return {'test_loss': 0.0, 'accuracy': 0.0}
    
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    results = {}
    
    if task_type == 'classification':
        if preds.dim() > 1 and preds.size(1) > 1:  # 多分类
            pred_labels = preds.argmax(dim=1)
            accuracy = (pred_labels == labels).float().mean().item()
            results['accuracy'] = accuracy
        else:  # 二分类
            pred_labels = (preds > 0).long() if preds.dim() > 1 else (preds.squeeze() > 0).long()
            accuracy = (pred_labels == labels).float().mean().item()
            results['accuracy'] = accuracy
    elif task_type == 'regression':
        # 回归任务指标
        mae = torch.abs(preds.squeeze() - labels).mean().item()
        mse = torch.nn.functional.mse_loss(preds.squeeze(), labels).item()
        results['mae'] = mae
        results['mse'] = mse
    
    return results


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Set configurations for each run
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        
        # 设置设备
        device = torch.device(cfg.accelerator if cfg.accelerator else 
                             'cuda' if torch.cuda.is_available() else 'cpu')
        
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, "
                     f"split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")
        
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        
        # custom_train expects three loggers for 'train', 'valid' and 'test'.
        if cfg.dataset.name in ['ogbn-arxiv', 'ogbn-proteins']:
            loggers_2 = create_logger()
            loggers_3 = create_logger()
            loggers_2[0].name = "val"
            loggers_3[0].name = "test"
            loggers.extend(loggers_2)
            loggers.extend(loggers_3)
            loaders = loaders * 3
        
        print(f"Using device: {device}")
        model = create_model().to(device)
        
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head
            )
        
        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
        
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        
        # 设置最佳模型保存路径
        best_model_path = os.path.join(cfg.run_dir, 'best_model.pth')
        final_model_path = os.path.join(cfg.run_dir, 'final_model.pth')
        
        # Start training
        mode = 'custom' 
        if mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the "
                                "default train.mode, set it to `custom`")
            train(loggers, loaders, model, optimizer, scheduler)
            # 训练结束后保存最终模型
            save_best_model(model, optimizer, scheduler, cfg.optim.max_epoch, 
                           "final_model", final_model_path)
        else:
            # 直接调用训练函数，不包装
            logging.info("Starting training...")
            train_result = train_dict[mode](loggers, loaders, model, optimizer, scheduler)
            logging.info(f"Training completed. Result: {train_result}")
            
            # 保存最终模型
            save_best_model(model, optimizer, scheduler, cfg.optim.max_epoch, 
                           "final_model", final_model_path)
            
            # 如果没有最佳模型，将最终模型复制为最佳模型
            if not os.path.exists(best_model_path):
                import shutil
                shutil.copy(final_model_path, best_model_path)
                logging.info("Copied final model as best model")
        
        # 加载模型并在测试集上评估
        model_path_to_load = best_model_path if os.path.exists(best_model_path) else final_model_path
        
        if os.path.exists(model_path_to_load):
            logging.info(f"Loading model from {model_path_to_load} for testing...")
            epoch, metric = load_best_model(model, optimizer, scheduler, model_path_to_load)
            logging.info(f"Loaded model from epoch {epoch} with metric: {metric}")
            
            # 找到测试集loader
            test_loader = None
            for i, loader in enumerate(loaders):
                # 尝试通过名称识别测试集
                if hasattr(loader, 'name') and 'test' in loader.name:
                    test_loader = loader
                    break
                # 如果loader没有名字，尝试通过索引判断
                if len(loaders) >= 3 and i == 2:  # 假设第三个是测试集
                    test_loader = loader
            
            if test_loader:
                # 尝试使用通用评估函数
                try:
                    # 根据数据集类型确定任务类型
                    task_type = 'classification'  # 默认分类任务
                    if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'task_type'):
                        task_type = cfg.dataset.task_type
                    
                    test_results = evaluate_model_general(model, test_loader, device, task_type)
                    logging.info("=== Final Test Results ===")
                    for metric_name, value in test_results.items():
                        logging.info(f"{metric_name}: {value:.6f}")
                    
                    # 保存测试结果到文件
                    results_file = os.path.join(cfg.run_dir, 'test_results.txt')
                    with open(results_file, 'w') as f:
                        for metric_name, value in test_results.items():
                            f.write(f"{metric_name}: {value:.6f}\n")
                    logging.info(f"Test results saved to: {results_file}")
                except Exception as e:
                    logging.error(f"Error during evaluation: {e}")
                    # 尝试更简单的评估方法
                    logging.info("Trying simplified evaluation...")
                    try:
                        # 只进行前向传播，不计算指标
                        model.eval()
                        with torch.no_grad():
                            for batch in test_loader:
                                if isinstance(batch, (list, tuple)):
                                    batch_data = [item.to(device) if hasattr(item, 'to') else item for item in batch]
                                    _ = model(batch_data[0] if len(batch_data) == 2 else batch_data[:-1])
                                else:
                                    batch = batch.to(device)
                                    _ = model(batch)
                        logging.info("Simplified evaluation completed successfully")
                    except Exception as e2:
                        logging.error(f"Simplified evaluation also failed: {e2}")
            else:
                logging.warning("Could not find test loader")
        else:
            logging.warning(f"No model found at {model_path_to_load}")
    
    # Aggregate results from different seeds
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")
    
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")