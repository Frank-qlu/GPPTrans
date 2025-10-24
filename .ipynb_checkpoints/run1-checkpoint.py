import os
import torch
import logging
import datetime
import argparse
from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from gpptrans.optimizer.extra_optimizers import ExtendedSchedulerConfig
from gpptrans.logger import create_logger
import gpptrans  # noqa, register custom modules
from dataclasses import dataclass

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义配置数据类
@dataclass
class OptimizerConfig:
    optimizer: str
    base_lr: float
    weight_decay: float
    momentum: float

def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum
    )

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps, 
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, 
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, 
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode='custom', 
        eval_period=cfg.train.eval_period
    )

def load_best_model(model, optimizer, scheduler, load_path):
    """加载最佳模型"""
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_metric', 0.0)

def evaluate_model_general(model, test_loader, device, task_type='classification'):
    """通用的模型评估函数"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0
    
    # 定义损失函数
    if task_type == 'classification':
        if hasattr(model, 'out_channels') and model.out_channels > 1:
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
    else:  # regression
        loss_fn = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            # 处理不同的数据格式
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
            
            # 计算损失
            if task_type == 'classification':
                if pred.dim() > 1 and pred.size(1) > 1:  # 多分类
                    loss = loss_fn(pred, labels.long())
                else:  # 二分类
                    loss = loss_fn(pred.squeeze(), labels.float())
            else:  # regression
                loss = loss_fn(pred.squeeze(), labels.float())
            
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
    
    # 计算指标
    if not all_preds:
        return {'test_loss': 0.0, 'accuracy': 0.0}
    
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    results = {'test_loss': avg_loss}
    
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

def test_model(config_file, model_path, output_dir=None):
    """测试已训练的模型"""
    
    # 创建一个简单的args对象来模拟训练时的参数
    class Args:
        def __init__(self, cfg_file):
            self.cfg_file = cfg_file
            self.opts = []  # 可选的其他配置参数
    
    args = Args(config_file)
    
    # 加载配置
    set_cfg(cfg)
    load_cfg(cfg, args)
    
    # 设置输出目录
    if output_dir:
        cfg.out_dir = output_dir
    else:
        cfg.out_dir = os.path.join('test_results', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(cfg.accelerator if cfg.accelerator else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # 创建数据加载器
    loaders = create_loader()
    
    # 创建模型
    model = create_model().to(device)
    
    # 创建优化器和调度器（用于加载状态）
    optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
    
    # 加载最佳模型
    logging.info(f"Loading model from: {model_path}")
    epoch, best_metric = load_best_model(model, optimizer, scheduler, model_path)
    logging.info(f"Loaded model from epoch {epoch} with best metric: {best_metric}")
    
    # 找到测试集loader
    test_loader = None
    for i, loader in enumerate(loaders):
        # 尝试通过名称识别测试集
        if hasattr(loader, 'name') and 'test' in loader.name.lower():
            test_loader = loader
            break
        # 如果loader没有名字，尝试通过索引判断
        if len(loaders) >= 3 and i == 2:  # 假设第三个是测试集
            test_loader = loader
    
    if test_loader is None:
        logging.warning("Could not find test loader, using the last loader")
        test_loader = loaders[-1]
    
    # 确定任务类型
    task_type = 'classification'  # 默认分类任务
    if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'task_type'):
        task_type = cfg.dataset.task_type
    elif hasattr(cfg, 'model') and hasattr(cfg.model, 'task'):
        task_type = cfg.model.task
    
    logging.info(f"Task type: {task_type}")
    
    # 在测试集上评估模型
    logging.info("Evaluating on test set...")
    test_results = evaluate_model_general(model, test_loader, device, task_type)
    
    # 输出结果
    logging.info("=== Test Results ===")
    for metric_name, value in test_results.items():
        logging.info(f"{metric_name}: {value:.6f}")
    
    # 保存测试结果
    results_file = os.path.join(cfg.out_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Loaded from epoch: {epoch}\n")
        f.write(f"Previous best metric: {best_metric}\n")
        f.write(f"Test date: {datetime.datetime.now()}\n\n")
        f.write("Test Results:\n")
        for metric_name, value in test_results.items():
            f.write(f"{metric_name}: {value:.6f}\n")
    
    logging.info(f"Test results saved to: {results_file}")
    
    return test_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained gpptrans model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    # 测试模型
    results = test_model(args.config, args.model, args.output)
    
    logging.info("Testing completed!")