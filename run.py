import datetime
import os
import torch
import logging
import warnings
import shutil

import gpptrans  # noqa, register custom modules
from gpptrans.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from gpptrans.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from gpptrans.logger import create_logger

# Environment
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
warnings.simplefilter('ignore')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# -------------------- Helper configs --------------------

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
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings(args):
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

# -------------------- Checkpoint utilities --------------------

def save_best_model(model, optimizer, scheduler, epoch, best_metric, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric
    }, save_path)
    logging.info(f"Best model saved to {save_path} with metric: {best_metric}")


def load_best_model(model, optimizer, scheduler, load_path, device=None):
    """ load the best model"""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    checkpoint = torch.load(load_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError(f"No 'model_state_dict' in checkpoint {load_path}")
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logging.warning(f"Could not load optimizer state: {e}")
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            logging.warning(f"Could not load scheduler state: {e}")
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', None)
    return epoch, best_metric

# -------------------- Robust evaluation --------------------

def evaluate_model_general(model, test_loader, device, task_type='classification', verbose=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    loss_fn = getattr(model, 'loss_fn', None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, (list, tuple)):
                moved = []
                for item in batch:
                    if hasattr(item, 'to'):
                        moved.append(item.to(device))
                    else:
                        moved.append(item)
                *model_inputs, labels = moved
                model_input = model_inputs[0] if len(model_inputs) == 1 else model_inputs
            else:
                batch = batch.to(device)
                if not hasattr(batch, 'y'):
                    logging.warning("Batch has no attribute 'y', skipping.")
                    continue
                labels = batch.y
                model_input = batch

            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels, device=device)

            pred = model(model_input)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]

            if not isinstance(pred, torch.Tensor):
                logging.warning(f"Unexpected prediction type: {type(pred)}; skipping batch {batch_idx}")
                continue

            batch_size = labels.size(0) if isinstance(labels, torch.Tensor) else 0
            try:
                if loss_fn is not None:
                    loss = loss_fn(pred, labels)
                else:
                    if pred.dim() > 1 and pred.size(1) > 1:
                        if labels.dtype != torch.long:
                            labels = labels.long()
                        loss = torch.nn.functional.cross_entropy(pred, labels)
                    else:
                        logits = pred.squeeze()
                        labels_float = labels.float()
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_float)
                total_loss += (loss.item() * batch_size) if batch_size > 0 else 0.0
            except Exception as e:
                logging.debug(f"Loss compute failed for batch {batch_idx}: {e}")

            total_samples += batch_size
            all_preds.append(pred.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0

    if len(all_preds) == 0:
        return {'test_loss': avg_loss, 'accuracy': 0.0}

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    results = {'test_loss': avg_loss}

    if task_type == 'classification':
        if preds.dim() > 1 and preds.size(1) > 1:
            pred_labels = preds.argmax(dim=1)
            if labels.dtype != torch.long:
                labels = labels.long()
            accuracy = (pred_labels == labels).float().mean().item()
            results['accuracy'] = accuracy
        else:
            logits = preds.squeeze()
            if logits.min().item() >= 0.0 and logits.max().item() <= 1.0:
                probs = logits
            else:
                probs = torch.sigmoid(logits)
            pred_labels = (probs >= 0.5).long()
            if labels.dtype != torch.long:
                labels = labels.long()
            accuracy = (pred_labels == labels).float().mean().item()
            results['accuracy'] = accuracy
    elif task_type == 'regression':
        preds_flat = preds.squeeze()
        mae = torch.abs(preds_flat - labels).mean().item()
        mse = torch.nn.functional.mse_loss(preds_flat, labels).item()
        results['mae'] = mae
        results['mse'] = mse

    if verbose:
        logging.info(f"Evaluation results: {results}")

    return results

# -------------------- Main flow --------------------

if __name__ == '__main__':
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    torch.set_num_threads(cfg.num_threads)

    for run_id, seed, split_index in zip(*run_loop_settings(args)):
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)

        device = torch.device(cfg.accelerator if cfg.accelerator else 
                             'cuda' if torch.cuda.is_available() else 'cpu')

        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)

        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")

        loaders = create_loader()
        loggers = create_logger()

        # Handle special OGB datasets where loader structure differs
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

        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)

        best_model_path = os.path.join(cfg.run_dir, 'best_model.pth')
        final_model_path = os.path.join(cfg.run_dir, 'final_model.pth')

        mode = 'custom'
        if mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging is not supported with the default train.mode, set it to `custom`")
            train(loggers, loaders, model, optimizer, scheduler)
            save_best_model(model, optimizer, scheduler, cfg.optim.max_epoch, "final_model", final_model_path)
        else:
            logging.info("Starting training...")
            train_result = train_dict[mode](loggers, loaders, model, optimizer, scheduler)
            logging.info(f"Training completed. Result: {train_result}")

            save_best_model(model, optimizer, scheduler, cfg.optim.max_epoch, "final_model", final_model_path)

            if not os.path.exists(best_model_path):
                shutil.copy(final_model_path, best_model_path)
                logging.info("Copied final model as best model")

        model_path_to_load = best_model_path if os.path.exists(best_model_path) else final_model_path

        if os.path.exists(model_path_to_load):
            logging.info(f"Loading model from {model_path_to_load} for testing...")
            model.to(device)
            epoch, metric = load_best_model(model, optimizer=None, scheduler=None, load_path=model_path_to_load, device=device)
            logging.info(f"Loaded model from epoch {epoch} with metric: {metric}")

            # select test loader robustly
            test_loader = None
            try:
                for i, loader in enumerate(loaders):
                    name = getattr(loader, 'name', None)
                    if name and 'test' in name.lower():
                        test_loader = loader
                        break
            except Exception:
                pass

            if test_loader is None:
                if isinstance(loaders, (list, tuple)) and len(loaders) > 0:
                    test_loader = loaders[-1]
                    logging.info("Test loader not found by name; using last loader as fallback.")
                else:
                    logging.warning("Could not determine test loader from loaders structure.")

            if test_loader:
                try:
                    task_type = 'classification'
                    if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'task_type'):
                        task_type = cfg.dataset.task_type

                    test_results = evaluate_model_general(model, test_loader, device, task_type, verbose=True)
                    logging.info("=== Final Test Results ===")
                    for metric_name, value in test_results.items():
                        logging.info(f"{metric_name}: {value:.6f}")

                    results_file = os.path.join(cfg.run_dir, 'test_results.txt')
                    with open(results_file, 'w') as f:
                        for metric_name, value in test_results.items():
                            f.write(f"{metric_name}: {value:.6f}\n")
                    logging.info(f"Test results saved to: {results_file}")
                except Exception as e:
                    logging.error(f"Error during evaluation: {e}", exc_info=True)
            else:
                logging.warning("Could not find test loader")
        else:
            logging.warning(f"No model found at {model_path_to_load}")

    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed when trying to aggregate multiple runs: {e}")

    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
