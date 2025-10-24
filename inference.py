"""
Evaluate a trained GPPTrans model on the test set using the best saved checkpoint.

Usage:
    python inference.py --cfg_file your_cfg.yaml --model_path path/to/best_model.pth
"""

import os
import torch
import logging
import warnings
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, set_cfg, load_cfg)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device

import gpptrans  # noqa
from gpptrans.finetuning import load_pretrained_model_cfg, init_model_from_pretrained

warnings.simplefilter('ignore')
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ---------------- Helper Functions ----------------

def load_best_model(model, load_path, device=None):
    """Load best model checkpoint and return the epoch and metric if available."""
    if device is None:
        device = next(model.parameters()).device
    checkpoint = torch.load(load_path, map_location=device)

    if 'model_state_dict' not in checkpoint:
        raise KeyError(f"No 'model_state_dict' found in checkpoint {load_path}")

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', None)
    logging.info(f"Loaded model from {load_path} (epoch={epoch}, metric={best_metric})")
    return epoch, best_metric


def evaluate_model(model, test_loader, device, task_type='classification'):
    """Robust test evaluation."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0
    loss_fn = getattr(model, 'loss_fn', None)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            labels = batch.y.to(device)
            preds = model(batch)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            # Compute loss
            try:
                if loss_fn is not None:
                    loss = loss_fn(preds, labels)
                else:
                    if preds.dim() > 1 and preds.size(1) > 1:
                        loss = torch.nn.functional.cross_entropy(preds, labels.long())
                    else:
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            preds.squeeze(), labels.float())
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
            except Exception as e:
                logging.warning(f"Loss computation failed: {e}")

            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    results = {'test_loss': avg_loss}
    if task_type == 'classification':
        if preds.dim() > 1 and preds.size(1) > 1:
            pred_labels = preds.argmax(dim=1)
        else:
            probs = torch.sigmoid(preds.squeeze())
            pred_labels = (probs >= 0.5).long()
        accuracy = (pred_labels == labels.long()).float().mean().item()
        results['accuracy'] = accuracy
    else:
        mae = torch.abs(preds.squeeze() - labels).mean().item()
        mse = torch.nn.functional.mse_loss(preds.squeeze(), labels).item()
        results['mae'] = mae
        results['mse'] = mse

    return results


# ---------------- Main Script ----------------

if __name__ == "__main__":
    args = parse_args()
    set_cfg(cfg)
    load_cfg(cfg, args)

    # Determine model path
    if hasattr(args, 'model_path') and args.model_path:
        model_path = args.model_path
    else:
        model_path = "best_model.pth"  # default
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model checkpoint: {model_path}")

    # Device setup
    device = torch.device(cfg.accelerator if cfg.accelerator else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Handle pretrained model settings
    if cfg.pretrained.dir:
        cfg = load_pretrained_model_cfg(cfg)

    # Data & model creation
    loaders = create_loader()
    model = create_model().to(device)

    if cfg.pretrained.dir:
        model = init_model_from_pretrained(
            model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
            cfg.pretrained.reset_prediction_head
        )

    # Load checkpoint
    epoch, metric = load_best_model(model, model_path, device)

    # Select test loader
    test_loader = None
    for loader in loaders:
        name = getattr(loader, 'name', None)
        if name and 'test' in name.lower():
            test_loader = loader
            break
    if test_loader is None and isinstance(loaders, (list, tuple)) and len(loaders) > 0:
        test_loader = loaders[-1]
        logging.info("Using last loader as test set (fallback).")

    if test_loader is None:
        raise RuntimeError("No valid test loader found!")

    # Determine task type
    task_type = getattr(cfg.dataset, 'task_type', 'classification')
    logging.info(f"Task type: {task_type}")

    # Evaluate
    test_results = evaluate_model(model, test_loader, device, task_type)
    logging.info("=== Test Results ===")
    for k, v in test_results.items():
        logging.info(f"{k}: {v:.6f}")

    # Save test results
    result_file = os.path.join(os.path.dirname(model_path), "test_results_eval.txt")
    with open(result_file, "w") as f:
        for k, v in test_results.items():
            f.write(f"{k}: {v:.6f}\n")

    logging.info(f"Test results saved to {result_file}")