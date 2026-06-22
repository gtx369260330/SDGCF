"""Training routines for SDGCF and comparison models."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import os
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import Config
from data_loader import create_dataloaders
from losses import HybridSleepStageLoss, compute_total_loss
from models import ConcatTransformer, SDGCFNet, SimpleConcatCNN, SingleModalityCNN
from utils import AverageMeter, EarlyStopping, calculate_metrics, get_checkpoint_path, save_json, set_seed


def _build_sdgcf(
    cfg: Config,
    *,
    use_graph: bool,
    graph_type: str = "dynamic",
) -> SDGCFNet:
    return SDGCFNet(
        input_channels=cfg.INPUT_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        embed_dim=cfg.EMBED_DIM,
        encoder_hidden=cfg.ENCODER_HIDDEN,
        kernel_sizes=cfg.KERNEL_SIZES,
        dropout=cfg.DROPOUT,
        graph_heads=cfg.GRAPH_HEADS,
        use_graph=use_graph,
        graph_type=graph_type,
        graph_alpha_init=cfg.GRAPH_ALPHA_INIT,
        use_auxiliary_heads=cfg.AUXILIARY_LOSS_WEIGHT > 0,
    )


def build_model(model_name: str, cfg: Config) -> nn.Module:
    """Build the proposed model, structural ablations or external baselines."""
    name = model_name.lower()
    if name == "sdgcf":
        return _build_sdgcf(cfg, use_graph=True, graph_type="dynamic")
    if name == "sdgcf_fixed_graph":
        return _build_sdgcf(cfg, use_graph=True, graph_type="fixed")
    if name == "multimodal_concat":
        return _build_sdgcf(cfg, use_graph=False)
    if name == "simple_concat":
        return SimpleConcatCNN(
            cfg.INPUT_CHANNELS,
            cfg.NUM_CLASSES,
            embed_dim=cfg.EMBED_DIM,
            dropout=cfg.DROPOUT,
        )
    if name == "concat_transformer":
        return ConcatTransformer(
            cfg.INPUT_CHANNELS,
            cfg.NUM_CLASSES,
            cfg.EMBED_DIM,
            cfg.ENCODER_HIDDEN,
            cfg.KERNEL_SIZES,
            cfg.DROPOUT,
            cfg.TRANSFORMER_HEADS,
            cfg.TRANSFORMER_LAYERS,
        )
    if name == "single_fpz":
        return SingleModalityCNN(0, cfg.NUM_CLASSES, cfg.EMBED_DIM, cfg.ENCODER_HIDDEN, cfg.KERNEL_SIZES, cfg.DROPOUT)
    if name == "single_pz":
        return SingleModalityCNN(1, cfg.NUM_CLASSES, cfg.EMBED_DIM, cfg.ENCODER_HIDDEN, cfg.KERNEL_SIZES, cfg.DROPOUT)
    if name == "single_eog":
        return SingleModalityCNN(2, cfg.NUM_CLASSES, cfg.EMBED_DIM, cfg.ENCODER_HIDDEN, cfg.KERNEL_SIZES, cfg.DROPOUT)
    raise ValueError(f"Unknown model_name: {model_name}")


def _is_amp_safe_model(model_name: str) -> bool:
    return model_name.lower() != "concat_transformer"


def _configure_safe_cuda_attention() -> None:
    if not torch.cuda.is_available():
        return
    for fn, value in [
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
        ("enable_cudnn_sdp", False),
    ]:
        try:
            getattr(torch.backends.cuda, fn)(value)
        except Exception:
            pass


def _amp_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


def apply_channel_augmentation(
    x: torch.Tensor,
    modality_dropout_prob: float = 0.0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """Apply lightweight augmentation while preserving at least one modality."""
    if modality_dropout_prob <= 0 and noise_std <= 0:
        return x
    x = x.clone()
    if noise_std > 0:
        x = x + torch.randn_like(x) * float(noise_std)
    if modality_dropout_prob > 0:
        batch_size, modality_count, _ = x.shape
        drop = torch.rand(batch_size, modality_count, device=x.device) < float(modality_dropout_prob)
        all_dropped = drop.all(dim=1)
        if all_dropped.any():
            rows = all_dropped.nonzero(as_tuple=False).flatten()
            keep = torch.randint(0, modality_count, (rows.numel(),), device=x.device)
            drop[rows, keep] = False
        x = x.masked_fill(drop.unsqueeze(-1), 0.0)
    return x


def _loss_kwargs(cfg: Config) -> Dict[str, float]:
    return {
        "auxiliary_weight": float(cfg.AUXILIARY_LOSS_WEIGHT),
        "attention_entropy_weight": float(cfg.ATTENTION_ENTROPY_WEIGHT),
        "attention_entropy_floor": float(cfg.ATTENTION_ENTROPY_FLOOR),
        "node_diversity_weight": float(cfg.NODE_DIVERSITY_WEIGHT),
        "node_diversity_margin": float(cfg.NODE_DIVERSITY_MARGIN),
    }


def _accumulate_loss_parts(meters: Dict[str, AverageMeter], parts: Dict[str, torch.Tensor], batch_size: int) -> None:
    for name, value in parts.items():
        if name not in meters:
            meters[name] = AverageMeter()
        meters[name].update(float(value.detach().item()), batch_size)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    cfg: Config,
    scaler=None,
    use_amp: bool = True,
    grad_clip: float = 5.0,
):
    model.train()
    loss_meters: Dict[str, AverageMeter] = {}
    y_true, y_pred = [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        x = apply_channel_augmentation(
            x,
            modality_dropout_prob=cfg.MODALITY_DROPOUT_PROB,
            noise_std=cfg.CHANNEL_NOISE_STD,
        )
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, use_amp):
            out = model(x)
            loss, parts = compute_total_loss(out, y, criterion, **_loss_kwargs(cfg))
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        _accumulate_loss_parts(loss_meters, parts, x.size(0))
        y_pred.extend(out["logits"].argmax(dim=1).detach().cpu().numpy().tolist())
        y_true.extend(y.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=loss_meters["total"].avg)
    return (
        {name: meter.avg for name, meter in loss_meters.items()},
        calculate_metrics(y_true, y_pred, class_names=["W", "N1", "N2", "N3", "REM"]),
    )


@torch.no_grad()
def validate(model, loader, criterion, device, cfg: Config, use_amp: bool = True):
    model.eval()
    loss_meters: Dict[str, AverageMeter] = {}
    y_true, y_pred = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        with _amp_context(device, use_amp):
            out = model(x)
            _, parts = compute_total_loss(out, y, criterion, **_loss_kwargs(cfg))
        _accumulate_loss_parts(loss_meters, parts, x.size(0))
        y_pred.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(y.cpu().numpy().tolist())
    return (
        {name: meter.avg for name, meter in loss_meters.items()},
        calculate_metrics(y_true, y_pred, class_names=["W", "N1", "N2", "N3", "REM"]),
    )


def _cuda_arch_supported(device: torch.device) -> tuple[bool, str]:
    try:
        index = 0 if device.index is None else int(device.index)
        major, minor = torch.cuda.get_device_capability(index)
        sm = f"sm_{major}{minor}"
        arch_list = list(torch.cuda.get_arch_list())
        if arch_list and sm not in arch_list:
            return False, f"GPU compute capability {sm} is not supported by this PyTorch build: {arch_list}"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _cuda_smoke_test(device: torch.device) -> tuple[bool, str]:
    try:
        torch.cuda.set_device(0 if device.index is None else int(device.index))
        x = torch.randn(16, 16, device=device)
        _ = (x @ x).mean().item()
        torch.cuda.synchronize(device)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _patch_broken_cuda_for_cpu_fallback(reason: str = "") -> None:
    try:
        torch.cuda.is_current_stream_capturing = lambda: False  # type: ignore[assignment]
        torch.cuda._is_in_bad_fork = lambda: False  # type: ignore[attr-defined,assignment]
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass
    os.environ["SDGCF_FORCE_CPU_FALLBACK"] = "1"
    if reason:
        print(f"[Device warning] CUDA disabled for this process: {reason}")


def _select_device(device_str: str | None = None) -> torch.device:
    requested = (device_str or "auto").strip().lower()
    if requested in ["", "auto", "default", "none"]:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        _patch_broken_cuda_for_cpu_fallback("CPU device requested")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            _patch_broken_cuda_for_cpu_fallback("CUDA unavailable")
            return torch.device("cpu")
        device = torch.device(requested)
        arch_ok, reason = _cuda_arch_supported(device)
        if not arch_ok:
            _patch_broken_cuda_for_cpu_fallback(reason)
            return torch.device("cpu")
        usable, reason = _cuda_smoke_test(device)
        if not usable:
            _patch_broken_cuda_for_cpu_fallback(reason)
            return torch.device("cpu")
        return device
    raise ValueError(f"Unknown device argument: {device_str}. Use auto, cpu, cuda, or cuda:0.")


def _print_device_info(device: torch.device) -> None:
    print(f"[Device] selected: {device}")
    print(f"[Device] torch version: {torch.__version__}")
    print(f"[Device] cuda available: {torch.cuda.is_available()}")
    print(f"[Device] torch cuda version: {torch.version.cuda}")


def _build_optimizer(model: nn.Module, cfg: Config) -> tuple[AdamW, list[float]]:
    """Use a modestly higher LR for the graph fusion parameters."""
    graph_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("graph_attention"):
            graph_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    groups = [{"params": backbone_parameters, "lr": float(cfg.LR)}]
    if graph_parameters:
        groups.append(
            {
                "params": graph_parameters,
                "lr": float(cfg.LR) * float(cfg.FUSION_LR_MULTIPLIER),
            }
        )
    optimizer = AdamW(
        groups,
        weight_decay=cfg.WEIGHT_DECAY,
        foreach=False,
        fused=False,
    )
    return optimizer, [float(group["lr"]) for group in optimizer.param_groups]


def _apply_warmup(optimizer: AdamW, target_lrs: list[float], epoch: int, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    factor = max(0.1, float(epoch) / float(warmup_epochs))
    for group, target_lr in zip(optimizer.param_groups, target_lrs):
        group["lr"] = target_lr * factor


def train_model(cfg: Config, model_name: str = "sdgcf", device_str: str | None = None) -> Dict[str, Any]:
    if model_name.lower() == "xgboost":
        from xgboost_baseline import train_xgboost_model

        return train_xgboost_model(cfg, model_name=model_name, device_str=device_str)
    if model_name.lower() in {"random_forest", "svm_linear", "logistic_regression"}:
        from classical_baselines import train_classical_model

        return train_classical_model(cfg, model_name=model_name, device_str=device_str)

    cfg.make_dirs()
    device = _select_device(device_str)
    set_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        _configure_safe_cuda_attention()
    if cfg.TORCH_NUM_THREADS:
        torch.set_num_threads(int(cfg.TORCH_NUM_THREADS))
    _print_device_info(device)

    train_loader, val_loader, _, class_weights, meta = create_dataloaders(cfg)
    model = build_model(model_name, cfg).to(device)
    weight = class_weights.to(device) if cfg.USE_CLASS_WEIGHT else None
    criterion = HybridSleepStageLoss(
        class_weights=weight,
        label_smoothing=cfg.LABEL_SMOOTHING,
        focal_gamma=cfg.FOCAL_GAMMA,
        focal_blend=cfg.FOCAL_BLEND,
    )
    optimizer, target_lrs = _build_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    use_amp = bool(cfg.USE_AMP and device.type == "cuda" and _is_amp_safe_model(model_name))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    early = EarlyStopping(cfg.EARLY_STOPPING_PATIENCE, mode="max")
    checkpoint_path = get_checkpoint_path(cfg.SAVE_DIR, model_name)
    history = []
    print(f"Model: {model_name} | Device: {device} | Checkpoint: {checkpoint_path}")
    print(f"Data shape: {meta['X_shape']} | DATA_FRACTION={meta.get('data_fraction', 1.0)}")
    print(
        "[Training policy] "
        f"label_smoothing={cfg.LABEL_SMOOTHING} | focal_gamma={cfg.FOCAL_GAMMA} | "
        f"focal_blend={cfg.FOCAL_BLEND} | aux={cfg.AUXILIARY_LOSS_WEIGHT} | "
        f"attention_entropy={cfg.ATTENTION_ENTROPY_WEIGHT} | "
        f"node_diversity={cfg.NODE_DIVERSITY_WEIGHT} | "
        f"graph_alpha_init={cfg.GRAPH_ALPHA_INIT} | "
        f"modality_dropout={cfg.MODALITY_DROPOUT_PROB} | noise={cfg.CHANNEL_NOISE_STD}"
    )

    for epoch in range(1, cfg.EPOCHS + 1):
        _apply_warmup(optimizer, target_lrs, epoch, cfg.WARMUP_EPOCHS)
        train_losses, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            scaler,
            use_amp,
            cfg.GRAD_CLIP_NORM,
        )
        val_losses, val_metrics = validate(model, val_loader, criterion, device, cfg, use_amp)
        if epoch >= cfg.WARMUP_EPOCHS:
            scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_losses["total"],
            "val_loss": val_losses["total"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": max(group["lr"] for group in optimizer.param_groups),
        }
        row.update({f"train_loss_{name}": value for name, value in train_losses.items() if name != "total"})
        row.update({f"val_loss_{name}": value for name, value in val_losses.items() if name != "total"})
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{cfg.EPOCHS} | train_loss={train_losses['total']:.4f} "
            f"val_loss={val_losses['total']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if early.step(val_metrics["macro_f1"]):
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "best_val_macro_f1": val_metrics["macro_f1"],
                },
                checkpoint_path,
            )
        if early.should_stop:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(Path(cfg.SAVE_DIR) / "logs" / f"training_history_{model_name}.csv", index=False)
    save_json(
        {
            "model_name": model_name,
            "best_val_macro_f1": early.best,
            "checkpoint": str(checkpoint_path),
            "training_policy": "hybrid_focal_ce_with_graph_regularization",
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "focal_gamma": cfg.FOCAL_GAMMA,
            "focal_blend": cfg.FOCAL_BLEND,
            "auxiliary_loss_weight": cfg.AUXILIARY_LOSS_WEIGHT,
            "attention_entropy_weight": cfg.ATTENTION_ENTROPY_WEIGHT,
            "node_diversity_weight": cfg.NODE_DIVERSITY_WEIGHT,
            "graph_alpha_init": cfg.GRAPH_ALPHA_INIT,
        },
        Path(cfg.SAVE_DIR) / "logs" / f"train_summary_{model_name}.json",
    )
    return {"history": history, "checkpoint": str(checkpoint_path), "best_val_macro_f1": early.best}
