from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import VGG19_Weights, vgg19

from linf_sr.data import (
    SatelliteCSVStreamDataset,
    summarize_filtered_csv,
)
from linf_sr.model import RRDBLINF, make_coord_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("RRDB-LINF (3x3) trainer for satellite SR")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/jacksju/jack_work/modulabs/Aiffelthon/dataset/24Patch_tif",
        help="Root folder containing Train/Valid/Test and CSV files.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/home/jacksju/jack_work/modulabs/Aiffelthon/dataset/24Patch_tif/train_24patch_list.csv",
    )
    parser.add_argument(
        "--valid-csv",
        type=str,
        default="/home/jacksju/jack_work/modulabs/Aiffelthon/dataset/24Patch_tif/valid_24patch_list.csv",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="4xSatSR,RefRSSRD,SEN2NAIPv2unet,SEN2VENuS",
        help="Comma-separated list of categories to include.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="2,4,10",
        help="Comma-separated scales used for bucketed training.",
    )
    parser.add_argument(
        "--max-samples-per-category",
        type=int,
        default=-1,
        help="Sample cap per category, for each split.",
    )
    parser.add_argument(
        "--batch-size-map",
        type=str,
        default="2:192,4:96,10:24",
        help="Per-scale batch size map. Example: 2:192,4:96,10:24",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=524288,
        help=(
            "Approximate streaming shuffle buffer size (rows) for train dataset. "
            "Larger improves randomness but increases host RAM usage per worker."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--train-mode",
        type=str,
        default="single",
        choices=["single", "two_stage"],
        help=(
            "single: legacy L1 training, "
            "two_stage: stage1 NLL-only then stage2 L1+lambda_nll*NLL+lambda_vgg*VGG."
        ),
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=3,
        help="Number of epochs for stage1 (NLL-only) when train-mode=two_stage.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=3,
        help="Number of epochs for stage2 (L1+lambda*NLL) when train-mode=two_stage.",
    )
    parser.add_argument(
        "--stage2-nll-lambda",
        type=float,
        default=2e-3,
        help="Lambda for stage2 loss: L1 + lambda * NLL.",
    )
    parser.add_argument(
        "--stage2-vgg-lambda",
        type=float,
        default=1e-3,
        help="Lambda for stage2 loss: L1 + lambda_nll * NLL + lambda_vgg * VGG.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Base LR used by single-mode training.",
    )
    parser.add_argument(
        "--stage1-start-lr",
        type=float,
        default=1e-4,
        help="Initial LR for the first mixed segment of stage1.",
    )
    parser.add_argument(
        "--stage2-start-lr",
        type=float,
        default=4e-4,
        help="Initial LR for the first mixed segment of stage2.",
    )
    parser.add_argument(
        "--restart-lr",
        type=float,
        default=1e-4,
        help="LR to reset to at epoch restarts and x10-only segment restarts.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-5,
        help="Cosine scheduler minimum LR for each epoch segment in two-stage mode.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision when using CUDA.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="AMP dtype for CUDA autocast.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=8192)
    parser.add_argument("--feat-channels", type=int, default=48)
    parser.add_argument("--num-rrdb-blocks", type=int, default=12)
    parser.add_argument("--growth-channels", type=int, default=24)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument(
        "--log-sigma-min",
        type=float,
        default=-3.5,
        help="Lower clamp bound for predicted per-pixel/channel log_sigma.",
    )
    parser.add_argument(
        "--log-sigma-max",
        type=float,
        default=1.0,
        help="Upper clamp bound for predicted per-pixel/channel log_sigma.",
    )
    parser.add_argument(
        "--film-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of FiLM meta MLP in encoder.",
    )
    parser.add_argument(
        "--film-scale",
        type=float,
        default=0.1,
        help="Global scale factor for FiLM gamma/beta modulation.",
    )
    parser.add_argument(
        "--sampling-temperature",
        "-sampling-temperature",
        type=float,
        default=0.0,
        help=(
            "Temperature for 3x3 local sampling weights in LINF decoder. "
            "0.0 keeps previous behavior."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/rrdb_linf_satellite",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="Path to resume checkpoint (e.g., latest_resume.pt or best_resume.pt).",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=-1,
        help="If >0, stop each epoch after this many train steps.",
    )
    parser.add_argument(
        "--max-valid-steps",
        type=int,
        default=-1,
        help="If >0, stop validation after this many steps per scale.",
    )
    parser.add_argument(
        "--skip-valid",
        action="store_true",
        help="Disable validation loop.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scale_map(raw: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid batch size map item: {item}")
        scale_raw, batch_raw = item.split(":", 1)
        out[int(scale_raw)] = int(batch_raw)
    return out


def summarize_counts(
    total: int,
    by_category: Mapping[str, int],
    by_scale: Mapping[int, int],
    title: str,
) -> None:
    print(f"\n[{title}] total={total}")
    print("  category counts:")
    for category, count in sorted(by_category.items()):
        print(f"    - {category}: {count}")
    print("  scale counts:")
    for scale, count in sorted(by_scale.items()):
        print(f"    - x{scale}: {count}")


def make_dataloaders(
    csv_path: str | Path,
    dataset_root: str | Path,
    counts_by_scale: Mapping[int, int],
    categories: Sequence[str],
    max_per_category: int,
    seed: int,
    scales: Iterable[int],
    batch_size_map: Mapping[int, int],
    shuffle_buffer_size: int,
    num_workers: int,
    shuffle: bool,
) -> Dict[int, DataLoader]:
    loaders: Dict[int, DataLoader] = {}
    for scale in sorted(scales):
        sample_count = int(counts_by_scale.get(scale, 0))
        if sample_count <= 0:
            continue
        dataset = SatelliteCSVStreamDataset(
            csv_path=csv_path,
            dataset_root=dataset_root,
            scale=scale,
            categories=categories,
            max_per_category=max_per_category,
            sample_count=sample_count,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed + scale,
        )
        batch_size = batch_size_map.get(scale, 1)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            # IterableDataset handles ordering internally.
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": False,
            "drop_last": False,
            "persistent_workers": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        loader = DataLoader(**loader_kwargs)
        loaders[scale] = loader
    return loaders


def iter_round_robin(loaders: Mapping[int, DataLoader]) -> Iterator[Tuple[int, dict]]:
    scales = sorted(loaders.keys())
    iterators = {scale: iter(loaders[scale]) for scale in scales}
    remaining = {scale: len(loaders[scale]) for scale in scales}

    while True:
        progressed = False
        for scale in scales:
            if remaining[scale] <= 0:
                continue
            batch = next(iterators[scale])
            remaining[scale] -= 1
            progressed = True
            yield scale, batch
        if not progressed:
            break


def build_coord_and_meta(
    batch: dict,
    device: torch.device,
    scale_to_index: Mapping[int, int],
    num_scales: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hr = batch["hr"]
    if not isinstance(hr, torch.Tensor):
        raise TypeError("Batch is missing HR tensor.")
    batch_size, _, hr_h, hr_w = hr.shape
    num_query = hr_h * hr_w

    coord = make_coord_grid(hr_h, hr_w, device=device)
    coord = coord.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    cell = torch.tensor(
        [2.0 / hr_w, 2.0 / hr_h], device=device, dtype=torch.float32
    ).view(1, 1, 2)
    cell = cell.expand(batch_size, num_query, 2)

    lr_resolution = batch["lr_resolution"].to(device=device, dtype=torch.float32).view(
        batch_size, 1
    )
    lr_resolution_log10 = torch.log10(torch.clamp(lr_resolution, min=1e-6))

    norm_group = batch["norm_group"].to(device=device, dtype=torch.long)
    norm_onehot = F.one_hot(norm_group, num_classes=2).to(torch.float32)
    scale_tensor = batch["scale"]
    if not isinstance(scale_tensor, torch.Tensor):
        scale_tensor = torch.tensor(scale_tensor, dtype=torch.long)
    scale_tensor = scale_tensor.to(device=device, dtype=torch.long).view(batch_size)
    scale_index = torch.full(
        (batch_size,),
        fill_value=-1,
        device=device,
        dtype=torch.long,
    )
    for scale_value, idx in scale_to_index.items():
        scale_index[scale_tensor == scale_value] = int(idx)
    if torch.any(scale_index < 0):
        unknown = torch.unique(scale_tensor[scale_index < 0]).tolist()
        raise ValueError(f"Unsupported scale values in batch: {unknown}")
    scale_onehot = F.one_hot(scale_index, num_classes=num_scales).to(torch.float32)

    encoder_meta = torch.cat([lr_resolution_log10, norm_onehot, scale_onehot], dim=-1)
    encoder_meta_expanded = encoder_meta.view(batch_size, 1, -1).expand(
        batch_size, num_query, -1
    )
    decoder_meta = torch.cat([cell, encoder_meta_expanded], dim=-1)
    return coord, decoder_meta, encoder_meta, norm_group


def compute_fixed_range_psnr_per_sample(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))


def compute_data_range_psnr_channels_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Per-sample, per-channel data-range PSNR.
    max_i = target.amax(dim=(2, 3)) - target.amin(dim=(2, 3))
    max_i = torch.clamp(max_i, min=eps)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(2, 3))
    mse = torch.clamp(mse, min=eps)
    psnr_ch = 20.0 * torch.log10(max_i) - 10.0 * torch.log10(mse)
    mean_psnr = psnr_ch.mean(dim=1)
    return psnr_ch, mean_psnr


def compute_channel_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean(dim=(2, 3))


def get_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


class VGG19Relu34Perceptual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.features.children())[:18])
        self.features.eval()
        self.features.requires_grad_(False)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)
        pred_feat = self.features(pred_norm)
        with torch.no_grad():
            target_feat = self.features(target_norm)
        return torch.abs(pred_feat - target_feat)


def build_vgg19_relu34_perceptual(device: torch.device) -> VGG19Relu34Perceptual:
    module = VGG19Relu34Perceptual().to(device)
    module.eval()
    return module


def gaussian_nll_map(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    diff2 = (pred - target) ** 2
    inv_var = torch.exp(-2.0 * log_sigma)
    nll = 0.5 * (diff2 * inv_var + 2.0 * log_sigma)
    return nll


def reduce_group_balanced_from_map(
    loss_map: torch.Tensor,
    norm_group: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_loss = loss_map.mean(dim=(1, 2, 3))
    mask_8 = norm_group == 0
    mask_14 = norm_group == 1

    if torch.any(mask_8):
        l8 = sample_loss[mask_8].mean()
    else:
        l8 = torch.tensor(float("nan"), device=loss_map.device, dtype=loss_map.dtype)

    if torch.any(mask_14):
        l14 = sample_loss[mask_14].mean()
    else:
        l14 = torch.tensor(float("nan"), device=loss_map.device, dtype=loss_map.dtype)

    if torch.any(mask_8) and torch.any(mask_14):
        balanced = 0.5 * l8 + 0.5 * l14
    elif torch.any(mask_8):
        balanced = l8
    elif torch.any(mask_14):
        balanced = l14
    else:
        raise RuntimeError("Group-balanced loss received an empty batch.")
    return balanced, l8, l14


def compose_stage_loss(
    stage_loss_mode: str,
    l1_loss: torch.Tensor,
    nll_loss: torch.Tensor,
    stage2_nll_lambda: float,
    vgg_loss: torch.Tensor | None = None,
    stage2_vgg_lambda: float = 0.0,
) -> torch.Tensor:
    if stage_loss_mode == "l1_only":
        return l1_loss
    if stage_loss_mode == "nll_only":
        return nll_loss
    if stage_loss_mode == "l1_plus_nll":
        total = l1_loss + stage2_nll_lambda * nll_loss
        if vgg_loss is not None and stage2_vgg_lambda > 0:
            total = total + stage2_vgg_lambda * vgg_loss
        return total
    raise ValueError(f"Unsupported stage_loss_mode: {stage_loss_mode}")


def compute_effective_train_steps_per_epoch(
    loaders: Mapping[int, DataLoader],
    max_train_steps: int,
) -> int:
    steps = sum(len(loader) for loader in loaders.values())
    if max_train_steps > 0:
        steps = min(steps, max_train_steps)
    return int(steps)


@dataclass(frozen=True)
class EpochLrSegment:
    name: str
    start_step: int
    end_step: int
    start_lr: float

    @property
    def num_steps(self) -> int:
        return self.end_step - self.start_step + 1


@dataclass(frozen=True)
class StageSchedulerConfig:
    initial_lr: float
    restart_lr: float
    eta_min: float
    x10_scale: int = 10


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    eta_min: float = 0.0,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(total_steps), 1),
        eta_min=eta_min,
    )


def compute_x10_only_start_step(
    loaders: Mapping[int, DataLoader],
    max_train_steps: int,
    x10_scale: int = 10,
) -> int | None:
    target_steps = compute_effective_train_steps_per_epoch(loaders, max_train_steps)
    if target_steps <= 0 or x10_scale not in loaders:
        return None

    scales = sorted(loaders.keys())
    if not any(scale != x10_scale for scale in scales):
        return None

    remaining = {scale: len(loaders[scale]) for scale in scales}
    step = 0
    while step < target_steps:
        active_scales = [scale for scale in scales if remaining[scale] > 0]
        if active_scales == [x10_scale]:
            return step + 1
        if not active_scales:
            break

        for scale in scales:
            if remaining[scale] <= 0:
                continue
            step += 1
            remaining[scale] -= 1
            if step >= target_steps:
                break

    return None


def build_epoch_lr_segments(
    loaders: Mapping[int, DataLoader],
    max_train_steps: int,
    epoch_start_lr: float,
    restart_lr: float,
    x10_scale: int = 10,
) -> List[EpochLrSegment]:
    target_steps = compute_effective_train_steps_per_epoch(loaders, max_train_steps)
    if target_steps <= 0:
        raise ValueError("Epoch LR segments require at least one train step.")

    x10_only_start_step = compute_x10_only_start_step(
        loaders=loaders,
        max_train_steps=max_train_steps,
        x10_scale=x10_scale,
    )
    if (
        x10_only_start_step is None
        or x10_only_start_step <= 1
        or x10_only_start_step > target_steps
    ):
        return [
            EpochLrSegment(
                name="full_epoch",
                start_step=1,
                end_step=target_steps,
                start_lr=epoch_start_lr,
            )
        ]

    return [
        EpochLrSegment(
            name="mixed",
            start_step=1,
            end_step=x10_only_start_step - 1,
            start_lr=epoch_start_lr,
        ),
        EpochLrSegment(
            name="x10_only",
            start_step=x10_only_start_step,
            end_step=target_steps,
            start_lr=restart_lr,
        ),
    ]


def describe_epoch_lr_segments(
    segments: Sequence[EpochLrSegment],
    eta_min: float,
) -> str:
    return " | ".join(
        (
            f"{segment.name} steps {segment.start_step}-{segment.end_step} "
            f"({segment.num_steps}) lr {segment.start_lr:.2e}->{eta_min:.2e}"
        )
        for segment in segments
    )


def compute_total_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    total_sq = 0.0
    has_grad = False
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = float(torch.linalg.vector_norm(param.grad.detach()).cpu())
        total_sq += grad_norm * grad_norm
        has_grad = True
    if not has_grad:
        return float("nan")
    return float(total_sq**0.5)


def compute_log_sigma_saturation_ratios(
    pred_log_sigma: torch.Tensor,
    norm_group: torch.Tensor,
    low_bound: float,
    high_bound: float,
    atol: float = 1e-6,
) -> Dict[str, float | None]:
    stats: Dict[str, float | None] = {}
    group_specs = ((0, "8bit"), (1, "14bit"))
    channel_names = ("r", "g", "b")

    for group_idx, group_name in group_specs:
        group_mask = norm_group == group_idx
        if not bool(torch.any(group_mask)):
            for channel_name in channel_names:
                stats[f"pred_log_sigma_low_ratio_{group_name}_{channel_name}"] = None
                stats[f"pred_log_sigma_high_ratio_{group_name}_{channel_name}"] = None
            continue

        group_values = pred_log_sigma[group_mask]  # [B_group, C, H, W]
        for channel_idx, channel_name in enumerate(channel_names):
            if channel_idx >= group_values.shape[1]:
                stats[f"pred_log_sigma_low_ratio_{group_name}_{channel_name}"] = None
                stats[f"pred_log_sigma_high_ratio_{group_name}_{channel_name}"] = None
                continue
            channel_values = group_values[:, channel_idx, :, :]
            low_ratio = (channel_values <= (low_bound + atol)).to(torch.float32).mean()
            high_ratio = (channel_values >= (high_bound - atol)).to(torch.float32).mean()
            stats[f"pred_log_sigma_low_ratio_{group_name}_{channel_name}"] = float(
                low_ratio.detach().cpu()
            )
            stats[f"pred_log_sigma_high_ratio_{group_name}_{channel_name}"] = float(
                high_ratio.detach().cpu()
            )
    return stats


def train_one_epoch(
    model: RRDBLINF,
    loaders: Mapping[int, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    stage_loss_mode: str,
    stage2_nll_lambda: float,
    stage2_vgg_lambda: float,
    vgg_perceptual: VGG19Relu34Perceptual | None,
    query_chunk_size: int,
    grad_clip: float,
    max_train_steps: int,
    scale_to_index: Mapping[int, int],
    num_scales: int,
    lr_segments: Sequence[EpochLrSegment] | None = None,
    eta_min: float = 0.0,
    debug_log_path: Path | None = None,
    debug_interval: int = 20,
    stage_name: str = "",
    global_epoch: int = 0,
) -> Tuple[Dict[str, float], torch.optim.lr_scheduler._LRScheduler | None]:
    model.train()
    total_loss = 0.0
    total_l1 = 0.0
    total_nll = 0.0
    total_vgg = 0.0
    total_steps = 0

    total_samples = 0
    total_fixed_psnr_sum = 0.0
    total_mean_data_range_psnr_sum = 0.0
    total_data_range_psnr_rgb_sum = np.zeros(3, dtype=np.float64)
    total_l1_rgb_sum = np.zeros(3, dtype=np.float64)

    target_steps = compute_effective_train_steps_per_epoch(loaders, max_train_steps)
    active_scheduler = scheduler
    active_segment_idx = 0
    if lr_segments:
        for segment in lr_segments:
            if segment.num_steps <= 0:
                raise ValueError(f"Invalid LR segment with non-positive length: {segment}")
        set_optimizer_lr(optimizer, lr_segments[0].start_lr)
        active_scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            total_steps=lr_segments[0].num_steps,
            eta_min=eta_min,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(iter_round_robin(loaders), total=target_steps, desc="train", leave=False)
    if debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    debug_handle_ctx = (
        debug_log_path.open("a")
        if (debug_interval > 0 and debug_log_path is not None)
        else nullcontext(None)
    )
    with debug_handle_ctx as debug_handle:
        for step, (scale, batch) in enumerate(progress, start=1):
            if lr_segments:
                while (
                    active_segment_idx + 1 < len(lr_segments)
                    and step == lr_segments[active_segment_idx + 1].start_step
                ):
                    active_segment_idx += 1
                    next_segment = lr_segments[active_segment_idx]
                    set_optimizer_lr(optimizer, next_segment.start_lr)
                    active_scheduler = build_cosine_scheduler(
                        optimizer=optimizer,
                        total_steps=next_segment.num_steps,
                        eta_min=eta_min,
                    )

            lr_step = float(optimizer.param_groups[0]["lr"])
            lr = batch["lr"].to(device=device, dtype=torch.float32, non_blocking=True)
            hr = batch["hr"].to(device=device, dtype=torch.float32, non_blocking=True)

            coord, meta, encoder_meta, norm_group = build_coord_and_meta(
                batch=batch,
                device=device,
                scale_to_index=scale_to_index,
                num_scales=num_scales,
            )
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                pred_q, pred_log_sigma_q = model(
                    lr,
                    coord,
                    meta,
                    encoder_meta=encoder_meta,
                    query_chunk_size=query_chunk_size,
                    return_log_sigma=True,
                )
                pred = pred_q.view(hr.shape[0], hr.shape[2], hr.shape[3], -1).permute(0, 3, 1, 2)
                pred_log_sigma = pred_log_sigma_q.view(
                    hr.shape[0], hr.shape[2], hr.shape[3], -1
                ).permute(0, 3, 1, 2)

            pred_f32 = pred.float()
            hr_f32 = hr.float()
            pred_log_sigma_f32 = pred_log_sigma.float()

            l1_map = F.l1_loss(pred_f32, hr_f32, reduction="none")
            nll_map = gaussian_nll_map(pred_f32, hr_f32, pred_log_sigma_f32)
            l1_loss, _, _ = reduce_group_balanced_from_map(l1_map, norm_group)
            nll_loss, _, _ = reduce_group_balanced_from_map(nll_map, norm_group)
            vgg_loss: torch.Tensor | None = None
            if (
                vgg_perceptual is not None
                and stage_loss_mode == "l1_plus_nll"
                and stage2_vgg_lambda > 0
            ):
                vgg_map = vgg_perceptual(pred_f32, hr_f32)
                vgg_loss, _, _ = reduce_group_balanced_from_map(vgg_map, norm_group)
            weighted_nll_loss = torch.zeros_like(nll_loss)
            weighted_vgg_loss = torch.zeros_like(nll_loss)
            if stage_loss_mode == "nll_only":
                weighted_nll_loss = nll_loss
            elif stage_loss_mode == "l1_plus_nll":
                weighted_nll_loss = stage2_nll_lambda * nll_loss
                if vgg_loss is not None and stage2_vgg_lambda > 0:
                    weighted_vgg_loss = stage2_vgg_lambda * vgg_loss
            loss = compose_stage_loss(
                stage_loss_mode=stage_loss_mode,
                l1_loss=l1_loss,
                nll_loss=nll_loss,
                stage2_nll_lambda=stage2_nll_lambda,
                vgg_loss=vgg_loss,
                stage2_vgg_lambda=stage2_vgg_lambda,
            )

            optimizer_step_happened = False
            amp_step_skipped = False
            grad_norm_value = float("nan")
            grad_clip_applied = False
            if scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm_value = float(grad_norm_tensor.detach().cpu())
                    grad_clip_applied = bool(
                        np.isfinite(grad_norm_value) and grad_norm_value > float(grad_clip)
                    )
                else:
                    grad_norm_value = compute_total_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_step_happened = scale_after >= scale_before
                amp_step_skipped = not optimizer_step_happened
            else:
                loss.backward()
                if grad_clip > 0:
                    grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm_value = float(grad_norm_tensor.detach().cpu())
                    grad_clip_applied = bool(
                        np.isfinite(grad_norm_value) and grad_norm_value > float(grad_clip)
                    )
                else:
                    grad_norm_value = compute_total_grad_norm(model.parameters())
                optimizer.step()
                optimizer_step_happened = True
            if active_scheduler is not None and optimizer_step_happened:
                active_scheduler.step()

            with torch.no_grad():
                pred_clamped = pred_f32.clamp(0.0, 1.0)
                fixed_psnr_per_sample = compute_fixed_range_psnr_per_sample(pred_clamped, hr_f32)
                data_range_psnr_rgb_per_sample, mean_data_range_psnr_per_sample = (
                    compute_data_range_psnr_channels_per_sample(pred_clamped, hr_f32)
                )
                l1_rgb_per_sample = compute_channel_l1_per_sample(pred_clamped, hr_f32)

                batch_size = int(pred_clamped.shape[0])
                total_samples += batch_size
                total_fixed_psnr_sum += float(fixed_psnr_per_sample.sum().detach().cpu())
                total_mean_data_range_psnr_sum += float(
                    mean_data_range_psnr_per_sample.sum().detach().cpu()
                )
                total_data_range_psnr_rgb_sum += (
                    data_range_psnr_rgb_per_sample.sum(dim=0).detach().cpu().numpy()
                )
                total_l1_rgb_sum += l1_rgb_per_sample.sum(dim=0).detach().cpu().numpy()

            total_steps += 1
            total_loss += float(loss.detach().cpu())
            total_l1 += float(l1_loss.detach().cpu())
            total_nll += float(nll_loss.detach().cpu())
            if vgg_loss is not None:
                total_vgg += float(vgg_loss.detach().cpu())
            progress.set_postfix(
                {
                    "scale": scale,
                    "loss": f"{total_loss / total_steps:.5f}",
                    "l1": f"{total_l1 / total_steps:.5f}",
                    "nll": f"{total_nll / total_steps:.5f}",
                    "vgg": f"{total_vgg / total_steps:.5f}",
                    "mean_dr_psnr": (
                        f"{(total_mean_data_range_psnr_sum / max(total_samples, 1)):.2f}"
                    ),
                }
            )
            if debug_handle is not None and debug_interval > 0 and step % debug_interval == 0:
                with torch.no_grad():
                    log_sigma_min_bound = float(getattr(model, "log_sigma_min", -3.0))
                    log_sigma_max_bound = float(getattr(model, "log_sigma_max", 1.0))
                    sigma_sat_ratios = compute_log_sigma_saturation_ratios(
                        pred_log_sigma=pred_log_sigma_f32,
                        norm_group=norm_group,
                        low_bound=log_sigma_min_bound,
                        high_bound=log_sigma_max_bound,
                    )
                    debug_record = {
                        "stage": stage_name,
                        "epoch": int(global_epoch),
                        "step": int(step),
                        "scale": int(scale),
                        "loss_batch": float(loss.detach().cpu()),
                        "l1_batch": float(l1_loss.detach().cpu()),
                        "nll_batch": float(nll_loss.detach().cpu()),
                        "weighted_nll_batch": float(weighted_nll_loss.detach().cpu()),
                        "vgg_batch": (
                            float(vgg_loss.detach().cpu()) if vgg_loss is not None else 0.0
                        ),
                        "weighted_vgg_batch": float(weighted_vgg_loss.detach().cpu()),
                        "psnr_batch": float(fixed_psnr_per_sample.mean().detach().cpu()),
                        "lr_step": lr_step,
                        "grad_norm": grad_norm_value,
                        "grad_clip_applied": bool(grad_clip_applied),
                        "amp_step_skipped": bool(amp_step_skipped),
                        "pred_abs_max": float(pred_f32.detach().abs().max().cpu()),
                        "pred_log_sigma_min": float(pred_log_sigma_f32.detach().amin().cpu()),
                        "pred_log_sigma_max": float(pred_log_sigma_f32.detach().amax().cpu()),
                        "pred_log_sigma_low_bound": log_sigma_min_bound,
                        "pred_log_sigma_high_bound": log_sigma_max_bound,
                    }
                    debug_record.update(sigma_sat_ratios)
                debug_handle.write(json.dumps(debug_record) + "\n")
                debug_handle.flush()

            if max_train_steps > 0 and step >= max_train_steps:
                break

    if total_steps == 0:
        raise RuntimeError("No training steps were executed.")
    if total_samples == 0:
        raise RuntimeError("No training samples were processed.")
    metrics = {
        "loss": total_loss / total_steps,
        "l1": total_l1 / total_steps,
        "nll": total_nll / total_steps,
        "vgg": total_vgg / total_steps,
        "psnr": total_fixed_psnr_sum / total_samples,
        "mean_data_range_psnr": total_mean_data_range_psnr_sum / total_samples,
        "data_range_psnr_r": total_data_range_psnr_rgb_sum[0] / total_samples,
        "data_range_psnr_g": total_data_range_psnr_rgb_sum[1] / total_samples,
        "data_range_psnr_b": total_data_range_psnr_rgb_sum[2] / total_samples,
        "l1_r": total_l1_rgb_sum[0] / total_samples,
        "l1_g": total_l1_rgb_sum[1] / total_samples,
        "l1_b": total_l1_rgb_sum[2] / total_samples,
        "steps": float(total_steps),
        "samples": float(total_samples),
    }
    if device.type == "cuda":
        metrics["max_mem_alloc_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        metrics["max_mem_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    return metrics, active_scheduler


@torch.no_grad()
def validate(
    model: RRDBLINF,
    loaders: Mapping[int, DataLoader],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    stage_loss_mode: str,
    stage2_nll_lambda: float,
    stage2_vgg_lambda: float,
    vgg_perceptual: VGG19Relu34Perceptual | None,
    query_chunk_size: int,
    max_valid_steps: int,
    scale_to_index: Mapping[int, int],
    num_scales: int,
) -> Dict[str, float]:
    model.eval()

    per_scale: Dict[int, Dict[str, float | int | np.ndarray]] = {}

    total_loss = 0.0
    total_l1 = 0.0
    total_nll = 0.0
    total_vgg = 0.0
    total_steps = 0
    total_samples = 0
    total_fixed_psnr_sum = 0.0
    total_mean_data_range_psnr_sum = 0.0
    total_data_range_psnr_rgb_sum = np.zeros(3, dtype=np.float64)
    total_l1_rgb_sum = np.zeros(3, dtype=np.float64)

    for scale, loader in sorted(loaders.items()):
        scale_state = {
            "loss_sum": 0.0,
            "l1_sum": 0.0,
            "nll_sum": 0.0,
            "vgg_sum": 0.0,
            "steps": 0,
            "samples": 0,
            "psnr_sum": 0.0,
            "mean_data_range_psnr_sum": 0.0,
            "data_range_psnr_rgb_sum": np.zeros(3, dtype=np.float64),
            "l1_rgb_sum": np.zeros(3, dtype=np.float64),
        }
        progress = tqdm(loader, desc=f"valid_x{scale}", leave=False)
        for step, batch in enumerate(progress, start=1):
            lr = batch["lr"].to(device=device, dtype=torch.float32, non_blocking=True)
            hr = batch["hr"].to(device=device, dtype=torch.float32, non_blocking=True)

            coord, meta, encoder_meta, norm_group = build_coord_and_meta(
                batch=batch,
                device=device,
                scale_to_index=scale_to_index,
                num_scales=num_scales,
            )
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                pred_q, pred_log_sigma_q = model(
                    lr,
                    coord,
                    meta,
                    encoder_meta=encoder_meta,
                    query_chunk_size=query_chunk_size,
                    return_log_sigma=True,
                )
                pred = pred_q.view(hr.shape[0], hr.shape[2], hr.shape[3], -1).permute(0, 3, 1, 2)
                pred_log_sigma = pred_log_sigma_q.view(
                    hr.shape[0], hr.shape[2], hr.shape[3], -1
                ).permute(0, 3, 1, 2)

            pred_f32 = pred.float()
            hr_f32 = hr.float()
            pred_log_sigma_f32 = pred_log_sigma.float()
            l1_map = F.l1_loss(pred_f32, hr_f32, reduction="none")
            nll_map = gaussian_nll_map(pred_f32, hr_f32, pred_log_sigma_f32)
            l1_loss, _, _ = reduce_group_balanced_from_map(l1_map, norm_group)
            nll_loss, _, _ = reduce_group_balanced_from_map(nll_map, norm_group)
            vgg_loss: torch.Tensor | None = None
            if (
                vgg_perceptual is not None
                and stage_loss_mode == "l1_plus_nll"
                and stage2_vgg_lambda > 0
            ):
                vgg_map = vgg_perceptual(pred_f32, hr_f32)
                vgg_loss, _, _ = reduce_group_balanced_from_map(vgg_map, norm_group)
            loss = compose_stage_loss(
                stage_loss_mode=stage_loss_mode,
                l1_loss=l1_loss,
                nll_loss=nll_loss,
                stage2_nll_lambda=stage2_nll_lambda,
                vgg_loss=vgg_loss,
                stage2_vgg_lambda=stage2_vgg_lambda,
            )

            pred_clamped = pred_f32.clamp(0.0, 1.0)
            fixed_psnr_per_sample = compute_fixed_range_psnr_per_sample(pred_clamped, hr_f32)
            data_range_psnr_rgb_per_sample, mean_data_range_psnr_per_sample = (
                compute_data_range_psnr_channels_per_sample(pred_clamped, hr_f32)
            )
            l1_rgb_per_sample = compute_channel_l1_per_sample(pred_clamped, hr_f32)

            batch_size = int(pred_clamped.shape[0])

            scale_state["steps"] = int(scale_state["steps"]) + 1
            scale_state["samples"] = int(scale_state["samples"]) + batch_size
            scale_state["loss_sum"] = float(scale_state["loss_sum"]) + float(loss.detach().cpu())
            scale_state["l1_sum"] = float(scale_state["l1_sum"]) + float(l1_loss.detach().cpu())
            scale_state["nll_sum"] = float(scale_state["nll_sum"]) + float(nll_loss.detach().cpu())
            if vgg_loss is not None:
                scale_state["vgg_sum"] = float(scale_state["vgg_sum"]) + float(
                    vgg_loss.detach().cpu()
                )
            scale_state["psnr_sum"] = float(scale_state["psnr_sum"]) + float(
                fixed_psnr_per_sample.sum().detach().cpu()
            )
            scale_state["mean_data_range_psnr_sum"] = float(
                scale_state["mean_data_range_psnr_sum"]
            ) + float(mean_data_range_psnr_per_sample.sum().detach().cpu())
            scale_state["data_range_psnr_rgb_sum"] = np.asarray(
                scale_state["data_range_psnr_rgb_sum"], dtype=np.float64
            ) + data_range_psnr_rgb_per_sample.sum(dim=0).detach().cpu().numpy()
            scale_state["l1_rgb_sum"] = np.asarray(
                scale_state["l1_rgb_sum"], dtype=np.float64
            ) + l1_rgb_per_sample.sum(dim=0).detach().cpu().numpy()

            total_steps += 1
            total_samples += batch_size
            total_loss += float(loss.detach().cpu())
            total_l1 += float(l1_loss.detach().cpu())
            total_nll += float(nll_loss.detach().cpu())
            if vgg_loss is not None:
                total_vgg += float(vgg_loss.detach().cpu())
            total_fixed_psnr_sum += float(fixed_psnr_per_sample.sum().detach().cpu())
            total_mean_data_range_psnr_sum += float(
                mean_data_range_psnr_per_sample.sum().detach().cpu()
            )
            total_data_range_psnr_rgb_sum += (
                data_range_psnr_rgb_per_sample.sum(dim=0).detach().cpu().numpy()
            )
            total_l1_rgb_sum += l1_rgb_per_sample.sum(dim=0).detach().cpu().numpy()

            if max_valid_steps > 0 and step >= max_valid_steps:
                break

        if int(scale_state["steps"]) == 0:
            continue

        per_scale[scale] = scale_state

    if total_steps == 0 or total_samples == 0:
        return {
            "loss": float("nan"),
            "l1": float("nan"),
            "nll": float("nan"),
            "vgg": float("nan"),
            "psnr": float("nan"),
            "mean_data_range_psnr": float("nan"),
            "steps": 0.0,
            "samples": 0.0,
        }

    metrics = {
        "loss": total_loss / total_steps,
        "l1": total_l1 / total_steps,
        "nll": total_nll / total_steps,
        "vgg": total_vgg / total_steps,
        "psnr": total_fixed_psnr_sum / total_samples,
        "mean_data_range_psnr": total_mean_data_range_psnr_sum / total_samples,
        "data_range_psnr_r": total_data_range_psnr_rgb_sum[0] / total_samples,
        "data_range_psnr_g": total_data_range_psnr_rgb_sum[1] / total_samples,
        "data_range_psnr_b": total_data_range_psnr_rgb_sum[2] / total_samples,
        "l1_r": total_l1_rgb_sum[0] / total_samples,
        "l1_g": total_l1_rgb_sum[1] / total_samples,
        "l1_b": total_l1_rgb_sum[2] / total_samples,
        "steps": float(total_steps),
        "samples": float(total_samples),
    }
    for scale in sorted(per_scale):
        state = per_scale[scale]
        steps = max(int(state["steps"]), 1)
        samples = max(int(state["samples"]), 1)
        metrics[f"loss_x{scale}"] = float(state["loss_sum"]) / steps
        metrics[f"l1_x{scale}"] = float(state["l1_sum"]) / steps
        metrics[f"nll_x{scale}"] = float(state["nll_sum"]) / steps
        metrics[f"vgg_x{scale}"] = float(state["vgg_sum"]) / steps
        metrics[f"psnr_x{scale}"] = float(state["psnr_sum"]) / samples
        metrics[f"mean_data_range_psnr_x{scale}"] = (
            float(state["mean_data_range_psnr_sum"]) / samples
        )
        data_rgb = np.asarray(state["data_range_psnr_rgb_sum"], dtype=np.float64)
        l1_rgb = np.asarray(state["l1_rgb_sum"], dtype=np.float64)
        metrics[f"data_range_psnr_r_x{scale}"] = data_rgb[0] / samples
        metrics[f"data_range_psnr_g_x{scale}"] = data_rgb[1] / samples
        metrics[f"data_range_psnr_b_x{scale}"] = data_rgb[2] / samples
        metrics[f"l1_r_x{scale}"] = l1_rgb[0] / samples
        metrics[f"l1_g_x{scale}"] = l1_rgb[1] / samples
        metrics[f"l1_b_x{scale}"] = l1_rgb[2] / samples
    return metrics


def save_checkpoint(
    save_path: Path,
    model: RRDBLINF,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr,
            "args": vars(args),
        },
        save_path,
    )


def capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        else:
            torch_state = torch.tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state)
    if "cuda" in state and torch.cuda.is_available():
        raw_cuda_states = state["cuda"]
        if isinstance(raw_cuda_states, torch.Tensor):
            raw_cuda_states = [raw_cuda_states]

        cuda_states: List[torch.Tensor] = []
        for cuda_state in raw_cuda_states:
            if isinstance(cuda_state, torch.Tensor):
                converted = cuda_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            else:
                converted = torch.tensor(cuda_state, dtype=torch.uint8)
            cuda_states.append(converted)
        torch.cuda.set_rng_state_all(cuda_states)


def save_resume_checkpoint(
    save_path: Path,
    model: RRDBLINF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
            "rng_state": capture_rng_state(),
            "args": vars(args),
        },
        save_path,
    )


def load_resume_checkpoint(
    resume_path: Path,
    model: RRDBLINF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected model keys in resume checkpoint: {unexpected}"
        )
    if missing:
        raise RuntimeError(
            f"Missing unsupported model keys in resume checkpoint: {missing}"
        )

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])

    last_epoch = int(checkpoint.get("epoch", 0))
    best_psnr = float(checkpoint.get("best_psnr", -1e9))
    return last_epoch, best_psnr


def load_model_checkpoint_only(
    checkpoint_path: Path,
    model: RRDBLINF,
    device: torch.device,
) -> Tuple[int, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected model keys in checkpoint: {unexpected}"
        )
    if missing:
        raise RuntimeError(
            f"Missing unsupported model keys in checkpoint: {missing}"
        )

    last_epoch = int(checkpoint.get("epoch", 0))
    best_psnr = float(checkpoint.get("best_psnr", -1e9))
    return last_epoch, best_psnr


def run_stage_training(
    stage_name: str,
    stage_epochs: int,
    stage_loss_mode: str,
    stage2_nll_lambda: float,
    stage2_vgg_lambda: float,
    vgg_perceptual: VGG19Relu34Perceptual | None,
    model: RRDBLINF,
    train_loaders: Mapping[int, DataLoader],
    valid_loaders: Mapping[int, DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    query_chunk_size: int,
    grad_clip: float,
    max_train_steps: int,
    max_valid_steps: int,
    save_dir: Path,
    args: argparse.Namespace,
    log_path: Path,
    global_epoch_start: int,
    scale_to_index: Mapping[int, int],
    num_scales: int,
    best_tracker: Dict[str, float],
    stage_scheduler_config: StageSchedulerConfig | None = None,
) -> Tuple[float, Path, int]:
    if stage_epochs <= 0:
        return -1e9, save_dir / f"{stage_name}_best_resume.pt", global_epoch_start - 1

    stage1_select_by_nll = stage_name == "stage1"
    stage_best_primary = float("inf") if stage1_select_by_nll else -1e9
    stage_best_resume_path = save_dir / f"{stage_name}_best_resume.pt"
    debug_log_path = save_dir / "debug_step_log.jsonl"
    global_epoch = global_epoch_start - 1

    for stage_epoch in range(1, stage_epochs + 1):
        global_epoch = global_epoch_start + stage_epoch - 1
        epoch_lr_segments: Sequence[EpochLrSegment] | None = None
        eta_min = 0.0
        if stage_scheduler_config is not None:
            epoch_start_lr = (
                stage_scheduler_config.initial_lr
                if stage_epoch == 1
                else stage_scheduler_config.restart_lr
            )
            eta_min = stage_scheduler_config.eta_min
            epoch_lr_segments = build_epoch_lr_segments(
                loaders=train_loaders,
                max_train_steps=max_train_steps,
                epoch_start_lr=epoch_start_lr,
                restart_lr=stage_scheduler_config.restart_lr,
                x10_scale=stage_scheduler_config.x10_scale,
            )
            print(
                f"[{stage_name}] Epoch {stage_epoch:03d} LR schedule | "
                f"{describe_epoch_lr_segments(epoch_lr_segments, eta_min)}"
            )

        train_metrics, scheduler = train_one_epoch(
            model=model,
            loaders=train_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            stage_loss_mode=stage_loss_mode,
            stage2_nll_lambda=stage2_nll_lambda,
            stage2_vgg_lambda=stage2_vgg_lambda,
            vgg_perceptual=vgg_perceptual,
            query_chunk_size=query_chunk_size,
            grad_clip=grad_clip,
            max_train_steps=max_train_steps,
            scale_to_index=scale_to_index,
            num_scales=num_scales,
            lr_segments=epoch_lr_segments,
            eta_min=eta_min,
            debug_log_path=debug_log_path,
            #debug_interval=20,
            debug_interval=1,
            stage_name=stage_name,
            global_epoch=global_epoch,
        )

        valid_metrics: Dict[str, float] = {}
        if valid_loaders:
            valid_metrics = validate(
                model=model,
                loaders=valid_loaders,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                stage_loss_mode=stage_loss_mode,
                stage2_nll_lambda=stage2_nll_lambda,
                stage2_vgg_lambda=stage2_vgg_lambda,
                vgg_perceptual=vgg_perceptual,
                query_chunk_size=query_chunk_size,
                max_valid_steps=max_valid_steps,
                scale_to_index=scale_to_index,
                num_scales=num_scales,
            )

        record = {
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            "epoch": global_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "valid": valid_metrics,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"[{stage_name}] Epoch {stage_epoch:03d}/{stage_epochs:03d} "
            f"(global {global_epoch:03d}) | "
            f"train loss={train_metrics['loss']:.5f} | "
            f"l1={train_metrics['l1']:.5f} | nll={train_metrics['nll']:.5f} | "
            f"vgg={train_metrics['vgg']:.5f} | "
            f"psnr={train_metrics['psnr']:.2f} | "
            f"mean_dr_psnr={train_metrics['mean_data_range_psnr']:.2f}"
        )
        if "max_mem_alloc_mb" in train_metrics:
            print(
                f"           max_mem alloc/resv = "
                f"{train_metrics['max_mem_alloc_mb']:.0f}/"
                f"{train_metrics['max_mem_reserved_mb']:.0f} MiB"
            )
        if valid_metrics:
            print(
                f"           valid loss={valid_metrics['loss']:.5f} | "
                f"l1={valid_metrics['l1']:.5f} | nll={valid_metrics['nll']:.5f} | "
                f"vgg={valid_metrics['vgg']:.5f} | "
                f"psnr={valid_metrics['psnr']:.2f} | "
                f"mean_dr_psnr={valid_metrics['mean_data_range_psnr']:.2f}"
            )

        current_primary = valid_metrics.get(
            "mean_data_range_psnr_x10",
            valid_metrics.get("mean_data_range_psnr", train_metrics["mean_data_range_psnr"]),
        )
        stage_selection_metric = current_primary
        if stage1_select_by_nll:
            stage_selection_metric = valid_metrics.get(
                "nll_x10",
                valid_metrics.get("nll", train_metrics["nll"]),
            )

        stage_improved = (
            stage_selection_metric < stage_best_primary
            if stage1_select_by_nll
            else stage_selection_metric > stage_best_primary
        )
        if stage_improved:
            stage_best_primary = stage_selection_metric
            save_checkpoint(
                save_path=save_dir / f"{stage_name}_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                best_psnr=stage_best_primary,
                args=args,
            )
            save_resume_checkpoint(
                save_path=stage_best_resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=global_epoch,
                best_psnr=stage_best_primary,
                args=args,
            )

        if current_primary > best_tracker["global_primary"]:
            best_tracker["global_primary"] = current_primary
            save_checkpoint(
                save_path=save_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                best_psnr=best_tracker["global_primary"],
                args=args,
            )
            save_resume_checkpoint(
                save_path=save_dir / "best_resume.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=global_epoch,
                best_psnr=best_tracker["global_primary"],
                args=args,
            )

        current_l1_x10 = valid_metrics.get("l1_x10")
        if current_l1_x10 is not None and current_l1_x10 < best_tracker["x10_l1"]:
            best_tracker["x10_l1"] = current_l1_x10
            save_checkpoint(
                save_path=save_dir / "best_l1_x10.pt",
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                best_psnr=best_tracker["x10_l1"],
                args=args,
            )
            save_resume_checkpoint(
                save_path=save_dir / "best_l1_x10_resume.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=global_epoch,
                best_psnr=best_tracker["x10_l1"],
                args=args,
            )

        current_mean_data_range_psnr_x10 = valid_metrics.get("mean_data_range_psnr_x10")
        if (
            current_mean_data_range_psnr_x10 is not None
            and current_mean_data_range_psnr_x10 > best_tracker["x10_mean_data_range_psnr"]
        ):
            best_tracker["x10_mean_data_range_psnr"] = current_mean_data_range_psnr_x10
            save_checkpoint(
                save_path=save_dir / "best_mean_data_range_psnr_x10.pt",
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                best_psnr=best_tracker["x10_mean_data_range_psnr"],
                args=args,
            )
            save_resume_checkpoint(
                save_path=save_dir / "best_mean_data_range_psnr_x10_resume.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=global_epoch,
                best_psnr=best_tracker["x10_mean_data_range_psnr"],
                args=args,
            )

        save_checkpoint(
            save_path=save_dir / f"{stage_name}_latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=global_epoch,
            best_psnr=stage_best_primary,
            args=args,
        )
        save_resume_checkpoint(
            save_path=save_dir / f"{stage_name}_latest_resume.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=global_epoch,
            best_psnr=stage_best_primary,
            args=args,
        )
        # Keep global latest alias.
        save_checkpoint(
            save_path=save_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=global_epoch,
            best_psnr=best_tracker["global_primary"],
            args=args,
        )
        save_resume_checkpoint(
            save_path=save_dir / "latest_resume.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=global_epoch,
            best_psnr=best_tracker["global_primary"],
            args=args,
        )

    return stage_best_primary, stage_best_resume_path, global_epoch


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    torch.backends.cudnn.benchmark = True
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available in this environment. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = get_amp_dtype(args.amp_dtype)
    print(f"AMP enabled: {use_amp} (dtype={args.amp_dtype})")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "train_config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    categories = parse_str_list(args.categories)
    scales = parse_int_list(args.scales)
    batch_size_map = parse_scale_map(args.batch_size_map)
    scale_to_index = {2: 0, 4: 1, 10: 2}
    num_scales = 3
    unknown_scales = [scale for scale in scales if scale not in scale_to_index]
    if unknown_scales:
        raise ValueError(
            f"Unsupported scales in --scales: {unknown_scales}. "
            "Supported scales are 2, 4, 10."
        )

    train_total, train_by_category, train_by_scale = summarize_filtered_csv(
        csv_path=args.train_csv,
        categories=categories,
        max_per_category=args.max_samples_per_category,
    )
    summarize_counts(train_total, train_by_category, train_by_scale, "train")

    train_loaders = make_dataloaders(
        csv_path=args.train_csv,
        dataset_root=args.dataset_root,
        counts_by_scale=train_by_scale,
        categories=categories,
        max_per_category=args.max_samples_per_category,
        seed=args.seed,
        scales=scales,
        batch_size_map=batch_size_map,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    if not train_loaders:
        raise RuntimeError("No train dataloaders were created. Check scales/categories.")

    valid_loaders = {}
    if not args.skip_valid:
        valid_total, valid_by_category, valid_by_scale = summarize_filtered_csv(
            csv_path=args.valid_csv,
            categories=categories,
            max_per_category=args.max_samples_per_category,
        )
        summarize_counts(valid_total, valid_by_category, valid_by_scale, "valid")
        valid_loaders = make_dataloaders(
            csv_path=args.valid_csv,
            dataset_root=args.dataset_root,
            counts_by_scale=valid_by_scale,
            categories=categories,
            max_per_category=args.max_samples_per_category,
            seed=args.seed + 10000,
            scales=scales,
            batch_size_map=batch_size_map,
            shuffle_buffer_size=1,
            num_workers=args.num_workers,
            shuffle=False,
        )

    model = RRDBLINF(
        in_channels=3,
        out_channels=3,
        feat_channels=args.feat_channels,
        num_blocks=args.num_rrdb_blocks,
        growth_channels=args.growth_channels,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_layers=args.decoder_layers,
        meta_dim=8,
        encoder_meta_dim=6,
        sampling_temperature=args.sampling_temperature,
        log_sigma_min=args.log_sigma_min,
        log_sigma_max=args.log_sigma_max,
        film_hidden_dim=args.film_hidden_dim,
        film_scale=args.film_scale,
    ).to(device)

    stage1_scheduler_config = StageSchedulerConfig(
        initial_lr=args.stage1_start_lr,
        restart_lr=args.restart_lr,
        eta_min=args.eta_min,
    )
    stage2_scheduler_config = StageSchedulerConfig(
        initial_lr=args.stage2_start_lr,
        restart_lr=args.restart_lr,
        eta_min=args.eta_min,
    )
    optimizer_lr = args.lr if args.train_mode == "single" else args.stage1_start_lr
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optimizer_lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and args.amp_dtype == "fp16"),
    )
    train_steps_per_epoch = compute_effective_train_steps_per_epoch(
        loaders=train_loaders,
        max_train_steps=args.max_train_steps,
    )
    if args.train_mode == "single":
        sched_total_steps = args.num_epochs * train_steps_per_epoch
        scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            total_steps=sched_total_steps,
        )
    else:
        scheduler = build_cosine_scheduler(
            optimizer=optimizer,
            total_steps=1,
            eta_min=args.eta_min,
        )
    best_tracker = {
        "global_primary": -1e9,
        "x10_l1": float("inf"),
        "x10_mean_data_range_psnr": -1e9,
    }

    log_path = save_dir / "train_log.jsonl"
    if args.train_mode == "single":
        start_epoch = 1
        if args.resume_path:
            last_epoch, best_psnr_loaded = load_resume_checkpoint(
                resume_path=Path(args.resume_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            start_epoch = last_epoch + 1
            best_tracker["global_primary"] = max(
                best_tracker["global_primary"], best_psnr_loaded
            )
            print(
                f"Resumed from: {args.resume_path} | "
                f"last_epoch={last_epoch} -> start_epoch={start_epoch} | "
                f"best_primary={best_psnr_loaded:.4f}"
            )

        if start_epoch > args.num_epochs:
            print(
                f"start_epoch ({start_epoch}) is greater than num_epochs ({args.num_epochs}). "
                "Nothing to train."
            )
            return

        stage_epochs = args.num_epochs - start_epoch + 1
        best_primary, _, _ = run_stage_training(
            stage_name="single",
            stage_epochs=stage_epochs,
            stage_loss_mode="l1_only",
            stage2_nll_lambda=args.stage2_nll_lambda,
            stage2_vgg_lambda=0.0,
            vgg_perceptual=None,
            model=model,
            train_loaders=train_loaders,
            valid_loaders=valid_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            query_chunk_size=args.query_chunk_size,
            grad_clip=args.grad_clip,
            max_train_steps=args.max_train_steps,
            max_valid_steps=args.max_valid_steps,
            save_dir=save_dir,
            args=args,
            log_path=log_path,
            global_epoch_start=start_epoch,
            scale_to_index=scale_to_index,
            num_scales=num_scales,
            best_tracker=best_tracker,
        )
        print(f"Training complete. Best x10 mean_data_range_psnr={best_primary:.2f}")
        print(f"Artifacts saved to: {save_dir.resolve()}")
        return

    # two-stage training: stage1 (NLL-only) -> load stage1 latest -> stage2 (L1+NLL+VGG)
    if args.resume_path:
        last_epoch, best_psnr_loaded = load_resume_checkpoint(
            resume_path=Path(args.resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        best_tracker["global_primary"] = max(best_tracker["global_primary"], best_psnr_loaded)
        print(
            f"Loaded initialization from resume: {args.resume_path} "
            f"(epoch={last_epoch}, best_primary={best_psnr_loaded:.4f})"
        )

    print(
        f"Two-stage config | stage1_epochs={args.stage1_epochs} (NLL-only), "
        f"stage2_epochs={args.stage2_epochs} "
        f"(L1 + {args.stage2_nll_lambda:.3e} * NLL + {args.stage2_vgg_lambda:.3e} * VGG)"
    )
    print(
        "Two-stage LR schedule | "
        f"stage1 start={args.stage1_start_lr:.2e}, "
        f"stage2 start={args.stage2_start_lr:.2e}, "
        f"epoch/x10 restart={args.restart_lr:.2e}, "
        f"eta_min={args.eta_min:.2e}"
    )

    _, _, stage1_last_global_epoch = run_stage_training(
        stage_name="stage1",
        stage_epochs=args.stage1_epochs,
        stage_loss_mode="nll_only",
        stage2_nll_lambda=args.stage2_nll_lambda,
        stage2_vgg_lambda=0.0,
        vgg_perceptual=None,
        model=model,
        train_loaders=train_loaders,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        query_chunk_size=args.query_chunk_size,
        grad_clip=args.grad_clip,
        max_train_steps=args.max_train_steps,
        max_valid_steps=args.max_valid_steps,
        save_dir=save_dir,
        args=args,
        log_path=log_path,
        global_epoch_start=1,
        scale_to_index=scale_to_index,
        num_scales=num_scales,
        best_tracker=best_tracker,
        stage_scheduler_config=stage1_scheduler_config,
    )

    stage1_latest_resume_path = save_dir / "stage1_latest_resume.pt"
    if stage1_latest_resume_path.exists():
        stage1_latest_epoch, stage1_tracked_best_nll = load_model_checkpoint_only(
            checkpoint_path=stage1_latest_resume_path,
            model=model,
            device=device,
        )
        print(
            "Stage1 complete. Loaded model-only checkpoint for Stage2: "
            f"{stage1_latest_resume_path} "
            f"(epoch={stage1_latest_epoch}, tracked_best_nll={stage1_tracked_best_nll:.4f})"
        )
    else:
        print("Stage1 latest checkpoint not found. Stage2 starts from current model state.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.stage2_start_lr, weight_decay=args.weight_decay
    )
    scheduler = build_cosine_scheduler(
        optimizer=optimizer,
        total_steps=1,
        eta_min=args.eta_min,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and args.amp_dtype == "fp16"),
    )
    print(
        "Stage2 optimizer reset. "
        f"Epoch 1 mixed segment starts at {args.stage2_start_lr:.2e}; "
        f"epoch/x10-only restarts use {args.restart_lr:.2e}; "
        f"eta_min={args.eta_min:.2e}"
    )

    vgg_perceptual = None
    if args.stage2_vgg_lambda > 0:
        vgg_perceptual = build_vgg19_relu34_perceptual(device)
        print(
            "Stage2 perceptual loss: VGG19 relu3_4 enabled "
            f"(lambda={args.stage2_vgg_lambda:.3e})"
        )

    stage2_best_primary, _, _ = run_stage_training(
        stage_name="stage2",
        stage_epochs=args.stage2_epochs,
        stage_loss_mode="l1_plus_nll",
        stage2_nll_lambda=args.stage2_nll_lambda,
        stage2_vgg_lambda=args.stage2_vgg_lambda,
        vgg_perceptual=vgg_perceptual,
        model=model,
        train_loaders=train_loaders,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        query_chunk_size=args.query_chunk_size,
        grad_clip=args.grad_clip,
        max_train_steps=args.max_train_steps,
        max_valid_steps=args.max_valid_steps,
        save_dir=save_dir,
        args=args,
        log_path=log_path,
        global_epoch_start=stage1_last_global_epoch + 1,
        scale_to_index=scale_to_index,
        num_scales=num_scales,
        best_tracker=best_tracker,
        stage_scheduler_config=stage2_scheduler_config,
    )

    print(
        "Two-stage training complete. "
        f"Stage2 best x10 mean_data_range_psnr={stage2_best_primary:.2f}"
    )
    print(f"Artifacts saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
