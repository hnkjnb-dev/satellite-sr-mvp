from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import tifffile
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

NORM_GROUP_MAP: Mapping[int, int] = {255: 0, 16383: 1}
ROBUST_CLAMP_QUANTILE_14BIT = 0.995
MIN_DYNAMIC_NORM_VALUE = 1e-6
LOG1P_KNEE_C = 1024.0
LOG1P_KNEE_LOG_MAX = math.log1p(16383.0 / LOG1P_KNEE_C)


@dataclass(frozen=True)
class PairRecord:
    category: str
    file_name: str
    lr_path: Path
    hr_path: Path
    scale: int
    lr_resolution: float
    normalize_value: int


def _to_tensor_hwc(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={tuple(image.shape)}")
    return image.permute(2, 0, 1).contiguous()


def _robust_upper_quantile(image_chw: torch.Tensor, q: float) -> float:
    if not (0.0 < q < 1.0):
        raise ValueError(f"Quantile must be in (0, 1), got {q}")
    upper = torch.quantile(image_chw.reshape(-1), q=q)
    upper_value = float(upper.item())
    if not math.isfinite(upper_value):
        return 0.0
    return max(upper_value, 0.0)


def normalize_pair_by_lr_max(
    lr: torch.Tensor,
    hr: torch.Tensor,
    normalize_value: int,
    robust_clamp_quantile_14bit: float = ROBUST_CLAMP_QUANTILE_14BIT,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    # Legacy signature is kept, but normalization now depends on bit depth.
    _ = robust_clamp_quantile_14bit
    lr_chw = _to_tensor_hwc(lr.to(torch.float32))
    hr_chw = _to_tensor_hwc(hr.to(torch.float32))

    norm_value = int(normalize_value)
    max_value = float(norm_value)
    lr_chw = torch.clamp(lr_chw, min=0.0, max=max_value)
    hr_chw = torch.clamp(hr_chw, min=0.0, max=max_value)

    if norm_value == 255:
        lr_chw = lr_chw / 255.0
        hr_chw = hr_chw / 255.0
        sample_norm_value = 255.0
    elif norm_value == 16383:
        lr_chw = torch.log1p(lr_chw / LOG1P_KNEE_C) / LOG1P_KNEE_LOG_MAX
        hr_chw = torch.log1p(hr_chw / LOG1P_KNEE_C) / LOG1P_KNEE_LOG_MAX
        sample_norm_value = float(LOG1P_KNEE_C)
    else:
        raise ValueError(f"Unsupported normalize_value={normalize_value}")

    # Kept for backward-compatible batch keys and logs.
    lr_clip_value = max_value
    hr_clip_value = max_value
    return lr_chw, hr_chw, sample_norm_value, lr_clip_value, hr_clip_value


def normalize_lr_by_lr_max(
    lr: torch.Tensor,
    normalize_value: int,
    robust_clamp_quantile_14bit: float = ROBUST_CLAMP_QUANTILE_14BIT,
) -> Tuple[torch.Tensor, float, float]:
    # Legacy signature is kept, but normalization now depends on bit depth.
    _ = robust_clamp_quantile_14bit
    lr_chw = _to_tensor_hwc(lr.to(torch.float32))
    norm_value = int(normalize_value)
    lr_clip_value = float(norm_value)
    lr_chw = torch.clamp(lr_chw, min=0.0, max=lr_clip_value)
    if norm_value == 255:
        lr_chw = lr_chw / 255.0
        sample_norm_value = 255.0
    elif norm_value == 16383:
        lr_chw = torch.log1p(lr_chw / LOG1P_KNEE_C) / LOG1P_KNEE_LOG_MAX
        sample_norm_value = float(LOG1P_KNEE_C)
    else:
        raise ValueError(f"Unsupported normalize_value={normalize_value}")

    # Kept for backward-compatible batch keys and logs.
    return lr_chw, sample_norm_value, lr_clip_value


def iter_filtered_csv_rows(
    csv_path: str | Path,
    categories: Sequence[str] | None = None,
    max_per_category: int = 1000,
) -> Iterator[dict]:
    csv_path = Path(csv_path)
    target_categories = set(categories) if categories else None
    counts: Dict[str, int] = defaultdict(int)

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = row["category"]
            if target_categories is not None and category not in target_categories:
                continue
            if max_per_category > 0 and counts[category] >= max_per_category:
                if (
                    target_categories is not None
                    and all(counts[c] >= max_per_category for c in target_categories)
                ):
                    break
                continue
            counts[category] += 1
            yield row


def summarize_filtered_csv(
    csv_path: str | Path,
    categories: Sequence[str] | None = None,
    max_per_category: int = 1000,
) -> Tuple[int, Dict[str, int], Dict[int, int]]:
    by_category: Dict[str, int] = defaultdict(int)
    by_scale: Dict[int, int] = defaultdict(int)
    total = 0

    for row in iter_filtered_csv_rows(
        csv_path=csv_path,
        categories=categories,
        max_per_category=max_per_category,
    ):
        total += 1
        by_category[row["category"]] += 1
        by_scale[int(row["scale"])] += 1

    return total, dict(sorted(by_category.items())), dict(sorted(by_scale.items()))


def load_records_from_csv(
    csv_path: str | Path,
    dataset_root: str | Path,
    max_per_category: int = 1000,
    categories: Sequence[str] | None = None,
) -> List[PairRecord]:
    dataset_root = Path(dataset_root)
    records: List[PairRecord] = []

    for row in iter_filtered_csv_rows(
        csv_path=csv_path,
        categories=categories,
        max_per_category=max_per_category,
    ):
        category = row["category"]
        file_name = row["file_name"]
        lr_path = dataset_root / row["LR_folder"] / file_name
        hr_path = dataset_root / row["HR_folder"] / file_name
        normalize_value = int(row["normalize_value"])
        if normalize_value not in NORM_GROUP_MAP:
            raise ValueError(
                f"Unsupported normalize_value={normalize_value} in {csv_path}."
            )

        records.append(
            PairRecord(
                category=category,
                file_name=file_name,
                lr_path=lr_path,
                hr_path=hr_path,
                scale=int(row["scale"]),
                lr_resolution=float(row["LR_resolution(m_pixel)"]),
                normalize_value=normalize_value,
            )
        )

    return records


def group_records_by_scale(records: Iterable[PairRecord]) -> Dict[int, List[PairRecord]]:
    grouped: Dict[int, List[PairRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.scale].append(rec)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


class SatellitePairDataset(Dataset[Dict[str, torch.Tensor | str | int | float]]):
    def __init__(self, records: Sequence[PairRecord]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("Dataset received zero records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int | float]:
        rec = self.records[index]

        lr = torch.from_numpy(tifffile.imread(rec.lr_path)).to(torch.float32)
        hr = torch.from_numpy(tifffile.imread(rec.hr_path)).to(torch.float32)
        lr, hr, sample_norm_value, lr_clip_value, hr_clip_value = normalize_pair_by_lr_max(
            lr=lr,
            hr=hr,
            normalize_value=rec.normalize_value,
        )

        return {
            "lr": lr,
            "hr": hr,
            "scale": rec.scale,
            "lr_resolution": torch.tensor(rec.lr_resolution, dtype=torch.float32),
            "norm_group": torch.tensor(
                NORM_GROUP_MAP[rec.normalize_value], dtype=torch.long
            ),
            "normalize_value": rec.normalize_value,
            "sample_normalize_value": torch.tensor(sample_norm_value, dtype=torch.float32),
            "lr_clip_value": torch.tensor(lr_clip_value, dtype=torch.float32),
            "hr_clip_value": torch.tensor(hr_clip_value, dtype=torch.float32),
            "category": rec.category,
            "file_name": rec.file_name,
        }


class SatelliteCSVStreamDataset(IterableDataset[Dict[str, torch.Tensor | str | int | float]]):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        scale: int,
        categories: Sequence[str] | None,
        max_per_category: int,
        sample_count: int,
        shuffle: bool = False,
        shuffle_buffer_size: int = 8192,
        seed: int = 42,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.scale = int(scale)
        self.categories = list(categories) if categories else None
        self.max_per_category = int(max_per_category)
        self.sample_count = int(sample_count)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = int(max(shuffle_buffer_size, 1))
        self.seed = int(seed)
        self._iter_calls = 0
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive for streaming dataset.")

    def __len__(self) -> int:
        return self.sample_count

    def _iter_round_robin_rows(self, rng_seed: int) -> Iterator[dict]:
        category_order = list(self.categories) if self.categories else []
        if not category_order:
            return

        if self.shuffle and len(category_order) > 1:
            rng = random.Random(rng_seed)
            start = rng.randrange(len(category_order))
            category_order = category_order[start:] + category_order[:start]

        per_category_iters = {
            category: iter_filtered_csv_rows(
                csv_path=self.csv_path,
                categories=[category],
                max_per_category=self.max_per_category,
            )
            for category in category_order
        }

        def next_row_for_scale(category: str) -> dict | None:
            for row in per_category_iters[category]:
                if int(row["scale"]) == self.scale:
                    return row
            return None

        buffered = {category: next_row_for_scale(category) for category in category_order}
        active_categories = [
            category for category in category_order if buffered[category] is not None
        ]

        while active_categories:
            next_active_categories: List[str] = []
            for category in active_categories:
                row = buffered[category]
                if row is None:
                    continue
                yield row
                nxt = next_row_for_scale(category)
                if nxt is not None:
                    buffered[category] = nxt
                    next_active_categories.append(category)
            active_categories = next_active_categories

    def _iter_rows_for_worker(
        self,
        worker_id: int,
        num_workers: int,
        rng_seed: int,
    ) -> Iterator[dict]:
        def selected_rows() -> Iterator[dict]:
            selected_index = 0
            for row in self._iter_round_robin_rows(rng_seed=rng_seed):
                if selected_index % num_workers == worker_id:
                    yield row
                selected_index += 1

        row_iter = selected_rows()
        if not self.shuffle or self.shuffle_buffer_size <= 1:
            yield from row_iter
            return

        # Streaming shuffle with bounded memory: approximately random within buffer.
        shuffle_seed = (
            int(rng_seed) * 1_000_003 + worker_id * 97_873 + self.scale * 8_191
        )
        rng = random.Random(shuffle_seed)
        buffer: List[dict] = []
        for row in row_iter:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(row)
                continue
            pick_idx = rng.randrange(len(buffer))
            yield buffer[pick_idx]
            buffer[pick_idx] = row

        while buffer:
            pick_idx = rng.randrange(len(buffer))
            yield buffer.pop(pick_idx)

    def _row_to_sample(self, row: dict) -> Dict[str, torch.Tensor | str | int | float]:
        file_name = row["file_name"]
        lr_path = self.dataset_root / row["LR_folder"] / file_name
        hr_path = self.dataset_root / row["HR_folder"] / file_name
        normalize_value = int(row["normalize_value"])
        if normalize_value not in NORM_GROUP_MAP:
            raise ValueError(
                f"Unsupported normalize_value={normalize_value} for {file_name}."
            )

        lr = torch.from_numpy(tifffile.imread(lr_path)).to(torch.float32)
        hr = torch.from_numpy(tifffile.imread(hr_path)).to(torch.float32)
        lr, hr, sample_norm_value, lr_clip_value, hr_clip_value = normalize_pair_by_lr_max(
            lr=lr,
            hr=hr,
            normalize_value=normalize_value,
        )

        return {
            "lr": lr,
            "hr": hr,
            "scale": int(row["scale"]),
            "lr_resolution": torch.tensor(
                float(row["LR_resolution(m_pixel)"]), dtype=torch.float32
            ),
            "norm_group": torch.tensor(
                NORM_GROUP_MAP[normalize_value], dtype=torch.long
            ),
            "normalize_value": normalize_value,
            "sample_normalize_value": torch.tensor(sample_norm_value, dtype=torch.float32),
            "lr_clip_value": torch.tensor(lr_clip_value, dtype=torch.float32),
            "hr_clip_value": torch.tensor(hr_clip_value, dtype=torch.float32),
            "category": row["category"],
            "file_name": file_name,
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor | str | int | float]]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
            rng_seed = self.seed + self._iter_calls
            self._iter_calls += 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Ensure all workers see the same round-robin stream, then shard by index.
            rng_seed = int(worker_info.seed - worker_info.id)

        row_iter = self._iter_rows_for_worker(
            worker_id=worker_id,
            num_workers=num_workers,
            rng_seed=rng_seed,
        )
        for row in row_iter:
            yield self._row_to_sample(row)
