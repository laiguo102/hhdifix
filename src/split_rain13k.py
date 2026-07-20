"""Create leakage-safe Rain13K manifests for the UNet -> DiFix pipeline.

The split is performed by the SHA-256 digest of each clean target file.  All
rainy variants that share an identical clean target therefore stay in the same
subset.  The script writes manifests only; it never moves or copies images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SPLIT_NAMES = ("unet_train", "difix_train", "validation")
DEFAULT_RATIOS = (0.6, 0.3, 0.1)
DEFAULT_SEED = 3407


@dataclass(frozen=True)
class Sample:
    sample_id: str
    input_path: Path
    target_path: Path
    target_sha256: str


def _natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _files_by_stem(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in files:
            raise ValueError(
                f"Duplicate stem '{path.stem}' in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
        files[path.stem] = path
    if not files:
        raise ValueError(f"No supported images found in: {directory}")
    return files


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_samples(dataset_root: str | Path) -> list[Sample]:
    root = Path(dataset_root).resolve()
    inputs = _files_by_stem(root / "input")
    targets = _files_by_stem(root / "target")
    input_stems = set(inputs)
    target_stems = set(targets)
    if input_stems != target_stems:
        missing_targets = sorted(input_stems - target_stems, key=_natural_key)
        missing_inputs = sorted(target_stems - input_stems, key=_natural_key)
        raise ValueError(
            "Rain13K input/target stems do not match; "
            f"missing targets={missing_targets[:10]}, missing inputs={missing_inputs[:10]}"
        )

    return [
        Sample(
            sample_id=stem,
            input_path=inputs[stem],
            target_path=targets[stem],
            target_sha256=_sha256(targets[stem]),
        )
        for stem in sorted(input_stems, key=_natural_key)
    ]


def target_counts(total: int, ratios: Sequence[float]) -> Dict[str, int]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} ratios, got {len(ratios)}")
    if any(value < 0 for value in ratios):
        raise ValueError(f"Ratios must be non-negative, got {ratios}")
    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    exact = [total * value for value in ratios]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    fractional_order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def group_samples(samples: Iterable[Sample]) -> Dict[str, list[Sample]]:
    groups: Dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.target_sha256, []).append(sample)
    return groups


def assign_groups(
    groups: Mapping[str, Sequence[Sample]],
    requested_counts: Mapping[str, int],
    seed: int,
) -> Dict[str, list[Sample]]:
    """Assign whole target groups while staying as close as possible to ratios."""

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    # Stable sorting preserves the seeded random order among equal-sized groups.
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    assigned: Dict[str, list[Sample]] = {name: [] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    split_priority = list(SPLIT_NAMES)
    rng.shuffle(split_priority)
    priority = {name: index for index, name in enumerate(split_priority)}

    for _, members in group_items:
        size = len(members)
        deficits = {name: requested_counts[name] - counts[name] for name in SPLIT_NAMES}
        fitting = [name for name in SPLIT_NAMES if deficits[name] >= size]
        if fitting:
            chosen = max(
                fitting,
                key=lambda name: (
                    deficits[name] / max(requested_counts[name], 1),
                    -priority[name],
                ),
            )
        else:
            chosen = min(
                SPLIT_NAMES,
                key=lambda name: (
                    max(0, counts[name] + size - requested_counts[name])
                    / max(requested_counts[name], 1),
                    priority[name],
                ),
            )
        assigned[chosen].extend(members)
        counts[chosen] += size

    for samples in assigned.values():
        samples.sort(key=lambda sample: _natural_key(sample.sample_id))
    return assigned


def validate_assignment(
    assigned: Mapping[str, Sequence[Sample]], expected_total: int
) -> None:
    all_ids: list[str] = []
    digest_owner: Dict[str, str] = {}
    for split, samples in assigned.items():
        for sample in samples:
            all_ids.append(sample.sample_id)
            owner = digest_owner.setdefault(sample.target_sha256, split)
            if owner != split:
                raise AssertionError(
                    f"Target group {sample.target_sha256} appears in both {owner} and {split}"
                )
    if len(all_ids) != expected_total or len(set(all_ids)) != expected_total:
        raise AssertionError(
            f"Assignment is not one-to-one: rows={len(all_ids)}, "
            f"unique IDs={len(set(all_ids))}, expected={expected_total}"
        )


def write_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    assigned: Mapping[str, Sequence[Sample]],
    requested_counts: Mapping[str, int],
    ratios: Sequence[float],
    seed: int,
    overwrite: bool = False,
) -> dict:
    root = Path(dataset_root).resolve()
    destination = Path(output_dir).resolve()
    filenames = [*(f"{name}.txt" for name in SPLIT_NAMES), "split_manifest.csv", "split_summary.json"]
    existing = [destination / name for name in filenames if (destination / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Split outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)

    for split in SPLIT_NAMES:
        text = "".join(f"{sample.sample_id}\n" for sample in assigned[split])
        (destination / f"{split}.txt").write_text(text, encoding="utf-8")

    manifest_path = destination / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "input_path", "target_path", "target_sha256", "split"),
        )
        writer.writeheader()
        for split in SPLIT_NAMES:
            for sample in assigned[split]:
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "input_path": sample.input_path.relative_to(root).as_posix(),
                        "target_path": sample.target_path.relative_to(root).as_posix(),
                        "target_sha256": sample.target_sha256,
                        "split": split,
                    }
                )

    actual_counts = {name: len(assigned[name]) for name in SPLIT_NAMES}
    group_counts = {
        name: len({sample.target_sha256 for sample in assigned[name]}) for name in SPLIT_NAMES
    }
    total = sum(actual_counts.values())
    summary = {
        "dataset_root": str(root),
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "total_pairs": total,
        "total_target_groups": sum(group_counts.values()),
        "requested_counts": dict(requested_counts),
        "actual_counts": actual_counts,
        "actual_ratios": {name: actual_counts[name] / total for name in SPLIT_NAMES},
        "target_group_counts": group_counts,
        "target_hash_leakage": False,
    }
    (destination / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def split_rain13k(
    dataset_root: str | Path,
    output_dir: str | Path,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict:
    samples = collect_samples(dataset_root)
    requested = target_counts(len(samples), ratios)
    groups = group_samples(samples)
    assigned = assign_groups(groups, requested, seed)
    validate_assignment(assigned, len(samples))
    return write_manifests(
        dataset_root,
        output_dir,
        assigned,
        requested,
        ratios,
        seed,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Rain13K by clean-target hash for UNet and DiFix training."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Directory containing aligned input/ and target/ folders.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("UNET", "DIFIX", "VALIDATION"),
        default=DEFAULT_RATIOS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = split_rain13k(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        ratios=args.ratios,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
