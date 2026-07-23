"""Materialize leakage-safe Rain13K views for the UNet -> DiFix pipeline.

The source dataset must already contain aligned, preprocessed 512x512 images in
``input/`` and ``target/``. Split text files contain one filename stem per line.
By default, the script creates hard links for ``difix_train`` and ``validation``
without duplicating image data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_SPLITS = ("difix_train", "validation")
DEFAULT_PROMPT = "remove rain streaks and restore a clean natural image"


def _files_by_stem(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    result: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(
                f"Duplicate image stem '{path.stem}' in {directory}: "
                f"{result[path.stem].name}, {path.name}"
            )
        result[path.stem] = path
    if not result:
        raise ValueError(f"No supported images found in: {directory}")
    return result


def read_split_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    if not sample_ids:
        raise ValueError(f"Split manifest is empty: {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Split manifest contains duplicate sample IDs: {path}")
    invalid = [
        sample_id
        for sample_id in sample_ids
        if Path(sample_id).name != sample_id or Path(sample_id).suffix
    ]
    if invalid:
        raise ValueError(
            "Split entries must be bare filename stems; invalid entries: "
            + ", ".join(invalid[:8])
        )
    return sample_ids


def _same_contents(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(1024 * 1024)
            right_chunk = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _materialize_file(source: Path, destination: Path, mode: str) -> None:
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        if _same_contents(source, destination):
            return
        raise FileExistsError(
            f"Destination exists with different contents: {destination}"
        )
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode}")


def materialize_split(
    processed_root: str | Path,
    split_file: str | Path,
    output_root: str | Path,
    split_name: str,
    mode: str = "hardlink",
) -> int:
    processed_root = Path(processed_root).resolve()
    output_root = Path(output_root).resolve()
    inputs = _files_by_stem(processed_root / "input")
    targets = _files_by_stem(processed_root / "target")
    sample_ids = read_split_ids(Path(split_file))

    missing_inputs = sorted(set(sample_ids) - set(inputs))
    missing_targets = sorted(set(sample_ids) - set(targets))
    if missing_inputs or missing_targets:
        raise ValueError(
            f"Split '{split_name}' references missing files; "
            f"input={missing_inputs[:8]}, target={missing_targets[:8]}"
        )

    input_output = output_root / split_name / "input"
    target_output = output_root / split_name / "target"
    input_output.mkdir(parents=True, exist_ok=True)
    target_output.mkdir(parents=True, exist_ok=True)
    expected_stems = set(sample_ids)
    for role, directory in (("input", input_output), ("target", target_output)):
        existing_stems = {
            path.stem
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        extra = sorted(existing_stems - expected_stems)
        if extra:
            raise ValueError(
                f"Existing {split_name}/{role} contains stems absent from the "
                f"split manifest: {extra[:8]}"
            )

    for sample_id in sample_ids:
        input_source = inputs[sample_id]
        target_source = targets[sample_id]
        _materialize_file(
            input_source,
            input_output / f"{sample_id}{input_source.suffix.lower()}",
            mode,
        )
        _materialize_file(
            target_source,
            target_output / f"{sample_id}{target_source.suffix.lower()}",
            mode,
        )
    for role, directory in (("input", input_output), ("target", target_output)):
        actual_stems = {
            path.stem
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if actual_stems != expected_stems:
            missing = sorted(expected_stems - actual_stems)
            extra = sorted(actual_stems - expected_stems)
            raise RuntimeError(
                f"Materialized {split_name}/{role} does not match its manifest; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
    return len(sample_ids)


def write_difix_config(
    output_root: str | Path,
    config_path: str | Path,
    prompt: str = DEFAULT_PROMPT,
) -> Path:
    output_root = Path(output_root).resolve()
    config_path = Path(config_path)
    config = {
        "train": {
            "image": str(output_root / "difix_train" / "input"),
            "ref_image": str(output_root / "difix_train" / "background"),
            "target_image": str(output_root / "difix_train" / "target"),
            "prompt": prompt,
        },
        "test": {
            "image": str(output_root / "validation" / "input"),
            "ref_image": str(output_root / "validation" / "background"),
            "target_image": str(output_root / "validation" / "target"),
            "prompt": prompt,
        },
    }
    if config_path.exists():
        raise FileExistsError(f"Config already exists: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path


def materialize_views(
    processed_root: str | Path,
    split_dir: str | Path,
    output_root: str | Path,
    splits: Iterable[str] = DEFAULT_SPLITS,
    mode: str = "hardlink",
) -> Dict[str, int]:
    split_dir = Path(split_dir)
    splits = list(splits)
    ids_by_split = {
        split: set(read_split_ids(split_dir / f"{split}.txt")) for split in splits
    }
    for index, first in enumerate(splits):
        for second in splits[index + 1 :]:
            overlap = sorted(ids_by_split[first] & ids_by_split[second])
            if overlap:
                raise ValueError(
                    f"Splits '{first}' and '{second}' overlap: {overlap[:8]}"
                )
    return {
        split: materialize_split(
            processed_root,
            split_dir / f"{split}.txt",
            output_root,
            split,
            mode,
        )
        for split in splits
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Rain13K DiFix train/validation views from split manifests."
    )
    parser.add_argument(
        "--processed_root",
        required=True,
        help="Aligned 512x512 Rain13K root containing input/ and target/.",
    )
    parser.add_argument(
        "--split_dir",
        required=True,
        help="Directory containing difix_train.txt and validation.txt.",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to materialize (default: difix_train validation).",
    )
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument(
        "--config_output",
        default=None,
        help="Optionally write a [background, rainy] DiFix dataset JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = materialize_views(
        args.processed_root,
        args.split_dir,
        args.output_root,
        splits=args.splits,
        mode=args.mode,
    )
    result = {"counts": counts, "output_root": str(Path(args.output_root).resolve())}
    if args.config_output:
        result["config"] = str(
            write_difix_config(args.output_root, args.config_output)
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
