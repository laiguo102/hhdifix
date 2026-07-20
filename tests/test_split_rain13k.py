import csv
import json

import pytest

from src.split_rain13k import split_rain13k, target_counts


def _make_dataset(root, group_sizes):
    input_dir = root / "input"
    target_dir = root / "target"
    input_dir.mkdir(parents=True)
    target_dir.mkdir()
    sample_id = 1
    for group_index, group_size in enumerate(group_sizes):
        target_bytes = f"clean-background-{group_index}".encode()
        for _ in range(group_size):
            (input_dir / f"{sample_id}.jpg").write_bytes(f"rain-{sample_id}".encode())
            (target_dir / f"{sample_id}.jpg").write_bytes(target_bytes)
            sample_id += 1


def test_target_counts_uses_largest_remainder():
    assert target_counts(13_711, (0.6, 0.3, 0.1)) == {
        "unet_train": 8_227,
        "difix_train": 4_113,
        "validation": 1_371,
    }


def test_grouped_split_is_complete_reproducible_and_leakage_free(tmp_path):
    dataset_root = tmp_path / "Rain13K"
    _make_dataset(dataset_root, [4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    first = split_rain13k(dataset_root, tmp_path / "split-a", seed=3407)
    second = split_rain13k(dataset_root, tmp_path / "split-b", seed=3407)

    assert sum(first["actual_counts"].values()) == 20
    assert first["target_hash_leakage"] is False
    assert first["actual_counts"] == second["actual_counts"]
    assert (tmp_path / "split-a" / "split_manifest.csv").read_bytes() == (
        tmp_path / "split-b" / "split_manifest.csv"
    ).read_bytes()

    with (tmp_path / "split-a" / "split_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert len({row["sample_id"] for row in rows}) == 20
    hash_owners = {}
    for row in rows:
        hash_owners.setdefault(row["target_sha256"], set()).add(row["split"])
    assert all(len(owners) == 1 for owners in hash_owners.values())

    summary = json.loads((tmp_path / "split-a" / "split_summary.json").read_text())
    assert summary["total_pairs"] == 20
    assert summary["total_target_groups"] == len(hash_owners)


def test_split_rejects_mismatched_stems(tmp_path):
    dataset_root = tmp_path / "Rain13K"
    (dataset_root / "input").mkdir(parents=True)
    (dataset_root / "target").mkdir()
    (dataset_root / "input" / "1.jpg").write_bytes(b"rain")
    (dataset_root / "target" / "2.jpg").write_bytes(b"clean")

    with pytest.raises(ValueError, match="stems do not match"):
        split_rain13k(dataset_root, tmp_path / "splits")


def test_split_refuses_to_overwrite_existing_manifests(tmp_path):
    dataset_root = tmp_path / "Rain13K"
    _make_dataset(dataset_root, [1] * 10)
    output_dir = tmp_path / "splits"
    split_rain13k(dataset_root, output_dir)

    with pytest.raises(FileExistsError, match="--overwrite"):
        split_rain13k(dataset_root, output_dir)
