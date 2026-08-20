"""High-level, validated XL-Sum loading for English/Hindi/Telugu.

Builds on src/data/xlsum_loader.py (raw Parquet access) by adding, in this
order:

  1. cleaning (drop rows with missing fields / empty article / empty
     summary) applied to the FULL split pool, BEFORE sampling -- so the
     configured split sizes (2000/300/500) are exactly what you get back.
     Dataset size never silently shrinks because a handful of examples
     turned out to be invalid; if there aren't enough valid rows to satisfy
     the configured size, this raises loudly instead.
  2. deterministic sampling (same rng.sample() approach as xlsum_loader,
     seeded from configs/data.yaml's sampling_seed).
  3. tagging every example with its language (config_name and ISO code).
  4. a train/validation/test id-overlap check per language.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from src.data.xlsum_loader import Split, load_split_table

REQUIRED_FIELDS = ["id", "url", "title", "summary", "text"]


@dataclass
class CleaningStats:
    language: str
    split: str
    raw_count: int
    missing_field_removed: int
    empty_text_removed: int
    empty_summary_removed: int
    valid_count: int
    sampled_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_row(row: dict) -> str | None:
    """Returns a removal reason, or None if the row is valid."""
    for field_name in REQUIRED_FIELDS:
        if field_name not in row or row[field_name] is None:
            return "missing_field"
    if not str(row["text"]).strip():
        return "empty_text"
    if not str(row["summary"]).strip():
        return "empty_summary"
    return None


def load_and_clean_split(
    config_name: str,
    iso_code: str,
    split: Split,
    n: int,
    seed: int,
    revision: str,
) -> tuple[list[dict], CleaningStats]:
    table = load_split_table(config_name, split, revision)
    rows = table.to_pylist()
    raw_count = len(rows)

    removal_counts = {"missing_field": 0, "empty_text": 0, "empty_summary": 0}
    valid_rows = []
    for row in rows:
        reason = _validate_row(row)
        if reason is None:
            valid_rows.append(row)
        else:
            removal_counts[reason] += 1

    if n > len(valid_rows):
        raise ValueError(
            f"{config_name}/{split}: only {len(valid_rows)} valid rows available "
            f"(of {raw_count} raw) after cleaning, but {n} were requested. "
            "Refusing to silently return a smaller sample -- lower the configured "
            "split size or investigate the data quality issue."
        )

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(valid_rows)), n))
    sampled = [dict(valid_rows[i]) for i in idx]
    for row in sampled:
        row["language"] = config_name
        row["language_iso"] = iso_code

    stats = CleaningStats(
        language=config_name,
        split=split,
        raw_count=raw_count,
        missing_field_removed=removal_counts["missing_field"],
        empty_text_removed=removal_counts["empty_text"],
        empty_summary_removed=removal_counts["empty_summary"],
        valid_count=len(valid_rows),
        sampled_count=len(sampled),
    )
    return sampled, stats


def verify_no_id_overlap(splits: dict[str, list[dict]]) -> dict:
    ids_by_split = {name: [row["id"] for row in rows] for name, rows in splits.items()}

    duplicates_within_split = {name: len(ids) - len(set(ids)) for name, ids in ids_by_split.items()}

    pairwise_overlaps = {}
    split_names = list(ids_by_split)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            overlap = set(ids_by_split[a]) & set(ids_by_split[b])
            pairwise_overlaps[f"{a}_vs_{b}"] = len(overlap)

    return {
        "duplicates_within_split": duplicates_within_split,
        "pairwise_id_overlap_count": pairwise_overlaps,
        "clean": all(v == 0 for v in duplicates_within_split.values())
        and all(v == 0 for v in pairwise_overlaps.values()),
    }


def load_language_bundle(
    config_name: str,
    iso_code: str,
    split_sizes: dict[str, int],
    seed: int,
    revision: str,
) -> dict:
    """Loads, cleans and samples all configured splits for one language and
    checks for id leakage across them.

    Returns {"splits": {...}, "cleaning_stats": {...}, "id_overlap_check": {...}}.
    """
    splits: dict[str, list[dict]] = {}
    cleaning_stats: dict[str, CleaningStats] = {}
    for split, n in split_sizes.items():
        rows, stats = load_and_clean_split(config_name, iso_code, split, n, seed, revision)
        splits[split] = rows
        cleaning_stats[split] = stats

    overlap_check = verify_no_id_overlap(splits)
    if not overlap_check["clean"]:
        raise RuntimeError(
            f"{config_name}: train/validation/test id overlap or in-split "
            f"duplicates detected: {overlap_check}"
        )

    return {
        "splits": splits,
        "cleaning_stats": cleaning_stats,
        "id_overlap_check": overlap_check,
    }


def load_all_languages(data_cfg: dict) -> dict[str, dict]:
    """Loads the full English/Hindi/Telugu bundle from a configs/data.yaml-shaped dict."""
    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    split_sizes = data_cfg["dataset"]["split_sizes"]

    bundles = {}
    for lang in data_cfg["dataset"]["languages"]:
        bundles[lang["config_name"]] = load_language_bundle(
            lang["config_name"], lang["iso_code"], split_sizes, seed, revision
        )
    return bundles
