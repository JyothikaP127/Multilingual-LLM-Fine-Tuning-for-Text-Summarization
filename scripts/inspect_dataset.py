"""XL-Sum dataset inspection.

For each of English/Hindi/Telugu, this:
  1. Loads the configured balanced subsample (2000 train / 300 val / 500 test)
     from the pinned Parquet revision -- no other rows are downloaded.
  2. Verifies there is no `id` overlap between train/validation/test, and no
     duplicate ids within a split (a subsampling bug could otherwise silently
     leak test examples into training).
  3. Tokenizes every sampled example (source `text` and target `summary`)
     with the real mT5 tokenizer and reports per-language length statistics
     and truncation rates against configs/data.yaml's max_source_length /
     max_target_length.

Writes results/qa/dataset_inspection_report.json and
results/qa/truncation_report.json. Downloads only the subsampled rows, not
the full dataset. Does not train anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.token_stats import compute_length_stats, tokenize_lengths  # noqa: E402
from src.data.xlsum_loader import load_balanced_subsample  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def check_id_overlap(splits: dict[str, list[dict]]) -> dict:
    ids_by_split = {name: [row["id"] for row in rows] for name, rows in splits.items()}

    duplicates_within_split = {
        name: len(ids) - len(set(ids)) for name, ids in ids_by_split.items()
    }

    pairwise_overlaps = {}
    split_names = list(ids_by_split)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            overlap = set(ids_by_split[a]) & set(ids_by_split[b])
            pairwise_overlaps[f"{a}_vs_{b}"] = len(overlap)

    return {
        "duplicates_within_split": duplicates_within_split,
        "pairwise_id_overlap_count": pairwise_overlaps,
        "clean": (
            all(v == 0 for v in duplicates_within_split.values())
            and all(v == 0 for v in pairwise_overlaps.values())
        ),
    }


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")

    set_seed(base_cfg["seed"])

    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    split_sizes = data_cfg["dataset"]["split_sizes"]
    max_source_length = data_cfg["preprocessing"]["max_source_length"]
    max_target_length = data_cfg["preprocessing"]["max_target_length"]
    languages = [lang["config_name"] for lang in data_cfg["dataset"]["languages"]]

    print(f"=== XL-Sum inspection (revision={revision[:12]}...) ===")
    print(f"License: {data_cfg['dataset']['license']}")
    print(f"Split sizes: {split_sizes}")
    print(f"max_source_length={max_source_length}, max_target_length={max_target_length}\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    dataset_report: dict = {
        "revision": revision,
        "license": data_cfg["dataset"]["license"],
        "split_sizes": split_sizes,
        "languages": {},
    }
    truncation_report: dict = {
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "languages": {},
    }

    for lang in languages:
        print(f"--- {lang} ---")
        splits = {
            split: load_balanced_subsample(lang, split, n, seed, revision)
            for split, n in split_sizes.items()
        }

        overlap_result = check_id_overlap(splits)
        print(f"  id overlap check: {'CLEAN' if overlap_result['clean'] else 'PROBLEM FOUND'}")
        if not overlap_result["clean"]:
            print(f"    details: {overlap_result}")

        all_rows = [row for rows in splits.values() for row in rows]
        source_lengths = tokenize_lengths(tokenizer, [row["text"] for row in all_rows])
        target_lengths = tokenize_lengths(tokenizer, [row["summary"] for row in all_rows])

        source_stats = compute_length_stats(source_lengths, "source_text", max_source_length)
        target_stats = compute_length_stats(target_lengths, "target_summary", max_target_length)

        print(
            f"  source tokens: mean={source_stats.mean:.1f} median={source_stats.median:.0f} "
            f"p90={source_stats.p90:.0f} p95={source_stats.p95:.0f} max={source_stats.max} "
            f"| truncated={source_stats.truncated_count} ({source_stats.truncated_pct:.1f}%)"
        )
        print(
            f"  target tokens: mean={target_stats.mean:.1f} median={target_stats.median:.0f} "
            f"p90={target_stats.p90:.0f} p95={target_stats.p95:.0f} max={target_stats.max} "
            f"| truncated={target_stats.truncated_count} ({target_stats.truncated_pct:.1f}%)"
        )

        dataset_report["languages"][lang] = {
            "rows_per_split": {name: len(rows) for name, rows in splits.items()},
            "id_overlap_check": overlap_result,
            "columns": sorted(all_rows[0].keys()) if all_rows else [],
        }
        truncation_report["languages"][lang] = {
            "n_examples_all_splits": len(all_rows),
            "source_text": source_stats.to_dict(),
            "target_summary": target_stats.to_dict(),
        }
        print()

    ds_out = save_json_report(dataset_report, repo_path("results/qa/dataset_inspection_report.json"))
    trunc_out = save_json_report(truncation_report, repo_path("results/qa/truncation_report.json"))

    print(f"Dataset inspection report written to: {ds_out}")
    print(f"Truncation report written to: {trunc_out}")


if __name__ == "__main__":
    main()
