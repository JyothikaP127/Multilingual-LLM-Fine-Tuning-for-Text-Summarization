"""Data pipeline inspection/QA.

Loads the already-cached 2,800-example-per-language XL-Sum subsample through
the real load_xlsum.py -> preprocess.py pipeline (no new downloads), prints
one raw and one tokenized example per language, runs the validation checks
(unicode script, excessive length, label correctness, id leakage), and
writes:

  results/qa/data_pipeline_report.json     -- dataset sizes, cleaning stats,
                                               id-overlap check, validation results
  results/qa/data_pipeline_token_stats.json -- per-language token-length stats
                                               and truncation rates at the
                                               ACTIVE max_source_length (768)
                                               and max_target_length (128)

Does not train anything, does not touch results/qa/truncation_comparison.*
(the 512/768/1024 sweep from the previous step).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_xlsum import load_all_languages  # noqa: E402
from src.data.preprocess import build_tokenizer, decode_labels, preprocess_examples  # noqa: E402
from src.data.token_stats import compute_length_stats, tokenize_lengths  # noqa: E402
from src.data.validation import (  # noqa: E402
    check_excessive_length,
    check_label_correctness,
    contains_expected_script,
    normalize_for_display,
)
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")
    model_cfg = load_yaml("model.yaml")

    set_seed(base_cfg["seed"])

    max_source_length = data_cfg["preprocessing"]["max_source_length"]
    max_target_length = data_cfg["preprocessing"]["max_target_length"]
    print(f"=== Data pipeline inspection (max_source_length={max_source_length}, "
          f"max_target_length={max_target_length}) ===\n")

    tokenizer = build_tokenizer(model_cfg["tokenizer_name_or_path"])

    print("Loading + cleaning + sampling all languages "
          "(reusing cached Parquet shards, no new downloads)...")
    bundles = load_all_languages(data_cfg)  # raises on any id overlap or short pool

    pipeline_report: dict = {"languages": {}}
    token_stats_report: dict = {
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "languages": {},
    }

    for lang, bundle in bundles.items():
        splits = bundle["splits"]
        sizes = {name: len(rows) for name, rows in splits.items()}
        print(f"\n--- {lang} ---")
        print(f"  sizes: {sizes}")
        print(f"  id overlap check: {'CLEAN' if bundle['id_overlap_check']['clean'] else 'PROBLEM'}")

        all_rows = [row for rows in splits.values() for row in rows]

        # --- validation: missing fields / empty already enforced upstream by
        # load_xlsum's cleaning step; re-check here as a QA assertion, plus
        # unicode-script and excessive-length checks that aren't part of
        # cleaning (they're diagnostic signals, not removal criteria).
        script_failures = [
            row["id"] for row in all_rows if not contains_expected_script(row["text"], lang)
        ]
        source_lengths_raw = tokenize_lengths(tokenizer, [row["text"] for row in all_rows])
        target_lengths_raw = tokenize_lengths(tokenizer, [row["summary"] for row in all_rows])
        excessive_length_ids = [
            row["id"]
            for row, length in zip(all_rows, source_lengths_raw)
            if check_excessive_length(length)
        ]
        print(f"  unicode-script check failures: {len(script_failures)} / {len(all_rows)}")
        print(f"  excessively long articles (>5000 tokens): {len(excessive_length_ids)}")

        # --- one raw example ---
        raw_example = splits["train"][0]
        print("  raw example (train[0]):")
        print(f"    id={raw_example['id']} language={raw_example['language']}")
        print(f"    title  : {normalize_for_display(raw_example['title'])}")
        print(f"    text   : {normalize_for_display(raw_example['text'])}")
        print(f"    summary: {normalize_for_display(raw_example['summary'])}")

        # --- preprocess all splits (deterministic tokenization) ---
        processed_by_split = {
            name: preprocess_examples(rows, tokenizer, max_source_length, max_target_length)
            for name, rows in splits.items()
        }
        all_processed = [ex for rows in processed_by_split.values() for ex in rows]

        # --- one processed/tokenized example (same underlying example) ---
        processed_example = processed_by_split["train"][0]
        actual_source_len = sum(processed_example["attention_mask"])
        actual_target_len = sum(1 for t in processed_example["labels"] if t != -100)
        decoded_input = tokenizer.decode(processed_example["input_ids"], skip_special_tokens=True)
        decoded_target = decode_labels(tokenizer, processed_example["labels"])
        print("  processed example (same id, after tokenization):")
        print(f"    source token length (post-truncation, non-pad): {actual_source_len}")
        print(f"    target token length (post-truncation, non-pad): {actual_target_len}")
        print(f"    decoded input (truncated)  : {normalize_for_display(decoded_input)}")
        print(f"    decoded target             : {normalize_for_display(decoded_target)}")

        # --- label-correctness validation ---
        label_check = check_label_correctness(all_processed, tokenizer)
        print(
            f"  label correctness: pad_id_leaked={label_check.pad_id_leaked_into_labels} "
            f"trailing_ignore_consistent={label_check.all_ignore_index_at_pad_positions} "
            f"empty_decoded_targets={label_check.empty_decoded_targets}"
        )

        # --- token-length QA (point 3) ---
        source_stats = compute_length_stats(source_lengths_raw, "source_text", max_source_length)
        target_stats = compute_length_stats(target_lengths_raw, "target_summary", max_target_length)
        print(
            f"  source tokens: mean={source_stats.mean:.1f} median={source_stats.median:.0f} "
            f"p90={source_stats.p90:.0f} p95={source_stats.p95:.0f} max={source_stats.max} "
            f"| truncated={source_stats.truncated_pct:.1f}%"
        )
        print(
            f"  target tokens: mean={target_stats.mean:.1f} median={target_stats.median:.0f} "
            f"p90={target_stats.p90:.0f} p95={target_stats.p95:.0f} max={target_stats.max} "
            f"| truncated={target_stats.truncated_pct:.1f}%"
        )

        pipeline_report["languages"][lang] = {
            "split_sizes": sizes,
            "cleaning_stats": {name: s.to_dict() for name, s in bundle["cleaning_stats"].items()},
            "id_overlap_check": bundle["id_overlap_check"],
            "unicode_script_check_failures": len(script_failures),
            "unicode_script_check_failure_ids": script_failures[:20],
            "excessive_length_count": len(excessive_length_ids),
            "excessive_length_ids": excessive_length_ids[:20],
            "label_correctness_check": label_check.to_dict(),
        }
        token_stats_report["languages"][lang] = {
            "n_examples_all_splits": len(all_rows),
            "source_text": source_stats.to_dict(),
            "target_summary": target_stats.to_dict(),
        }

    report_path = save_json_report(pipeline_report, repo_path("results/qa/data_pipeline_report.json"))
    stats_path = save_json_report(
        token_stats_report, repo_path("results/qa/data_pipeline_token_stats.json")
    )
    print(f"\nPipeline report written to: {report_path}")
    print(f"Token-stats report written to: {stats_path}")


if __name__ == "__main__":
    main()
