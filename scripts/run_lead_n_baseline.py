"""BASELINE 2: Lead-N extractive baseline (deterministic, no model).

GENERATION ONLY -- this script does not compute ROUGE or BERTScore. It
extracts the first N sentences (src/eval/sentence_split.py) for N=1 and
N=3, and saves the predictions. Metrics are computed separately by
scripts/evaluate_predictions.py.

N is a configured parameter (configs/eval.yaml: lead_n.values) evaluated
directly on the test set without being tuned against it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_xlsum import load_and_clean_split  # noqa: E402
from src.eval.sentence_split import lead_n_summary  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_csv_rows, save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def run_qa_smoke_test(languages_rows, n_values) -> None:
    print("=== QA smoke test: Lead-1 and Lead-3 on one example per language ===")
    for lang, iso, rows in languages_rows:
        row = rows[0]
        for n in n_values:
            summary = lead_n_summary(row["text"], n)
            print(f"  {lang} lead-{n}: id={row['id']} non_empty={bool(summary.strip())}")
            print(f"    output: {summary[:150]!r}")
            if not summary.strip():
                raise RuntimeError(f"QA smoke test failed: empty lead-{n} output for {lang}")
    print()


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")
    eval_cfg = load_yaml("eval.yaml")

    set_seed(base_cfg["seed"])

    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    test_n = data_cfg["dataset"]["split_sizes"]["test"]
    languages = data_cfg["dataset"]["languages"]
    n_values = eval_cfg["lead_n"]["values"]

    print("=== BASELINE 2: Lead-N extractive (GENERATION ONLY) ===")
    print(f"N values (not tuned on test): {n_values}\n")

    languages_rows = []
    for lang in languages:
        rows, _ = load_and_clean_split(
            lang["config_name"], lang["iso_code"], "test", test_n, seed, revision
        )
        assert len(rows) == test_n
        languages_rows.append((lang["config_name"], lang["iso_code"], rows))
        print(f"Loaded {lang['config_name']} test split: {len(rows)} examples (same test set as zero-shot baseline)")
    print()

    run_qa_smoke_test(languages_rows, n_values)

    all_predictions_rows = []
    timing_report = {}

    for n in n_values:
        print(f"--- Lead-{n} ---")
        for lang, iso, rows in languages_rows:
            t0 = time.time()
            predictions = [lead_n_summary(r["text"], n) for r in rows]
            elapsed = time.time() - t0
            timing_report[f"lead_{n}_{lang}"] = {
                "total_inference_time_sec": elapsed,
                "avg_inference_time_sec": elapsed / len(rows),
            }
            print(f"  {lang}: {len(rows)} examples extracted ({elapsed:.3f}s)")

            for row, pred in zip(rows, predictions):
                all_predictions_rows.append(
                    {
                        "id": row["id"],
                        "language": lang,
                        "model": f"lead_{n}",
                        "n_sentences": n,
                        "source_article": row["text"],
                        "reference_summary": row["summary"],
                        "generated_summary": pred,
                    }
                )

    pred_path = save_csv_rows(
        all_predictions_rows, repo_path("results/qualitative_samples/baseline_lead_n_predictions.csv")
    )
    save_json_report(
        {
            "n_values": n_values,
            "timing_by_language_and_n": timing_report,
            "note": "Predictions only -- no ROUGE/BERTScore computed here. Run "
            "scripts/evaluate_predictions.py against the saved predictions CSV for metrics.",
        },
        repo_path("results/qa/baseline_lead_n_generation_report.json"),
    )

    print(f"\nPredictions written to: {pred_path}")
    print("Next step: python scripts/evaluate_predictions.py --predictions "
          "results/qualitative_samples/baseline_lead_n_predictions.csv --group-by n_sentences")


if __name__ == "__main__":
    main()
