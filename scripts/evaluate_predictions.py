"""Evaluation stage -- reads a predictions CSV (written by
scripts/run_baseline_zero_shot.py or scripts/run_lead_n_baseline.py) and
computes ROUGE-1/2/L (multilingual SentencePiece tokenizer,
src/eval/rouge_tokenizer.py) and BERTScore (pinned multilingual backbone,
loaded ONCE via bert_score.BERTScorer and reused across every model/language
group in the file -- see src/eval/bertscore_eval.py) for every (model,
language) group present, plus a macro-averaged Overall row per model.

Deliberately separate from generation: this script does no inference beyond
the BERTScore backbone's forward pass, does not touch the summarization
model, and can be re-run against the same predictions CSV to try a
different metric configuration without regenerating anything.

Usage:
  python scripts/evaluate_predictions.py --predictions <csv> --output <csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.aggregate import aggregate_language_scores  # noqa: E402
from src.eval.bertscore_eval import load_bertscorer, score_with_scorer  # noqa: E402
from src.eval.rouge_eval import build_rouge_scorer, score_pairs  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_csv_rows, save_json_report  # noqa: E402

MODEL_NAME = "google/mt5-small"


def _read_predictions(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to a predictions CSV")
    parser.add_argument("--output", required=True, help="Path to write the metrics CSV")
    args = parser.parse_args()

    eval_cfg = load_yaml("eval.yaml")
    iso_by_config_name = {
        lang["config_name"]: lang["iso_code"] for lang in load_yaml("data.yaml")["dataset"]["languages"]
    }

    predictions_path = repo_path(args.predictions) if not Path(args.predictions).is_absolute() else Path(args.predictions)
    rows = _read_predictions(predictions_path)
    print(f"Loaded {len(rows)} predictions from {predictions_path}")

    # Group by model (e.g. zero_shot_mt5_small, lead_1, lead_3), then by language.
    by_model_language: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model_language[row["model"]][row["language"]].append(row)

    print("Building ROUGE scorer (SentencePiece tokenizer) and BERTScore model "
          f"({eval_cfg['bertscore']['model_type']}) -- each loaded ONCE for this whole run...")
    rouge_scorer = build_rouge_scorer(MODEL_NAME)
    bertscorer = load_bertscorer(eval_cfg["bertscore"]["model_type"], eval_cfg["bertscore"]["num_layers"])
    print("Scorers ready.\n")

    all_metrics_rows = []
    for model_name, lang_groups in by_model_language.items():
        print(f"--- Scoring model={model_name} ---")
        per_language_metric_rows: dict[str, list[dict]] = {}
        n_by_language: dict[str, int] = {}

        for lang, lang_rows in lang_groups.items():
            predictions = [r["generated_summary"] for r in lang_rows]
            references = [r["reference_summary"] for r in lang_rows]

            rouge_results = score_pairs(rouge_scorer, predictions, references)
            bs_result = score_with_scorer(bertscorer, predictions, references)

            example_rows = [
                {"rouge1": rr["rouge1"], "rouge2": rr["rouge2"], "rougeL": rr["rougeL"], "bertscore_f1": f1}
                for rr, f1 in zip(rouge_results, bs_result["f1"])
            ]
            per_language_metric_rows[lang] = example_rows
            n_by_language[lang] = len(lang_rows)
            print(f"  {lang}: {len(lang_rows)} examples scored")

        aggregated = aggregate_language_scores(per_language_metric_rows)
        for lang in lang_groups:
            m = aggregated[lang]
            all_metrics_rows.append(
                {
                    "model": model_name,
                    "language": lang,
                    "rouge1": m["rouge1"],
                    "rouge2": m["rouge2"],
                    "rougeL": m["rougeL"],
                    "bertscore_f1": m["bertscore_f1"],
                    "n_examples": n_by_language[lang],
                }
            )
        all_metrics_rows.append(
            {
                "model": model_name,
                "language": "overall",
                "rouge1": aggregated["overall"]["rouge1"],
                "rouge2": aggregated["overall"]["rouge2"],
                "rougeL": aggregated["overall"]["rougeL"],
                "bertscore_f1": aggregated["overall"]["bertscore_f1"],
                "n_examples": sum(n_by_language.values()),
            }
        )

    output_path = repo_path(args.output) if not Path(args.output).is_absolute() else Path(args.output)
    metrics_path = save_csv_rows(all_metrics_rows, output_path)
    save_json_report(
        {"predictions_file": str(predictions_path), "metrics_rows": all_metrics_rows},
        output_path.with_suffix("").parent.parent / "qa" / f"{output_path.stem}_eval_report.json",
    )

    print("\n=== Summary ===")
    for row in all_metrics_rows:
        print(
            f"  {row['model']:22s} {row['language']:10s} rouge1={row['rouge1']:.4f} "
            f"rouge2={row['rouge2']:.4f} rougeL={row['rougeL']:.4f} bertscore={row['bertscore_f1']:.4f}"
        )
    print(f"\nMetrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
