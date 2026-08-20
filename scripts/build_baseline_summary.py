"""Combines baseline_zero_shot.csv and baseline_lead_n.csv (metrics, from
scripts/evaluate_predictions.py) with the generation-time timing already
recorded in results/qa/baseline_*_generation_report.json into the single
requested summary table: results/metrics/baseline_summary.csv.

Columns: Model, Language, ROUGE-1, ROUGE-2, ROUGE-L, BERTScore, Number of
test examples, Inference time, Average inference time/example.

"Overall" ROUGE/BERTScore rows are macro-averages across the three
languages (see src/eval/aggregate.py's docstring) -- an unweighted mean of
the three per-language means, not a pooled average over all 1500 examples,
so English cannot dominate just by being computed first or having examples
processed faster. With the test set balanced at 500/500/500 this happens to
equal a pooled average numerically, but the macro-average is what is
reported and is what stays correct if that balance ever changes.

Overall *timing* (a physical duration, not a quality score) is summed
across languages for total time, and total-time/total-n for the average --
weighting by language would misrepresent wall-clock reality here, unlike
the quality-metric case above.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import repo_path  # noqa: E402
from src.utils.reporting import save_csv_rows  # noqa: E402

MODEL_DISPLAY_NAMES = {
    "zero_shot_mt5_small": "Zero-shot mT5-small",
    "lead_1": "Lead-1",
    "lead_3": "Lead-3",
}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_timing() -> dict[tuple[str, str], dict]:
    """Returns {(model, language): {total_inference_time_sec, avg_inference_time_sec}}."""
    timing: dict[tuple[str, str], dict] = {}

    zs = json.load(open(repo_path("results/qa/baseline_zero_shot_generation_report.json"), encoding="utf-8"))
    for lang, t in zs["timing_by_language"].items():
        timing[("zero_shot_mt5_small", lang)] = t
    total = sum(t["total_inference_time_sec"] for t in zs["timing_by_language"].values())
    n_total = 500 * len(zs["timing_by_language"])
    timing[("zero_shot_mt5_small", "overall")] = {
        "total_inference_time_sec": total,
        "avg_inference_time_sec": total / n_total,
    }

    ln = json.load(open(repo_path("results/qa/baseline_lead_n_generation_report.json"), encoding="utf-8"))
    per_n_totals: dict[str, list[float]] = {}
    for key, t in ln["timing_by_language_and_n"].items():
        # key format: "lead_{n}_{language}"
        parts = key.split("_")
        n_value, lang = parts[1], "_".join(parts[2:])
        model = f"lead_{n_value}"
        timing[(model, lang)] = t
        per_n_totals.setdefault(model, []).append(t["total_inference_time_sec"])
    for model, totals in per_n_totals.items():
        total = sum(totals)
        n_total = 500 * len(totals)
        timing[(model, "overall")] = {
            "total_inference_time_sec": total,
            "avg_inference_time_sec": total / n_total,
        }

    return timing


def main() -> None:
    rows_in = _read_csv(repo_path("results/metrics/baseline_zero_shot.csv")) + _read_csv(
        repo_path("results/metrics/baseline_lead_n.csv")
    )
    timing = _load_timing()

    summary_rows = []
    for row in rows_in:
        t = timing.get((row["model"], row["language"]), {})
        summary_rows.append(
            {
                "Model": MODEL_DISPLAY_NAMES.get(row["model"], row["model"]),
                "Language": row["language"],
                "ROUGE-1": round(float(row["rouge1"]), 4),
                "ROUGE-2": round(float(row["rouge2"]), 4),
                "ROUGE-L": round(float(row["rougeL"]), 4),
                "BERTScore": round(float(row["bertscore_f1"]), 4),
                "Number of test examples": row["n_examples"],
                "Inference time (sec)": round(t.get("total_inference_time_sec", 0.0), 3),
                "Average inference time/example (sec)": round(t.get("avg_inference_time_sec", 0.0), 5),
            }
        )

    out_path = save_csv_rows(summary_rows, repo_path("results/metrics/baseline_summary.csv"))
    print(f"Summary written to: {out_path} ({len(summary_rows)} rows)")
    for row in summary_rows:
        print(f"  {row['Model']:22s} {row['Language']:10s} "
              f"R1={row['ROUGE-1']:.4f} R2={row['ROUGE-2']:.4f} RL={row['ROUGE-L']:.4f} "
              f"BS={row['BERTScore']:.4f} n={row['Number of test examples']} "
              f"time={row['Inference time (sec)']}s")


if __name__ == "__main__":
    main()
