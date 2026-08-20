"""Builds a deterministic (not cherry-picked) set of representative examples
per language, merging source/reference/prediction across all three
baselines for direct side-by-side comparison.

Selection is a fixed-seed random sample (configs/eval.yaml:
qualitative_analysis.seed/samples_per_language) over each language's full
500-example test set -- not hand-picked "good" outputs.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_csv_rows  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    eval_cfg = load_yaml("eval.yaml")
    data_cfg = load_yaml("data.yaml")
    qa_cfg = eval_cfg["qualitative_analysis"]

    zero_shot_rows = _read_csv(repo_path("results/qualitative_samples/baseline_zero_shot_predictions.csv"))
    lead_n_rows = _read_csv(repo_path("results/qualitative_samples/baseline_lead_n_predictions.csv"))

    zero_shot_by_id = {row["id"]: row for row in zero_shot_rows}
    lead1_by_id = {row["id"]: row for row in lead_n_rows if row["n_sentences"] == "1"}
    lead3_by_id = {row["id"]: row for row in lead_n_rows if row["n_sentences"] == "3"}

    languages = [lang["config_name"] for lang in data_cfg["dataset"]["languages"]]
    samples_per_language = qa_cfg["samples_per_language"]
    seed = qa_cfg["seed"]

    representative_rows = []
    for lang in languages:
        lang_ids = [row["id"] for row in zero_shot_rows if row["language"] == lang]
        rng = random.Random(seed)
        selected_ids = rng.sample(lang_ids, min(samples_per_language, len(lang_ids)))

        for example_id in selected_ids:
            zs = zero_shot_by_id[example_id]
            l1 = lead1_by_id.get(example_id, {})
            l3 = lead3_by_id.get(example_id, {})
            representative_rows.append(
                {
                    "id": example_id,
                    "language": lang,
                    "source_article": zs["source_article"],
                    "reference_summary": zs["reference_summary"],
                    "zero_shot_prediction": zs["generated_summary"],
                    "lead_1_prediction": l1.get("generated_summary", ""),
                    "lead_3_prediction": l3.get("generated_summary", ""),
                }
            )

    out_path = save_csv_rows(
        representative_rows, repo_path("results/qualitative_samples/representative_examples.csv")
    )
    print(f"Representative examples written to: {out_path} ({len(representative_rows)} rows, "
          f"{samples_per_language}/language, seed={seed})")


if __name__ == "__main__":
    main()
