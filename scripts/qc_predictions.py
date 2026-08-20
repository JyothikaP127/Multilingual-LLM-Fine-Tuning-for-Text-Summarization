"""Pre-evaluation quality control on the already-generated prediction files.

Does NOT regenerate anything and does NOT compute ROUGE/BERTScore -- it only
checks that the saved predictions are structurally sound before the
evaluation stage runs:

  1. Exactly 500 examples per language in each baseline.
  2. Prediction ids match the project's canonical test-set ids exactly
     (same seed/revision path as everywhere else in the pipeline).
  3. No missing predictions (every expected id is present).
  4. No empty generated summaries.
  5. Prints 3 deterministic (seeded, not cherry-picked) examples per
     language: source / reference / zero-shot / lead-1 / lead-3.
  6. Flags zero-shot outputs for known red flags (very short, <extra_id_*>
     sentinel tokens, repetition, script mismatch, near-verbatim copying)
     as OBSERVATIONS -- these are reported, not auto-labelled as
     "hallucination" or treated as failures.
"""
from __future__ import annotations

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_xlsum import load_and_clean_split  # noqa: E402
from src.data.validation import contains_expected_script  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_very_short(text: str, min_chars: int = 15) -> bool:
    return len(text.strip()) < min_chars


def _has_sentinel_token(text: str) -> bool:
    return bool(re.search(r"<extra_id_\d+>", text))


def _has_repetition(text: str, min_word_len: int = 2, max_word_frac: float = 0.4) -> bool:
    words = text.split()
    if len(words) < 4:
        return False
    counts = Counter(w for w in words if len(w) >= min_word_len)
    if not counts:
        return False
    most_common_word, count = counts.most_common(1)[0]
    return count / len(words) > max_word_frac


def _is_near_verbatim_copy(prediction: str, source: str, min_len: int = 20) -> bool:
    pred = prediction.strip()
    if len(pred) < min_len:
        return False
    return pred in source


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")

    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    test_n = data_cfg["dataset"]["split_sizes"]["test"]
    languages = [lang["config_name"] for lang in data_cfg["dataset"]["languages"]]

    print("=== QC: reloading canonical test-set ids ===")
    reference_ids = {}
    reference_rows_by_id = {}
    for lang in languages:
        cfg = next(l for l in data_cfg["dataset"]["languages"] if l["config_name"] == lang)
        rows, _ = load_and_clean_split(lang, cfg["iso_code"], "test", test_n, seed, revision)
        reference_ids[lang] = {r["id"] for r in rows}
        for r in rows:
            reference_rows_by_id[(lang, r["id"])] = r
        print(f"  {lang}: {len(rows)} canonical test ids (expected {test_n})")
        assert len(rows) == test_n

    zero_shot_rows = _read_csv(repo_path("results/qualitative_samples/baseline_zero_shot_predictions.csv"))
    lead_n_rows = _read_csv(repo_path("results/qualitative_samples/baseline_lead_n_predictions.csv"))

    qc_report: dict = {"languages": {}}
    all_ok = True

    print("\n=== QC: counts, id match, missing/empty predictions ===")
    for lang in languages:
        zs_lang_rows = [r for r in zero_shot_rows if r["language"] == lang]
        zs_ids = {r["id"] for r in zs_lang_rows}
        count_ok = len(zs_lang_rows) == test_n
        ids_match = zs_ids == reference_ids[lang]
        missing_ids = reference_ids[lang] - zs_ids
        empty_preds = [r["id"] for r in zs_lang_rows if not r["generated_summary"].strip()]

        lead1_lang_rows = [r for r in lead_n_rows if r["language"] == lang and r["n_sentences"] == "1"]
        lead3_lang_rows = [r for r in lead_n_rows if r["language"] == lang and r["n_sentences"] == "3"]
        lead1_ids_match = {r["id"] for r in lead1_lang_rows} == reference_ids[lang]
        lead3_ids_match = {r["id"] for r in lead3_lang_rows} == reference_ids[lang]

        print(
            f"  {lang}: zero_shot n={len(zs_lang_rows)} (expected {test_n}) "
            f"count_ok={count_ok} ids_match={ids_match} missing={len(missing_ids)} "
            f"empty_predictions={len(empty_preds)} | lead1_ids_match={lead1_ids_match} "
            f"lead3_ids_match={lead3_ids_match} (n={len(lead1_lang_rows)}/{len(lead3_lang_rows)})"
        )

        lang_ok = count_ok and ids_match and not missing_ids and not empty_preds and lead1_ids_match and lead3_ids_match
        all_ok = all_ok and lang_ok
        qc_report["languages"][lang] = {
            "zero_shot_count": len(zs_lang_rows),
            "zero_shot_count_ok": count_ok,
            "zero_shot_ids_match_reference": ids_match,
            "zero_shot_missing_count": len(missing_ids),
            "zero_shot_empty_predictions": empty_preds,
            "lead1_ids_match_reference": lead1_ids_match,
            "lead3_ids_match_reference": lead3_ids_match,
        }

    if not all_ok:
        raise RuntimeError(f"QC FAILED -- see printed detail above. Report: {qc_report}")
    print("\nAll structural QC checks PASSED.\n")

    # --- 3 deterministic examples per language ---
    print("=== 3 deterministic examples per language ===")
    zero_shot_by_id = {(r["language"], r["id"]): r for r in zero_shot_rows}
    lead1_by_id = {(r["language"], r["id"]): r for r in lead_n_rows if r["n_sentences"] == "1"}
    lead3_by_id = {(r["language"], r["id"]): r for r in lead_n_rows if r["n_sentences"] == "3"}

    qc_report["sample_examples"] = {}
    for lang in languages:
        rng = random.Random(base_cfg["seed"])
        sample_ids = rng.sample(sorted(reference_ids[lang]), 3)
        qc_report["sample_examples"][lang] = []
        for example_id in sample_ids:
            zs = zero_shot_by_id[(lang, example_id)]
            l1 = lead1_by_id[(lang, example_id)]
            l3 = lead3_by_id[(lang, example_id)]
            print(f"\n--- {lang} id={example_id} ---")
            print(f"  SOURCE   : {zs['source_article'][:200]}...")
            print(f"  REFERENCE: {zs['reference_summary'][:200]}")
            print(f"  ZERO-SHOT: {zs['generated_summary']!r}")
            print(f"  LEAD-1   : {l1['generated_summary'][:200]!r}")
            print(f"  LEAD-3   : {l3['generated_summary'][:200]!r}")
            qc_report["sample_examples"][lang].append(
                {
                    "id": example_id,
                    "source_preview": zs["source_article"][:200],
                    "reference": zs["reference_summary"],
                    "zero_shot": zs["generated_summary"],
                    "lead_1": l1["generated_summary"],
                    "lead_3": l3["generated_summary"],
                }
            )

    # --- zero-shot anomaly observations ---
    print("\n=== Zero-shot output anomaly observations (not auto-classified as failures) ===")
    qc_report["zero_shot_anomalies"] = {}
    for lang in languages:
        zs_lang_rows = [r for r in zero_shot_rows if r["language"] == lang]
        very_short = [r["id"] for r in zs_lang_rows if _is_very_short(r["generated_summary"])]
        sentinel = [r["id"] for r in zs_lang_rows if _has_sentinel_token(r["generated_summary"])]
        repetitive = [r["id"] for r in zs_lang_rows if _has_repetition(r["generated_summary"])]
        script_mismatch = [
            r["id"] for r in zs_lang_rows if not contains_expected_script(r["generated_summary"], lang)
        ]
        copying = [
            r["id"] for r in zs_lang_rows if _is_near_verbatim_copy(r["generated_summary"], r["source_article"])
        ]
        n = len(zs_lang_rows)
        print(
            f"  {lang}: very_short={len(very_short)}/{n} ({100*len(very_short)/n:.1f}%) "
            f"sentinel_tokens={len(sentinel)}/{n} ({100*len(sentinel)/n:.1f}%) "
            f"repetitive={len(repetitive)}/{n} ({100*len(repetitive)/n:.1f}%) "
            f"script_mismatch={len(script_mismatch)}/{n} ({100*len(script_mismatch)/n:.1f}%) "
            f"near_verbatim_copy={len(copying)}/{n} ({100*len(copying)/n:.1f}%)"
        )
        qc_report["zero_shot_anomalies"][lang] = {
            "n": n,
            "very_short_count": len(very_short),
            "sentinel_token_count": len(sentinel),
            "repetitive_count": len(repetitive),
            "script_mismatch_count": len(script_mismatch),
            "near_verbatim_copy_count": len(copying),
        }

    out_path = save_json_report(qc_report, repo_path("results/qa/baseline_evaluation_qc_report.json"))
    print(f"\nQC report written to: {out_path}")


if __name__ == "__main__":
    main()
