"""BASELINE 1: zero-shot google/mt5-small (no fine-tuning, weights untouched).

GENERATION ONLY -- this script does not compute ROUGE or BERTScore. It
loads the pretrained model once, generates predictions for the full
500-example test split per language, and saves them. Metrics are computed
separately by scripts/evaluate_predictions.py, which reads the CSV this
script writes. Splitting these apart means predictions can be regenerated
without recomputing metrics, metrics can be recomputed (or run on a
different machine, e.g. Kaggle) without regenerating predictions, and the
BERTScore backbone is never loaded during generation.

Model weights are loaded read-only (model.eval(), torch.no_grad()) --
nothing is trained, nothing is written back to the model.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_xlsum import load_and_clean_split  # noqa: E402
from src.data.preprocess import build_tokenizer  # noqa: E402
from src.data.validation import contains_expected_script  # noqa: E402
from src.eval.generation import build_generation_kwargs, generate_summaries, load_base_model  # noqa: E402
from src.models.inspect_utils import compute_parameter_breakdown  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_csv_rows, save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

MODEL_NAME = "google/mt5-small"


def run_qa_smoke_test(model, tokenizer, gen_kwargs, max_source_length, languages_rows) -> None:
    print("=== QA smoke test: one example per language ===")
    for lang, iso, rows in languages_rows:
        row = rows[0]
        summaries, elapsed = generate_summaries(model, tokenizer, [row["text"]], max_source_length, gen_kwargs)
        output = summaries[0]
        non_empty = bool(output.strip())
        script_ok = contains_expected_script(output, lang)
        print(f"  {lang}: id={row['id']} elapsed={elapsed:.1f}s non_empty={non_empty} "
              f"script_plausible={script_ok}")
        print(f"    output: {output!r}")
        if not non_empty:
            raise RuntimeError(f"QA smoke test failed: empty zero-shot output for {lang}")
        if not script_ok:
            print(
                f"    NOTE: generated output does not contain the expected {lang} script. "
                "This is EXPECTED for a non-fine-tuned mT5 checkpoint (pretrained with a "
                "span-corruption objective, not summarization) -- flagged, not treated as a failure."
            )
    print()


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")
    eval_cfg = load_yaml("eval.yaml")

    set_seed(base_cfg["seed"])

    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    test_n = data_cfg["dataset"]["split_sizes"]["test"]
    max_source_length = data_cfg["preprocessing"]["max_source_length"]
    languages = data_cfg["dataset"]["languages"]

    gen_cfg = eval_cfg["generation"]
    gen_kwargs = build_generation_kwargs(gen_cfg)
    batch_size = gen_cfg["batch_size"]

    print("=== BASELINE 1: zero-shot google/mt5-small (GENERATION ONLY) ===")
    print(f"Generation config (explicit, from configs/eval.yaml): {gen_kwargs}")
    print(f"batch_size={batch_size}, max_source_length={max_source_length}\n")

    tokenizer = build_tokenizer(MODEL_NAME)

    print("Loading pretrained mT5-small (read-only, no fine-tuning)...")
    t0 = time.time()
    model = load_base_model(MODEL_NAME)
    model_load_time = time.time() - t0

    breakdown = compute_parameter_breakdown(model)
    tied = model.lm_head.weight is model.shared.weight
    print(f"Model loaded in {model_load_time:.1f}s.")
    print(
        f"Actual total parameters: {breakdown.total_params:,} "
        f"(embedding {model.shared.weight.numel():,} + lm_head {model.lm_head.weight.numel():,} "
        f"[tied={tied}] + backbone {breakdown.total_params - model.shared.weight.numel() - model.lm_head.weight.numel():,}). "
        f"See results/qa/model_parameters_actual_weights.json for the full, dedicated verification.\n"
    )

    # --- Load the exact same test split used throughout the project ---
    languages_rows = []
    for lang in languages:
        rows, _ = load_and_clean_split(
            lang["config_name"], lang["iso_code"], "test", test_n, seed, revision
        )
        assert len(rows) == test_n, f"{lang['config_name']}: expected {test_n} test rows, got {len(rows)}"
        languages_rows.append((lang["config_name"], lang["iso_code"], rows))
        print(f"Loaded {lang['config_name']} test split: {len(rows)} examples (matches project-wide test set)")
    print()

    run_qa_smoke_test(model, tokenizer, gen_kwargs, max_source_length, languages_rows)

    # --- Full generation (no scoring here) ---
    all_predictions_rows = []
    timing_by_language: dict[str, dict] = {}

    for lang, iso, rows in languages_rows:
        print(f"--- Generating for {lang} ({len(rows)} examples) ---")
        predictions: list[str] = []
        total_elapsed = 0.0

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["text"] for r in batch]
            summaries, elapsed = generate_summaries(model, tokenizer, texts, max_source_length, gen_kwargs)
            predictions.extend(summaries)
            total_elapsed += elapsed
            if (i // batch_size) % 10 == 0:
                print(f"  batch {i // batch_size + 1}/{-(-len(rows)//batch_size)} "
                      f"({i + len(batch)}/{len(rows)} examples, {total_elapsed:.1f}s elapsed so far)")

        avg_time = total_elapsed / len(rows)
        examples_per_sec = len(rows) / total_elapsed
        timing_by_language[lang] = {
            "total_inference_time_sec": total_elapsed,
            "avg_inference_time_sec": avg_time,
            "examples_per_sec": examples_per_sec,
        }
        print(f"  {lang}: total={total_elapsed:.1f}s avg={avg_time:.3f}s/ex rate={examples_per_sec:.3f} ex/s")

        for row, pred in zip(rows, predictions):
            all_predictions_rows.append(
                {
                    "id": row["id"],
                    "language": lang,
                    "model": "zero_shot_mt5_small",
                    "source_article": row["text"],
                    "reference_summary": row["summary"],
                    "generated_summary": pred,
                }
            )

    # --- Save predictions + generation run metadata (no metrics) ---
    pred_path = save_csv_rows(
        all_predictions_rows, repo_path("results/qualitative_samples/baseline_zero_shot_predictions.csv")
    )
    save_json_report(
        {
            "model_name": MODEL_NAME,
            "actual_total_params": breakdown.total_params,
            "embedding_params": model.shared.weight.numel(),
            "lm_head_params": model.lm_head.weight.numel(),
            "tie_word_embeddings_actual": tied,
            "model_load_time_sec": model_load_time,
            "generation_config": gen_kwargs,
            "batch_size": batch_size,
            "max_source_length": max_source_length,
            "timing_by_language": timing_by_language,
            "note": "Predictions only -- no ROUGE/BERTScore computed here. Run "
            "scripts/evaluate_predictions.py against the saved predictions CSV for metrics.",
        },
        repo_path("results/qa/baseline_zero_shot_generation_report.json"),
    )

    print("\n=== Generation complete (no metrics computed) ===")
    for lang, _, rows in languages_rows:
        t = timing_by_language[lang]
        print(f"  {lang:10s} n={len(rows)} total={t['total_inference_time_sec']:.1f}s "
              f"avg={t['avg_inference_time_sec']:.3f}s/ex rate={t['examples_per_sec']:.3f} ex/s")
    print(f"\nPredictions written to: {pred_path}")
    print("Next step: python scripts/evaluate_predictions.py --predictions "
          "results/qualitative_samples/baseline_zero_shot_predictions.csv --model-name zero_shot_mt5_small")


if __name__ == "__main__":
    main()
