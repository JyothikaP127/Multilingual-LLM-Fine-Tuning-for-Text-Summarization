"""LoRA mT5-small test-set inference.

Loads the trained LoRA adapter (results/checkpoints/lora_full/ by default)
on top of the base google/mt5-small model via PeftModel.from_pretrained
(never by reading adapter_model.safetensors directly), and generates
summaries for the exact same canonical 1,500-example test set used by the
baselines (500/language x English/Hindi/Telugu).

Reuses, unmodified:
  - src/data/load_xlsum.py's load_and_clean_split -- the SAME dataset
    loading path used by scripts/run_baseline_zero_shot.py and
    scripts/run_lead_n_baseline.py. No second dataset-loading
    implementation was written.
  - src/eval/generation.py's build_generation_kwargs -- the SAME generation
    configuration (configs/eval.yaml's `generation` section) used by
    scripts/run_baseline_zero_shot.py. No new generation parameters.
  - src/train/train_lora.py's load_pretrained_model -- the SAME base-model
    loading call used during training.
  - src/utils/hardware.py's detect_hardware/resolve_precision -- the SAME
    "auto" precision policy used during training (bf16 if truly supported,
    else fp32, never fp16).

Does NOT touch: the train/validation splits, any baseline prediction or
metrics file, dataset configuration, max_source_length, or the generation
config. Does NOT modify src/eval/generation.py or any training file --
device-aware generation is a small local wrapper here only because the
existing generate_summaries() never needed device placement (every prior
script that used it ran on CPU).

Usage:
  python scripts/run_lora_test_inference.py --smoke-only
      Runs only the 1-example-per-language smoke check, then stops.

  python scripts/run_lora_test_inference.py
      Runs the smoke check, then the full 1,500-example generation.

  python scripts/run_lora_test_inference.py --adapter-dir <path> --smoke-only
      Points at a different adapter directory (e.g. the local smoke-test
      checkpoint) -- used for validating this script's mechanics before the
      real trained adapter is available on this machine.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_xlsum import load_and_clean_split  # noqa: E402
from src.data.preprocess import build_tokenizer  # noqa: E402
from src.eval.generation import build_generation_kwargs  # noqa: E402
from src.train.train_lora import load_pretrained_model  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.hardware import detect_hardware, resolve_precision  # noqa: E402
from src.utils.reporting import save_csv_rows, save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

MODEL_NAME = "google/mt5-small"
DEFAULT_ADAPTER_DIR = "results/checkpoints/lora_full"
DISPLAY_MODEL_NAME = "LoRA mT5-small"
EXPECTED_LORA_PARAMS = 344_064

_SENTINEL_RE = re.compile(r"<extra_id_\d+>")


def _dtype_for(precision: str):
    import torch

    return {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]


def load_lora_model(model_name: str, adapter_dir: str, device: str, precision: str):
    """Loads the base model, then attaches the trained adapter via the
    correct PEFT mechanism. is_trainable=False deliberately -- this is
    inference-only, so no gradient bookkeeping is set up on the adapter
    weights (that would be wasteful and is not what "trainable" means in an
    inference context).
    """
    from peft import PeftModel

    dtype = _dtype_for(precision)
    base_model = load_pretrained_model(model_name)  # same call used in src/train/train_lora.py
    base_model = base_model.to(dtype=dtype)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    peft_model = peft_model.to(device)
    peft_model.eval()
    return peft_model


def verify_adapter(peft_model, expected_lora_cfg: dict) -> dict:
    """Verifies the loaded adapter matches the expected LoRA configuration
    and that exactly the expected number of LoRA parameters exist -- not an
    accidentally-duplicated full base model. Checks parameter COUNT (by
    name), not requires_grad, since is_trainable=False is correct for
    inference and would make a requires_grad-based check report zero.
    """
    loaded_cfg = peft_model.peft_config["default"]
    config_matches = (
        loaded_cfg.r == expected_lora_cfg["r"]
        and loaded_cfg.lora_alpha == expected_lora_cfg["lora_alpha"]
        and abs(loaded_cfg.lora_dropout - expected_lora_cfg["lora_dropout"]) < 1e-9
        and set(loaded_cfg.target_modules) == set(expected_lora_cfg["target_modules"])
    )

    total_params = sum(p.numel() for p in peft_model.parameters())
    lora_params = sum(p.numel() for name, p in peft_model.named_parameters() if "lora_" in name)
    unexpectedly_trainable = [name for name, p in peft_model.named_parameters() if p.requires_grad]

    return {
        "loaded_lora_config": {
            "r": loaded_cfg.r,
            "lora_alpha": loaded_cfg.lora_alpha,
            "lora_dropout": loaded_cfg.lora_dropout,
            "target_modules": sorted(loaded_cfg.target_modules),
        },
        "config_matches_expected": config_matches,
        "total_params": total_params,
        "lora_adapter_param_count": lora_params,
        "lora_param_count_matches_expected": lora_params == EXPECTED_LORA_PARAMS,
        "unexpectedly_trainable_params": unexpectedly_trainable,
        "clean": config_matches and lora_params == EXPECTED_LORA_PARAMS and not unexpectedly_trainable,
    }


def generate_on_device(model, tokenizer, texts, max_source_length, generation_kwargs, device):
    """Device-aware generation call, reusing the exact same
    generation_kwargs (built via src/eval/generation.py's
    build_generation_kwargs, unmodified) as scripts/run_baseline_zero_shot.py.
    """
    import torch

    inputs = tokenizer(texts, max_length=max_source_length, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - start

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return decoded, elapsed


def is_sentinel_only(text: str) -> bool:
    """True if the (non-empty) text consists solely of <extra_id_N> tokens
    and whitespace -- i.e. no real generated content at all.
    """
    if not text.strip():
        return False
    remainder = _SENTINEL_RE.sub("", text).strip()
    return remainder == ""


def run_smoke_check(model, tokenizer, gen_kwargs, max_source_length, languages_rows, device) -> None:
    print("=== Smoke check: one example per language ===")
    for lang, iso, rows in languages_rows:
        row = rows[0]
        summaries, elapsed = generate_on_device(model, tokenizer, [row["text"]], max_source_length, gen_kwargs, device)
        output = summaries[0]
        non_empty = bool(output.strip())
        print(f"  {lang}: id={row['id']} elapsed={elapsed:.2f}s non_empty={non_empty}")
        print(f"    output: {output!r}")
        if not non_empty:
            raise RuntimeError(f"Smoke check failed: empty LoRA generation output for {lang}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the 1-example-per-language smoke check, then stop (no full 1,500-example generation).",
    )
    parser.add_argument(
        "--adapter-dir",
        default=DEFAULT_ADAPTER_DIR,
        help=f"Path to the trained LoRA adapter directory (default: {DEFAULT_ADAPTER_DIR}).",
    )
    args = parser.parse_args()

    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")
    eval_cfg = load_yaml("eval.yaml")
    lora_cfg = load_yaml("lora.yaml")

    set_seed(base_cfg["seed"])
    hw = detect_hardware()
    device = hw.device

    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    test_n = data_cfg["dataset"]["split_sizes"]["test"]
    max_source_length = data_cfg["preprocessing"]["max_source_length"]
    languages = data_cfg["dataset"]["languages"]

    gen_cfg = eval_cfg["generation"]
    gen_kwargs = build_generation_kwargs(gen_cfg)  # unchanged, from src/eval/generation.py
    batch_size = gen_cfg["batch_size"]

    precision = resolve_precision(lora_cfg["training"]["precision"], hw)  # same "auto" policy as training
    if precision == "fp16":
        raise RuntimeError("precision resolved to fp16 -- refusing, same mT5 NaN-risk policy used in training.")

    adapter_dir = str(repo_path(args.adapter_dir))

    print("=== LoRA mT5-small test inference ===")
    print(f"Device: {device} | Precision: {precision}")
    print(f"Generation config (reused, unchanged, from configs/eval.yaml): {gen_kwargs}")
    print(f"batch_size={batch_size}, max_source_length={max_source_length}")
    print(f"Adapter: {adapter_dir}\n")

    tokenizer = build_tokenizer(MODEL_NAME)
    print("Loading base model + trained LoRA adapter (PeftModel.from_pretrained)...")
    model = load_lora_model(MODEL_NAME, adapter_dir, device, precision)
    print("Base model loaded successfully. Adapter loaded successfully.")

    verification = verify_adapter(model, lora_cfg["lora"])
    print(f"Adapter verification: {verification}")
    if not verification["clean"]:
        raise RuntimeError(f"Adapter verification FAILED: {verification}")
    print(
        "Adapter verification OK: expected LoRA config present, "
        f"{verification['lora_adapter_param_count']:,} LoRA params (matches expected "
        f"{EXPECTED_LORA_PARAMS:,}), no accidental full-model trainability.\n"
    )

    # --- load the EXACT SAME canonical test split as the baselines ---
    languages_rows = []
    for lang in languages:
        rows, _ = load_and_clean_split(
            lang["config_name"], lang["iso_code"], "test", test_n, seed, revision
        )
        assert len(rows) == test_n, f"{lang['config_name']}: expected {test_n} test rows, got {len(rows)}"
        languages_rows.append((lang["config_name"], lang["iso_code"], rows))
        print(f"Loaded {lang['config_name']} test split: {len(rows)} examples (matches project-wide test set)")
    print()

    run_smoke_check(model, tokenizer, gen_kwargs, max_source_length, languages_rows, device)

    if args.smoke_only:
        print(
            "--smoke-only was set: stopping here, per instruction not to run the full "
            "1,500-example generation automatically."
        )
        return

    # --- Full generation (1,500 examples) ---
    all_predictions_rows = []
    timing_by_language: dict[str, dict] = {}
    empty_predictions = 0
    sentinel_only_predictions = 0

    for lang, iso, rows in languages_rows:
        print(f"--- Generating for {lang} ({len(rows)} examples) ---")
        predictions: list[str] = []
        total_elapsed = 0.0

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts = [r["text"] for r in batch]
            summaries, elapsed = generate_on_device(model, tokenizer, texts, max_source_length, gen_kwargs, device)
            predictions.extend(summaries)
            total_elapsed += elapsed
            if (i // batch_size) % 10 == 0:
                print(
                    f"  batch {i // batch_size + 1}/{-(-len(rows)//batch_size)} "
                    f"({i + len(batch)}/{len(rows)} examples, {total_elapsed:.1f}s elapsed so far)"
                )

        avg_time = total_elapsed / len(rows)
        examples_per_sec = len(rows) / total_elapsed
        timing_by_language[lang] = {
            "total_inference_time_sec": total_elapsed,
            "avg_inference_time_sec": avg_time,
            "examples_per_sec": examples_per_sec,
        }
        print(f"  {lang}: total={total_elapsed:.1f}s avg={avg_time:.3f}s/ex rate={examples_per_sec:.3f} ex/s")

        for row, pred in zip(rows, predictions):
            if not pred.strip():
                empty_predictions += 1
            elif is_sentinel_only(pred):
                sentinel_only_predictions += 1
            all_predictions_rows.append(
                {
                    "id": row["id"],
                    "language": lang,
                    "model": DISPLAY_MODEL_NAME,
                    "source_article": row["text"],
                    "reference_summary": row["summary"],
                    "generated_summary": pred,
                }
            )

    # Same 6-column schema as results/qualitative_samples/baseline_zero_shot_predictions.csv
    # (id, language, model, source_article, reference_summary, generated_summary). That
    # existing schema has no inference-time column -- timing lives in the generation
    # report JSON instead, exactly as it does for the zero-shot baseline.
    pred_path = save_csv_rows(all_predictions_rows, repo_path("results/qualitative_samples/lora_test_predictions.csv"))

    overall_total_time = sum(t["total_inference_time_sec"] for t in timing_by_language.values())
    overall_n = sum(len(rows) for _, _, rows in languages_rows)

    save_json_report(
        {
            "model_name": DISPLAY_MODEL_NAME,
            "base_model": MODEL_NAME,
            "adapter_path": args.adapter_dir,
            "examples_per_language": {lang: len(rows) for lang, _, rows in languages_rows},
            "total_examples": overall_n,
            "generation_config": gen_kwargs,
            "batch_size": batch_size,
            "max_source_length": max_source_length,
            "device": device,
            "precision": precision,
            "total_inference_time_sec": overall_total_time,
            "avg_inference_time_sec": overall_total_time / overall_n,
            "timing_by_language": timing_by_language,
            "empty_predictions": empty_predictions,
            "sentinel_only_predictions": sentinel_only_predictions,
            "adapter_verification": verification,
        },
        repo_path("results/qa/lora_test_generation_report.json"),
    )

    print("\n=== Generation complete ===")
    for lang, _, rows in languages_rows:
        t = timing_by_language[lang]
        print(
            f"  {lang:10s} n={len(rows)} total={t['total_inference_time_sec']:.1f}s "
            f"avg={t['avg_inference_time_sec']:.3f}s/ex rate={t['examples_per_sec']:.3f} ex/s"
        )
    print(f"\nempty_predictions={empty_predictions}, sentinel_only_predictions={sentinel_only_predictions}")
    print(f"Predictions written to: {pred_path}")
    print(
        "Next step: python scripts/evaluate_predictions.py --predictions "
        "results/qualitative_samples/lora_test_predictions.csv --output results/metrics/lora_test.csv"
    )


if __name__ == "__main__":
    main()
