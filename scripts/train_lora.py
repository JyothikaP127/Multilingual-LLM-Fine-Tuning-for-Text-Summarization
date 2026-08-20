"""LoRA fine-tuning entry point.

Usage:
  python scripts/train_lora.py --mode smoke   (default; tiny pipeline validation, CPU-safe)
  python scripts/train_lora.py --mode full    (NOT approved yet -- refuses to run)

The smoke test uses a small deterministic PREFIX of the project's real
canonical train/validation splits (same ids as everywhere else in the
project), not a separately re-sampled subset -- so it is a genuine slice of
real training data, not synthetic or independently-drawn data. It writes to
results/checkpoints/smoke_test/, never the full-run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load_xlsum import load_and_clean_split  # noqa: E402
from src.data.preprocess import build_tokenizer, preprocess_examples  # noqa: E402
from src.train.train_lora import (  # noqa: E402
    TokenizedDataset,
    attach_lora,
    build_parameter_report,
    build_trainer,
    build_training_arguments,
    load_pretrained_model,
    save_lora_checkpoint,
)
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.hardware import detect_hardware  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

MODEL_NAME = "google/mt5-small"
EXPECTED_TRAINABLE_PARAMS = 344_064  # r=8, target_modules=["q","v"] -- verified in results/qa/


def _param_checksum(tensor) -> str:
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def load_multilingual_split(data_cfg: dict, split: str, n_per_language: int | None) -> list[dict]:
    """n_per_language=None -> the full canonical project split size. Otherwise
    takes the first n_per_language rows of that canonical split -- a genuine
    prefix subset, never a separately-seeded re-sample.
    """
    revision = data_cfg["dataset"]["revision"]
    seed = data_cfg["dataset"]["sampling_seed"]
    canonical_n = data_cfg["dataset"]["split_sizes"][split]
    rows = []
    for lang in data_cfg["dataset"]["languages"]:
        lang_rows, _ = load_and_clean_split(
            lang["config_name"], lang["iso_code"], split, canonical_n, seed, revision
        )
        if n_per_language is not None:
            lang_rows = lang_rows[:n_per_language]
        rows.extend(lang_rows)
    return rows


def verify_gradient_flow(peft_model, sample_example: dict) -> dict:
    """Explicit, dedicated check (separate from the real training loop):
    one manual forward+backward pass, then inspect .grad directly. This is
    the definitive answer to "did LoRA params receive gradients" -- weight
    change alone is suggestive but this is direct evidence.
    """
    import torch

    peft_model.train()
    peft_model.zero_grad()

    input_ids = torch.tensor(sample_example["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(sample_example["attention_mask"]).unsqueeze(0)
    labels = torch.tensor(sample_example["labels"]).unsqueeze(0)

    out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss_value = out.loss.item()
    out.loss.backward()

    lora_grad_nonzero = any(
        p.grad is not None and torch.any(p.grad != 0)
        for name, p in peft_model.named_parameters()
        if "lora_" in name and p.requires_grad
    )
    frozen_params_with_grad = [
        name
        for name, p in peft_model.named_parameters()
        if "lora_" not in name and p.grad is not None
    ]

    peft_model.zero_grad()

    return {
        "manual_forward_loss": loss_value,
        "loss_finite": math.isfinite(loss_value),
        "lora_params_received_nonzero_gradient": lora_grad_nonzero,
        "frozen_params_with_gradient": frozen_params_with_grad,
        "clean": lora_grad_nonzero and not frozen_params_with_grad and math.isfinite(loss_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    if args.mode == "full":
        raise RuntimeError(
            "Full LoRA training run has not been approved yet. This script refuses to run "
            "--mode full until that approval is explicit. Re-run with --mode smoke."
        )

    base_cfg = load_yaml("base.yaml")
    data_cfg = load_yaml("data.yaml")
    lora_cfg = load_yaml("lora.yaml")

    set_seed(base_cfg["seed"])
    hw = detect_hardware()

    smoke_cfg = lora_cfg["smoke_test"]
    train_max_src = lora_cfg["training"]["max_source_length"]
    train_max_tgt = lora_cfg["training"]["max_target_length"]
    assert train_max_src == data_cfg["preprocessing"]["max_source_length"], (
        "max_source_length mismatch between configs/lora.yaml and configs/data.yaml -- "
        "these must stay in sync; this run does not change max_source_length."
    )
    assert train_max_tgt == data_cfg["preprocessing"]["max_target_length"]

    print("=== LoRA training: SMOKE TEST ===")
    print(f"Hardware: device={hw.device} cuda_available={hw.cuda_available} torch={hw.torch_version}")

    print("\nLoading data (prefix subset of the canonical deterministic splits, no new downloads)...")
    train_rows = load_multilingual_split(data_cfg, "train", smoke_cfg["train_examples_per_language"])
    val_rows = load_multilingual_split(data_cfg, "validation", smoke_cfg["val_examples_per_language"])
    print(f"  train: {len(train_rows)} examples total ({smoke_cfg['train_examples_per_language']}/language)")
    print(f"  val:   {len(val_rows)} examples total ({smoke_cfg['val_examples_per_language']}/language)")
    for lang in sorted({r["language"] for r in train_rows}):
        n_tr = sum(1 for r in train_rows if r["language"] == lang)
        n_va = sum(1 for r in val_rows if r["language"] == lang)
        print(f"    {lang}: train={n_tr} val={n_va}")

    tokenizer = build_tokenizer(MODEL_NAME)
    train_processed = preprocess_examples(train_rows, tokenizer, train_max_src, train_max_tgt)
    val_processed = preprocess_examples(val_rows, tokenizer, train_max_src, train_max_tgt)
    train_dataset = TokenizedDataset(train_processed)
    val_dataset = TokenizedDataset(val_processed)

    print("\nLoading pretrained mT5-small (ACTUAL weights via from_pretrained)...")
    model = load_pretrained_model(MODEL_NAME)
    total_before = sum(p.numel() for p in model.parameters())
    print(f"  actual total params (pre-LoRA): {total_before:,}")

    print("\nVerifying LoRA target modules against the ACTUAL loaded model's named_modules(), "
          "then attaching LoRA...")
    peft_model = attach_lora(model, lora_cfg["lora"])

    report = build_parameter_report(peft_model)
    print(f"  total params                 : {report.total_params:,}")
    print(f"  trainable params             : {report.trainable_params:,}")
    print(f"  frozen params                : {report.frozen_params:,}")
    print(f"  trainable % of total         : {report.trainable_pct_of_total:.4f}%")
    print(f"  trainable % of non-emb backbone: {report.trainable_pct_of_non_embedding_backbone:.4f}%")
    print(f"  freezing clean (no leaks)    : {report.freezing_clean}")

    if not report.freezing_clean:
        raise RuntimeError(f"Freezing verification FAILED: {report.to_dict()}")
    if report.trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise RuntimeError(
            f"Trainable param count {report.trainable_params:,} does not match the expected/"
            f"verified {EXPECTED_TRAINABLE_PARAMS:,} for r=8, target_modules=['q','v']. "
            "Investigate before proceeding -- do not silently accept a mismatch."
        )
    print("  Trainable parameter count matches the pre-verified expected value. OK.")

    param_report_path = save_json_report(
        {**report.to_dict(), "model_name": MODEL_NAME, "lora_config": lora_cfg["lora"]},
        repo_path("results/qa/lora_parameter_report.json"),
    )
    print(f"  parameter report written to: {param_report_path}")

    # --- explicit gradient-flow check (separate from the Trainer loop) ---
    print("\n=== Gradient-flow verification (manual forward+backward on one example) ===")
    grad_check = verify_gradient_flow(peft_model, train_processed[0])
    print(f"  {grad_check}")
    if not grad_check["clean"]:
        raise RuntimeError(f"Gradient-flow verification FAILED: {grad_check}")

    # --- snapshot LoRA + a frozen base parameter for before/after comparison ---
    lora_param_name = next(
        name for name, p in peft_model.named_parameters() if "lora_A" in name and p.requires_grad
    )
    base_param_name = next(
        name
        for name, _ in peft_model.named_parameters()
        if name.endswith("encoder.block.0.layer.0.SelfAttention.k.weight")
    )
    params_by_name = dict(peft_model.named_parameters())
    lora_param_before = params_by_name[lora_param_name].detach().clone()
    base_param_before = params_by_name[base_param_name].detach().clone()
    checksum_before = _param_checksum(lora_param_before)
    print(f"\nTracking LoRA param: {lora_param_name} (checksum before: {checksum_before})")
    print(f"Tracking frozen base param: {base_param_name}")

    output_dir = str(repo_path(smoke_cfg["output_dir"]))
    training_hp = {
        "per_device_train_batch_size": smoke_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": smoke_cfg["gradient_accumulation_steps"],
        "num_train_epochs": smoke_cfg["num_train_epochs"],
        "learning_rate": lora_cfg["training"]["learning_rate"],
        "warmup_ratio": lora_cfg["training"]["warmup_ratio"],
        "weight_decay": lora_cfg["training"]["weight_decay"],
        "precision": smoke_cfg["precision"],
        "logging_steps": smoke_cfg["logging_steps"],
        "seed": base_cfg["seed"],
    }
    training_args, resolved_precision = build_training_arguments(output_dir, training_hp, hw)
    print(f"\nResolved precision: {resolved_precision}")
    print(
        f"Training args: batch_size={training_args.per_device_train_batch_size} "
        f"grad_accum={training_args.gradient_accumulation_steps} "
        f"epochs={training_args.num_train_epochs} lr={training_args.learning_rate}"
    )

    trainer = build_trainer(peft_model, tokenizer, training_args, train_dataset, val_dataset)

    print("\n=== Training (smoke test) ===")
    t0 = time.time()
    train_result = trainer.train()
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    final_train_loss = train_result.training_loss
    print(f"Final training_loss: {final_train_loss}")
    if not math.isfinite(final_train_loss):
        raise RuntimeError(f"Training loss is not finite: {final_train_loss}")

    print("\n=== Validation ===")
    eval_metrics = trainer.evaluate()
    print(f"Eval metrics: {eval_metrics}")
    if not math.isfinite(eval_metrics["eval_loss"]):
        raise RuntimeError(f"Eval loss is not finite: {eval_metrics['eval_loss']}")

    # --- post-training verification ---
    params_by_name_after = dict(peft_model.named_parameters())
    lora_param_after = params_by_name_after[lora_param_name].detach().clone()
    base_param_after = params_by_name_after[base_param_name].detach().clone()
    checksum_after = _param_checksum(lora_param_after)

    import torch

    lora_changed = not torch.equal(lora_param_before, lora_param_after)
    base_unchanged = torch.equal(base_param_before, base_param_after)

    print(f"\nLoRA parameter checked: {lora_param_name}")
    print(f"  checksum before: {checksum_before}")
    print(f"  checksum after : {checksum_after}")
    print(f"  changed        : {lora_changed}")
    print(f"Base (frozen) parameter checked: {base_param_name}")
    print(f"  unchanged      : {base_unchanged}")

    if not lora_changed:
        raise RuntimeError("LoRA parameter did NOT change after training -- something is wrong.")
    if not base_unchanged:
        raise RuntimeError("Base (frozen) parameter CHANGED during training -- freezing is broken.")

    # --- save adapter + tokenizer + config + logs ---
    save_lora_checkpoint(peft_model, tokenizer, output_dir)
    save_json_report(
        {"lora_config": lora_cfg, "smoke_test_config": smoke_cfg, "resolved_precision": resolved_precision},
        repo_path(f"{smoke_cfg['output_dir']}/training_config_used.json"),
    )
    save_json_report(
        {
            "log_history": trainer.state.log_history,
            "final_training_loss": final_train_loss,
            "eval_metrics": eval_metrics,
            "train_time_sec": train_time,
            "gradient_flow_check": grad_check,
            "lora_param_checked": lora_param_name,
            "lora_param_checksum_before": checksum_before,
            "lora_param_checksum_after": checksum_after,
            "lora_param_changed": lora_changed,
            "base_param_checked": base_param_name,
            "base_param_unchanged": base_unchanged,
        },
        repo_path(f"{smoke_cfg['output_dir']}/training_metrics.json"),
    )

    ckpt_files = [f for f in Path(output_dir).rglob("*") if f.is_file()]
    ckpt_size_bytes = sum(f.stat().st_size for f in ckpt_files)
    print(f"\nCheckpoint dir: {output_dir}")
    print(f"Checkpoint files: {[f.name for f in ckpt_files]}")
    print(f"Checkpoint size: {ckpt_size_bytes / 1024:.1f} KB")
    if ckpt_size_bytes > 50 * 1024 * 1024:
        raise RuntimeError(
            f"Checkpoint size ({ckpt_size_bytes / 1024 / 1024:.1f} MB) is far larger than an "
            "adapter-only save should be (~1-2 MB for r=8 on q/v) -- the base model may have "
            "been saved by mistake."
        )

    # --- reload adapter and generate, to prove the saved checkpoint actually works ---
    print("\n=== Reload adapter + generate ===")
    from peft import PeftModel

    fresh_base = load_pretrained_model(MODEL_NAME)
    reloaded = PeftModel.from_pretrained(fresh_base, output_dir)
    reloaded.eval()

    sample_text = val_rows[0]["text"]
    inputs = tokenizer(sample_text, max_length=train_max_src, truncation=True, return_tensors="pt")
    with torch.no_grad():
        gen_ids = reloaded.generate(**inputs, max_new_tokens=32, num_beams=2)
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(f"  reload+generate output: {decoded[0]!r}")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
