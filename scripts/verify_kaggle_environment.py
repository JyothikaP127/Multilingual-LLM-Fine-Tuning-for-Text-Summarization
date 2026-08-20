"""Kaggle/Colab GPU environment verification -- run this ON Kaggle/Colab
after `pip install -r requirements.txt`, not locally (this machine has no
CUDA GPU to verify against).

Checks, in order:
  1-14. Every project-critical package imports and reports its version.
  CUDA availability, GPU name, CUDA version, BF16/FP16 support.
  google/mt5-small loads (actual pretrained weights).
  PEFT attaches the verified LoRA config (r=8, alpha=16, dropout=0.05,
    target_modules=["q","v"]) via the SAME src/train/train_lora.py code
    path used and verified locally -- not a reimplementation.
  Trainable parameter count == 344,064 (raises loudly if not).
  A real forward pass on the GPU.
  A tiny generation call on the GPU.

Does NOT run training. Does NOT modify requirements.txt or any package.
Writes results/qa/kaggle_environment_report.json and prints a clear
PASS/FAIL summary.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_NAME = "google/mt5-small"
EXPECTED_TRAINABLE_PARAMS = 344_064

CRITICAL_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "evaluate",
    "sentencepiece",
    "rouge_score",
    "bert_score",
    "pyarrow",
    "pandas",
    "numpy",
    "yaml",  # pyyaml's import name
    "huggingface_hub",
]


def check_imports() -> dict:
    print("=== 1. Project-critical imports ===")
    results = {}
    for mod_name in CRITICAL_PACKAGES:
        display_name = "pyyaml" if mod_name == "yaml" else mod_name
        try:
            mod = __import__(mod_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK   {display_name:<16} {version}")
            results[display_name] = {"ok": True, "version": version}
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {display_name:<16} {e}")
            results[display_name] = {"ok": False, "error": str(e)}
    return results


def check_hardware() -> dict:
    import torch

    print("\n=== 2-5. CUDA / GPU / BF16 / FP16 ===")
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_available else False

    fp16_supported = None
    if cuda_available:
        try:
            x = torch.randn(4, 4, dtype=torch.float16, device="cuda")
            _ = x @ x
            fp16_supported = True
        except Exception:  # noqa: BLE001
            fp16_supported = False

    print(f"  torch.cuda.is_available() : {cuda_available}")
    print(f"  GPU name                  : {gpu_name}")
    print(f"  CUDA version (torch build): {cuda_version}")
    print(f"  BF16 supported            : {bf16_supported}")
    print(f"  FP16 hardware-usable       : {fp16_supported} "
          "(hardware capability only -- mT5 should still use BF16 if available, "
          "not FP16, per the documented NaN-loss risk)")

    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "bf16_supported": bf16_supported,
        "fp16_hardware_usable": fp16_supported,
    }


def check_model_and_lora(device: str) -> dict:
    print(f"\n=== 6-8. mT5-small load + LoRA attach + trainable-param verification (device={device}) ===")
    from src.train.train_lora import attach_lora, build_parameter_report, load_pretrained_model
    from src.utils.config import load_yaml

    model = load_pretrained_model(MODEL_NAME)
    total_before = sum(p.numel() for p in model.parameters())
    print(f"  mT5-small loaded. Total params (pre-LoRA): {total_before:,}")

    lora_cfg = load_yaml("lora.yaml")["lora"]
    peft_model = attach_lora(model, lora_cfg)
    report = build_parameter_report(peft_model)

    print(f"  trainable params: {report.trainable_params:,} (expected {EXPECTED_TRAINABLE_PARAMS:,})")
    print(f"  trainable % of total: {report.trainable_pct_of_total:.4f}%")
    print(f"  freezing clean: {report.freezing_clean}")

    if report.trainable_params != EXPECTED_TRAINABLE_PARAMS:
        raise RuntimeError(
            f"Trainable param count {report.trainable_params:,} != expected "
            f"{EXPECTED_TRAINABLE_PARAMS:,}. Investigate before proceeding."
        )
    if not report.freezing_clean:
        raise RuntimeError(f"Freezing check failed on Kaggle: {report.to_dict()}")

    peft_model = peft_model.to(device)
    return {"model": peft_model, "report": report.to_dict(), "total_params_pre_lora": total_before}


def check_forward_and_generate(peft_model, device: str) -> dict:
    import torch

    from src.data.preprocess import build_tokenizer

    print("\n=== 9. Forward pass on GPU ===")
    tokenizer = build_tokenizer(MODEL_NAME)
    sample_text = "The government announced a new policy today after weeks of debate in parliament."
    sample_summary = "A new policy was announced after parliamentary debate."

    inputs = tokenizer(sample_text, max_length=768, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(text_target=sample_summary, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels_ids = labels["input_ids"].to(device)

    peft_model.train()
    out = peft_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels_ids)
    loss_value = out.loss.item()
    import math

    forward_ok = math.isfinite(loss_value)
    print(f"  forward loss: {loss_value} (finite: {forward_ok})")
    if not forward_ok:
        raise RuntimeError(f"Forward pass produced a non-finite loss: {loss_value}")

    print("\n=== 10. Tiny generation test on GPU ===")
    peft_model.eval()
    gen_inputs = tokenizer(sample_text, max_length=768, truncation=True, return_tensors="pt")
    gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}
    with torch.no_grad():
        gen_ids = peft_model.generate(**gen_inputs, max_new_tokens=32, num_beams=2)
    decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    print(f"  generated: {decoded[0]!r}")

    return {"forward_loss": loss_value, "forward_loss_finite": forward_ok, "generated_text": decoded[0]}


def main() -> None:
    from src.utils.config import repo_path
    from src.utils.reporting import save_json_report

    report: dict = {}
    passed = True

    try:
        report["imports"] = check_imports()
        if not all(v["ok"] for v in report["imports"].values()):
            passed = False

        report["hardware"] = check_hardware()

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_result = check_model_and_lora(device)
        report["lora_verification"] = model_result["report"]
        report["actual_total_params_pre_lora"] = model_result["total_params_pre_lora"]

        gen_result = check_forward_and_generate(model_result["model"], device)
        report["forward_and_generation"] = gen_result

        report["overall_status"] = "PASSED"
        print("\n=== ALL CHECKS PASSED ===")
    except Exception as e:  # noqa: BLE001
        passed = False
        report["overall_status"] = "FAILED"
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()
        print(f"\n=== CHECK FAILED: {e} ===")
        print(report["traceback"])

    out_path = save_json_report(report, repo_path("results/qa/kaggle_environment_report.json"))
    print(f"\nReport written to: {out_path}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
