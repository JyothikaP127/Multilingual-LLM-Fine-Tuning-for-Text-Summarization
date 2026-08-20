"""Resolves the mT5-small embedding-tying question using the ACTUAL loaded
pretrained checkpoint, not the config-only random-init model used in the
earlier architecture audit.

Why this exists: results/qa/model_architecture_report.json (from the
earlier stage) instantiated MT5ForConditionalGeneration(config) with random
weights. transformers' MT5Config forces tie_word_embeddings=True in that
case, making the input embedding and output projection the SAME tensor --
which is why that report measured 172,119,424 total parameters. But when
the real pretrained weights are loaded via .from_pretrained(), transformers
detects the checkpoint's embedding and lm_head matrices have DIFFERENT
values and refuses to tie them (this is a documented library behavior, not
a bug), so the real deployed model has ~300M parameters, not ~172M. This
was discovered while running the zero-shot baseline (results/qa/
baseline_zero_shot_run_report.json also records it) and is confirmed here
as a dedicated, citable QA artifact.

This script does NOT modify or delete results/qa/model_architecture_report.json
-- it adds a new, separate artifact:
results/qa/model_parameters_actual_weights.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.inspect_utils import compute_parameter_breakdown, verify_lora_application  # noqa: E402
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

MODEL_NAME = "google/mt5-small"


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    lora_cfg = load_yaml("lora.yaml")["lora"]
    set_seed(base_cfg["seed"])

    from transformers import MT5ForConditionalGeneration

    print(f"Loading ACTUAL pretrained weights for {MODEL_NAME} (not config-only random init)...")
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    tied = model.lm_head.weight is model.shared.weight
    import torch

    values_equal = torch.equal(model.shared.weight, model.lm_head.weight)

    breakdown = compute_parameter_breakdown(model)

    print(f"\nActual total parameters   : {breakdown.total_params:,}")
    print(f"Embedding (shared) params : {breakdown.embedding_params:,}")
    print(f"LM head params            : {model.lm_head.weight.numel():,}")
    print(f"Non-embedding backbone    : {breakdown.total_params - breakdown.embedding_params - model.lm_head.weight.numel():,}")
    print(f"shared.weight is lm_head.weight (same object)? {tied}")
    print(f"shared.weight == lm_head.weight (same values)? {values_equal}")

    if tied:
        raise RuntimeError(
            "Unexpected: pretrained checkpoint loaded with tied embeddings. "
            "The parameter breakdown below assumes untied weights based on prior "
            "empirical verification -- investigate before trusting these numbers."
        )

    # Real total = embedding + lm_head (both separate, real-valued) + backbone
    embedding_params = model.shared.weight.numel()
    lm_head_params = model.lm_head.weight.numel()
    non_embedding_backbone = breakdown.total_params - embedding_params - lm_head_params

    lora_result = verify_lora_application(
        model,
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
    )
    lora_total = breakdown.total_params  # LoRA adapters add a negligible amount; report against real total
    print(f"\nLoRA (r={lora_cfg['r']}, target_modules={lora_cfg['target_modules']}) on ACTUAL weights:")
    print(f"  trainable params: {lora_result['trainable_params']:,}")
    print(f"  % of ACTUAL total ({lora_result['total_params']:,}): {lora_result['trainable_pct_of_total']:.4f}%")
    print(
        f"  % of non-embedding backbone ({non_embedding_backbone:,}): "
        f"{100.0 * lora_result['trainable_params'] / non_embedding_backbone:.4f}%"
    )

    report = {
        "model_name": MODEL_NAME,
        "loaded_from": "pretrained checkpoint via .from_pretrained() -- ACTUAL weights, not config-only random init",
        "tie_word_embeddings_same_object": tied,
        "tie_word_embeddings_same_values": values_equal,
        "actual_total_params": breakdown.total_params,
        "embedding_params": embedding_params,
        "lm_head_params": lm_head_params,
        "non_embedding_backbone_params": non_embedding_backbone,
        "comparison_to_earlier_config_only_report": {
            "earlier_config_only_total_params": 172119424,
            "earlier_config_only_embedding_params_tied": 128057344,
            "note": (
                "The earlier figure came from a randomly-initialized model where "
                "MT5Config forces tie_word_embeddings=True. It is NOT wrong for that "
                "model object, but it does not describe the actual deployed checkpoint. "
                "Use the actual_* fields in this file for any efficiency/trainable-% "
                "reporting going forward."
            ),
        },
        "lora_verification_on_actual_weights": lora_result,
        "lora_trainable_pct_of_non_embedding_backbone": (
            100.0 * lora_result["trainable_params"] / non_embedding_backbone
        ),
    }
    out_path = save_json_report(report, repo_path("results/qa/model_parameters_actual_weights.json"))
    print(f"\nReport written to: {out_path}")
    print("(results/qa/model_architecture_report.json left untouched, per instruction not to delete QA artifacts.)")


if __name__ == "__main__":
    main()
