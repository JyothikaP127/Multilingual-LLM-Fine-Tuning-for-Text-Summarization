"""mT5-small architecture inspection: named_modules(), attention projection
naming, parameter breakdown (total / embedding / non-embedding), and an
actual (not assumed) verification that PEFT LoRA applies cleanly to the
configured target_modules.

Does not download pretrained weights (config-only instantiation -- see
src/models/inspect_utils.py docstring for why that's architecturally
equivalent for this purpose). Does not train anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.inspect_utils import (  # noqa: E402
    compute_parameter_breakdown,
    find_attention_projection_names,
    list_linear_module_names,
    load_config_and_model,
    verify_lora_application,
    verify_seq2seq_trainer_compat,
)
from src.utils.config import load_yaml, repo_path  # noqa: E402
from src.utils.reporting import save_json_report  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> None:
    base_cfg = load_yaml("base.yaml")
    model_cfg = load_yaml("model.yaml")
    lora_cfg = load_yaml("lora.yaml")["lora"]

    set_seed(base_cfg["seed"])

    model_name = model_cfg["model_name_or_path"]
    print(f"=== Inspecting {model_name} ===")

    config, model = load_config_and_model(model_name)

    linear_names = list_linear_module_names(model)
    grouped_projections = find_attention_projection_names(linear_names)
    param_breakdown = compute_parameter_breakdown(model)

    print(f"Total Linear modules found: {len(linear_names)}")
    print("Linear modules grouped by final name component:")
    for suffix, names in sorted(grouped_projections.items()):
        print(f"  {suffix}: {len(names)} occurrences (e.g. {names[0]})")

    print("\nParameter breakdown:")
    print(f"  total params           : {param_breakdown.total_params:,}")
    print(f"  embedding (tied) params: {param_breakdown.embedding_params:,}")
    print(f"  non-embedding params   : {param_breakdown.non_embedding_params:,}")
    print(f"  tie_word_embeddings effective: {param_breakdown.tie_word_embeddings_effective}")

    target_modules = lora_cfg["target_modules"]
    missing = [t for t in target_modules if t not in grouped_projections]
    if missing:
        raise RuntimeError(
            f"Configured LoRA target_modules {target_modules} include names not "
            f"found as Linear modules on this model: {missing}. "
            f"Available suffixes: {sorted(grouped_projections)}"
        )

    lora_result = verify_lora_application(
        model,
        target_modules=target_modules,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
    )

    print(f"\nLoRA application check (target_modules={target_modules}, r={lora_cfg['r']}):")
    print(f"  trainable params : {lora_result['trainable_params']:,}")
    print(f"  total params     : {lora_result['total_params']:,}")
    print(f"  trainable % of total: {lora_result['trainable_pct_of_total']:.4f}%")
    print(
        f"  trainable % of non-embedding backbone: "
        f"{100.0 * lora_result['trainable_params'] / param_breakdown.non_embedding_params:.4f}%"
    )
    print(f"  LoRA injection points (A/B pairs): {lora_result['num_lora_injection_points']}")

    # Rank ablation, computed the same verified way (fresh model per r so
    # counts aren't polluted by the previous LoRA injection).
    rank_ablation = {}
    for r in load_yaml("lora.yaml").get("rank_ablation", []):
        _, fresh_model = load_config_and_model(model_name)
        result = verify_lora_application(
            fresh_model,
            target_modules=target_modules,
            r=r,
            lora_alpha=r * 2,
            lora_dropout=lora_cfg["lora_dropout"],
        )
        rank_ablation[str(r)] = result
        print(
            f"  r={r}: trainable={result['trainable_params']:,} "
            f"({result['trainable_pct_of_total']:.4f}% of total)"
        )

    trainer_compat = verify_seq2seq_trainer_compat(model, model_name)
    print(
        f"\nSeq2SeqTrainer compatibility: "
        f"{'OK' if trainer_compat['trainer_constructed_with_model'] else 'FAILED'}"
    )

    report = {
        "model_name": model_name,
        "config": config.to_dict(),
        "linear_module_count": len(linear_names),
        "attention_projection_suffixes": {k: len(v) for k, v in grouped_projections.items()},
        "sample_module_names": {k: v[:2] for k, v in grouped_projections.items()},
        "parameter_breakdown": param_breakdown.to_dict(),
        "lora_verification": lora_result,
        "lora_rank_ablation": rank_ablation,
        "seq2seq_trainer_compat": trainer_compat,
    }
    out_path = save_json_report(report, repo_path("results/qa/model_architecture_report.json"))
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
