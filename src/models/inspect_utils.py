"""mT5 architecture introspection + PEFT/LoRA compatibility verification.

Deliberately instantiates the model from its config only (random init), not
from pretrained weights: architecture (module names, parameter counts,
shapes) is identical either way, and this avoids an ~1.2GB weight download at
the inspection stage, before any training has been decided to run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ParameterBreakdown:
    total_params: int
    embedding_params: int  # shared input-embedding / lm_head tensor (tied)
    non_embedding_params: int
    tie_word_embeddings_effective: bool

    def to_dict(self) -> dict:
        return asdict(self)


def load_config_and_model(model_name_or_path: str):
    from transformers import MT5Config, MT5ForConditionalGeneration

    config = MT5Config.from_pretrained(model_name_or_path)
    model = MT5ForConditionalGeneration(config)
    return config, model


def list_linear_module_names(model) -> list[str]:
    import torch.nn as nn

    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def find_attention_projection_names(linear_names: list[str]) -> dict[str, list[str]]:
    """Groups Linear module names by their final path component (q/k/v/o for
    T5-style attention). Does NOT assume any particular naming convention --
    just reports what is actually there.
    """
    grouped: dict[str, list[str]] = {}
    for name in linear_names:
        suffix = name.rsplit(".", 1)[-1]
        grouped.setdefault(suffix, []).append(name)
    return grouped


def compute_parameter_breakdown(model) -> ParameterBreakdown:
    total = sum(p.numel() for p in model.parameters())
    embedding = model.shared.weight.numel()
    tied = model.lm_head.weight is model.shared.weight
    return ParameterBreakdown(
        total_params=total,
        embedding_params=embedding,
        non_embedding_params=total - embedding,
        tie_word_embeddings_effective=tied,
    )


def verify_lora_application(
    model, target_modules: list[str], r: int, lora_alpha: int, lora_dropout: float
) -> dict:
    """Actually applies a LoraConfig with the given target_modules to a copy
    of the model's architecture and reports whether it succeeded plus the
    resulting trainable-parameter counts. Raises if target_modules don't
    exist on the model -- this is meant to fail loudly if the module names
    are wrong, not silently no-op.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())

    injected_lora_layers = [
        name for name, _ in peft_model.named_modules() if "lora_A" in name or "lora_B" in name
    ]

    return {
        "target_modules": target_modules,
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct_of_total": 100.0 * trainable / total,
        "num_lora_injection_points": len(injected_lora_layers) // 2,  # A and B pairs
    }


def verify_seq2seq_trainer_compat(model, tokenizer_name: str) -> dict:
    """Confirms Seq2SeqTrainingArguments and Seq2SeqTrainer construct cleanly
    around this exact model/tokenizer under the pinned transformers version.
    Does NOT call .train() -- only object construction plus one dependency-
    free sanity property access, so nothing is trained and nothing is
    downloaded beyond the already-cached tokenizer.
    """
    import tempfile

    from transformers import (
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = Seq2SeqTrainingArguments(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,
            num_train_epochs=1,
            predict_with_generate=True,
            report_to=[],
        )
        trainer = Seq2SeqTrainer(model=model, args=args, processing_class=tokenizer)
        constructed = trainer.model is model

    return {
        "seq2seq_training_arguments_importable": True,
        "seq2seq_trainer_importable": True,
        "trainer_constructed_with_model": constructed,
        "predict_with_generate_supported": True,
    }
