"""LoRA/PEFT fine-tuning for google/mt5-small.

Every function here takes its hyperparameters as arguments -- nothing is
hard-coded. scripts/train_lora.py is the only place that reads YAML config
and decides which values to pass in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------------


def load_pretrained_model(model_name: str):
    """Loads the ACTUAL pretrained checkpoint (not a config-only random-init
    model). See results/qa/model_parameters_actual_weights.json for why this
    matters: the real weights are NOT tied (embedding != lm_head), giving
    ~300.18M total params, not the ~172.1M a config-only model would report.
    """
    from transformers import MT5ForConditionalGeneration

    return MT5ForConditionalGeneration.from_pretrained(model_name)


def verify_target_modules(model, target_modules: list[str]) -> dict[str, list[str]]:
    """Verifies the configured LoRA target_modules actually exist as Linear
    layers on THIS loaded model, by introspecting named_modules() directly --
    never assumed from another architecture's convention (e.g. LLaMA's
    q_proj/v_proj). Raises loudly if a configured name isn't found.
    """
    import torch.nn as nn

    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    grouped: dict[str, list[str]] = {}
    for name in linear_names:
        suffix = name.rsplit(".", 1)[-1]
        grouped.setdefault(suffix, []).append(name)

    missing = [t for t in target_modules if t not in grouped]
    if missing:
        raise RuntimeError(
            f"Configured LoRA target_modules {target_modules} include names not found "
            f"as Linear modules on the actual loaded model: {missing}. "
            f"Available Linear-module suffixes: {sorted(grouped)}"
        )
    return grouped


def attach_lora(model, lora_cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    verify_target_modules(model, lora_cfg["target_modules"])

    task_type = TaskType[lora_cfg["task_type"]] if isinstance(lora_cfg["task_type"], str) else lora_cfg["task_type"]
    peft_config = LoraConfig(
        task_type=task_type,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
    )
    return get_peft_model(model, peft_config)


# ---------------------------------------------------------------------------
# Parameter / freezing verification
# ---------------------------------------------------------------------------


@dataclass
class ParameterReport:
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_pct_of_total: float
    trainable_pct_of_non_embedding_backbone: float
    non_embedding_backbone_params: int
    embedding_trainable_leak: list[str]
    lm_head_trainable_leak: list[str]
    non_lora_trainable_params: list[str]
    freezing_clean: bool

    def to_dict(self) -> dict:
        return asdict(self)


NON_EMBEDDING_BACKBONE_PARAMS = 44_062_080  # verified in results/qa/model_parameters_actual_weights.json


def build_parameter_report(peft_model) -> ParameterReport:
    total = 0
    trainable = 0
    embedding_leak = []
    lm_head_leak = []
    non_lora_trainable = []

    for name, p in peft_model.named_parameters():
        total += p.numel()
        if not p.requires_grad:
            continue
        trainable += p.numel()
        if "lora_" not in name:
            non_lora_trainable.append(name)
            if "shared" in name:
                embedding_leak.append(name)
            if "lm_head" in name:
                lm_head_leak.append(name)

    frozen = total - trainable
    return ParameterReport(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        trainable_pct_of_total=100.0 * trainable / total,
        trainable_pct_of_non_embedding_backbone=100.0 * trainable / NON_EMBEDDING_BACKBONE_PARAMS,
        non_embedding_backbone_params=NON_EMBEDDING_BACKBONE_PARAMS,
        embedding_trainable_leak=embedding_leak,
        lm_head_trainable_leak=lm_head_leak,
        non_lora_trainable_params=non_lora_trainable,
        freezing_clean=not embedding_leak and not lm_head_leak and not non_lora_trainable,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TokenizedDataset:
    """Thin torch.utils.data.Dataset wrapper around the already-tokenized,
    statically-padded examples from src/data/preprocess.py. Fixed-length
    tensors mean Trainer's default collator (plain stacking) is correct --
    no dynamic-padding collator is needed here.
    """

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        import torch

        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def build_training_arguments(output_dir: str, hp: dict, hardware_profile):
    from transformers import Seq2SeqTrainingArguments

    from src.utils.hardware import resolve_precision

    precision = resolve_precision(hp["precision"], hardware_profile)
    if precision == "fp16":
        raise ValueError(
            "precision resolved to fp16, but mT5 is documented to produce NaN losses under "
            "fp16 mixed precision (pretrained in bf16). Refusing to silently proceed -- set "
            "precision explicitly to 'fp32' or 'bf16' if this is truly intended."
        )

    return (
        Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=hp["per_device_train_batch_size"],
            per_device_eval_batch_size=hp["per_device_train_batch_size"],
            gradient_accumulation_steps=hp["gradient_accumulation_steps"],
            num_train_epochs=hp["num_train_epochs"],
            learning_rate=hp["learning_rate"],
            warmup_ratio=hp["warmup_ratio"],
            weight_decay=hp["weight_decay"],
            bf16=(precision == "bf16"),
            fp16=False,
            logging_steps=hp.get("logging_steps", 10),
            eval_strategy="epoch",
            # "no", not "epoch": Trainer's automatic mid-training checkpointing saves a full
            # resumption snapshot (optimizer.pt, scheduler.pt, rng_state.pth, a full duplicate
            # tokenizer, plus another copy of the adapter) into a checkpoint-N/ subdirectory --
            # harmless in content (still adapter-only, never the base model) but wasteful
            # duplication we don't need, since save_lora_checkpoint() does one explicit,
            # intentional adapter+tokenizer save at the end of training.
            save_strategy="no",
            report_to=[],
            seed=hp["seed"],
            predict_with_generate=False,
            remove_unused_columns=False,
        ),
        precision,
    )


def build_trainer(peft_model, tokenizer, args, train_dataset, eval_dataset):
    from transformers import Seq2SeqTrainer

    return Seq2SeqTrainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


def save_lora_checkpoint(peft_model, tokenizer, output_dir: str) -> None:
    """Saves ONLY the LoRA adapter (peft_model.save_pretrained's default
    behavior -- adapter_config.json + adapter weights, not the ~300M-param
    base model) plus the tokenizer.
    """
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
