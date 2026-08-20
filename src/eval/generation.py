"""Zero-shot generation utilities for the un-fine-tuned google/mt5-small
baseline. Generation parameters are always loaded from configs/eval.yaml's
`generation` section and passed explicitly to `.generate()` -- never left
as unstated Hugging Face library defaults, since those differ across model
checkpoints/versions and would make the baseline non-reproducible.
"""
from __future__ import annotations

import time


def load_base_model(model_name: str):
    from transformers import MT5ForConditionalGeneration

    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return model


def build_generation_kwargs(gen_cfg: dict) -> dict:
    return {
        "max_new_tokens": gen_cfg["max_new_tokens"],
        "num_beams": gen_cfg["num_beams"],
        "length_penalty": gen_cfg["length_penalty"],
        "no_repeat_ngram_size": gen_cfg["no_repeat_ngram_size"],
        "early_stopping": gen_cfg["early_stopping"],
        "do_sample": gen_cfg["do_sample"],
    }


def generate_summaries(
    model,
    tokenizer,
    texts: list[str],
    max_source_length: int,
    generation_kwargs: dict,
) -> tuple[list[str], float]:
    """Runs batched generation on already-frozen model weights (model.eval(),
    no gradient tracking). Returns (decoded_summaries, wall_clock_seconds).
    """
    import torch

    inputs = tokenizer(
        texts,
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.time() - start

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return decoded, elapsed
