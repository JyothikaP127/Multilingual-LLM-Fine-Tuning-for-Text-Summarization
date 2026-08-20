"""Seq2seq tokenization for XL-Sum examples.

One code path handles all three languages: the mT5 SentencePiece tokenizer
is inherently multilingual, so there is no English-specific branching,
regex, or language-conditional logic anywhere in this module. Language is
carried through as metadata only (for later per-language evaluation), never
used to change how tokenization happens.
"""
from __future__ import annotations


def build_tokenizer(tokenizer_name: str = "google/mt5-small"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def preprocess_examples(
    examples: list[dict],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
) -> list[dict]:
    """Tokenizes raw XL-Sum rows (each with at least id/text/summary/language)
    into model-ready seq2seq examples: input_ids/attention_mask for the
    source article, labels for the reference summary.

    Deterministic: tokenization is a pure function of the input text and the
    fixed max_length/truncation/padding settings -- no randomness involved.

    Padding is static ("max_length") rather than dynamic, so every example
    in this QA/inspection pipeline has a uniform, predictable shape; a
    dynamic per-batch padding collator can be swapped in later at actual
    training time for memory efficiency without changing this function.

    Label padding positions are set to -100 (not tokenizer.pad_token_id) --
    this is the standard seq2seq convention: -100 is the ignore_index that
    cross-entropy loss skips, so the model is never penalized for not
    predicting pad tokens. Getting this wrong (leaving pad_token_id in
    labels) is a common seq2seq bug and is exactly what
    src/data/validation.py's check_label_correctness checks for.
    """
    texts = [ex["text"] for ex in examples]
    summaries = [ex["summary"] for ex in examples]

    model_inputs = tokenizer(
        texts,
        max_length=max_source_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=summaries,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    pad_id = tokenizer.pad_token_id
    label_ids = [
        [(token_id if token_id != pad_id else -100) for token_id in seq]
        for seq in labels["input_ids"]
    ]

    processed = []
    for i, ex in enumerate(examples):
        processed.append(
            {
                "id": ex["id"],
                "language": ex["language"],
                "language_iso": ex.get("language_iso"),
                "input_ids": model_inputs["input_ids"][i],
                "attention_mask": model_inputs["attention_mask"][i],
                "labels": label_ids[i],
            }
        )
    return processed


def decode_labels(tokenizer, labels: list[int]) -> str:
    """Reverses the -100 substitution before decoding, so processed labels
    can be inspected/decoded back to text for QA purposes.
    """
    pad_id = tokenizer.pad_token_id
    restored = [pad_id if token_id == -100 else token_id for token_id in labels]
    return tokenizer.decode(restored, skip_special_tokens=True)
