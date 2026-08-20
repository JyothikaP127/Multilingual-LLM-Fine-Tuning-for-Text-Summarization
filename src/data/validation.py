"""Data-quality validation checks for the XL-Sum pipeline.

These are diagnostic/QA checks run by scripts/inspect_data_pipeline.py --
they report problems, they do not silently fix or drop data (cleaning of
genuinely invalid rows already happened in src/data/load_xlsum.py, on the
full pool, before sampling).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

UNICODE_RANGES = {
    "hindi": (0x0900, 0x097F),  # Devanagari
    "telugu": (0x0C00, 0x0C7F),  # Telugu
}


def contains_expected_script(text: str, language: str) -> bool:
    """For Hindi/Telugu, checks the text actually contains at least one
    character from the expected Unicode block -- catches encoding
    corruption (e.g. mojibake, wrong-language rows) that a plain
    non-empty-string check would miss. English has no single expected
    block, so it always passes this check.
    """
    if language not in UNICODE_RANGES:
        return True
    lo, hi = UNICODE_RANGES[language]
    return any(lo <= ord(ch) <= hi for ch in text)


def check_missing_fields(row: dict, required_fields: list[str]) -> list[str]:
    return [f for f in required_fields if f not in row or row[f] is None]


def check_empty(row: dict, field: str) -> bool:
    return not str(row.get(field, "")).strip()


def check_excessive_length(token_length: int, threshold: int = 5000) -> bool:
    """Flags outlier articles far beyond typical length (a QA signal for
    scraping artifacts, e.g. multiple concatenated articles) -- separate
    from ordinary truncation at max_source_length, which is expected and
    already measured in the truncation report.
    """
    return token_length > threshold


@dataclass
class LabelCheckResult:
    total_examples: int
    pad_id_leaked_into_labels: int  # labels containing pad_token_id where -100 was expected
    all_ignore_index_at_pad_positions: bool
    empty_decoded_targets: int

    def to_dict(self) -> dict:
        return asdict(self)


def check_label_correctness(processed_examples: list[dict], tokenizer) -> LabelCheckResult:
    """Verifies labels were built correctly:
      - every pad_token_id position in the raw label token stream must have
        been replaced by -100 (the cross-entropy ignore_index), not left as
        pad_token_id, and vice versa (once -100 appears, everything after it
        should also be -100, since padding is only ever a trailing block for
        this tokenizer/padding strategy).
      - decoding the non-ignored label tokens must not be empty (an empty
        decoded target would mean a summary was tokenized into nothing).
    """
    from src.data.preprocess import decode_labels

    pad_id = tokenizer.pad_token_id
    pad_leak = 0
    empty_targets = 0
    inconsistent_trailing = 0

    for ex in processed_examples:
        labels = ex["labels"]
        if pad_id in labels:
            pad_leak += 1

        # once -100 begins, it should never revert to a real token afterwards
        seen_ignore = False
        for token_id in labels:
            if token_id == -100:
                seen_ignore = True
            elif seen_ignore:
                inconsistent_trailing += 1
                break

        decoded = decode_labels(tokenizer, labels)
        if not decoded.strip():
            empty_targets += 1

    return LabelCheckResult(
        total_examples=len(processed_examples),
        pad_id_leaked_into_labels=pad_leak,
        all_ignore_index_at_pad_positions=(inconsistent_trailing == 0),
        empty_decoded_targets=empty_targets,
    )


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_display(text: str, max_chars: int = 200) -> str:
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    return collapsed[:max_chars] + ("..." if len(collapsed) > max_chars else "")
