"""Token-length distribution and truncation reporting.

Used to quantify -- rather than assume -- how much source content a given
max_source_length cuts off, per language. mT5's SentencePiece tokenizer
produces very different token counts per language for comparable articles
(see results/qa/dataset_inspection_report.json), so this must be measured
per language, not assumed uniform.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass


@dataclass
class LengthStats:
    field: str
    n: int
    mean: float
    median: float
    p90: float
    p95: float
    max: int
    truncated_count: int
    truncated_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * p), len(sorted_values) - 1)
    return float(sorted_values[idx])


def compute_length_stats(token_lengths: list[int], field: str, max_length: int) -> LengthStats:
    lengths = sorted(token_lengths)
    n = len(lengths)
    truncated = sum(1 for length in lengths if length > max_length)
    return LengthStats(
        field=field,
        n=n,
        mean=statistics.mean(lengths) if lengths else 0.0,
        median=_percentile(lengths, 0.5),
        p90=_percentile(lengths, 0.90),
        p95=_percentile(lengths, 0.95),
        max=lengths[-1] if lengths else 0,
        truncated_count=truncated,
        truncated_pct=100.0 * truncated / n if n else 0.0,
    )


def compute_stats_at_thresholds(
    token_lengths: list[int], field: str, thresholds: list[int]
) -> dict[int, LengthStats]:
    """Same underlying distribution, evaluated against several candidate
    max-length cutoffs at once (mean/median/p90/p95/max are threshold-
    independent; truncated_count/pct are recomputed per threshold).
    """
    return {threshold: compute_length_stats(token_lengths, field, threshold) for threshold in thresholds}


def tokenize_lengths(tokenizer, texts: list[str]) -> list[int]:
    return [len(ids) for ids in tokenizer(texts)["input_ids"]]
