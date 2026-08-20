"""Aggregation of per-example metric scores into per-language and Overall rows.

"Overall" is the unweighted mean of the three per-language mean scores (a
macro-average across languages), not a pooled mean over every example
lumped together. This is the methodologically robust choice: it does not
let whichever language happens to have more test examples dominate the
headline number. With the current balanced 500/500/500 test split the two
approaches are numerically identical, but the macro-average is what is
reported here and is what stays correct if the split sizes ever change.
"""
from __future__ import annotations

import statistics


def aggregate_language_scores(per_language_examples: dict[str, list[dict]]) -> dict[str, dict]:
    """per_language_examples: {language: [{metric_name: value, ...}, ...]}.
    Returns {language: {metric_name: mean_value}, "overall": {metric_name: mean_of_language_means}}.
    """
    metric_names = list(next(iter(per_language_examples.values()))[0].keys())

    per_language_means = {
        lang: {metric: statistics.mean(row[metric] for row in rows) for metric in metric_names}
        for lang, rows in per_language_examples.items()
    }

    overall = {
        metric: statistics.mean(per_language_means[lang][metric] for lang in per_language_means)
        for metric in metric_names
    }

    return {**per_language_means, "overall": overall}
