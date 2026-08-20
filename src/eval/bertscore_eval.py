"""BERTScore evaluation using an explicitly pinned multilingual backbone.

bert_score's built-in lang2model default mapping has no entry for 'hi' or
'te' and defaults 'en' to an English-only backbone (roberta-large) -- see
the earlier environment audit. This module always passes model_type
explicitly (configs/eval.yaml: bert-base-multilingual-cased) so the same
backbone scores all three languages and results stay cross-lingually
comparable.

Uses bert_score.BERTScorer (a class you construct once) rather than the
bert_score.score() functional API, which reloads the underlying model from
disk on every call -- wasteful when scoring multiple languages/models in
one evaluation run. Build one BERTScorer with load_bertscorer() and reuse
it across every scoring call in a script.
"""
from __future__ import annotations


def load_bertscorer(model_type: str, num_layers: int):
    from bert_score import BERTScorer

    return BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        rescale_with_baseline=False,  # no verified baseline file for this backbone across hi/te
    )


def score_with_scorer(scorer, predictions: list[str], references: list[str]) -> dict:
    precision, recall, f1 = scorer.score(predictions, references)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }
