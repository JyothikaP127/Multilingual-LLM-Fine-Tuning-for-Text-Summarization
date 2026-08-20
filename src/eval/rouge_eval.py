"""ROUGE-1/2/L scoring using the multilingual-safe SentencePiece tokenizer
from src/eval/rouge_tokenizer.py -- never rouge_score's default
whitespace/regex tokenizer, which is tuned for English and mis-segments
Hindi/Telugu.
"""
from __future__ import annotations

from src.eval.rouge_tokenizer import SentencePieceRougeTokenizer


def build_rouge_scorer(tokenizer_name: str = "google/mt5-small"):
    from rouge_score import rouge_scorer

    sp_tokenizer = SentencePieceRougeTokenizer(tokenizer_name)
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=sp_tokenizer)


def score_pairs(scorer, predictions: list[str], references: list[str]) -> list[dict]:
    """rouge_score.RougeScorer.score(target, prediction) -- reference first,
    prediction second; this order is preserved here to match that API.
    """
    results = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        results.append(
            {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )
    return results
