"""SentencePiece-based tokenizer adapter for rouge_score.

rouge_score.RougeScorer's default tokenizer splits on whitespace/regex word
boundaries tuned for English -- it under-segments or mis-segments Hindi and
Telugu (Telugu in particular does not reliably whitespace-delimit the way
English does). rouge_score.RougeScorer accepts any object exposing a
`tokenize(text) -> list[str]` method (see rouge_score.tokenizers.Tokenizer),
so this wraps the mT5 SentencePiece tokenizer to produce ROUGE overlap units
that are consistent across English, Hindi and Telugu -- the same subword
vocabulary is used for every language, so no language-specific rule set is
needed and no language goes through a different code path than the others.
"""
from __future__ import annotations


class SentencePieceRougeTokenizer:
    """Minimal implementation of rouge_score's Tokenizer interface."""

    def __init__(self, tokenizer_name: str = "google/mt5-small"):
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(self, text: str) -> list[str]:
        # convert_ids_to_tokens keeps the SentencePiece subword pieces (e.g.
        # "▁word") rather than decoding back to characters, so ROUGE's
        # n-gram/LCS overlap operates on consistent subword units.
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return self._tokenizer.convert_ids_to_tokens(ids)
