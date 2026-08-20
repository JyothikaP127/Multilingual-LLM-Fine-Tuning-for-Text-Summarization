"""Loads XL-Sum language subsets directly from the Hub's auto-converted Parquet
revision (csebuetnlp/xlsum), avoiding the legacy trust_remote_code loading
script entirely. Only the requested split's shard files are ever downloaded --
never the full multi-language dataset.

File layout at the pinned commit (verified 2026-08-20, see configs/data.yaml
for the exact commit sha):

    english/{train/0000,train/0001,validation/0000,test/0000}.parquet
    hindi/{train/0000,train/0001,validation/0000,test/0000}.parquet
    telugu/{train/0000,validation/0000,test/0000}.parquet   (Telugu has one train shard)

Columns: id, url, title, summary, text.
"""
from __future__ import annotations

import random
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HUB_ID = "csebuetnlp/xlsum"

LANGUAGE_SHARDS: dict[str, dict[str, list[str]]] = {
    "english": {
        "train": ["english/train/0000.parquet", "english/train/0001.parquet"],
        "validation": ["english/validation/0000.parquet"],
        "test": ["english/test/0000.parquet"],
    },
    "hindi": {
        "train": ["hindi/train/0000.parquet", "hindi/train/0001.parquet"],
        "validation": ["hindi/validation/0000.parquet"],
        "test": ["hindi/test/0000.parquet"],
    },
    "telugu": {
        "train": ["telugu/train/0000.parquet"],
        "validation": ["telugu/validation/0000.parquet"],
        "test": ["telugu/test/0000.parquet"],
    },
}

Split = Literal["train", "validation", "test"]

COLUMNS = ["id", "url", "title", "summary", "text"]


def load_split_table(config_name: str, split: Split, revision: str) -> pa.Table:
    """Downloads (and caches, via huggingface_hub's local cache) only the
    parquet shard(s) for one language/split, and returns them as one table.
    """
    tables = []
    for rel_path in LANGUAGE_SHARDS[config_name][split]:
        local_path = hf_hub_download(
            HUB_ID, rel_path, repo_type="dataset", revision=revision
        )
        tables.append(pq.read_table(local_path, columns=COLUMNS))
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def load_balanced_subsample(
    config_name: str, split: Split, n: int, seed: int, revision: str
) -> list[dict]:
    """Deterministically samples n rows (without replacement) from one
    language/split, seeded so the same subsample is produced every run.
    """
    table = load_split_table(config_name, split, revision)
    rows = table.to_pylist()
    if n > len(rows):
        raise ValueError(
            f"{config_name}/{split} has only {len(rows)} rows available at "
            f"revision {revision}, but {n} were requested."
        )
    rng = random.Random(seed)
    idx = rng.sample(range(len(rows)), n)
    return [rows[i] for i in idx]
