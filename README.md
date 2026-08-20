# Multilingual LLM Fine-Tuning for Text Summarization

**Research question:** Can parameter-efficient LoRA fine-tuning improve multilingual text
summarization while substantially reducing the number of trainable parameters compared
with full fine-tuning?

Status: **scaffolding + environment/model/dataset verification stage.** No training has
been run yet.

## Fixed decisions

| Item | Decision |
|---|---|
| Model | `google/mt5-small` |
| Framework | PyTorch + Hugging Face Transformers + PEFT (LoRA) |
| Dataset | [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) (`csebuetnlp/xlsum`), CC-BY-NC-SA-4.0 |
| Languages | English, Hindi, Telugu |
| Split sizes (per language, balanced) | train 2,000 / validation 300 / test 500 |
| Experiments | zero-shot base, Lead-N extractive baseline, full fine-tuning, LoRA fine-tuning |
| Evaluation | ROUGE-1/2/L (multilingual-safe tokenization) + multilingual BERTScore, per-language breakdown, bootstrap CIs, qualitative error analysis |

## Hardware model

This project is developed on a **CPU-only laptop** (no CUDA GPU) and trained on a
**free-tier cloud GPU** (Colab / Kaggle). Nothing in the code assumes a specific machine:

- Device and BF16 support are auto-detected at runtime (`src/utils/hardware.py`); the
  local machine falls back to CPU/fp32, a cloud GPU auto-selects bf16 if the GPU is
  Ampere-or-newer, otherwise fp32. FP16 is intentionally never auto-selected — mT5 was
  pretrained in bf16 and is known to produce NaN losses under fp16 mixed precision.
- Config (`configs/hardware/*.yaml`) can force a precision/batch profile regardless of
  what's detected.
- No absolute or machine-specific paths appear anywhere in the code; all paths are
  relative to the repo root or come from config.

## Known non-obvious facts baked into this design (verified, not assumed)

- **mT5-small has ~172.1M total parameters, not ~300M.** Its `config.json` says
  `tie_word_embeddings: false`, but `transformers`' `MT5Config` source forces
  `tie_word_embeddings = True` at load time regardless (a documented override in the
  library itself) — the input embedding and output projection share one tensor. That
  tied embedding matrix is ~128.1M params, ~74% of the entire model. Trainable-parameter
  percentages are therefore reported both against the full total and against the
  ~44M-parameter non-embedding backbone.
- **LoRA target modules are `["q", "v"]`**, verified directly against the
  `MT5Attention` source in `transformers==5.14.1` (layers are named `q`/`k`/`v`/`o`, not
  `q_proj`/`v_proj`) and cross-checked against PEFT's own
  `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`, which lists the same for both
  `t5` and `mt5`.
- **Token length differs sharply by language under the mT5 SentencePiece tokenizer**
  (e.g. Telugu articles run ~2.2x longer in tokens than English articles covering
  similar stories). A fixed `max_source_length` truncates Telugu far more aggressively
  than English — this is measured and reported per language as a QA artifact
  (`results/qa/`), not left as a silent confound.
- **`bert_score`'s default `lang2model` mapping has no entry for Hindi or Telugu** and
  defaults English to an English-only backbone (`roberta-large`). This project pins one
  multilingual backbone (`bert-base-multilingual-cased`) explicitly for all three
  languages so BERTScore numbers are actually comparable across languages.
- **XL-Sum is loaded from the Hub's auto-converted Parquet revision**
  (`csebuetnlp/xlsum`, commit `ab04842511fc357feae53281fd104aa3c36dff07`), not the
  legacy `trust_remote_code` loading script, since `datasets` is moving away from
  script-based loading and pinning a commit (rather than the mutable
  `refs/convert/parquet` ref) keeps the dataset snapshot reproducible.

## License note

XL-Sum is distributed under **CC-BY-NC-SA-4.0** (attribution, non-commercial,
share-alike). This project uses it for non-commercial research/portfolio purposes only.
Citation: Hasan et al., "XL-Sum: Large-Scale Multilingual Abstractive Summarization for
44 Languages", Findings of ACL-IJCNLP 2021.

## Setup

```bash
# 1. Install torch for your platform (see requirements.txt header for exact commands)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest
pip install -r requirements.txt
```

## Repository layout

```
configs/            YAML configs (base, data, model, lora, full_finetune, eval, hardware/)
src/
  data/              XL-Sum loading (Parquet-based) + tokenizer-based length stats
  models/            Model/PEFT architecture inspection helpers
  train/             (not yet implemented)
  eval/              (not yet implemented)
  utils/             hardware auto-detection, seeding, JSON report saving
scripts/
  verify_environment.py   environment inspection -> results/qa/environment_report.json
  inspect_model.py        mT5 architecture + LoRA compatibility -> results/qa/model_architecture_report.json
  inspect_dataset.py      XL-Sum structure, ID-overlap check, truncation report -> results/qa/dataset_inspection_report.json
results/             QA/inspection artifacts and (later) experiment metrics
report/              Final write-up (not yet started)
```

## Running the verification scripts

```bash
python scripts/verify_environment.py
python scripts/inspect_model.py
python scripts/inspect_dataset.py
```

Each writes a JSON report under `results/qa/` and prints a human-readable summary.
None of these scripts train a model, download the full dataset, or write checkpoints.

## Reproducibility

- Global seed fixed at `42` (`configs/base.yaml`, applied via `src/utils/seed.py`).
- All package versions pinned in `requirements.txt`, verified as a mutually compatible
  set against `google/mt5-small` + PEFT LoRA before pinning (see commit history / dev
  notes for the verification steps).
- Dataset snapshot pinned to an exact Hub commit, not a mutable ref or branch.
