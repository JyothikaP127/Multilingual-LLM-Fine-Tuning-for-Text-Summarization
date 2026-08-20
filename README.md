# Multilingual LLM Fine-Tuning for Text Summarization

Fine-tuning `google/mt5-small` for multilingual abstractive text summarization in **English, Hindi, and Telugu** using **Parameter-Efficient Fine-Tuning (LoRA)**.

The project implements an end-to-end pipeline covering:

- Multilingual dataset preparation
- Data cleaning and validation
- Tokenization and truncation analysis
- Zero-shot baseline evaluation
- Extractive Lead-N baselines
- LoRA fine-tuning of mT5-small
- Adapter verification and checkpoint validation
- Multilingual ROUGE and BERTScore evaluation
- Qualitative comparison of generated summaries

---

## Overview

This project investigates whether a relatively small multilingual language model can be adapted for text summarization across English, Hindi, and Telugu using LoRA instead of updating the entire model.

The base model used is:

**`google/mt5-small`**

The model is fine-tuned on a deterministic subset of the **XL-Sum** dataset.

### Languages

| Language | Train | Validation | Test |
|---|---:|---:|---:|
| English | 2,000 | 300 | 500 |
| Hindi | 2,000 | 300 | 500 |
| Telugu | 2,000 | 300 | 500 |
| **Total** | **6,000** | **900** | **1,500** |

The test set is kept separate from training and validation and is used only for final evaluation.

---

# Project Pipeline

```text
                 XL-Sum Dataset
                       │
                       ▼
              Data Cleaning & QA
                       │
                       ▼
             Deterministic Sampling
                       │
                       ▼
               Tokenization
                       │
                       ▼
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
  Baseline Models                LoRA Fine-Tuning
        │                             │
        │                        mT5-small
        │                             │
        │                       LoRA q/v layers
        │                             │
        └──────────────┬──────────────┘
                       ▼
              Test-set Generation
                       │
                       ▼
             ROUGE + BERTScore
                       │
                       ▼
             Multilingual Analysis