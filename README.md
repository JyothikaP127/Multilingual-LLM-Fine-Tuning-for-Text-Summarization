# Multilingual LLM Fine-Tuning for Text Summarization

Parameter-efficient fine-tuning of mT5-small for multilingual news
summarization across English, Hindi, and Telugu using LoRA.

## Overview

This project investigates whether parameter-efficient fine-tuning of
`google/mt5-small` can improve multilingual text summarization compared
with zero-shot generation and simple extractive baselines.

The model is fine-tuned using LoRA (Low-Rank Adaptation), allowing only a
small fraction of the model parameters to be updated while keeping the
base model frozen.

Languages:
- English
- Hindi
- Telugu

Dataset:
- XL-Sum

## Dataset

For each language:

| Split | Examples |
|---|---:|
| Train | 2,000 |
| Validation | 300 |
| Test | 500 |

Total:
- 6,000 training examples
- 900 validation examples
- 1,500 test examples

The dataset pipeline performs:
- deterministic sampling
- missing/empty-field validation
- train/validation/test ID-overlap checks
- Unicode/script validation
- source/target token-length analysis
- seq2seq preprocessing with `-100` label masking

## Model

Base model:

`google/mt5-small`

Fine-tuning method:

**LoRA (Low-Rank Adaptation)**

Configuration:

| Parameter | Value |
|---|---:|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q, v |
| Task type | SEQ_2_SEQ_LM |
| Trainable parameters | 344,064 |
| Base model parameters | 556,291,456 |
| Trainable percentage | 0.0618% |

The base model parameters remain frozen during training.

## Sequence Length Analysis

Source truncation was evaluated at 512, 768 and 1024 tokens.

The final configuration uses:

- `max_source_length = 768`
- `max_target_length = 128`

Measured source truncation at 768:

| Language | Truncated |
|---|---:|
| English | 34.0% |
| Hindi | 52.6% |
| Telugu | 76.5% |

Telugu has substantially longer tokenized source articles, resulting in
higher truncation rates. Increasing the global sequence length reduces
absolute truncation but does not eliminate the language disparity.

## Baselines

Three baselines were evaluated:

### 1. Zero-shot mT5-small

The pretrained mT5-small model is evaluated without fine-tuning.

### 2. Lead-1

The first sentence of each article is used as the summary.

### 3. Lead-3

The first three sentences are used as the summary.

## Results

All models were evaluated on the same 1,500-example test set.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|---|---:|---:|---:|---:|
| Zero-shot mT5-small | 0.0794 | 0.0171 | 0.0752 | 0.5743 |
| Lead-1 | 0.2517 | 0.0846 | 0.1880 | 0.6935 |
| Lead-3 | 0.2647 | 0.1004 | 0.1757 | 0.6918 |
| **LoRA mT5-small** | **0.2411** | **0.0888** | **0.1980** | **0.6788** |

## Language-wise Results

### English

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|---|---:|---:|---:|---:|
| Zero-shot | 0.0936 | 0.0182 | 0.0869 | 0.5867 |
| Lead-1 | 0.2401 | 0.0376 | 0.1815 | 0.6915 |
| Lead-3 | 0.2407 | 0.0564 | 0.1643 | 0.6855 |
| **LoRA** | **0.2645** | **0.0682** | **0.2137** | 0.6738 |

LoRA achieves the strongest ROUGE performance for English.

### Hindi

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|---|---:|---:|---:|---:|
| Zero-shot | 0.0856 | 0.0260 | 0.0804 | 0.5952 |
| Lead-1 | **0.3010** | 0.1228 | **0.2173** | **0.7074** |
| Lead-3 | 0.3002 | **0.1348** | 0.1936 | 0.7022 |
| LoRA | 0.2679 | 0.1144 | 0.2157 | 0.6916 |

Extractive baselines remain stronger than LoRA for Hindi.

### Telugu

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|---|---:|---:|---:|---:|
| Zero-shot | 0.0591 | 0.0070 | 0.0583 | 0.5410 |
| Lead-1 | 0.2141 | 0.0933 | 0.1654 | 0.6815 |
| Lead-3 | **0.2532** | **0.1100** | **0.1692** | **0.6876** |
| LoRA | 0.1911 | 0.0838 | 0.1647 | 0.6709 |

Telugu has the lowest LoRA performance and the highest source truncation.

## Key Findings

1. LoRA fine-tuning substantially improves mT5-small over zero-shot
   summarization.

2. LoRA achieves the highest overall ROUGE-L score among the evaluated
   systems.

3. Simple Lead-N extractive baselines remain highly competitive and
   outperform LoRA on several overall and language-specific metrics.

4. English benefits most clearly from LoRA fine-tuning.

5. Telugu remains challenging and has substantially higher source-side
   truncation.

6. The relationship between Telugu's truncation and lower performance is
   consistent with the observed results, but the current experiment does
   not establish truncation as a causal factor.

## Inference

The final LoRA model was evaluated on 1,500 test examples using a Kaggle
Tesla T4 GPU.

Inference configuration:

- batch size: 8
- maximum source length: 768
- maximum generated tokens: 128
- beam size: 4
- precision: BF16

Measured inference time:

| Language | Time | Examples/sec |
|---|---:|---:|
| English | 71.1 s | 7.03 |
| Hindi | 128.5 s | 3.89 |
| Telugu | 104.3 s | 4.79 |
| **Total** | **303.9 s** | **4.94** |

## Evaluation

The project evaluates:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

ROUGE uses SentencePiece tokenization and BERTScore uses the multilingual
BERT backbone.

## Project Structure

```text
configs/
├── base.yaml
├── data.yaml
├── eval.yaml
├── lora.yaml
└── hardware/

src/
├── data/
├── eval/
├── models/
├── train/
└── utils/

scripts/
├── train_lora.py
├── run_lora_test_inference.py
├── run_baseline_zero_shot.py
├── run_lead_n_baseline.py
├── evaluate_predictions.py
└── ...

results/
├── metrics/
├── qa/
└── qualitative_samples/

requirements.txt
README.md