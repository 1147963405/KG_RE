# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical KG Relation Extraction — a cascading (two-stage) binary tagging framework for extracting (subject, predicate, object) triples from Chinese clinical text, based on CMeKG. BERT encoder + sequence labeling.

## Commands

```bash
# Install dependencies
pip install torch transformers numpy

# Train the model
python model_re/train.py

# Run inference demo (5 test scenarios)
python model_re/demo_re.py

# Run quick demo (single COVID-19 text)
python model_re/demo.py
```

## Architecture

```
Input text → split by "。" → extract_spoes(sentence) → triples
```

**Two-stage cascading model:**

1. **Model4s** (subject extraction): BERT → Dropout → Linear(768→2) → Sigmoid
   - Output: (batch, seq_len, 2) — per-token prob of subject start/end
   - Returns output + BERT hidden states for stage 2

2. **Model4po** (object + predicate extraction): Hidden + subject embedding → Dropout → Linear(768→num_p*2) → Sigmoid
   - Input: BERT hidden states + subject position embedding
   - Output: (batch, seq_len, num_p*2) — per-token per-predicate object start/end probs

## Key Files

| File | Purpose |
|------|---------|
| `model_re/medical_re.py` | Core module: config, data pipeline, Model4s, Model4po, training loop, inference, evaluation |
| `model_re/train.py` | Training entry point |
| `model_re/demo_re.py` | Inference demo with 5 test scenarios |
| `model/medical_re/predicate.json` | 23 medical relation predicates |
| `model/medical_re/train_example.json` | 442 training samples with SPO annotations |
| `docs/clinical_relation_plan.md` | Proposed expansion to 59 predicates (not yet implemented) |

## Critical Notes

- **Hardcoded paths**: All file paths in `config` class point to `D:/CMe_KG/CMeKG/...` — update to `D:/AI/KG_RE/...` before running.
- **CPU-only**: All `.cuda()` calls are commented out. The model runs on CPU — enable CUDA by uncommenting those lines if GPU is available.
- **Num_p hardcoded**: `Model4po` output layer is `Linear(768, num_p*2)` where `num_p = 23`. If adding predicates (see docs/clinical_relation_plan.md), update `config.num_p` and `predicate.json`.
- **No requirements.txt**: Dependencies are `torch`, `transformers`, `numpy`.
- **Long text**: Handled via sentence splitting on "。" with 128-char truncation per sentence.

## Model Weights (gitignored)

- `model/medical_re/model_re.pkl` (1.22 GB) — trained cascading model checkpoint
- `model/medical_re/pytorch_model.bin` (411 MB) — BERT-base-chinese pretrained weights
- `save/model_re_trained.pkl` — output after running train.py (currently 0 bytes)

## Evaluation

F1 / Precision / Recall computed via exact SPO triple matching (set-based). Run `run_train()` in `medical_re.py` which splits 8:2 and evaluates on the held-out set after training.