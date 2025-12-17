# SGTA — Synalytic Graph Transformer Augmentation

SGTA is a lightweight geometric inductive bias for Transformer fine-tuning.
It introduces a synalytic memory that induces an additive bias to attention / logits
based on spherical proximity of representations, without modifying the architecture
or adding trainable parameters.

---

## Motivation

While Transformer attention is typically learned implicitly, SGTA explores whether
explicit geometric structure can shape attention dynamics and internal organization
of representations during fine-tuning.

The goal of this work is **not** to maximize benchmark performance, but to study
how a controlled inductive bias affects training dynamics, calibration, and internal structure.

---

## Experimental setup

- Task: SST-2 (GLUE)
- Models: BERT-base, DistilBERT
- Evaluation: multi-seed (5 seeds)
- Metrics: validation accuracy, steps-to-target accuracy, ECE, spectral diagnostics
- No architectural changes; identical training protocol to baseline

---

## Key findings

### BERT-base

- SGTA achieves **comparable validation accuracy** to the baseline.
- No acceleration of convergence is observed; SGTA reaches target accuracy in more steps on average.
- Multi-seed analysis shows that gains are **not systematic**, highlighting sensitivity to initialization.
- Despite similar performance, SGTA consistently **modifies the attention graph structure**,
  as reflected by changes in the Laplacian spectral gap λ₂.

### DistilBERT (capacity-limited regime)

- SGTA shows a **clearer and more stable improvement** over the baseline.
- This suggests that the proposed inductive bias is more effective when representational capacity is limited.

---

## Interpretation

These results indicate that SGTA acts as a structural regularizer rather than
a performance-oriented optimization.
Its primary effect is on **how** the model learns, not only **how well** it performs.

---

## What this is NOT

- This is not a drop-in performance boost.
- This is not a new architecture.
- This is not a SOTA-oriented method.

---

## Why this matters

Understanding and controlling internal learning dynamics is critical for:
- interpretability,
- stability,
- capacity-limited models,
- and principled architectural design.

SGTA provides a minimal and diagnostic-friendly testbed for such investigations.
