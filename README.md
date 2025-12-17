# SGTA (Synalytic Graph Transformer Augmentation)
**Research preview**

This repository presents a research preview of **SGTA**, a lightweight geometric inductive bias for Transformer fine-tuning based on a self-organizing spherical memory.

The goal of this preview is to expose **clean experimental signals and mechanistic diagnostics**, rather than to provide a full benchmark or production-ready system.


## Core idea (high-level)

SGTA introduces a small, non-parametric **synalytic memory** that induces an additive bias to attention or logits based on spherical proximity between hidden representations and adaptive memory caps.

The method:
- does **not** modify the Transformer architecture
- does **not** add trainable parameters to the backbone
- acts as a soft geometric inductive bias during fine-tuning

## Key observations (SST-2)

### BERT-base: validation accuracy
Baseline vs SGTA
![Val acc](figures/fig1_BERT_SST2_val_acc_vs_steps_baseline_vs_SGTA.png)

**Observation:**  
On a saturated model (BERT-base on SST-2), SGTA does not consistently(in some seed it does) improve peak validation accuracy across random seeds.

### BERT-base: calibration (ECE, lower is better)
![ECE](figures/fig2_BERT_SST2_ECE_vs_steps_baseline_vs_SGTA.png)

**Observation:**  
SGTA exhibits slightly improved or comparable calibration behavior during training, without degrading accuracy.

### Mechanistic diagnostic (SGTA only)
Laplacian spectral gap λ₂ of the **synalytic memory graph**
![Gap](figures/fig3_BERT_SST2_laplacian_gap_vs_steps_SGTA.png)

**Observation:**  
The synalytic memory graph exhibits structured spectral dynamics during training, suggesting progressive stabilization of the induced memory topology.

This diagnostic is intended as a **mechanistic probe**, not a causal claim.


## Capacity-limited regime: DistilBERT

SGTA shows a **substantially stronger and more consistent effect** when model capacity is limited.

### DistilBERT: validation accuracy
![Distil curve](figures/fig4_DistilBERT_SST2_val_acc_vs_steps_baseline_vs_SGlogit.png)

### DistilBERT: best validation accuracy
![Distil best](figures/fig5_DistilBERT_SST2_best_val_acc_baseline_vs_SGlogit.png)

**Observation:**  
SG-logit achieves higher peak validation accuracy and reaches competitive performance earlier, indicating that the synalytic bias is most effective in low-capacity regimes.


## What this preview shows

- SGTA acts as a **low-overhead geometric inductive bias**
- Effects are **limited on saturated models**
- Effects are **pronounced in capacity-limited settings**
- The synalytic memory exhibits **non-trivial structural dynamics** during training

## Limitations and scope

This preview does **not claim**:
- superiority on large, saturated models
- multi-task or large-scale benchmark dominance
- causal guarantees from spectral diagnostics
- wall-clock or cost improvements

These aspects are subject to ongoing work.


## Reproducibility

This repository contains:
- experiment configurations
- multi-seed training histories
- result summaries
- all figures shown above

Full training code is intentionally **not released at this stage** to avoid fragmentary or misleading reuse prior to peer-reviewed publication.

## Status

Research preview.  
Paper and full code release are in preparation.
