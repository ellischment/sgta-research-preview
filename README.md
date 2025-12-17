# SGTA (Synalytic Graph Transformer Augmentation)

**Research preview**

This repository presents a **research preview** of **SGTA**, a lightweight geometric inductive bias for Transformer fine-tuning based on a self-organizing spherical memory.

The purpose of this preview is to expose **clean experimental signals and mechanistic diagnostics**, not to provide a full benchmark or production-ready implementation.

## Core ideф

SGTA introduces a small, non-parametric **synalytic memory** that induces an additive bias to attention or logits based on spherical proximity between hidden representations and a set of adaptive memory caps.

Key properties:

* does **not** modify the Transformer architecture
* does **not** add trainable parameters to the backbone
* acts as a soft geometric inductive bias during fine-tuning
* can be injected at the attention or logit level


## Experimental setting

* Task: SST-2 (binary sentiment classification)
* Backbones: BERT-base, DistilBERT
* Evaluation: validation accuracy, calibration (ECE), training efficiency
* All curves are **aggregated over multiple random seeds** using a fixed training configuration
* Target accuracy for efficiency analysis: **≥ 0.91 validation accuracy**


## Key results (SST-2)

### BERT-base: validation accuracy

Mean validation accuracy over seeds
Baseline vs SGTA

![Val acc](figures/fig1_BERT_SST2_val_acc_baseline_vs_SGTA.png)

### BERT-base: calibration

Expected Calibration Error (ECE, lower is better), mean over seeds

![ECE](figures/fig2_BERT_SST2_ECE_baseline_vs_SGTA.png)

### Training efficiency

Steps to reach ≥ 0.91 validation accuracy
Aggregated over seeds

![Steps](figures/fig0_BERT_SST2_steps_to_target_baseline_vs_SGTA.png)

### Mechanistic diagnostic (SGTA only)

Laplacian spectral gap λ₂ of the **synalytic memory graph** during training
Mean over seeds

![Gap](figures/fig3_BERT_SST2_gap_lambda2_SGTA.png)

## Capacity-limited regime: DistilBERT

To evaluate the effect of SGTA under reduced model capacity, we repeat the experiment with **DistilBERT** and a simplified **logit-level synalytic bias**.

### DistilBERT: validation accuracy

Mean over seeds

![Distil curve](figures/fig4_DistilBERT_SST2_val_acc_baseline_vs_SGlogit.png)


### DistilBERT: best validation accuracy

Peak validation accuracy aggregated over seeds

![Distil best](figures/fig5_DistilBERT_SST2_best_val_acc_baseline_vs_SGlogit.png)

**Observation:**
SG-logit reaches the target accuracy significantly faster and achieves higher peak accuracy compared to the baseline, indicating that the synalytic bias is most effective in **capacity-limited regimes**.


## What this preview shows

* SGTA provides a **consistent, low-overhead inductive bias** during fine-tuning
* On saturated models (BERT-base on SST-2), the effect is **modest but stable**
* On capacity-limited models (DistilBERT), the effect is **substantially stronger**
* The synalytic memory exhibits **structural stabilization** during training, reflected by the evolution of the spectral gap


## Limitations and scope

This preview **does not claim**:

* large-scale multi-seed benchmarking
* superiority across diverse tasks or datasets
* causal guarantees from the spectral diagnostics
* wall-clock or cost improvements (only step-based efficiency is reported)

These aspects are the subject of ongoing work.


## Reproducibility

This repository contains:

* experiment configurations
* aggregated training histories
* result summaries
* all figures shown above (generated via a unified plotting pipeline)

The full training code and implementation details are intentionally **not released at this stage** to avoid fragmentary or misleading reuse prior to peer-reviewed publication.


## Status

Research preview.
Full paper and code release are in preparation.
