# SGTA — Synalytic Graph Transformer Augmentation
Research preview

SGTA is a geometric inductive bias for Transformer fine-tuning.
It introduces a small non-parametric synalytic memory that influences attention
and output logits based on the angular proximity of internal representations.
The approach does not modify the Transformer architecture and does not introduce
additional trainable parameters.

This repository presents a mechanistic research preview.
It is not intended as a benchmark submission or a claim of systematic
performance improvement.

## Motivation

In standard Transformer models, attention structure and semantic organization
emerge implicitly through optimization.
While this is sufficient for strong empirical performance,
it makes learning dynamics difficult to analyze and control.

SGTA explores whether introducing an explicit geometric structure
can influence internal representation organization during fine-tuning,
without altering the base architecture.

The focus of this work is on understanding how a controlled inductive bias
affects learning dynamics, stability across random initializations,
calibration behavior, and the structure of attention-induced interaction graphs.

## Experimental setup

Experiments are conducted under a strictly controlled protocol.

- Task: SST-2 (GLUE)
- Models: BERT-base, DistilBERT
- Evaluation: five random seeds
- Training: identical protocol for baseline and SGTA
- Architecture: unchanged, no additional trainable parameters

Evaluation includes validation accuracy, steps to reach a fixed target accuracy,
calibration error, and spectral diagnostics of attention-induced graphs.

## Summary of observations

For BERT-base, SGTA follows a learning trajectory comparable to the baseline.
Mean validation accuracy and variance largely overlap.
SGTA does not accelerate convergence and is not a speed optimization method.

Despite similar external performance metrics, SGTA consistently alters the
structure of attention-induced interaction graphs, as reflected by changes
in spectral properties.

In capacity-limited models such as DistilBERT, SGTA exhibits a clearer effect,
including higher peak accuracy, faster stabilization of learning dynamics,
and reduced sensitivity to initialization.

## Interpretation

SGTA is not designed as an optimization or performance enhancement method.
Its primary objective is to study how an explicit geometric inductive bias
affects internal learning dynamics and representation geometry.

The experimental results indicate that such structural effects can be observed
even when standard task-level metrics remain largely unchanged.

## Limitations and scope

This work does not claim state-of-the-art performance,
systematic accuracy gains, convergence acceleration,
causal guarantees from spectral diagnostics,
or validation across diverse tasks or domains.

All results should be interpreted as exploratory and diagnostic.

## Repository structure

- README.md — project overview and experimental findings
- RESEARCH_NOTES.md — conceptual framing and open research questions
- THEORY_OVERVIEW.md — overview of the formal theoretical framework
- DIRECTION.md — directions for theoretical and empirical collaboration
- figures/ — plots and diagnostics referenced in the analysis

## Reproducibility and status

The repository contains experiment configurations, training histories,
result summaries, and all figures shown in this preview.

Full training code and extended experiments are not released at this stage
to avoid premature or fragmentary reuse prior to peer-reviewed publication.

Status: research preview.
A full paper and code release are in preparation.
