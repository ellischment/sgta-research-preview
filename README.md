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

## Conceptual framework

Internal representations of a Transformer are treated as points on a
low-dimensional unit sphere, where semantic information is encoded
primarily through direction rather than magnitude.

Concepts are modeled as spherical caps — regions on the sphere characterized
by a center direction and an angular radius.
A finite set of such regions constitutes a synalytic memory that evolves
during training.

The memory interacts with the model through representation-dependent biasing.
Three bias modes are supported:
- **logit bias** — applied at the classifier output,
- **attention bias** — applied before self-attention,
- **dual** — both simultaneously.

Centers adapt to the stream of hidden states via stochastic approximation,
with explicit update rules for both centers and radii.
Overlap between concepts is penalized to maintain interpretability.
A weighted graph is constructed over concepts, with edges reflecting
geometric proximity and empirical co-activation.
Spectral properties of this graph (e.g., algebraic connectivity λ₂)
serve as diagnostic signals during training.

A formal mathematical treatment of the model —
including the variational justification of spherical concept shape,
convergence proofs for center dynamics, and graph construction —
is available in the accompanying draft paper (in preparation, contact for details).

## Experimental setup

Experiments are conducted under a strictly controlled protocol.

- Task: SST-2 (GLUE)
- Models: BERT-base, DistilBERT
- Evaluation: five random seeds
- Training: identical protocol for baseline and SGTA
- Architecture: unchanged, no additional trainable parameters

Evaluation includes validation accuracy, steps to reach a fixed target accuracy,
calibration error, and spectral diagnostics of attention-induced graphs.

## Experimental pipeline

A complete experimental pipeline is implemented in `SGTA_first.ipynb`
and organized around nine diagnostic maps:

1. **Geometry** — sphere validity, comparison with GMM and PCA baselines
2. **Dynamics** — birth/death/drift of concepts, overlap, NDC over training
3. **Functional** — comparison of all bias modes (none, logit, attn, dual)
4. **Graph** — structure and spectral properties of the concept graph
5. **Stability** — concept alignment across random seeds (Hungarian matching)
6. **OOD** — distance-to-concept as error predictor (ROC-AUC)
7. **Ablation** — component-wise comparison of bias modes
8. **Worst-seed** — diagnostic on the hardest random initialization
9. **Attribute alignment** — plug-in for external concept labels (CUB/VAW)

The pipeline is configurable via dataclass-based configuration objects
covering dataset, training, concept memory, and experiment settings.

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

## Current status (May 2026)

The notebook contains a functional implementation of all
components described above. It has been tested on a minimal setup
(small synthetic SST-2 subset, BERT-base-uncased, single seed, 1 epoch)
and produces expected outputs.

The code has not yet been systematically verified across
all configurations and model scales. Several parts are noted
as smoke tests (checkpointing disabled, minimal data) and require
validation before full release.

Extended experiments, cleaner package structure, and full public release
are planned following peer review.

A draft paper describing the mathematical framework
is available for private discussion upon request.

## Limitations and scope

This work does not claim state-of-the-art performance,
systematic accuracy gains, convergence acceleration,
causal guarantees from spectral diagnostics,
or validation across diverse tasks or domains.

All results should be interpreted as exploratory and diagnostic.

## Repository structure

- README.md — project overview and experimental findings
- SGTA_first.ipynb — full experimental pipeline
- RESEARCH_NOTES.md — conceptual framing and open research questions
- THEORY_OVERVIEW.md — overview of the formal theoretical framework
- DIRECTION.md — directions for theoretical and empirical collaboration
- QUESTIONS.md — core open research questions and discussion entry points
- figures/ — plots and diagnostics referenced in the analysis

## Reproducibility and status

The repository contains experiment configurations, training histories,
result summaries, and all figures shown in this preview.

Full training code and extended experiments are not released at this stage
to avoid premature or fragmentary reuse prior to peer-reviewed publication.

Status: research preview.
A full paper and code release are in preparation.
