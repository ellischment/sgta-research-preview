# Research notes

This document summarizes the current conceptual position of the SGTA project
and records open questions that arise from the existing results.
It is intended for research discussion rather than benchmarking or release.

## Conceptual framing

SGTA treats internal representations of a Transformer as points on a
low-dimensional unit sphere, where semantic information is encoded
primarily through direction.

Concepts are modeled as regions in this space, characterized by a center
direction and an angular radius.
A finite set of such regions constitutes a synalytic memory.

The memory layer evolves during training and interacts with the model
through representation-dependent biasing, without modifying the base
architecture.

## Memory dynamics

Concept regions evolve according to explicit update rules.

Repeated activation leads to increased concentration of a concept.
Lack of activation leads to gradual expansion.
Centers adapt to changes in the representation distribution.
Excessive overlap between concepts is penalized.

These dynamics are designed to admit stable regimes and to avoid collapse
into a single dominant region.

Convergence of a single center to its population prototype has been proved
under a hemispherical coherence assumption (draft paper, Theorem 5.5).
The proof uses a stochastic approximation argument on the sphere and
requires standard Robbins-Monro conditions on step sizes.

## Graph structure

A weighted graph is constructed over the set of concepts.
Edges reflect geometric proximity and empirical co-activation.

This graph is not imposed externally.
It emerges from training dynamics and reflects internal organization
of representations rather than task-specific labels.

Spectral properties of this graph are used as diagnostic signals
to study structural changes induced by the synalytic bias.

A formal construction of the graph and its spectral diagnostics
(algebraic connectivity λ₂, modularity, density) is given in the
draft paper (sections 6.2-6.5). Under the assumption that concept
weights converge, the graph topology stabilizes at a finite time
(Theorem A.26 in the Appendix).

## Bias modes

The framework supports four modes of interaction with the base model:

- **none** — baseline, no bias applied,
- **logit** — bias added to classifier output,
- **attn** — bias added to attention mask before self-attention,
- **dual** — both biases applied simultaneously.

The two bias pathways operate at different stages of computation
(attention vs. classification head). Whether their combination is
additive or interactive is not yet understood.

## Theoretical results

A formal mathematical treatment of the model is in preparation.
Key results include:

- A variational justification for modeling concepts as spherical caps:
  under radial monotonicity of density, spherical caps are the exact
  minimizers of the functional balancing coverage and compactness
  (Theorem 3.18).

- A convergence proof for center dynamics in the single-concept case
  (Theorem 5.5).

- A construction of the concept graph and proof of its topological
  stabilization under convergence (Theorem A.26).

Several open theoretical questions remain, including extension of the
convergence proof to the multi-concept case with overlap regularization
and a rigorous continuous-time (ODE) formulation of the dynamics.

## Open research questions

- Under what conditions do the memory dynamics admit stable partitions
  of representation space into multiple concepts?

- How does the behavior of SGTA depend on the dimensionality
  of the spherical embedding space?

- Can changes in graph spectral properties be linked to known notions
  of generalization or robustness?

- Is there a regime in which the synalytic bias improves calibration
  or stability across random initializations?

- How does the effect scale with model capacity and dataset complexity?

- Does the dual bias mode combine additively, or does interaction
  between the two bias pathways emerge?

## Status

The current results should be viewed as exploratory.
The emphasis is on diagnostic insight rather than performance improvement.

A draft paper with the formal theoretical framework exists and
a functional experimental pipeline has been implemented.
Further theoretical analysis and broader empirical validation
are planned before public code release.
