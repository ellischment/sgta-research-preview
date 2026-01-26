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

## Graph structure

A weighted graph is constructed over the set of concepts.
Edges reflect geometric proximity and empirical co-activation.

This graph is not imposed externally.
It emerges from training dynamics and reflects internal organization
of representations rather than task-specific labels.

Spectral properties of this graph are used as diagnostic signals
to study structural changes induced by the synalytic bias.

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

## Status

The current results should be viewed as exploratory.
The emphasis is on diagnostic insight rather than performance improvement.

Further theoretical analysis and broader empirical validation
are planned before public code release.
