# Research notes and open questions

This document summarizes the current conceptual position of the SGTA project,
as well as open research directions that are not yet resolved.
It is intended for research discussion rather than for benchmarking or release.

## Conceptual framing

SGTA treats internal representations of a Transformer as points on a low-dimensional
unit sphere, where semantic information is encoded primarily through direction.
Concepts are modeled as regions in this space, characterized by a center direction
and an angular radius.

The memory layer maintains a finite set of such regions and updates them dynamically
based on model activity during training.
This allows concepts to sharpen, drift, or dissolve over time.

A key assumption of the project is that introducing an explicit geometric structure
at the level of representations can affect learning dynamics without modifying
the base architecture.

## Memory dynamics

Concept regions evolve according to simple update rules.

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

Several directions remain open and are of primary interest.

- Under what conditions do the memory dynamics admit stable partitions
  of representation space into multiple concepts?

- How does the behavior of SGTA depend on the dimensionality
  of the spherical embedding space?

- Can changes in graph spectral properties be linked to known notions
  of generalization or robustness?

- Is there a regime in which the synalytic bias provably improves
  calibration or stability across random initializations?

- How does the effect scale with model capacity and dataset complexity?

## Relation to theory

The project is informed by kinetic models of memory formation
and results on critical dimensionality in concept organization.
At present, these connections are primarily qualitative.

One long-term goal is to formalize the observed dynamics
in terms of simplified analytical models that capture
the interaction between optimization, geometry, and memory updates.

## Status

The current results should be viewed as exploratory.
The emphasis is on diagnostic insight rather than performance improvement.

Further theoretical analysis and broader empirical validation
are planned before public code release.
