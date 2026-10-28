# Theory overview

The SGTA project is supported by a formal mathematical formulation
covering representation geometry, memory dynamics,
graph construction, and stability considerations.

The theoretical material is currently under internal revision
and is not publicly released at this stage.

The formulation includes:

- A spherical representation space: hidden states are L2-normalized
  to the unit sphere S^{d-1}, where angular distance replaces
  Euclidean distance as the natural measure of similarity
  (section 3.1).

- A geometric model of concepts: each concept is a spherical cap
  defined by a center direction and an angular radius.
  A variational argument (Theorem 3.18) shows that under the
  assumption of radially decreasing density around a prototype,
  the optimal concept region is exactly a spherical cap,
  balancing coverage and compactness.

- A synalytic memory with explicit dynamics: centers are updated
  via stochastic approximation on the sphere, radii adapt via
  exponential moving average, and overlap between concepts
  is penalized to maintain interpretability (sections 4.2-4.5).

- Convergence guarantees: for the single-concept case, the center
  converges almost surely to the population prototype under
  a hemispherical coherence assumption and standard Robbins-Monro
  conditions on step sizes (Theorem 5.5).

- A graph structure over concepts: a weighted graph is constructed
  from geometric proximity and empirical co-activation.
  Under convergence of concept weights, the graph topology
  stabilizes at a finite time (Theorem A.26).
  Spectral diagnostics include algebraic connectivity λ₂,
  modularity, and density (sections 6.2-6.5).

Open theoretical questions include:

- Extension of the convergence proof to the multi-concept case
  with overlap regularization, where coupling between trajectories
  breaks independence.

- Rigorous continuous-time (ODE) formulation of the dynamics
  with stability analysis.

- Bounds on the number of stable, non-overlapping concepts
  as a function of spherical dimension.

- Formal connections between graph spectral properties and
  generalization or robustness.



The formal material is available for private discussion.
Specific results — including the variational justification
of spherical concept shape (Theorem 3.18), the center
convergence proof (Theorem 5.5), the graph stabilization
theorem (Theorem A.26), and the full appendix with proofs —
can be shared upon request.

The formal material is available for private discussion.
