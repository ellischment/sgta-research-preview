# Open research questions

This document lists the core research questions raised by the SGTA project.
These questions are intended as entry points for discussion and collaboration.

## Memory dynamics and stability

- Under what conditions do the memory update rules admit stable partitions
  of the representation space into multiple concepts?

- How sensitive are the stationary regimes to update rates,
  regularization strength, and stochasticity of training?

  We know that the convergence proof imposes standard
  Robbins-Monro conditions on step sizes. Radius dynamics use an
  exponential moving average which has not been
  analyzed theoretically.

  The interaction  between the three rates (center step γ_t, radius step β_t,
  overlap penalty λ) is unexplored.

- Can the discrete-time dynamics be approximated by a continuous-time system
  with analyzable stability properties?

## Geometry and dimensionality

- How does the behavior of SGTA depend on the dimensionality
  of the spherical representation space?

  *What is known:* Higher dimensionality increases sphere surface area
  and reduces expected overlap between random caps via concentration
  of measure. The draft paper discusses this in sections 2.4 and 3.6.
  *What is open:* No bounds are proved on the number of stable concepts
  as a function of dimension d. No systematic empirical study exists.

- Is there a critical dimension at which the number of stable,
  non-overlapping concepts is maximized?

- The variational argument (draft paper, Theorem 3.18) justifies
  spherical shape under the assumption of radial monotonicity
  of density around a prototype (Assumption 3.17). Under this
  assumption, the optimal concept region is a spherical cap.
  How robust is this to deviations from radial symmetry in real
  hidden states?

  *What is known:* Under the assumption, spherical caps are the exact
  minimizers of the functional F(A) = λ·σ(A) − ∫_A p(z)dσ(z).
  *What is open:* Whether hidden states of trained Transformers
  exhibit approximately radial density around semantic prototypes.
  If they do not, the model may need extension to elliptical or
  non-parametric region shapes.

## Internal structure and generalization

- Can changes in the synalytic graph structure be related to known notions
  of generalization, robustness, or representation compression?

  *What is open:* The graph and its spectral metrics (λ₂, modularity,
  density) are defined (section 6.2, Algorithm 2), but no formal
  connection to generalization bounds or compression theory
  has been established.

- Are there structural indicators that precede or predict
  changes in task-level performance?  This would require temporal analysis across training
  to check whether graph metrics change before accuracy changes.
  The question is formulated but has not been investigated.

- The pipeline tracks λ₂ (algebraic connectivity), overlap,
  and NDC (number of detected concepts) during training.
  Do any of these metrics carry information about model behavior
  beyond what is visible in validation accuracy?

  *What is open:* The metrics are defined and computable, but their
  relationship to task-level behavior has not been studied.

## Capacity effects

- Why does the synalytic bias exhibit a clearer and more stable effect
  in capacity-limited models?
  A plausible hypothesis is that capacity-limited
  models have less representational redundancy, so an external
  geometric structure provides a stronger signal. This has not been
  tested theoretically or empirically.

- Can this behavior be explained by simplified theoretical models?
*What is open:* A toy model with controlled capacity could
  potentially isolate the mechanism, but this has not been
  attempted.

## Bias mode interactions

- The framework supports four bias modes: none, logit (classifier bias),
  attn (attention mask bias), and dual (both). Is the effect of dual
  mode additive, or does interaction between the two bias pathways emerge?

  *What is open:* The two bias pathways operate at different stages
  of computation (attention vs. classification head). It is not known
  whether their combination is simply additive or whether attention
  bias reshapes representations in a way that changes the effect
  of the logit bias.

- Under what conditions might attention-level bias behave differently
  from logit-level bias?

  *What is open:* Hypothesis: attention bias may be more impactful
  when the task requires reorganizing which tokens attend to which,
  while logit bias may be sufficient when only the decision boundary
  needs adjustment. This has not been tested.

## Empirical calibration

- Does SGTA affect model calibration (expected calibration error,
  negative log-likelihood, Brier score)?

  *What is open:* The pipeline is instrumented to compute ECE, NLL,
  and Brier alongside accuracy. No systematic study of calibration
  effects across modes, seeds, or data scales has been conducted.
