"""SGTA-lab: a hypothesis tournament over the internal dynamics of LLM inference.

Layout of a run directory (one per model):

    runs/<model_slug>/
        info.json        model id, n_layers, hidden size, conventions
        meta.csv         one row per example (labels, confidence, error type, split)
        states.npy       float16 [N, L+1, D]  last prompt token, index 0 = embeddings,
                         index l = output of decoder block l-1 (pre final norm)
        env.txt          pip freeze of the environment that produced the run (versions also in info.json)
        lens.npz         logit-lens arrays [N, L+1]: prob/rank of eventual first answer token, entropy;
                         ent_attn / ent_mlp: lens entropy of each block's attention / MLP output (TriLens-style)
        norms.npy        float32 [N, L+1] raw L2 norms
        fit/             standardisation stats and per-layer partitions (caps / euclid / random)
        perturb.npz      layer-resolved perturbation responses (basin depth profile: random, structured and
                         probe directions) and gradient margins (cheapest flip angle, H1b)
        intervene.csv    region-swap interventions followed 3 layers ahead, with static / delta inertia baselines
        results/         hypothesis verdicts, metrics, figures, REPORT.md
"""

__version__ = "0.7.0"
