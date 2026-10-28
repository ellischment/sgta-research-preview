"""Sprint configuration. JSON on disk, dataclasses in memory, no extra dependencies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataCfg:
    # dataset name -> number of questions to take
    datasets: dict[str, int] = field(default_factory=lambda: {"triviaqa": 3000, "popqa": 3000, "nq_open": 1500})
    seed: int = 0
    test_fraction: float = 0.4
    system_prompt: str = "Answer the question with a short factual phrase only. No explanation."
    prompt_style: str = "chat"  # "chat" for instruct models, "qa" (3-shot) for base models


@dataclass
class GenCfg:
    max_new_tokens: int = 16
    batch_size: int = 32
    n_samples: int = 5  # stochastic samples used only to measure answer consistency
    temperature: float = 0.7
    top_p: float = 0.95
    consistency_threshold: float = 0.8  # wrong and consistency >= thr  ->  CW (associated-like)


@dataclass
class FitCfg:
    ks: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    pca_dim: int = 256  # clustering is done in a PCA view, centers are recomputed in full space
    radius_quantile: float = 0.9
    seed: int = 0


@dataclass
class PerturbCfg:
    n_examples: int = 3000  # stratified: CW, consistent FA, UW, inconsistent FA (store.stratified_pick)
    layer_fracs: list[float] = field(default_factory=lambda: [0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
    kappas: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])  # x median nearest-neighbour angle
    margin_grid: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    margin: bool = True  # H1b: gradient-directed minimal flip angle
    grad_batch_size: int = 16  # backward passes hold activations: keep this smaller than batch_size
    n_random: int = 8
    batch_size: int = 64
    # region-swap interventions (causal test of H2)
    n_intervene: int = 800
    intervene_k: int = 16
    intervene_layer_fracs: list[float] = field(default_factory=lambda: [0.375, 0.5, 0.625, 0.75])
    intervene_horizon: int = 3  # follow the moved state this many layers ahead (H2 vs inertia)


@dataclass
class SprintCfg:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    run_dir: str = "runs/qwen2.5-1.5b"
    dtype: str = "bfloat16"
    device: str = "cuda"
    data: DataCfg = field(default_factory=DataCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    fit: FitCfg = field(default_factory=FitCfg)
    perturb: PerturbCfg = field(default_factory=PerturbCfg)

    @staticmethod
    def load(path: str) -> SprintCfg:
        raw = json.loads(Path(path).read_text())
        cfg = SprintCfg()
        for k, v in raw.items():
            if k in ("data", "gen", "fit", "perturb"):
                sub = getattr(cfg, k)
                for kk, vv in v.items():
                    if not hasattr(sub, kk):
                        raise KeyError(f"unknown config key {k}.{kk}")
                    setattr(sub, kk, vv)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
            elif not k.startswith("_"):
                raise KeyError(f"unknown config key {k}")
        return cfg

    def dump(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
