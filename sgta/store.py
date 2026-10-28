"""Reading and writing run directories. Everything downstream of extraction is CPU-only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(run_dir: Path) -> None:
    files = sorted(
        p for p in run_dir.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256" and "results" not in p.parts
    )
    lines = [f"{sha256_file(p)}  {p.relative_to(run_dir)}" for p in files]
    (run_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


@dataclass
class Run:
    """Lazy view of a run directory."""

    path: Path

    def __post_init__(self):
        self.path = Path(self.path)
        self._cache: dict[str, object] = {}

    # ---------- core ----------
    @property
    def info(self) -> dict:
        if "info" not in self._cache:
            self._cache["info"] = json.loads((self.path / "info.json").read_text())
        return self._cache["info"]

    @property
    def meta(self) -> pd.DataFrame:
        if "meta" not in self._cache:
            self._cache["meta"] = pd.read_csv(self.path / "meta.csv")
        return self._cache["meta"]

    @property
    def states(self) -> np.ndarray:
        """float16 [N, L+1, D], memory-mapped."""
        if "states" not in self._cache:
            self._cache["states"] = np.load(self.path / "states.npy", mmap_mode="r")
        return self._cache["states"]

    @property
    def n_layers(self) -> int:
        return int(self.info["n_layers"])

    @property
    def lens(self) -> dict[str, np.ndarray]:
        if "lens" not in self._cache:
            z = np.load(self.path / "lens.npz")
            self._cache["lens"] = {k: z[k] for k in z.files}
        return self._cache["lens"]

    @property
    def norms(self) -> np.ndarray:
        if "norms" not in self._cache:
            self._cache["norms"] = np.load(self.path / "norms.npy")
        return self._cache["norms"]

    # ---------- labels ----------
    @property
    def y_err(self) -> np.ndarray:
        return (1 - self.meta["correct"].to_numpy()).astype(int)

    @property
    def is_train(self) -> np.ndarray:
        return (self.meta["split"] == "train").to_numpy()

    @property
    def etype(self) -> np.ndarray:
        return self.meta["etype"].to_numpy()

    @property
    def consistency_threshold(self) -> float:
        """The threshold that defined CW at extraction (config.json), 0.8 when the run has no config."""
        try:
            return float(json.loads((self.path / "config.json").read_text())["gen"]["consistency_threshold"])
        except Exception:
            return 0.8

    @property
    def consistent(self) -> np.ndarray:
        """Answer repeats across samples (consistency >= threshold). Missing consistency counts as not consistent,
        the same rule extraction uses for CW. CW are consistent by definition; FA are a mix."""
        return self.meta["consistency"].fillna(0).to_numpy(float) >= self.consistency_threshold

    # ---------- fitted objects ----------
    def fit_stats(self) -> dict[str, np.ndarray]:
        if "stats" not in self._cache:
            z = np.load(self.path / "fit" / "stats.npz")
            self._cache["stats"] = {k: z[k] for k in z.files}
        return self._cache["stats"]

    def partition(self, kind: str, k: int) -> dict[str, np.ndarray]:
        key = f"part_{kind}_K{k}"
        if key not in self._cache:
            z = np.load(self.path / "fit" / f"{key}.npz")
            self._cache[key] = {kk: z[kk] for kk in z.files}
        return self._cache[key]

    def has(self, rel: str) -> bool:
        return (self.path / rel).exists()

    def perturb(self) -> dict[str, np.ndarray] | None:
        if not self.has("perturb.npz"):
            return None
        if "perturb" not in self._cache:
            z = np.load(self.path / "perturb.npz")
            self._cache["perturb"] = {k: z[k] for k in z.files}
        return self._cache["perturb"]

    def intervene(self) -> pd.DataFrame | None:
        if not self.has("intervene.csv"):
            return None
        return pd.read_csv(self.path / "intervene.csv")

    # ---------- derived views ----------
    def layer_z(self, layer: int) -> np.ndarray:
        """Standardised, unit-normalised state at one layer: float32 [N, D]."""
        st = self.fit_stats()
        h = np.asarray(self.states[:, layer, :], dtype=np.float32)
        s = (h - st["mu"][layer]) / st["sd"][layer]
        n = np.linalg.norm(s, axis=1, keepdims=True)
        return s / np.clip(n, 1e-8, None)

    def layer_s(self, layer: int) -> np.ndarray:
        st = self.fit_stats()
        h = np.asarray(self.states[:, layer, :], dtype=np.float32)
        return (h - st["mu"][layer]) / st["sd"][layer]

    def results_dir(self) -> Path:
        p = self.path / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p


def stratified_pick(etype: np.ndarray, consistent: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Examples for the perturbation stage, sized for the main test (CW vs consistent FA).

    Quotas: CW n/3, consistent FA n/3, UW n/6, inconsistent FA n/6. A stratum that is short is taken
    whole and the remainder is filled at random from everything not yet picked. Sorted indices.
    """
    rng = np.random.default_rng(seed)
    fa = etype == "FA"
    strata = [(etype == "CW", n // 3), (fa & consistent, n // 3), (etype == "UW", n // 6), (fa & ~consistent, n // 6)]
    picks = [rng.permutation(np.flatnonzero(m))[:q] for m, q in strata]
    got = np.concatenate(picks).astype(int)
    if len(got) < n:
        rest = np.setdiff1d(np.arange(len(etype)), got)
        got = np.concatenate([got, rng.permutation(rest)[: n - len(got)]])
    return np.sort(got)


def assign_split(ids, test_fraction: float, seed: int) -> np.ndarray:
    """Deterministic split by hash of id so that re-extraction never moves examples across splits."""
    out = []
    for i in ids:
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out.append("test" if h < test_fraction else "train")
    return np.array(out)
