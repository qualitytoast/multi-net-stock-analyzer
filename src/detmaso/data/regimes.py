"""K-means market regimes, fit once on the training period and frozen (paper Sec 3.2.1; plan A13-A16)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Paper numbering 1..5 -> code index 0..4 (agent i <-> regime i).
REGIME_NAMES: tuple[str, ...] = ("Low-Vol Bull", "High-Vol Bull", "Sideways", "Low-Vol Bear", "Tail-Risk")


@dataclass
class RegimeModel:
    """Frozen scaler + centroids (in agent order). The single shared regime->agent mapping."""

    mean: np.ndarray  # (12,)
    std: np.ndarray  # (12,)
    centroids: np.ndarray  # (n_agents, 12), standardized units, row i = agent i
    names: list[str]
    state_names: list[str]
    silhouette: dict[int, float]
    fit_end: str  # last date used for fitting (look-ahead guard)

    def standardize(self, s: np.ndarray) -> np.ndarray:
        return (s - self.mean) / self.std

    def assign(self, s: np.ndarray) -> np.ndarray:
        """Agent index (0..n-1) of the nearest centroid for each row of raw s (unclipped, A13)."""
        z = self.standardize(np.asarray(s, dtype=np.float64))
        d2 = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=-1)
        return d2.argmin(axis=1)

    def to_json(self, path: Path) -> None:
        payload = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "centroids": self.centroids.tolist(),
            "names": self.names,
            "state_names": self.state_names,
            "silhouette": {str(k): v for k, v in self.silhouette.items()},
            "fit_end": self.fit_end,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "RegimeModel":
        p = json.loads(path.read_text())
        return cls(
            mean=np.array(p["mean"]),
            std=np.array(p["std"]),
            centroids=np.array(p["centroids"]),
            names=p["names"],
            state_names=p["state_names"],
            silhouette={int(k): v for k, v in p["silhouette"].items()},
            fit_end=p["fit_end"],
        )


def name_clusters(centroids_z: np.ndarray, state_names: list[str]) -> list[int]:
    """Deterministic naming rule (plan A16). Returns order[i] = cluster index for agent i.

    V = mean(sigma20, sigma60), M = mean(mu20, mu60) on standardized centroids.
    Tail-Risk = argmin(M - V); of the rest, the two highest M are the Bulls (lower V = Low-Vol
    Bull, other = High-Vol Bull); of the last two, lower M = Low-Vol Bear, other = Sideways.
    Written for exactly 5 clusters, the paper's realized n.
    """
    if centroids_z.shape[0] != 5:
        raise ValueError("the naming rule is defined for 5 regimes")
    idx = {name: i for i, name in enumerate(state_names)}
    V = (centroids_z[:, idx["sigma20"]] + centroids_z[:, idx["sigma60"]]) / 2
    M = (centroids_z[:, idx["mu20"]] + centroids_z[:, idx["mu60"]]) / 2
    remaining = list(range(5))
    tail = int(np.argmin(M - V))
    remaining.remove(tail)
    bulls = sorted(remaining, key=lambda c: M[c], reverse=True)[:2]
    low_vol_bull, high_vol_bull = sorted(bulls, key=lambda c: V[c])
    rest = [c for c in remaining if c not in bulls]
    low_vol_bear, sideways = sorted(rest, key=lambda c: M[c])
    return [low_vol_bull, high_vol_bull, sideways, low_vol_bear, tail]


def fit_regimes(
    state: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    n_agents: int,
    k_range: tuple[int, int],
    n_init: int,
    seed: int,
) -> RegimeModel:
    """Fit the scaler and K-means on training-period states only; log the silhouette curve."""
    s_train = state.loc[train_dates].dropna()
    X = s_train.to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / std
    silhouette: dict[int, float] = {}
    fitted: dict[int, KMeans] = {}
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed).fit(Z)
        silhouette[k] = float(silhouette_score(Z, km.labels_, metric="euclidean"))
        fitted[k] = km
    km = fitted[n_agents]
    order = name_clusters(km.cluster_centers_, list(state.columns))
    return RegimeModel(
        mean=mean,
        std=std,
        centroids=km.cluster_centers_[order],
        names=list(REGIME_NAMES),
        state_names=list(state.columns),
        silhouette=silhouette,
        fit_end=str(s_train.index.max().date()),
    )
