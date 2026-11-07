"""Build the model selected by model.kind (default: the paper's Det-MASO)."""

from __future__ import annotations

from torch import nn

from detmaso.config import Config
from detmaso.model.detmaso import DetMASO
from detmaso.model.single import SingleMLP


def build_model(cfg: Config, d_in: int) -> nn.Module:
    if cfg.model.kind == "detmaso":
        return DetMASO.from_config(cfg, d_in)
    if cfg.model.kind == "single_mlp":
        return SingleMLP.from_config(cfg, d_in)
    raise ValueError(f"unknown model.kind {cfg.model.kind!r}")
