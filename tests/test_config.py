"""Step 1 config machinery: the paper config is frozen, variants inherit it, flags are validated."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from detmaso.config import load_config
from tests.conftest import ROOT, synthetic_config

PAPER = ROOT / "configs/csi300.yaml"
VARIANTS = {
    "csi300_nodiv.yaml": {"variant": "nodiv", "loss.c_div": 0.0},
    "csi300_single_mlp.yaml": {"variant": "single_mlp", "model.kind": "single_mlp", "loss.c_div": 0.0, "loss.c_ent": 0.0},
    "csi300_single_mlp_wide.yaml": {
        "variant": "single_mlp_wide", "model.kind": "single_mlp", "loss.c_div": 0.0, "loss.c_ent": 0.0, "model.hidden": (180, 90, 45),
    },
    "csi300_single_mlp_state.yaml": {
        "variant": "single_mlp_state", "model.kind": "single_mlp", "loss.c_div": 0.0, "loss.c_ent": 0.0, "model.market_state_input": True,
    },
}  # fmt: skip


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_paper_config_hashes_match_the_test_lock():
    """The paper config may never change: its hashes are the ones the test period was opened with."""
    cfg = load_config(PAPER)
    entry = json.loads((ROOT / "locks/csi300_test.json").read_text())["entries"][0]
    assert cfg.hash() == entry["config_hash"] == "d2a65e9a6a237d2c"
    assert cfg.data_hash() == entry["data_hash"] == "8e81084fb70adad6"
    assert cfg.variant == "paper" and cfg.model.kind == "detmaso" and not cfg.model.market_state_input


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_configs_change_only_what_they_declare(name):
    paper, variant = load_config(PAPER), load_config(ROOT / "configs" / name)
    a, b = _flat(paper.to_dict()), _flat(variant.to_dict())
    changed = {k: b[k] for k in a if a[k] != b[k]}
    assert changed == VARIANTS[name]
    assert variant.data_hash() == paper.data_hash()  # same prepared data, splits and regimes
    assert variant.hash() != paper.hash()


def test_post_paper_flags_enter_the_hash_only_when_changed(tmp_path):
    base = synthetic_config(tmp_path)
    explicit_defaults = dataclasses.replace(base, model=dataclasses.replace(base.model, kind="detmaso", market_state_input=False))
    assert explicit_defaults.hash() == base.hash()
    single = synthetic_config(tmp_path, variant="s", model={"kind": "single_mlp"}, loss={"c_div": 0.0, "c_ent": 0.0})
    plain_c = synthetic_config(tmp_path, variant="s", loss={"c_div": 0.0, "c_ent": 0.0})
    assert single.hash() != plain_c.hash()  # only model.kind differs
    assert single.data_hash() == base.data_hash()


def test_variant_label_is_never_hashed(tmp_path):
    base = synthetic_config(tmp_path)
    assert dataclasses.replace(base, variant="renamed").hash() == base.hash()


def test_base_chains_merge_sections_key_by_key(tmp_path):
    paper_text = PAPER.read_text()
    (tmp_path / "paper.yaml").write_text(paper_text)
    (tmp_path / "child.yaml").write_text("base: paper.yaml\nvariant: child\nloss:\n  c_div: 0.0\n")
    (tmp_path / "grandchild.yaml").write_text("base: child.yaml\nvariant: grandchild\nloss:\n  c_ent: 0.02\n")
    paper = load_config(tmp_path / "paper.yaml")
    child = load_config(tmp_path / "child.yaml")
    grandchild = load_config(tmp_path / "grandchild.yaml")
    assert child == dataclasses.replace(paper, variant="child", loss=dataclasses.replace(paper.loss, c_div=0.0))
    assert grandchild == dataclasses.replace(paper, variant="grandchild", loss=dataclasses.replace(paper.loss, c_div=0.0, c_ent=0.02))


def test_base_cycles_and_unknown_keys_raise(tmp_path):
    (tmp_path / "a.yaml").write_text("base: b.yaml\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")
    (tmp_path / "paper.yaml").write_text(PAPER.read_text())
    (tmp_path / "typo.yaml").write_text("base: paper.yaml\nvariant: typo\nmodle:\n  kind: single_mlp\n")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_config(tmp_path / "typo.yaml")
    (tmp_path / "typo2.yaml").write_text("base: paper.yaml\nvariant: typo2\nmodel:\n  knd: single_mlp\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(tmp_path / "typo2.yaml")


@pytest.mark.parametrize(
    "variant, overrides, message",
    [
        ("s", {"model": {"kind": "single_mlp"}}, "IC and Sharpe terms only"),  # c_div / c_ent left at the paper values
        ("s", {"model": {"market_state_input": True}}, "only supported with model.kind single_mlp"),
        ("paper", {"model": {"kind": "single_mlp"}, "loss": {"c_div": 0.0, "c_ent": 0.0}}, "give it a variant name"),
        ("s", {"model": {"kind": "two_mlps"}}, "model.kind must be one of"),
    ],
)
def test_invalid_flag_combinations_raise(tmp_path: Path, variant, overrides, message):
    with pytest.raises(ValueError, match=message):
        synthetic_config(tmp_path, variant=variant, **overrides)
