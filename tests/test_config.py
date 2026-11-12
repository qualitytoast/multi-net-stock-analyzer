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
        ("s", {"loss": {"c_bal": -0.1}}, "c_bal must be >= 0"),
        ("s", {"loss": {"balance_target": "regime"}}, "only matters with loss.c_bal > 0"),
        ("s", {"loss": {"c_bal": 0.1, "balance_target": "equal"}}, "balance_target must be one of"),
        ("s", {"model": {"kind": "single_mlp"}, "loss": {"c_div": 0.0, "c_ent": 0.0, "c_bal": 0.1}}, "needs model.kind detmaso"),
        ("paper", {"loss": {"c_bal": 0.1}}, "give it a variant name"),
        ("paper", {"placebo": {"shuffle_seed": 0}}, "give it a variant name"),
        ("s", {"placebo": {"shuffle_seed": 0, "block_days": 0}}, "block_days must be >= 1"),
    ],
)
def test_invalid_flag_combinations_raise(tmp_path: Path, variant, overrides, message):
    with pytest.raises(ValueError, match=message):
        synthetic_config(tmp_path, variant=variant, **overrides)


def test_scheduler_balance_switches_enter_the_hash_only_when_changed(tmp_path):
    from detmaso.config import PlaceboConfig

    base = synthetic_config(tmp_path)
    explicit = dataclasses.replace(
        base, loss=dataclasses.replace(base.loss, c_bal=0.0, balance_target="uniform"), train=dataclasses.replace(base.train, validate=True),
        placebo=PlaceboConfig(),
    )  # fmt: skip
    assert explicit.hash() == base.hash() and explicit.data_hash() == base.data_hash()
    one_int = synthetic_config(tmp_path, variant="b", loss={"c_bal": 1, "balance_target": "regime"})
    one_float = synthetic_config(tmp_path, variant="b", loss={"c_bal": 1.0, "balance_target": "regime"})
    assert one_int.hash() == one_float.hash() != base.hash() and isinstance(one_int.loss.c_bal, float)
    assert synthetic_config(tmp_path, variant="t", train={"validate": False}).hash() != base.hash()
    for cfg in (one_int, synthetic_config(tmp_path, variant="p", placebo={"shuffle_seed": 3})):
        assert cfg.data_hash() == base.data_hash()  # the prepared data is reused


ALLOWED_BALANCE_CHANGES = {"variant", "loss.c_bal", "loss.balance_target", "train.validate", "train.max_epochs", "train.seeds", "placebo.shuffle_seed"}
BALANCE_CONFIGS = sorted(str(p.relative_to(ROOT / "configs")) for p in (ROOT / "configs/scheduler_balance").rglob("*.yaml"))


@pytest.mark.parametrize("name", BALANCE_CONFIGS)
def test_scheduler_balance_configs_change_only_balance_settings(name):
    """Every pilot, strength and placebo file is its root config (paper or single network) plus balance settings."""
    cfg = load_config(ROOT / "configs" / name)
    root = load_config(ROOT / "configs" / ("csi300_single_mlp.yaml" if cfg.model.kind == "single_mlp" else "csi300.yaml"))
    a, b = _flat(root.to_dict()), _flat(cfg.to_dict())
    changed = {k for k in a if a[k] != b[k]}
    assert changed <= ALLOWED_BALANCE_CHANGES, changed - ALLOWED_BALANCE_CHANGES
    assert cfg.data_hash() == load_config(PAPER).data_hash() and cfg.variant != "paper"
    if cfg.loss.c_bal > 0:
        assert cfg.loss.balance_target == "regime"  # docs/RULES.md: mood-share targets (decision D2 b)
    if "placebo" in name:
        assert cfg.is_placebo and cfg.placebo.shuffle_seed == 0


# More data (docs/RULES.md, Rules ND and CU): membership filters, QA options ---------------------------------------
def test_membership_filters_enter_the_hashes_only_when_set(tmp_path):
    from detmaso.config import DataConfig, QAConfig

    base = synthetic_config(tmp_path)
    assert base.data.min_traded_days == 0 and base.data.exclude_codes == ()
    explicit = dataclasses.replace(base, data=dataclasses.replace(base.data, min_traded_days=0, exclude_codes=()))
    assert explicit.hash() == base.hash() and explicit.data_hash() == base.data_hash()
    listed = synthetic_config(tmp_path, variant="listed", data={"min_traded_days": 120})
    longer = synthetic_config(tmp_path, variant="listed", data={"min_traded_days": 121})
    cut = synthetic_config(tmp_path, variant="cut", data={"exclude_codes": ("SZ399300", "sh000300")})
    assert len({base.data_hash(), listed.data_hash(), longer.data_hash(), cut.data_hash()}) == 4  # each is its own prepared dataset
    assert len({base.hash(), listed.hash(), longer.hash(), cut.hash()}) == 4
    # codes are kept upper-case and sorted, so the order in the file never changes the fingerprint
    assert cut.data.exclude_codes == ("SH000300", "SZ399300")
    assert synthetic_config(tmp_path, variant="cut", data={"exclude_codes": ("SH000300", "SZ399300")}).data_hash() == cut.data_hash()
    # the QA report options change no data and no model: never hashed, and allowed under the name "paper"
    checked = dataclasses.replace(base, qa=QAConfig(membership_checks=True, worst_spans=25))
    assert checked.variant == "paper" and checked.hash() == base.hash() and checked.data_hash() == base.data_hash()
    assert DataConfig.__dataclass_fields__["min_traded_days"].metadata == DataConfig.__dataclass_fields__["exclude_codes"].metadata == {"hash": "post_paper"}


@pytest.mark.parametrize(
    "variant, overrides, message",
    [
        ("paper", {"data": {"min_traded_days": 120}}, "give it a variant name"),
        ("paper", {"data": {"exclude_codes": ("SH000300",)}}, "give it a variant name"),
        ("s", {"data": {"min_traded_days": -1}}, "min_traded_days must be a whole number >= 0"),
        ("s", {"data": {"min_traded_days": 120.0}}, "min_traded_days must be a whole number >= 0"),
        ("s", {"data": {"min_traded_days": True}}, "min_traded_days must be a whole number >= 0"),
        ("s", {"data": {"exclude_codes": "SH000300"}}, "must be a list of codes"),
        ("s", {"data": {"exclude_codes": ("SH000300", "sh000300")}}, "lists a code twice"),
        ("paper", {"qa": {"membership_checks": "yes"}}, "membership_checks must be true or false"),
        ("paper", {"qa": {"worst_spans": 0}}, "worst_spans must be a whole number >= 1"),
    ],
)
def test_invalid_membership_and_qa_settings_raise(tmp_path: Path, variant, overrides, message):
    with pytest.raises(ValueError, match=message):
        synthetic_config(tmp_path, variant=variant, **overrides)


def test_qa_and_membership_settings_load_from_yaml(tmp_path):
    (tmp_path / "paper.yaml").write_text(PAPER.read_text())
    (tmp_path / "qa.yaml").write_text("base: paper.yaml\nqa:\n  membership_checks: true\n")
    (tmp_path / "listed.yaml").write_text(
        "base: qa.yaml\nvariant: listed\ndata:\n  min_traded_days: 120\n  exclude_codes: [SZ399300, SH000300]\n"
    )
    (tmp_path / "typo.yaml").write_text("base: paper.yaml\nqa:\n  membership_check: true\n")
    paper, qa, listed = (load_config(tmp_path / n) for n in ("paper.yaml", "qa.yaml", "listed.yaml"))
    assert not paper.qa.membership_checks and qa.qa.membership_checks and qa.variant == "paper"
    assert qa.hash() == paper.hash() and qa.data_hash() == paper.data_hash()
    assert listed.qa.membership_checks and listed.data.min_traded_days == 120 and listed.data.exclude_codes == ("SH000300", "SZ399300")
    assert listed.hash() != paper.hash() and listed.data_hash() != paper.data_hash()
    assert qa.to_dict()["qa"] == {"membership_checks": True, "worst_spans": 10}  # written to a run group's config.json
    with pytest.raises(ValueError, match="unknown keys for QAConfig"):
        load_config(tmp_path / "typo.yaml")
