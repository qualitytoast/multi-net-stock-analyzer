"""Placebo answers (docs/RULES.md, Rule P) and the refusals that keep placebo runs away from real ones."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from detmaso.config import PlaceboConfig
from detmaso.data.placebo import shuffle_answers
from tests.conftest import ROOT, synthetic_config
from tests.test_end_to_end import synthetic_panel_and_splits


def _panel_with_holes():
    panel, _ = synthetic_panel_and_splits(n_days=60, n_stocks=12)
    lab = panel.label_ok.clone()
    lab[5, 3] = False  # stock 3 misses a label on day 5 (block 0: days 0-19)
    lab[25:27, 7] = False  # stock 7 misses days 25-26 (block 1: days 20-39)
    trd = panel.tradable.clone()
    trd[30, 2] = False  # stock 2 cannot be bought on day 30
    return dataclasses.replace(panel, label_ok=lab, tradable=trd)


def test_only_stocks_labelled_all_block_take_part_and_others_are_dropped():
    panel = _panel_with_holes()
    p = shuffle_answers(panel, 20, seed=0)
    lab, trd = p.label_ok.numpy(), p.tradable.numpy()
    assert not lab[0:20, 3].any() and lab[20:, 3].all()  # out for its whole block only
    assert not lab[20:40, 7].any() and lab[0:20, 7].all() and lab[40:, 7].all()
    assert not trd[0:20, 3].any() and not trd[20:40, 7].any()  # left out of the portfolio too
    assert not trd[30, 2] and trd[31, 2]  # members keep their own "can be bought" status
    assert lab.sum(axis=1).min() == 11


def test_answers_are_permuted_within_blocks_with_one_partner_map_for_labels_and_returns():
    panel = _panel_with_holes()
    p = shuffle_answers(panel, 20, seed=0)
    y, y2 = panel.y.numpy(), p.y.numpy()
    r, r2 = panel.r_next.numpy(), p.r_next.numpy()
    lab = p.label_ok.numpy()
    partners = {}
    for t in range(60):
        m = np.flatnonzero(lab[t])
        assert np.array_equal(np.sort(y[t, m]), np.sort(y2[t, m]))  # each day's set of answers is unchanged
        assert np.all(y2[t, m] != y[t, m])  # nobody keeps its own answers
        for c in m:
            partner = int(np.flatnonzero(y[t] == y2[t, c])[0])
            assert r2[t, c] == r[t, partner]  # the next-day return comes from the same partner
            partners.setdefault((t // 20, c), set()).add(partner)
    assert all(len(v) == 1 for v in partners.values())  # one partner for the whole block


def test_shuffle_is_deterministic_and_seeded():
    panel = _panel_with_holes()
    a, b, c = shuffle_answers(panel, 20, 0), shuffle_answers(panel, 20, 0), shuffle_answers(panel, 20, 1)
    assert torch.equal(a.y, b.y) and torch.equal(a.r_next, b.r_next)
    assert not torch.equal(a.y, c.y)
    assert torch.equal(a.X, panel.X) and torch.equal(a.universe, panel.universe) and torch.equal(a.s_sched, panel.s_sched)


def test_placebo_is_off_by_default_and_never_part_of_the_data_hash(tmp_path):
    cfg = synthetic_config(tmp_path)
    placebo = synthetic_config(tmp_path, variant="p", placebo={"shuffle_seed": 0})
    assert not cfg.is_placebo and placebo.is_placebo
    assert placebo.data_hash() == cfg.data_hash() and placebo.hash() != cfg.hash()
    assert dataclasses.replace(cfg, placebo=PlaceboConfig()).hash() == cfg.hash()


def test_placebo_and_real_runs_are_never_compared(tmp_path):
    from detmaso.eval.compare import CompareError, check_comparable
    from tests.test_compare import _market, _variant

    cfg = synthetic_config(tmp_path)
    placebo = dataclasses.replace(cfg, variant="fake", placebo=PlaceboConfig(shuffle_seed=0))
    ic, port = _market(2)
    with pytest.raises(CompareError, match="placebo"):
        check_comparable([_variant(cfg, "real", ic, port), _variant(placebo, "fake", ic, port)])


def test_placebo_and_training_only_configs_never_open_the_test_period(tmp_path):
    from detmaso.eval.lock import TestLockError, open_test_period

    with pytest.raises(TestLockError, match="placebo"):
        open_test_period(synthetic_config(tmp_path, variant="p", placebo={"shuffle_seed": 0}), "x", [0])
    with pytest.raises(TestLockError, match="training-only"):
        open_test_period(synthetic_config(tmp_path, variant="t", train={"validate": False}), "x", [0])


def test_evaluate_refuses_training_only_runs(tmp_path):
    from detmaso.cli import main
    from detmaso.config import load_config

    yaml = tmp_path / "pilot.yaml"
    yaml.write_text(
        f"base: {ROOT / 'configs/csi300.yaml'}\nvariant: pilot_t\nrun_dir: {tmp_path / 'runs'}\n"
        "loss:\n  c_bal: 0.1\n  balance_target: regime\ntrain:\n  validate: false\n"
    )
    cfg = load_config(yaml)
    (tmp_path / "runs" / "g").mkdir(parents=True)
    (tmp_path / "runs" / "g" / "config.json").write_text(json.dumps({"hash": cfg.hash()}))
    with pytest.raises(SystemExit, match="training-only"):
        main(["evaluate", "--config", str(yaml), "--tag", "g"])
