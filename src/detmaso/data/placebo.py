"""Placebo answers (docs/RULES.md, Rule P): shuffle the answers so there is nothing real to find.

Within each block of ``block_days`` consecutive model dates (counted from the first model date), the stocks
that have a 20-day label on every day of the block take part. They are put in a random order and each one
gets the next one's 20-day labels and next-day returns for the whole block (a single random cycle, so no
stock keeps its own answers). Stocks without a label on every day of the block are dropped from label_ok
and tradable for that block, so they are left out of training, IC and the portfolio.

One shuffle per block rather than per day keeps the day-to-day overlap of the real 20-day labels. Pairing
stocks at random still removes the shared moves of similar stocks, so placebo numbers are calmer than real
ones and are judged against their own uncertainty. Features, the universe (score z-scoring), market state
and regimes are untouched.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from detmaso.data.panel import Panel


def shuffle_answers(panel: Panel, block_days: int, seed: int) -> Panel:
    """A copy of ``panel`` with shuffled y, r_next, label_ok and tradable (see the module docstring)."""
    code = panel.code_idx.numpy()
    lab = panel.label_ok.numpy()
    y, rn, trd = panel.y.numpy(), panel.r_next.numpy(), panel.tradable.numpy()
    y2, rn2 = y.copy(), rn.copy()
    lab2 = np.zeros_like(lab)
    trd2 = np.zeros_like(trd)
    n_codes = len(panel.code_list)
    T = lab.shape[0]
    for b, start in enumerate(range(0, T, block_days)):
        rows = np.arange(start, min(start + block_days, T))
        labelled = np.zeros((len(rows), n_codes), dtype=bool)
        for i, t in enumerate(rows):
            labelled[i, code[t, lab[t]]] = True
        members = np.flatnonzero(labelled.all(axis=0))  # stock codes labelled on every day of the block
        if len(members) < 2:
            continue  # nobody to swap with: the whole block is left out
        order = np.random.default_rng([seed, b]).permutation(members)
        partner = np.empty(n_codes, dtype=np.int64)
        partner[order] = np.roll(order, -1)  # one cycle: no fixed points
        for t in rows:
            col_of = np.full(n_codes, -1, dtype=np.int64)
            present = code[t] >= 0
            col_of[code[t, present]] = np.flatnonzero(present)
            mine, theirs = col_of[members], col_of[partner[members]]
            y2[t, mine] = y[t, theirs]
            rn2[t, mine] = rn[t, theirs]
            lab2[t, mine] = True
            trd2[t, mine] = trd[t, mine]  # members keep their own "can be bought today" status
    return dataclasses.replace(
        panel,
        y=torch.from_numpy(y2),
        r_next=torch.from_numpy(rn2),
        label_ok=torch.from_numpy(lab2),
        tradable=torch.from_numpy(trd2),
    )
