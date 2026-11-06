# Deviations from the approved plan

Every change to a planned default is logged here with its evidence. All evidence comes from the
training and validation periods only; the test period (2024-03-01 to 2026-03-01) was not looked at.

## 1. Agent activation: ReLU -> GELU (plan A17)

- **Planned default:** ReLU (matches the paper's only explicit network equation, the scheduler).
- **What happened:** with ReLU, training aborted in epoch 2 (seed 0) because one agent's output
  became constant across all stocks (the dead-agent safeguard, plan A27). Diagnosis: the agent tied
  to the rarest regime (Tail-Risk, 59 of 2,653 training days) had 15 of 16 last-layer units
  permanently inactive. Rare-regime agents get almost no learning signal (the IC term scales each
  agent's gradient by its scheduler weight and regime mask), Adam turns that weak, noisy signal into
  full-size random-walk steps, and ReLU units that drift negative for every input never recover.
  The same death happened with decoupled weight decay (epoch 3).
- **Change:** GELU, which never has an exactly zero gradient. The paper's own remark on
  activations (l.529-531) lists "GELU, ELU, Tanh" as the smooth choices it has in mind.
- **Result:** no dead agents in diagnostic runs.

## 2. Weight decay: coupled Adam L2 -> decoupled AdamW (plan A29)

- **Planned default:** `torch.optim.Adam(weight_decay=1e-4)` (classic L2 added to the gradient).
- **What happened:** with GELU and coupled L2, the rare-regime agents' output spread shrank steadily
  (agent 4 median from 0.18 to 0.023 over 6 epochs, about 35% per epoch). When an agent's loss
  gradient is tiny, Adam's per-parameter normalization makes the L2 term dominate, so the weights
  march toward zero; because agent outputs are z-scored, nothing in the loss resists the shrinking.
  Extrapolated, this would trip the dead-agent safeguard well within the 200-epoch budget.
- **Change:** decoupled weight decay (`torch.optim.AdamW`, same value 1e-4). The paper says only
  "Adam" and "weight decay is 10^-4", which fits either reading.
- **Result:** all five agents keep a stable output spread over 6 diagnostic epochs.

Of the four combinations tried (ReLU or GELU x coupled or decoupled), GELU with decoupled decay is the
only one that trains stably. Both settings are config values (`model.activation`,
`train.weight_decay_mode`), so the original defaults can be restored with a two-line config change.

## 3. Network precision: float32 -> float64 (plan A35)

- **Planned default:** float32 networks (float64 only for soft rank, Sigma and log det).
- **What happened:** agent outputs are z-scored (paper Eq. 2), so the loss never sees an agent's
  overall output scale, and that scale drifts freely during training: the two dominant agents'
  raw output spread grew to about 280, while starved agents' spread fell below 1e-6, where float32
  (about 7 significant digits) can no longer represent the pattern the z-score is meant to recover.
- **Change:** run the agents and scheduler in float64. This changes no math, only precision.

## 4. Constant agent outputs: abort -> score of 0 for that date (plan A27)

- **Planned default:** skip a training date if any agent's output is constant across stocks, and
  abort the run if that happens on more than 1% of an epoch's dates or 20 dates in a row.
- **What happened:** even in float64, seed 0 aborted at epoch 11. Diagnosis (instrumented run): the
  scheduler becomes a hard switch between the two common-regime agents (Low-Vol Bull, Sideways),
  each day giving one of them weight of about 1.0. The other three agents' learning signal is
  multiplied by their weight (about 0), and once the softmax saturates the gradient that could raise
  that weight vanishes too; the entropy bonus (c_ent = 0.01) is too weak to break this. Starved
  agents drift until all units in their last hidden layer are off for every stock on some dates
  (agent 3: 16 live units, then 7, then 0 within about 1,000 steps), so their output is constant.
- **Change:** the abort rule was this project's own safeguard, not part of the paper. A constant
  output cannot be standardized, so on such a date that agent's score is defined as 0 (its natural
  limit): it gets IC 0, a zero row in the correlation matrix (the log-det shrinkage keeps the penalty
  finite and constant), and no gradient that day; the other agents train normally. The number of
  dead agent-days is logged per epoch and reported. Non-finite losses still skip and abort as planned.
- **Finding for the write-up:** with the paper's coefficients, the scheduler does not keep weight on
  all five agents; the log-det barrier protects diversity in correlation space but, because outputs
  are z-scored, not against an agent's output collapsing. This is reported, not "fixed".
- **Details (added after code review):** the "constant output" threshold is 1e-12 in float64 (the
  plan's 1e-6 was set with float32 in mind; see entry 3). Diversity diagnostics (mean abs
  correlation, log det, smallest eigenvalue) are computed over the agents that are live that day,
  and the number of live agents is logged next to them. Shapley coalitions whose score is constant
  on a day (for example, only dead agents) get IC 0 for that day, consistent with the definition above.

## 5. Excess-return benchmark: all index members -> eligible members (plan A40)

- **Planned default:** excess returns against the equal-weighted average of all index members.
- **Change:** against the equal-weighted average of the *eligible* members, i.e. index members that
  traded on t-1 and on t, the same pool the portfolio picks from. This is the natural investable
  benchmark; members that cannot be traded that day are left out of both. Headline metrics use
  absolute returns and are unaffected.

## Finding (not a deviation): one regime name does not fit its cluster

The regime-naming rule (plan A16) was fixed before seeing data and is applied as written. On the
CSI300 training period it names a cluster "Low-Vol Bear" whose volatility is the second highest of
the five (above "High-Vol Bull"); it looks more like a high-volatility post-crash phase. Names are
labels only (any one-to-one mapping between agents and clusters trains the same way), so the rule
is not retuned after the fact; the data QA report flags the mismatch automatically.
