> **About this document.** This is the implementation plan approved on 2026-11-05, before any code was written. It was written for the project owner, so it addresses "you". Changes made to its defaults during the build are logged in [DEVIATIONS.md](../DEVIATIONS.md).

# Det-MASO reproduction plan (MVP; plan only, no code yet)

## TLDR

- **Decisions: all answered** (Section 13, recorded 2026-11-05):
  - recommended defaults everywhere;
  - the MVP is CSI300 only;
  - of the optional extras, only the two extra plots (D8.4) are in;
  - S&P 500 waits until you have checked WRDS access.
- **The paper:** I read the whole paper from its LaTeX source (there is no appendix, no released code, no data snapshot) and extracted the plotted data from its figure files.
  - **Given:** the loss weights, learning-rate schedule, most scheduler sizes, dates, and the Shapley formula.
  - **Not given (about 45 items):** agent layer sizes, the soft-rank "softness", about 15 of the 34 feature formulas, portfolio rebalancing, how returns are annualized, and more. Each has a default in **Section 3**, marked ASSUMPTION.
- **Your summary** was mostly right (corrections in **Section 1**). The main misses:
  - c_div is given (0.1).
  - There is a regime "gradient mask" that your summary does not mention.
  - There are 5 metrics, not 4 (ICIR too).
- **Data:**
  - CSI300 works today: a community-maintained Qlib dataset is current through 2026-11-04 and can be pinned to an exact checksum.
  - S&P 500 is the main risk. Qlib's official US data stops in Nov 2020, and no free source has complete prices for companies that stopped trading (bought out or bankrupt).
  - Per D1, the MVP is CSI300 only. The S&P 500 step comes later, after you check WRDS access.
- **The paper has serious problems** (**Section 12**: 27 items; checked by 5 separate AI review passes, none refuted). The biggest:
  - **Figure 4's "CSI300 index" is not the real index.** It has values on Chinese public holidays, shows an 8% one-day drop that never happened, and stops in April 2025, 10 months before the stated test period ends.
  - **Figure 3's diversity numbers are impossible for 5 agents.** They fit a 2-agent system.
  - **The "no diversity is worse than a single MLP" sentence contradicts the paper's own results table** (the one you were told about).
  - **So:** we treat the paper's numbers as claims to test, not ground truth, and keep the test period locked until the final run.
- **Build:** 11 milestones, smallest runnable piece first. A laptop CPU is enough (minutes per training run).

---

## 0. Context

**Goal:** a faithful MVP reproduction of **Det-MASO** (arXiv 2610.34936, Wu and Yang, 26 Oct 2026). How Det-MASO works:
- Five small neural networks ("agents") each specialize in one market regime.
- A scheduler mixes their per-stock scores into one final score.
- A log-determinant penalty keeps the agents from becoming copies of each other.

**Deliverable:** a Python/PyTorch codebase that:
1. rebuilds the paper's data pipeline, regimes, model, loss, training, top-decile backtest, metrics and Shapley interaction analysis;
2. reports our numbers next to the paper's;
3. documents every place we had to guess.

**Out of scope** (your rules):
- baselines (AlphaGen, AlphaForge, AlphaSage, FactorVAE, RVRAE, Single-MLP);
- ablations, including the Det-MASO variants (no-div, SND, MMD) and the hyperparameter sensitivity sweeps;
- transaction costs;
- walk-forward retraining;
- different agent counts;
- new features;
- any other improvement the paper does not describe.

**Follow-up after the MVP (your separate step, D8.2):** single-network and no-diversity runs. These are not built here; c_div is already a config value.

**Test-period discipline:** all development and debugging use the training and validation periods. The test period (Mar 2024 to Mar 2026) is opened **once**, in Milestone 9, with a frozen config. This prevents us from (even accidentally) tuning toward the paper's numbers.

The project folder is empty, so everything below is new code.

---

## 1. What the paper actually says (corrections to your summary)

Table and equation numbers below follow the compiled PDF. LaTeX numbers all 24 equations in order, including ones inside proofs.

| Your summary | What the paper says (source of truth) |
|---|---|
| Qlib; CSI300 and S&P 500 | Correct: CSI300 (300 stocks) and S&P 500 (500 stocks), "daily price, volume, and fundamental data ... obtained from the Qlib platform" (Sec 4.1.1). No code or data snapshot is released; the Data Availability statement only says "available through the Qlib platform". No feature actually uses fundamental data. |
| 34 z-scored stock features | Correct: D = 34 in 9 categories (Table 2), z-scored across stocks each day using data up to the previous day only. About 15 of the 34 formulas are ambiguous (Section 3.2). |
| Target: 20-day forward return | Correct: tau = 20 trading days, r = Close(t+20) / Close(t) - 1. Timing detail: features use data through day t-1, while the return starts at day t's close, a one-day gap. |
| Test about Mar 2024 to Mar 2026 | Correct: test 2024-03-01 to 2026-03-01. Also train 2012-01-01 to 2022-12-31 and validation 2023-01-01 to 2024-02-28 (Sec 4.1.2). |
| 5 MLP agents, 3 hidden layers | Correct, but "5" is the output of a procedure. The paper picks the agent count with K-means plus a silhouette score over K = 2..K_max, and says to use whatever count wins. That gave 5 on both markets (Sec 3.2.1, 4.1.6). K_max and the hidden-layer widths are never given; Sec 3.2.2 says "see Section 4", and Section 4 does not have them. |
| One agent per K-means regime; frozen centroids | Correct. K-means is fit once on the training period and frozen for validation and test. Agent i is tied to regime i. |
| Attention scheduler over a 12-dim state | Correct. It uses 4 windows (5/20/60/120 days) x 3 statistics (volatility, momentum, dispersion). Each window's 3 numbers go through their own small 2-layer MLP to a 16-dim vector. Attention weighs the 4 windows, then a 16 -> 32 -> 5 MLP with softmax gives the 5 agent weights (Eq. 6-8). The output layer starts at zero, so every agent starts at weight 1/5. There is one weight vector per day, shared by all stocks. |
| Loss = soft-rank IC + rolling Sharpe (stop-grad) + c_div * (-log det Sigma) + entropy | Mostly correct, with 3 fixes. (1) The IC term sums each agent's **own** IC, each multiplied by its scheduler weight w_i and a **regime mask** m_i: 1 for the agent whose regime is today's, lambda_mask = 0.1 for the others (Eq. 11, 14, 20). It is not the IC of the mixed score. (2) Sigma is the correlation of the agents' soft **ranks**, not raw outputs (Eq. 3). (3) The Sharpe term uses a long-short portfolio whose weights are the mixed score itself, not the top-decile portfolio (Eq. 12). |
| c_div and learning rate unknown | Both given: c_IC = 1.0, c_SR = 0.5, c_div = 0.1, c_ent = 0.01, and lambda_mask = 0.1 (Table 7 default). Optimizer: a single Adam, cosine learning rate from 1e-3 to 1e-5 (Eq. 21), weight decay 1e-4, gradient clipping at max-norm 1.0 (Sec 3.2.6, 4.1.5). |
| Up to 200 epochs, patience 8, 5 seeds | 200 and 8 are correct. Patience is measured on a 3-epoch smoothed validation IC, and the best weights are restored. "5 seeds" appears only in the Figure 3 caption; the main results table (Table 5) does not say how many runs it averages. |
| Shapley interaction, exact over 32 coalitions | Correct. This is the Grabisch-Roubens interaction index: v(S) = the average daily IC of the mixed score using only the agents in S, with scheduler weights renormalized (Eq. 22-23). |
| Top-decile long-only; IC, AR, Sharpe, MDD | Correct, plus a fifth metric, **ICIR** (Sec 4.1.4). The rebalancing frequency is never given. |
| CSI300: IC 0.081, Sharpe 1.85, AR 31.5%, MDD 8.5% | Correct (Table 5). ICIR is 0.62 (Tables 6 and 7). |
| S&P 500: IC 0.072, Sharpe 1.58, AR 26.8% | Correct, plus MDD 8.2% (Table 5). No ICIR is reported for S&P 500. |
| (not in your summary) | Other Det-MASO outputs we can compare against: regime coverage (Table 4); agent diversity (Figure 3: mean abs correlation 0.49, log det -0.28, but see Section 12); scheduler weights over time (Figure 4, also see Section 12); and the claim that the Low-Vol Bull x Tail-Risk pair has the strongest positive Shapley interaction (Sec 4.5). |

---

## 2. Spec table (component -> where it comes from)

"l." = line number in the paper's LaTeX source (Det_MASO.tex in the arXiv source bundle).

| # | Component | What the paper specifies | Source |
|---|---|---|---|
| S1 | Universes | CSI300 (~300 stocks), S&P 500 (~500), daily data from Qlib | Sec 4.1.1 (l.720) |
| S2 | Splits | train 2012-01-01..2022-12-31; val 2023-01-01..2024-02-28; test 2024-03-01..2026-03-01; "strictly temporal with no overlap" | Sec 4.1.2 (l.725) |
| S3 | Label and IC | r(x, t+tau) = Close(x, t+tau) / Close(x, t) - 1, tau = 20 trading days; IC = Spearman correlation (Eq. 1) | Sec 3.1 (l.204-214), Sec 4.1.2 |
| S4 | Features | D = 34 in 9 categories; cross-sectional z-score per date; built only from data strictly before t | Sec 3.2.2, Table 2 (l.290-313), l.214 |
| S5 | Market state | s_t in R^12; windows {5, 20, 60, 120}; sigma_w = std of the equal-weighted market return; mu_w = sum of market returns; delta_w = mean over days of the cross-sectional return std; data in [t-w, t-1] | Sec 3.2.1, Table 1 (l.253-271), Sec 3.2.3 |
| S6 | Agent count | K-means on training-period s_t; n* = argmax silhouette over K in {2..K_max}; set n = n*; result n = 5 on both markets | Eq. 4 (l.273-283), Alg. 1 (l.622), Sec 4.1.6 (l.754) |
| S7 | Regimes | centroids fit once on train and frozen; each day assigned to its nearest centroid; labels re-indexed so agent i <-> regime i | l.281-283, l.369 |
| S8 | Agent network | 3-hidden-layer MLP, D -> 1 per stock; weights shared across stocks, independent across agents; output z-scored across stocks each day | Sec 3.2.2 (l.288), Eq. 2 (l.215-221) |
| S9 | Factor matrix | F_t = [alpha_1..alpha_n] in R^(N x n) | Eq. 5 (l.315) |
| S10 | Window encoders | per window: a two-layer MLP R^3 -> R^16 (hidden width not given) | Sec 3.2.3 (l.325) |
| S11 | Window attention | a_t = softmax(W_a concat(h_5, h_20, h_60, h_120)), W_a in R^(4x64); h_t = sum_k a_k h_k in R^16 | Eq. 6-7 (l.326-336) |
| S12 | Scheduler head | w_t = softmax(W_2 ReLU(W_1 h_t + b_1) + b_2); W_1 in R^(32x16), W_2 in R^(n x 32); output layer zero-initialized | Eq. 8 (l.342-348) |
| S13 | Final score | alpha_hat_t = sum_i w_i alpha_i (no re-standardization) | Eq. 9 (l.351-354) |
| S14 | Sigma | Sigma_t = F_tilde^T F_tilde / N; column i = z-score(SoftRank_kappa(alpha_i)); unit diagonal | Eq. 3 (l.224-229) |
| S15 | Soft rank | Blondel et al. 2020 operator; kappa_scaled = kappa_soft * N / 100 ("scaled by the cross-section size"); exact Spearman at evaluation | l.229, l.601 |
| S16 | Per-agent IC | IC_i = Spearman(alpha_i, r), computed from the cached soft ranks during training | l.229, l.385 |
| S17 | Sharpe term | SR = mean / std over window T_t of p_s = sum_x alpha_hat(x,s) * r(x, s+tau); past alpha_hat_s are constants (stop-gradient) | Eq. 12 (l.387-396), l.601 |
| S18 | Diversity penalty | phi(Sigma) = -log det(Sigma) | l.398-399, l.419-428 |
| S19 | Entropy | H(w) = -sum_i w_i log w_i | Eq. 13 (l.400-404) |
| S20 | Regime mask | m_i,t = 1 if regime(t) = i, else lambda_mask | Eq. 14 (l.406-417) |
| S21 | Objective / loss | J_t (Eq. 11); training minimizes L_t = -c_IC sum_i w_i m_i IC_i - c_SR SR + c_div (-log det Sigma_t) - c_ent H(w_t) | Eq. 11 (l.374-384), Eq. 20 (l.590-598) |
| S22 | Coefficients | c_IC 1.0, c_SR 0.5, c_div 0.1, c_ent 0.01; lambda_mask 0.1 (CSI300; Table 7 default) | Sec 4.1.5 (l.749), Table 7 (l.912-916) |
| S23 | Optimizer | one Adam over all parameters; weight decay 1e-4; gradient clip max-norm 1.0 | l.586-588, l.749 |
| S24 | LR schedule | cosine from 1e-3 to 1e-5 over T_max = num_train_dates x epochs | Eq. 21 (l.603-608) |
| S25 | Training loop | each epoch, for each training date t in order: one Adam step on L_t | Algorithm 1 (l.616-637) |
| S26 | Early stopping | 3-epoch smoothed validation IC, patience 8, restore best weights; E_max = 200 | l.609, l.749 |
| S27 | Seeds | 5 independent runs (stated for Figure 3 only) | l.833 |
| S28 | Metrics | IC (mean daily Spearman), ICIR (mean / std of daily IC), annualized return, Sharpe, max drawdown of a top-decile long portfolio | Sec 4.1.4 (l.744), Table 5 caption (l.796) |
| S29 | Shapley | v(S) = mean over T_eval of IC(alpha_hat^S_t, r), with w renormalized over S; v(empty) = 0; I(i,j) = sum_S omega(abs(S), n) * Delta_ij v(S), omega = abs(S)! (n - abs(S) - 2)! / (n-1)!; exact over 2^5 = 32 coalitions | Eq. 22-23 (l.659-677), l.701 |
| S30 | Reported targets | Table 5 Det-MASO rows; Table 4 regime coverage; Figure 3 diversity; Figure 4 scheduler weights; Low-Vol Bull x Tail-Risk strongest interaction | l.754-857 |

---

## 3. Gaps in the paper and proposed defaults (ASSUMPTIONS)

**Rules for defaults:**
1. Use the most literal reading of the paper.
2. If ambiguous, use the closest **Qlib / Alpha158** convention, since the authors built on Qlib. Alpha158 is Qlib's standard library of 158 stock features.
3. Otherwise use standard practice.
4. **Any alternative to a default is chosen on validation data only, never by looking at test results.**

Most defaults become config values; a few (feature formulas, the regime-naming rule) are small, isolated functions. Rows that are really your call point to a decision ("-> D#") or to the "already settled" list in Section 13.

### 3.1 Data, universe and timing

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A1 | Which CSI300 dataset | Community dataset **chenditc/investment_data**, release tag `2026-11-04`, sha256 `dc917293925635189e982f47d35b960c3db8a9c3f11312c937673c8afdec21aa`. Check it with its `validate_archive.py` taken from commit `b8c129b4d9b838f050eac8b1135111b3b79fc894`, and extract to a versioned folder. | Qlib's official CN download still works but stops at 2020-09-25. Qlib's README points to this source. It has daily releases to 2026-11-04, keeps delisted stocks, and has historical CSI300 membership. |
| A2 | Which S&P 500 dataset | Not in the MVP (D1 = a). Decided after the CSI300 results, once you have checked WRDS access. | No free point-in-time S&P 500 dataset reaches 2026 |
| A3 | Which stocks count on day t (two masks) | **universe_t** = index members on t that traded on t-1 (features computable). This mask drives agent-output z-scoring, the scores and the portfolio pool; to be bought, a stock also needs a price on t. **label_ok_t** = universe_t stocks whose 20-day label exists. This mask is used **only** inside the loss, IC and Shapley. Features and labels come from each stock's full price history, so a new index member has a full feature history, and a stock leaving the index within 20 days keeps its label. | Prevents a subtle look-ahead: if label availability shaped test-day scores or stock picks, the backtest would quietly avoid stocks that are about to be suspended or delisted. Also avoids survivorship bias. |
| A4 | Price and volume fields | Ratio features use Qlib's adjusted `$open $high $low $close` and split-adjusted `$volume`. The one level feature (logvol20) uses raw volume `$volume * $factor`; it is counted in lots of 100 shares, which does not matter after the log and z-score. | Community-data prices are rescaled so each stock's first close is 1, so only ratios are comparable across stocks |
| A5 | Timing | Sample date t = entry day. Features use data through the close of t-1. Label = Close(t+20) / Close(t) - 1 on adjusted closes. Implementation: compute each feature with windows ending on day d, then shift one row, so row t uses d = t-1. That is **exactly one shift**; Section 3.2 already writes the formulas in row-t form. | Literal reading of l.204 and l.214 |
| A6 | Purge at split boundaries | Drop the last 20 training dates and the last 20 validation dates from training and early stopping, because their labels use the next period's prices. -> settled by your brief (Section 13) | "No overlap" holds for the date ranges, but 20-day labels cross the boundaries |
| A7 | Exact boundaries | Use the literal dates. 2024-02-29 falls in no split (a paper quirk). Test dates run 2024-03-01 to 2026-02-27, and their labels need prices to about 2026-03-27. Load data 2011-01-01 to 2026-04-30, which gives warm-up for the 120-day windows. | Literal |
| A8 | Suspended days and missing data | Rules:<br>(1) A stock's daily return = its close / its previous **traded** close - 1. The day trading resumes carries the whole move, and no fake 0% days are created (no forward-filling).<br>(2) Rolling statistics use the last w market days, skipping missing days. They are valid only if at least 80% of those days have data; otherwise NaN.<br>(3) Point-to-point features (momentum, MA deviation, range) use the last traded close at or before each end point.<br>(4) Labels are NaN if either end-point close is missing.<br>(5) After z-scoring, NaN becomes 0, i.e. the day's average (Qlib's CSZFillna convention).<br>(6) Skip a training date with fewer than 50 label_ok stocks. | Suspended stocks are NaN in Qlib (85 of 300 CSI300 members on 2015-07-08). The paper is silent, and pandas defaults would silently create 0% returns. |
| A9 | Outliers | After z-scoring, clip to [-5, 5], as a numerical safeguard only | The paper says plain z-score; +-5 almost never binds |
| A10 | Standardization formula | Population std (ddof 0) with eps 1e-8, everywhere | Sigma needs an exact unit diagonal (l.229) |
| A11 | Membership lag | Accept the dataset's CSI300 membership changes at the end of June and December, about 2 weeks after the real rebalances, and document this | Only free source; small effect |

### 3.2 The 34 features

Formulas are for row t and use data through day t-1. C, O, H, L = adjusted close, open, high and low. V = split-adjusted volume. ret = daily return per A8.

| # | Name | Formula | Status |
|---|---|---|---|
| 1-6 | mom_w, w in {1,5,10,20,60,120} | C(t-1) / C(t-1-w) - 1 | **ASSUMPTION** (compounded). The paper's mu notation hints at a plain sum of daily returns, which is nearly identical after z-scoring. |
| 7-10 | vol_w, w in {5,10,20,60} | std of ret over the last w days | Clear. Scale factors cancel in the z-score. |
| 11-14 | rturn_w, w in {5,10,20,60} ("relative turnover") | V(t-1) / mean(V over the last w days) | **ASSUMPTION**: reciprocal of Alpha158 VMA. Within one stock, a turnover ratio equals a volume ratio because shares outstanding cancel. |
| 15-17 | pvcorr_w, w in {5,10,20} | Spearman correlation of C and V over the last w days | **ASSUMPTION**: rank version of Alpha158 CORR |
| 18-20 | upfrac_w, w in {5,10,20} ("RSI proxy") | fraction of the last w days with ret > 0 | Clear (Alpha158 CNTP) |
| 21-23 | madev_w, w in {5,20,60} | C(t-1) / mean(C over the last w days) - 1 | **ASSUMPTION** on form |
| 24 | macd | mom_20 - mom_60 | Clear. Stock-level; the paper reuses the market symbols mu_20 and mu_60 here (Section 12, item 20). |
| 25 | overnight | O(t-1) / C(t-2) - 1 | Clear |
| 26 | intraday | C(t-1) / O(t-1) - 1 | Clear (Alpha158 KMID) |
| 27 | closepos | (C - L) / (H - L) on day t-1; 0.5 if H = L | **ASSUMPTION** |
| 28 | hl_1 (first of "high-low spread x2") | (H - L) / C on day t-1 | **ASSUMPTION** |
| 29 | hl_5 (second of "high-low spread x2") | mean of (H - L) / C over the last 5 days | **ASSUMPTION**: the paper never names the two spreads |
| 30 | skew60 | skewness of ret over the last 60 days | Clear |
| 31 | ac1_60 | correlation of ret with its 1-day lag over the last 60 days | Clear |
| 32 | momsharpe | mom_60 / (vol_60 + 1e-12) | Clear (stock-level; same symbol reuse as #24) |
| 33 | range20 | (max H over the last 20 days - min L over the last 20 days) / C(t-1) | **ASSUMPTION** (Alpha158 MAX20 - MIN20 style) |
| 34 | logvol20 | log(mean raw volume over the last 20 days + 1) | **ASSUMPTION**: share volume, the literal "volume"; dollar volume is the alternative |

### 3.3 Market state and regimes

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A12 | "Market" definition | Each day d: the equal-weighted mean of one-day returns over index members that traded on both d-1 and d. Dispersion = the cross-sectional std of the same returns. sigma_w uses ddof 1; mu_w is the plain sum, as written. | Literal Table 1, using point-in-time members |
| A13 | Scaling s_t | Standardize each of the 12 dims with training-period mean and std. K-means, silhouette and regime assignment use this **unclipped** standardized s_t. Only the scheduler's input is clipped to [-5, 5]. | Raw dims differ in scale by about 10x. Clipping before clustering would flatten the crash days that define the Tail-Risk regime. |
| A14 | K-means settings | k-means++, n_init = 50, random_state = 0, Euclidean; silhouette on all training days for K = 2..10 | K_max is never given. The phrase "n <= 5" might hint at 5, but that is speculation. |
| A15 | If silhouette does not pick 5 | Keep n = 5 and report what the clustering picked (D2 = a) | The paper says to use whatever it picks; your brief says 5 agents and rules out other counts |
| A16 | Naming regimes | On standardized centroids, let V = mean(sigma20, sigma60) and M = mean(mu20, mu60). **Tail-Risk** = argmin(M - V). Of the rest, the two with the highest M are the Bulls: lower V = **Low-Vol Bull (1)**, the other = **High-Vol Bull (2)**. Of the last two, lower M = **Low-Vol Bear (4)**, the other = **Sideways (3)**. Agents are numbered to match. One shared mapping object feeds training, plots, coverage and Shapley. | The paper names regimes "by inspection". A fixed rule lets us compare with Table 4 and check the Low-Vol Bull x Tail-Risk claim. |

### 3.4 Model

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A17 | Agent widths and activation | 34 -> 64 -> 32 -> 16 -> 1, ReLU, no dropout or batch norm, PyTorch default init, each agent initialized independently | Never given. ReLU matches the paper's only explicit network equation; the paper lists only weight decay and clipping as regularizers. |
| A18 | Output standardization | Inside the computation graph, over universe_t, population std + 1e-8 | Eq. 2 |
| A19 | Scheduler details | Each window: Linear(3,16) -> ReLU -> Linear(16,16). W_a = Linear(64, 4, **no bias**), as written in Eq. 6. W_1 = Linear(16,32) + ReLU. W_2 = Linear(32,5) with weights **and** bias zero-initialized, so w starts exactly uniform. | The window-MLP hidden width and activation are not given |

### 3.5 Loss

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A20 | Soft-rank implementation and eps | Blondel et al. L2 soft rank with ascending ranks, written as our own small PyTorch function (exact PAV algorithm, float64, CPU). torchsort 0.1.10 serves as a test oracle. eps = kappa_soft * N_t / 100, with N_t = the number of stocks in that day's loss cross-section (label_ok_t). | l.601 says the temperature is "scaled by the cross-section size"; l.204 calls N the universe size, which differs only by a few suspended stocks. torchsort has no Apple-Silicon wheel and no Apple-GPU support. |
| A21 | kappa_soft value | **1.0** (D4 = a). At kappa_soft >= 0.1, the agent side of the ranking becomes exactly linear: training IC = correlation of the agent's z-scored scores with the return **ranks**, and Sigma = the ordinary correlation of agent outputs. At 0.01 this holds on almost all days. We log the daily share of non-linear soft ranks. | Never given. Analysis in D4. |
| A22 | Training IC_i | Over label_ok_t: Pearson correlation between z-scored soft ranks of alpha_i and z-scored hard ranks (average ties) of the label. w and m are not detached, as written. | l.229, l.385 |
| A23 | log det safeguards | Sigma in float64; shrink toward identity: Sigma_eps = (1 - 1e-4) Sigma + 1e-4 I; log det via Cholesky (Section 6.3). Reported diversity numbers use the unshrunk Sigma. | The paper has no safeguard. Shrinkage keeps the unit diagonal, so log det(I) stays exactly 0. |
| A24 | Sharpe window T_train | T_train = **the number of training dates**. The "rolling" window is then the expanding set of all dates processed so far in the current epoch. Past p_s values are cached (detached), the cache resets each epoch, and the term turns on once 20 values exist. SR = mean / (std(ddof 1) + 1e-8) on raw 20-day returns, computed from running float64 sums. Side effect: the Sharpe term's pull on each date shrinks roughly as 1 / (dates so far), so it acts mostly early in each epoch. Per-term gradient norms are logged. A fixed 252-day window is an alternative only if results are far off, and only after asking you first (D8.3). | T_train is overloaded and never valued (Section 12, item 5). This reading invents no number, and it makes Algorithm 1's loop, the cosine schedule and the paper's parallel notation (T_eval = size of the evaluation set, l.667) consistent. |
| A25 | Entropy | computed with log_softmax (no log 0) | numerical |
| A26 | lambda_mask | 0.1 for both markets (D5 = a) | Table 7 default for CSI300; no S&P 500 value is given |
| A27 | Skipped steps | A training step is skipped if the loss or any gradient is non-finite, or if an agent's output is constant on that day (std < 1e-6). On a skip: no Adam update, p_t is not cached, and the LR schedule still advances. Training stops with an error if more than 1% of an epoch's dates are skipped, 20 consecutive dates are skipped, or there are more than 10 non-finite events in a run. | Fail loudly instead of training silently on a broken agent |

### 3.6 Training

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A28 | Loop | Each epoch is one chronological pass over the purged training dates. Each Adam step uses one date's cross-section. No shuffling. | Algorithm 1; the Sharpe cache needs time order |
| A29 | Optimizer variant | `torch.optim.Adam`, betas (0.9, 0.999), eps 1e-8, weight_decay 1e-4 (classic L2, not AdamW), on all parameters | The paper says "Adam" |
| A30 | Clipping | global gradient norm over all parameters, 1.0, every step | l.749 |
| A31 | LR schedule | cosine per step, T_max = (number of purged training dates) x 200; early stopping does not rescale it | Eq. 21 |
| A32 | Early stopping | After each epoch, val IC = the mean daily exact Spearman IC of alpha_hat over purged validation dates. Smoothed value for epoch e = mean of epochs e-2, e-1, e. Only epochs e >= 3 can be "best". Stop after 8 epochs without improvement, restore epoch e's weights, and log the full curve. | l.609; prevents a lucky unsmoothed epoch 1 from winning |
| A33 | Seeds | 5 seeds (0-4) that vary only network initialization; K-means fixed; report mean +- std | Figure 3 caption |
| A34 | Test-time updates | None: train once, freeze, evaluate. -> settled by your brief (Section 13) | Algorithm 1; walk-forward retraining is out of scope |
| A35 | Device and precision | `device = auto`: an NVIDIA GPU (CUDA) if present, otherwise CPU. On your Mac this means the CPU (D9 = a). Networks run in float32; soft rank, Sigma and log det in float64. Deterministic algorithms on (CUDA also needs `CUBLAS_WORKSPACE_CONFIG`). | Your brief: "use a GPU if available" |

### 3.7 Backtest and metrics

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A36 | Rebalance frequency | Daily: at the close of day t, hold the top decile by alpha_hat_t until the close of t+1 (D3 = a). | The paper is silent |
| A37 | Decile and weights | k_t = ceil(10% of the eligible stocks) (about 30 for CSI300, about 50 for S&P 500), equally weighted. Eligible = universe_t stocks with a price on t. | "top-decile" |
| A38 | Suspended or delisted holdings | A stock with no price on t+1 earns 0 that day. If it is still suspended at the next rebalance, it is assumed sold at its last traded price (a frictionless approximation, tiny for CSI300 large caps). A delisted stock takes its last available return and is then removed. | The paper is silent |
| A39 | Frictions | None (no costs, no Chinese price-limit or lot-size rules) | Your scope rule; the paper never mentions costs |
| A40 | Absolute vs relative returns | Absolute portfolio returns are the headline. Excess returns are reported against the **equal-weighted average of all index members**, built from the same dividend-adjusted prices. | The paper's wording is "annualized return of the top-decile long portfolio". The official CSI300 index excludes dividends, so beating it would flatter us by about 2-3% a year. |
| A41 | Annualization | Compound the daily returns: AR = final value ^ (1 / years) - 1, with years = calendar length of the test period. Sharpe = mean / std of daily returns x sqrt(trading days per year actually observed in the test period: about 242 for China, 252 for the US); risk-free rate 0. MDD = the largest drop from a running peak of the value curve, with the starting value 1 counted as a peak. Qlib's `risk_analysis` form is also reported (summed returns, 238 days). The standard form is the headline (D6 = a). | The paper is silent; the two recipes differ by a few points |
| A42 | IC and ICIR | IC_t = Spearman (average ties) between alpha_hat_t and the label over label_ok_t. IC = mean over all test days; ICIR = mean / std (ddof 1), not annualized. The Table 5 caption says metrics are "computed on the same top-decile long portfolio". We read that as describing AR, SR and MDD; IC stays cross-sectional per Eq. 1 and Sec 4.1.4. The alternative reading (IC within the top decile) is noted. | Matches Qlib's "Rank IC" and "Rank ICIR" |

### 3.8 Shapley and reporting

| ID | Gap | Default (ASSUMPTION) | Why |
|---|---|---|---|
| A43 | Shapley period and aggregation | Development runs use the validation period; the final run uses the test period (M9). Exact daily Spearman IC; computed per seed, then mean +- std; agents named per A16. | The paper says "held-out evaluation period" |
| A44 | Diversity numbers (Figure 3 analog) | Mean abs pairwise rank correlation and mean log det of the unshrunk Sigma over the last min(40, E) training epochs, ending at the stop epoch, across the 5 seeds. Also measured over the test period (D8.4). | The Figure 3 protocol; its own numbers are suspect (Section 12) |
| A45 | Sub-window | Also report all metrics on 2024-03-01..2025-04-18, the span Figure 4's data actually covers, labeled as a comparison only | Hints the paper's numbers may come from a shorter test window |

---

## 4. Environment and project structure

**Environment** (from research on what actually installs on your Mac):
- **Python 3.12**, managed with `uv`. Your machine has 3.13, but pyqlib 0.9.7, the latest on PyPI, only ships wheels up to 3.12; Qlib's own CI runs 3.12 on Apple Silicon.
- **Pins:** `pyqlib==0.9.7`, `pandas>=2.2,<3`, and `mlflow<3.13` (pulled in by pyqlib; newer versions break it). Also numpy 2.x, the standard torch wheel, scipy, scikit-learn, pyarrow, pyyaml, matplotlib, pytest. Everything is locked in `uv.lock`.
- **Optional dev-only:** `torchsort==0.1.10` built with `--no-build-isolation`, used only as a test oracle. If it will not build, tests fall back to a brute-force reference.

**Layout:**

```
multi-net-stock-analyzer/
├── pyproject.toml               # Python 3.12, pinned deps
├── uv.lock
├── README.md                    # setup, data download, how to run, assumption list
├── configs/
│   ├── csi300.yaml              # every ASSUMPTION above is a named key
│   └── sp500.yaml
├── scripts/
│   ├── get_cn_data.sh           # pinned download + sha256 + validate + extract
│   └── build_us_data.py         # only if D1 says so (Milestone 10)
├── src/detmaso/
│   ├── config.py                # typed dataclasses, YAML loading, run metadata and config hash
│   ├── data/
│   │   ├── qlib_io.py           # qlib.init, calendar, membership spans, raw OHLCV -> parquet cache
│   │   ├── returns.py           # suspension-aware daily returns and window-validity rule (A8)
│   │   ├── features.py          # the 34 features, one function per category
│   │   ├── labels.py            # 20-day forward return
│   │   ├── market_state.py      # 12-dim s_t
│   │   ├── regimes.py           # scaler, K-means, silhouette curve, naming rule, shared mapping
│   │   ├── splits.py            # date splits + purge
│   │   └── panel.py             # per-date tensors (padded) + universe / tradable / label_ok masks
│   ├── model/
│   │   ├── agent.py             # one MLP agent + masked cross-sectional standardization
│   │   ├── scheduler.py         # window encoders + attention + weight head
│   │   └── detmaso.py           # agents + scheduler -> alphas, w, a, alpha_hat
│   ├── losses/
│   │   ├── softrank.py          # Blondel L2 soft rank (PAV), torch.autograd.Function
│   │   ├── stats.py             # masked z-score, hard ranks, exact Spearman
│   │   └── objective.py         # IC terms, Sigma + log det, Sharpe sums, entropy, mask, total
│   ├── train/trainer.py         # epoch loop, cosine LR, clipping, skips, early stopping, checkpoints, logs
│   ├── eval/
│   │   ├── inference.py         # frozen model over a period -> alphas, weights, attention, regimes
│   │   ├── metrics.py           # IC, ICIR, AR, SR, MDD (+ Qlib-style forms)
│   │   ├── backtest.py          # top-decile long-only portfolio
│   │   ├── shapley.py           # v(S) over 32 coalitions, I(i,j)
│   │   └── report.py            # tables and plots next to the paper's numbers
│   └── cli.py                   # prepare | regimes | train | evaluate | shapley | report | all
├── tests/                       # pytest suite (Section 9)
└── runs/                        # gitignored: data cache, per-seed checkpoints and logs, reports
```

**Principles:**
- Data transforms are pure functions, each testable for look-ahead.
- One config object goes everywhere.
- Each run records its config hash, data tag, sha256 and git commit next to its results.
- The folder becomes a git repo at Milestone 0.

---

## 5. Data pipeline

1. **Download (pinned).** `scripts/get_cn_data.sh`:
   - fetch `qlib_bin.tar.gz` and `qlib_bin.manifest.json` for tag `2026-11-04`;
   - check the full sha256 from A1;
   - run the dataset's own validator (commit from A1);
   - extract to `~/.qlib/qlib_data/cn_data_2026-11-04`.
2. **Init and checks.**
   - Call `qlib.init(provider_uri=..., region="cn")`.
   - The calendar must reach at least 2026-04-30; the pinned tag ends 2026-11-04.
   - The membership file must give exactly 300 CSI300 members on 2012-01-04, 2022-12-30, 2024-03-01 and 2026-03-02, with at least 290 of them trading each day.
   - Spot-check: Kweichow Moutai's raw close on 2024-03-01 is about 1685.
3. **Membership.**
   - `D.list_instruments(D.instruments("csi300"), "2011-01-01", "2026-04-30", as_list=False)` gives each stock's membership spans.
   - Build a member[date, stock] table for the roughly 950 stocks that were ever members.
4. **Raw panel.**
   - Load `$open $high $low $close $volume $factor` for all ever-members over 2011-01-01..2026-04-30, **without** the membership filter, so each stock has its full history.
   - Pivot to date x stock tables and cache them as parquet.
5. **Returns** per A8: each stock's close divided by its previous traded close, so there are no fake 0% days.
6. **Features** (Section 3.2):
   - windows ending on day d, with the 80% validity rule;
   - then one shift, so row t uses data through t-1.
7. **Cross-sectional processing per date** over universe_t:
   - z-score each feature (population std);
   - clip to +-5;
   - set NaN to 0.
8. **Labels.** Close(t+20) / Close(t) - 1 from the full price history; NaN only if an end-point price is missing.
9. **Market state** per A12:
   - each day, the equal-weighted return and the cross-sectional std over members that traded on both days;
   - then rolling 5/20/60/120-day statistics ending at t-1, giving 12 columns.
10. **Regimes.**
    - Fit the scaler on training-period s_t.
    - Compute the silhouette curve for K = 2..10 and save the plot.
    - Fit K = 5 on the unclipped standardized s_t (D2), apply the naming rule (A16), and assign every date to its nearest frozen centroid.
    - Report coverage next to Table 4 (CSI300: 35/12/32/16/5%).
    - Save the scaler, centroids and the mapping object. They never change afterwards.
11. **Splits and purge** (A6, A7).
12. **Panel tensors.** Per date: X [N_max, 34], label [N_max], masks `universe`, `tradable` and `label_ok` [N_max], s [12], regime id. Stored padded as [T, N_max, ...] float32.
13. **Data QA report** (`runs/<market>/data_report.md`), built up milestone by milestone:
    - members per day;
    - share of label_ok stocks;
    - calendar range;
    - spot prices;
    - NaN rate per feature;
    - silhouette curve and regime coverage;
    - single-feature IC sanity check on the **training period only** (test 30).

---

## 6. Model, loss and training loop

### 6.1 Forward pass for one date t

```
U = universe_t                                           # stocks whose features exist (known at t-1)
alphas[i] = zscore_over(agent_i(X_t), U)                 # [5, N], each row mean 0, std 1 over U
w_t, a_t  = scheduler(clip(scale(s_t), -5, 5))           # w_t: [5] on the simplex; a_t: [4] window attention
alpha_hat = sum_i w_t[i] * alphas[i]                     # [N] final score (also used for inference)
```

### 6.2 Loss for one training date t (on L = label_ok_t only)

```
N_t   = count(L)
eps   = kappa_soft * N_t / 100
R     = soft_rank(alphas[:, L].double(), eps)            # [5, N_t]
A     = zscore(R, dim=1)                                 # standardized soft ranks (reused for IC and Sigma)
Y     = zscore(hard_rank(y_t[L]))                        # label ranks; no gradient needed
IC    = (A * Y).mean(dim=1)                              # [5] per-agent soft-rank IC
Sigma = (1 - 1e-4) * (A @ A.T / N_t) + 1e-4 * I          # [5, 5] float64, exact unit diagonal
neg_logdet = -2 * sum(log(diag(cholesky(Sigma))))
p_t   = sum(alpha_hat[L] * y_t[L])                       # long-short "portfolio return" of day t
n, s1, s2 = n_prev + 1, s1_prev + p_t, s2_prev + p_t**2  # running float64 sums of cached p_s (detached) + live p_t
SR    = mean_from(n, s1) / (std_from(n, s1, s2, ddof=1) + 1e-8) if n >= 20 else 0
H     = -sum(w_t * log_softmax(logits_t))
m     = [0.1]*5; m[agent_of(regime(t))] = 1.0           # shared mapping object (A16)
loss  = -1.0*sum(w_t*m*IC) - 0.5*SR + 0.1*neg_logdet - 0.01*H
```

Then:
1. `loss.backward()`
2. clip the global gradient norm to 1.0
3. Adam step
4. cosine LR step
5. add `p_t.detach()` to the running sums

A skipped step (A27) does steps 4 and the skip counter only.

### 6.3 Numerical safeguards (log det and friends)

1. **float64** for soft ranks, Sigma and log det (5x5, so it costs nothing).
2. **Shrinkage toward identity** (A23).
   - Every eigenvalue is at least 1e-4, so the penalty is capped at about 46 (5 x ln(1e4)) and its gradient stays bounded.
   - It keeps diag = 1, so log det(I) = 0 exactly.
3. **Symmetrize** Sigma, then `torch.linalg.cholesky_ex`. If it fails, retry with 10x shrinkage and log the event. Expected count: 0.
4. **Masked z-score** with eps in the variance; the dead-agent and non-finite skip rules come from A27.
5. **Sharpe:** at least 20 values, plus eps in the denominator.
6. **Entropy** via log_softmax.
7. **Gradient clipping** at 1.0 (from the paper).
8. **Per-epoch logs:**
   - each loss term, and each term's gradient norm;
   - mean abs correlation, log det (unshrunk), smallest eigenvalue, condition number;
   - scheduler entropy and mean weight per agent;
   - the soft-vs-hard IC gap and the share of non-linear soft ranks;
   - skip counts.

### 6.4 Training loop and cost

- **Loop:** for each epoch (up to 200):
  1. reset the Sharpe sums;
  2. make one chronological pass over the purged training dates (about 2,650 for CSI300);
  3. evaluate validation IC, smooth it, and apply early stopping.

  Restore the best weights at the end.
- **Cost:** each step is tiny (5 MLPs on a 300 x 34 matrix, a 5x5 log det, and soft ranks of 5 x 300 numbers). Estimate: 3-5 ms per step, so about 10-15 s per epoch on CPU. A run takes about 5-20 minutes with early stopping, 50 minutes at worst. Seeds can run in parallel processes.

---

## 7. Backtest and metrics

Inputs per evaluated day t: the score alpha_hat_t (data through t-1) and the label r_t = Close(t+20) / Close(t) - 1.

| Metric | Definition (headline) | Also reported |
|---|---|---|
| IC | mean over days of Spearman(alpha_hat_t, r_t) over label_ok_t | - |
| ICIR | mean(IC_t) / std(IC_t), ddof 1, not annualized | - |
| Portfolio | Each day t: from the eligible stocks (A37), buy the top ceil(10%) by alpha_hat_t at the close of t, in equal amounts, and hold to the close of t+1. Daily return R_{t+1} = the average of each stock's next-day return. | 20-day-hold version only via D8 |
| AR | (final value) ^ (1 / years) - 1, years = calendar span | Qlib form: mean(R) x 238 |
| SR | mean(R) / std(R) x sqrt(observed trading days per year), risk-free 0 | Qlib form: x sqrt(238) |
| MDD | the largest (1 - value / running peak), with the starting value 1 included as a peak | Qlib form: min(cumsum - running max of cumsum) |
| Excess | AR, SR and MDD on R minus the equal-weighted member return (A40) | - |

All metrics are computed on validation during development. They are computed on the test period once, at M9. They are also computed on the A45 sub-window.

**Reports** (M9):
- results table: our mean +- std over 5 seeds next to Table 5's Det-MASO rows;
- cumulative-return curve with the real CSI300 index as a buy-and-hold line, as in the paper's Figure 2;
- stacked scheduler weights over the full test period with the real CSI300 index on top (Figure 4 analog; the paper's index line is not real data, see Section 12);
- regime coverage next to Table 4;
- diversity numbers next to Figure 3 (A44);
- the Shapley matrix;
- two extra plots (D8.4): the window-attention weights a_t over the test period (the paper promises this analysis but never shows it), and agent diversity over the test period.

---

## 8. Shapley interaction analysis

1. Run the frozen model over the evaluation period: validation during development, test at M9. Save alpha_i,t for all 5 agents, w_t, and the labels.
2. For each of the 31 non-empty coalitions S of the 5 agents:
   - alpha_hat^S_t = sum over i in S of (w_i,t / sum over j in S of w_j,t) * alpha_i,t
   - v(S) = mean over days of the exact Spearman IC of alpha_hat^S_t
   - v(empty) = 0
3. For each pair (i, j), I(i,j) = sum over subsets S of the other 3 agents of omega(abs(S)) * [v(S+i+j) - v(S+i) - v(S+j) + v(S)]. For n = 5, omega(0) = 1/4, omega(1) = 1/12, omega(2) = 1/12, omega(3) = 1/4. These sum to 1 over the 8 subsets, which I checked.
4. Output a 5x5 symmetric matrix per seed plus mean +- std, shown as a heatmap with regime names (A16).
5. Check the paper's claim in two parts and report each separately:
   - (a) I(Low-Vol Bull, Tail-Risk) > 0;
   - (b) it is the largest pair.

   Count how many of the 5 seeds agree with each part.
6. Cost: 31 coalitions x about 485 days x one Spearman each, which takes seconds.

---

## 9. Sanity checks and unit tests

**Math and loss**
1. `neg_logdet(I) == 0` exactly, including after shrinkage. For [[1, r], [r, 1]] it equals -log(1 - r^2).
2. Identical agents (rank-1 Sigma) give a finite but large penalty. The penalty rises as correlation rises.
3. `torch.autograd.gradcheck` in float64 on tiny inputs (8 stocks, 3 agents) for: masked z-score, the soft rank (in **both** the linear regime with large eps and the merged-block regime with small eps), the IC term, the log det term, the Sharpe term, entropy, and the full loss.
4. **Repulsion:** one gradient step on the log det term alone lowers the abs correlation of two correlated agents.
5. **Sign checks:** one step on the IC term alone raises IC; on the Sharpe term alone raises SR; on entropy alone raises H. This catches flipped signs, which gradcheck cannot.
6. **Stop-gradient:** cached Sharpe values carry no gradient.
7. **Soft rank:**
   - matches torchsort (or brute force) to 1e-8;
   - equals hard ranks at tiny eps and is exactly linear at large eps;
   - preserves order and handles ties;
   - eps equals kappa_soft * N_t / 100 on a known date.
8. Exact Spearman matches `scipy.stats.spearmanr`, with ties and NaN masking.
9. **Scheduler:** zero-init gives w = 0.2 for every agent at step 0; w and a each sum to 1; they do not depend on stock data.
10. Agent outputs have mean 0 and std 1 over universe_t; padding never leaks into statistics.
11. **Mask wiring:** with lambda_mask = 0 and c_SR = c_div = c_ent = 0, only the agent of regime(t) gets a nonzero gradient. The test uses the same mapping object that coverage, plots and Shapley use.
12. **Skip path:** a constant agent output triggers a skip: no update, no cache entry, the LR still advances, and the counter increments.

**Look-ahead and data**

13. **Features, two-sided check:**
    - changing any price on day t or later leaves row t unchanged;
    - changing a price on day t-1 does change row t (for each feature).

    Together these catch both leaks and accidental double shifts.
14. The same two-sided check for the market state s_t (M3).
15. **No label leak into decisions:** setting every label and every price after day t to NaN leaves alpha_hat_t and the stocks bought on t unchanged.
16. **Label alignment** on a hand-built price series: label(t) = C(t+20)/C(t) - 1 exactly.
17. **Suspensions** on a synthetic series:
    - the resumption day carries the full move;
    - there are no fake 0% days;
    - the 80% window rule applies;
    - the market state skips stocks that did not trade on both days.
18. **Splits and purge:**
    - every training label ends on or before the last training day;
    - every early-stopping label ends on or before the last validation day;
    - no date is in two splits.
19. **Frozen regimes:** the scaler and K-means see only training dates (assert the max fitted date is at most 2022-12-31). Centroids are byte-identical before and after val/test assignment.
20. **Membership:**
    - 300 members on the sample dates;
    - non-members never enter universe_t;
    - a stock leaving the index within 20 days keeps its label.
21. **Cross-sectional processing:**
    - before clipping: mean 0, std 1 over non-NaN members;
    - after clipping: all values in [-5, 5];
    - NaN becomes exactly 0;
    - non-members are absent.

**Backtest and metrics**

22. Toy return series with known answers: AR with calendar-time compounding, Sharpe with observed days per year, MDD including a day-1 loss, and both Qlib forms.
23. Portfolio timing: the return earned on day t+1 depends only on scores from day t, which use data through t-1.
24. The decile contains exactly ceil(10%) of the eligible stocks.

**Shapley**

25. An additive game (v(S) = sum of c_i) gives every I(i,j) = 0.
26. A pure pair-synergy game (v(S) = 1 if {i,j} is in S) gives I(i,j) = 1 and all other pairs 0.
27. Symmetry I(i,j) = I(j,i); weights sum to 1; all 32 coalitions are evaluated.

**End-to-end**

28. **Synthetic-market smoke test:**
    - build a fake panel where the label depends on feature 3 in regime A and on feature 7 in regime B;
    - train for a few epochs;
    - expect positive validation IC and agent correlation well below 1.
29. **Determinism:** the same seed on CPU gives identical metrics twice.
30. **Pipeline sanity on the training period (2012-2022 only):**
    - the IC of a few raw features (20-day momentum, 20-day volatility) should have the signs well documented for China's market; short-term reversal and the low-volatility effect usually make both negative;
    - a wrong sign points to a label or timing bug;
    - this never looks at the test period.

---

## 10. Success criteria

**Why an exact match is unrealistic:**
- About 45 details had to be assumed.
- The data vintage differs from the authors'.
- The test window is only about 485 trading days.
- The paper's ICIR of 0.62 implies a daily IC standard deviation of about 0.13. With 20-day overlapping labels, the test period holds only about 25 independent periods.
- We share the same market period, so market luck is common to both. Implementation choices and seeds still move results.

**The paper's own numbers may not be reliable targets.**
- Figures 3 and 4 cannot come from the experiment as described (Section 12, items 1-2), so a large gap may reflect the paper, not our code.
- Even the Single-MLP floor below comes from the same results table, so it is only a rough guide.
- Our four qualitative checks use our own thresholds.
- We never tune toward the paper's numbers: defaults are fixed and the config is frozen before the test period is opened (M9).

**Bands** (applied to the mean over 5 seeds, headline conventions):

| Metric | Paper CSI300 | Paper S&P 500 | "Reproduced" if within | "Close" if within |
|---|---|---|---|---|
| IC | 0.081 | 0.072 | +-0.015 | +-0.030 |
| ICIR | 0.62 | (none) | +-0.15 | +-0.30 |
| Sharpe | 1.85 | 1.58 | +-0.40 | +-0.80 |
| Annual return | 31.5% | 26.8% | +-7 points | +-14 points |
| Max drawdown | 8.5% | 8.2% | +-4 points | +-8 points |

A result better than a band (e.g. a higher IC, a smaller drawdown) counts as inside it, and is flagged as "exceeds paper".

**Qualitative checks:**
- **q1 (agents stay different):** the test-period mean abs correlation between agents is below 0.7.
- **q2 (the scheduler does not collapse onto one agent):** average scheduler entropy is above 0.80, which is half the maximum of ln 5 = 1.61.
- **q3 (crashes are rare):** the Tail-Risk regime covers 3-7% of training days.
- **q4 (the Shapley claim):** I(Low-Vol Bull, Tail-Risk) > 0 in at least 4 of 5 seeds. Whether it is also the largest pair is reported separately.

**Verdict per market, applied in this order:**
1. **Not reproduced:** IC or Sharpe is below that market's Single-MLP result in Table 5 (CSI300: 0.055 / 1.05; S&P 500: 0.048 / 0.92).
2. **Reproduced:** IC and Sharpe are within "Reproduced", annual return and drawdown are at least within "Close", and q1-q4 all hold.
3. **Partially reproduced:** IC and Sharpe are at least within "Close", and at least 3 of q1-q4 hold.
4. **Inconclusive:** anything else, with a list of what failed.

**If results are far off,** check in this order (cheapest and most likely first). Steps 1-6 use training or validation data only.
1. **Data:**
   - members per day;
   - NaN rates;
   - calendar end;
   - spot prices;
   - single-feature IC signs (test 30).
2. **Timing:** re-run tests 13-17.
3. **Regimes:** coverage vs Table 4; centroid profiles; silhouette argmax.
4. **Training dynamics:**
   - validation-IC curve and stop epoch;
   - per-term gradient norms (does one term dominate?);
   - scheduler entropy;
   - diversity trajectory;
   - skip counts.
5. **Soft rank:** the logged regime and the soft-vs-hard IC gap.
6. **Seed spread:** is the paper's number inside our 5-seed range?
7. **Evaluation conventions:** Qlib's form vs standard, excess vs absolute, and the A45 sub-window. These are reported anyway. The 20-day hold and the no-gap label timing (D8.1) are not planned; I will ask you before running them.
8. **Extra runs with other assumptions** (D8.3) are not planned either; I will ask you first. If you approve, winners are picked on validation and reported separately from "the" reproduction.
9. **If everything checks out,** the gap may be in the paper (Section 12). Report it as such.

---

## 11. Milestones (build order, smallest runnable piece first)

| # | Milestone | Done when |
|---|---|---|
| M0 | `git init`; uv project on Python 3.12 with pinned deps; `pytest` wired up; pinned CN data download + validation (A1) | `pytest` runs (empty suite passes); `qlib.init` works; the checks in Section 5 step 2 pass |
| M1 | Raw panel, membership, labels, splits + purge, the three masks, parquet cache | Tests 16, 18, 20 pass; QA report part 1 (members/day, label coverage, calendar, spot prices) |
| M2 | Suspension-aware returns, 34 features, cross-sectional processing | Tests 13, 17, 21, 30 pass; QA report part 2 (NaN rates, training-period single-feature ICs) |
| M3 | Market state + regimes + the shared regime-to-agent mapping | Tests 14, 19 pass; QA report part 3 (silhouette curve, coverage vs Table 4) |
| M4 | Agents + scheduler forward pass | Tests 9, 10 pass; one forward pass over a real date runs |
| M5 | Soft rank + all loss terms + safeguards + skip rules | Tests 1-8, 11, 12 pass (including gradcheck in both soft-rank regimes) |
| M6 | Trainer: smoke run on a 1-year slice, then the synthetic test, then a full seed-0 run with early stopping | Tests 28, 29 pass; seed-0 run finishes with zero skips and zero non-finite events; validation curve and per-term gradient norms logged |
| M7 | Inference + metrics + backtest, on **validation** (seed 0) | Tests 15, 22-24 pass; first validation results table |
| M8 | Shapley analysis on **validation** (seed 0) | Tests 25-27 pass; matrix and heatmap |
| M9 | Freeze config (hash recorded); run 5 seeds; **open the test period once**; full CSI300 report (Table 5 and 4 analogs, Figure 2/3/4 analogs, Shapley, sub-window, and the two D8.4 plots) | Report written; verdict per Section 10 |
| M10 | **Deferred, not part of the MVP** (D1 = a). After the CSI300 results, and once you have checked WRDS access, we pick a D1 option (b, c or d), build or obtain the data, then rerun M1-M9 with `configs/sp500.yaml` | Same as M9 for S&P 500 |

---

## 12. Inconsistencies and problems found in the paper

**How these were checked:**
1. I read the paper.
2. Two AI auditors re-read it for contradictions in numbers and in method.
3. Three AI skeptics then tried to refute every item, each with a different lens: exact wording, recomputed math, and a charitable expert reading with web fact-checks.

**No item was refuted.** Items 23 and 26 were judged only partly right, and the AlphaForge point in item 19 is an anomaly rather than a strict contradiction. For every other item, any detail a reviewer corrected is already fixed in the wording below. Items 1 and 2 I also re-derived myself.

### 12.1 High impact: changes what we build or what we can compare against

| # | Problem | Evidence | What we do |
|---|---|---|---|
| 1 | **Figure 4 does not show real data over the stated test period** | Decoded from the figure's vector data:<br>- The plotted data runs 2024-02-29 to 2025-04-18, not to 2026-03-01.<br>- The "CSI300 index" line has a value on every weekday, including 11 days when Chinese exchanges were closed (Oct 1-7 2024, Jan 28 - Feb 4 2025), and it moves on those days.<br>- Per web checks, it reads 3,991 on 2024-02-29 (real close about 3,516). It shows -8.3% on 2025-02-21 (real: about +1.1%) and only -0.9% on 2025-04-07 (real: about -7%).<br>- The text's "February 2025 selloff" looks nothing like the plotted drop: CSI300's worst February day was about -2% (Feb 28), and the month ended up about 2%. | Not a target. Our version plots the real index over the full test period. |
| 2 | **Figure 3's diversity numbers are impossible for 5 agents** | For 5 agents with mean abs pairwise correlation m, log det is at most ln(1+4m) + 4 ln(1-m). At m = 0.49 that is -1.61, yet the paper reports -0.28; all four bars violate the bound, and averaging over epochs or seeds cannot fix it. The numbers are consistent with a **2-agent** system: ln(1 - rho^2) at rho = 0.978 / 0.748 / 0.637 / 0.494 gives exactly -3.13 / -0.82 / -0.52 / -0.28. | No valid diversity target for 5 agents; we report ours with this caveat |
| 3 | The test-time procedure is never stated | Algorithm 1 trains once with early stopping, i.e. a frozen model. But l.611-613 says "at each rebalance date the model is updated on the T_train most recent observations". | Frozen (settled by your brief) |
| 4 | 20-day labels cross the split boundaries | "No overlap" is true of the date ranges. But the last ~20 training labels use validation-period prices, and the last ~20 validation labels, which drive early stopping, use test-period prices. No purge is described. 2024-02-29, a trading day, belongs to no split. | Purge (settled by your brief) |
| 5 | T_train is overloaded and never given a value | It serves as the loop bound in Algorithm 1 (l.625), the Sharpe window {max(1, t - T_train + 1) .. t} (l.396), and "updated on the T_train most recent observations" (l.612). These are consistent only if T_train = the number of training dates, in which case the "rolling" Sharpe window is really an expanding one. | A24 |
| 6 | Promised architecture details are missing, and other required values are never given | "Architecture details are given in Section 4" (l.288), but Section 4 has no agent widths, activation or window-MLP width. K_max, kappa_soft and T_train are never valued either. | A14, A17, A19, A21 |
| 7 | Soft-rank softness is never given, and most values turn it into plain correlation | kappa_scaled = kappa_soft * N / 100 (l.601), but kappa_soft has no value. With z-scored inputs, any kappa_soft >= 0.1 makes the agent side of the "rank" correlation the paper stresses exactly linear (at 0.01, almost always). The scaling also makes bigger universes softer, without explanation. | D4 |
| 8 | lambda_mask is "selected on the validation split" (l.749), but no validation result appears anywhere | The only scan (Table 7) is on the test period, and every default sits exactly at the test-IC peak. Suggestive, not proof. No S&P 500 value is given. | D5 |
| 9 | Where the S&P 500 data came from is unclear | "Obtained from the Qlib platform" for 2023-2026 is possible only through Qlib's Yahoo collector, which downloads currently listed stocks only and so builds in survivorship bias. Qlib's official US data ends 2020-11-10. The authors' actual source is unknown. | D1 |

### 12.2 Medium impact: reported results contradict each other

| # | Problem | Evidence |
|---|---|---|
| 10 | **"Removing diversity drops IC below the single MLP on both markets" (l.793) is contradicted by the paper's own Table 5.** This is the one you were told about, with one correction: the sentence is in the main-results section. The ablation section and its Table 6 (IC 0.058) agree with Table 5. | Det-MASO (no div.) IC is 0.058 vs Single-MLP 0.055 on CSI300, and 0.050 vs 0.048 on S&P 500: above, not below. Sharpe and annual return are also above; only drawdown is worse. Proposition 1 assumes c_div > 0, so it cannot "predict" the c_div = 0 case either. |
| 11 | The abstract mixes comparison bases | The abstract claims "IC +1.1%, AR +6.4%, Sharpe +0.33". IC and AR match CSI300 vs AlphaForge, but CSI300's Sharpe gain is +0.43. +0.33 matches S&P 500 vs AlphaForge, or CSI300 vs the MMD variant. "1.1%" is an absolute IC difference; the body calls the same gain +15.7%. |
| 12 | Promised analyses are absent | Missing: tracking of lambda_min (Sigma's smallest eigenvalue) "on the training curves" (l.577-581); det(Sigma) "throughout training" with a cost comparison "across both datasets" (l.519-524); an analysis of the window-attention weights a_t (l.336); epsilon "calibrated on the validation split" (l.577), never done or valued; the Shapley matrix itself (only "I(1,5) > 0" is stated). |
| 13 | Figure 3 averages "the last 40 training epochs" | With patience-8 early stopping and best-epoch restore, those epochs are not the deployed model and may not even exist |
| 14 | The number of runs is unclear | Only Figure 3 mentions 5 seeds; Table 5 shows single numbers with no spread |
| 15 | The mask description is looser than the math | "Each agent receives full gradient updates primarily on ... its assigned regime" (l.281), but only the IC term is masked. Sharpe and diversity gradients reach every agent every day, which l.367 concedes. |
| 16 | "Fundamental data" is claimed but never used | l.720 vs the 34 features, which are all price/volume based |
| 17 | "Five metrics" are promised, but the main table has no ICIR | ICIR appears only for the CSI300 Det-MASO variants (Tables 6-7) |
| 18 | "Throughout" vs "for most" | The text says Det-MASO leads "throughout the test period" (l.783); the Figure 2 caption says "for most of the test period" |
| 19 | Three issues affect only baseline or variant rows (out of scope) | (a) AlphaForge's S&P 500 IC (0.047) ranks last while its Sharpe and return rank 3rd; possibly a typo. (b) Table 5 underlines MMD's CSI300 drawdown of 10.5% as second best, but AlphaForge's 10.2% is better. (c) SND and MMD are two other diversity penalties the paper compares against. SND is called (often) non-differentiable, yet a gradient-trained SND variant is reported, and neither formula is given. |

### 12.3 Low impact: notation, theory, wording

| # | Problem |
|---|---|
| 20 | Two stock features reuse the market-level symbols mu_w and sigma_w ("mu_20 - mu_60", "mu_60 / sigma_60"). Read literally, they would be identical for every stock and vanish after z-scoring; we use each stock's own values. D = 34 also needs the unnamed "high-low spread x2" to be two features. |
| 21 | "8.5 x 10^4 multiply-adds per step" = 500 x 34 x 5, the cost of one linear layer per agent, not a 3-hidden-layer MLP. The soft-rank cost is also left out. |
| 22 | "c_div >> lambda_mask" in sensitivity: their IC spans are 0.012 vs 0.011. ">" holds; ">>" does not. |
| 23 | (Partly right) "Scaling all loss coefficients leaves Adam unchanged" holds only approximately; clipping, weight decay and Adam's epsilon break it. |
| 24 | Proposition 1 assumes steady ascent on one fixed objective, but training takes one Adam step per date on a different objective each time. Proposition 2's conclusion about raw outputs does not follow from its soft-rank matrix. |
| 25 | Symbol clashes: tau (horizon and a dummy day index), kappa (temperature and an unrelated constant), and others. |
| 26 | (Partly right) "Without explicit regime labels" (intro l.146, related work l.185, conclusion l.930) vs training that uses hard K-means labels through the mask. Inference needs none. |
| 27 | The Shapley "evaluation period" is never named; test is the natural reading. |

**What this means for the plan:**
- Items 1-2 mean two of the paper's four figures cannot be reproduction targets, and they raise doubts about how its tables were produced. Section 10 is built with that in mind.
- Items 3-9 each map to an assumption or decision.
- The rest go in the write-up and do not change the code.

---

## 13. Decisions (answered 2026-11-05)

**Summary of your answers** (the full explanations follow, for reference):

| Item | Decision |
|---|---|
| Settled by your brief: purge, frozen model | Kept as-is |
| D1 Markets / S&P 500 data | a: CSI300 only for the MVP. S&P 500 is decided after the CSI300 results; you will check WRDS access first. |
| D2 Agent count | a: keep 5, and report what the clustering picked |
| D3 Rebalancing | a: daily |
| D4 Ranking softness | a: kappa_soft = 1.0 |
| D5 lambda_mask | a: 0.1 for both markets |
| D6 Annualization | a: standard recipe as the headline; Qlib's form alongside |
| D7 Success bands | a: as written in Section 10 |
| D8.1 Alternative evaluations | No for now. Revisit only if results are far off, and ask you first. |
| D8.2 Single-network and no-diversity runs | No. You will run these as a separate step after the MVP. |
| D8.3 Re-runs with other guesses | No for now. Revisit only if results are far off, and ask you first. |
| D8.4 Two extra plots | Yes |
| D9 Mac GPU | a: CPU on your Mac (an NVIDIA GPU is still used automatically if present) |
| D10 Other defaults | a: accept all other Section 3 defaults |

**Already settled by your brief** (you confirmed: keep as-is).
- **Purge (A6):** the 20 days before each split boundary are dropped, so training answers cannot peek into later periods. Your sanity-check list asks for this.
- **Frozen model (A34):** the model is trained once, frozen, then tested. Continued learning during the test would be walk-forward retraining, which your brief excludes.

Each decision below has a plain-English explanation, the options, and my recommendation. Terms are explained in Section 14.

### D1. Which markets, and where the S&P 500 data comes from

**Plain version:**
- The paper tests on China's 300 biggest companies (CSI300) and the US S&P 500.
- Good free Chinese data exists, including which companies were in the index on each past date.
- The US is harder. A company can leave the S&P 500 in two ways:
  - it shrinks but keeps trading (free sources still have its prices);
  - it **stops trading** because it was bought out or went bankrupt ("delisted"). Free sources usually lack those prices.
- Leaving delisted companies out secretly keeps only the "survivors" and makes any strategy look better than it was. This is **survivorship bias**. Qlib's official US data also stops in 2020.

**Options:**
- **(a)** CSI300 only for this MVP; decide on S&P 500 after the CSI300 results are in.
- **(b)** Then S&P 500 from free sources: historical membership lists on GitHub plus Yahoo prices. About 2-3 extra days of work. Roughly a third of the companies that left the index will have no prices; this gets documented.
- **(c)** Then S&P 500 from a source that keeps delisted companies. Either EODHD (paid, roughly $20-60 for a month), or **WRDS/CRSP**. WRDS is a research-data service many universities subscribe to; CRSP is its US stock database, which includes delisted companies and past S&P 500 lists. To check access, search your university library's database list for "WRDS" (it is often business-school only).
- **(d)** S&P 500 with today's members only. Simplest, and possibly what the authors did, but biased.

**My recommendation:** (a) now. Then (c) if you have WRDS access, otherwise (b).

**Your answer:** a. You will check WRDS access before the S&P 500 step.

### D2. Your brief vs the paper, if the clustering does not say 5

**Plain version:**
- The paper picks the number of agents with a clustering method that counts distinct market "moods" (regimes), trying 2 to 10 of them. Its rule is: use whatever number wins. On its data, that was 5.
- Your brief asks for both "a 5-agent system" and "exactly what the paper describes". If our clustering picks, say, 4, those two requests conflict.

**Options:**
- **(a)** Keep 5 and report what the clustering picked. Your brief also rules out "different agent counts", and all the paper's results are for 5.
- **(b)** Use the number the clustering picks.

**My recommendation:** (a).

**Your answer:** a

### D3. How often the simulated portfolio swaps stocks

**Plain version:**
- To measure money made, we simulate holding the 10% of stocks with the highest scores.
- The paper never says how often we re-pick them.
- No trading fees are charged (out of scope). Re-picking every day would cost the most in real life, so it gives the most flattering numbers. It is also the most common choice in Qlib-based papers, and probably what the authors did.

**Options:**
- **(a)** Every day.
- **(b)** Every 20 trading days. This matches the 20-day prediction horizon, but results then depend on which day you start.

**My recommendation:** (a). The 20-day version can be added as a side check via D8.

**Your answer:** a

### D4. The "softness" of the differentiable ranking (kappa_soft)

**Plain version:**
- The model is trained to rank stocks well. Ranking is not differentiable, so the paper uses a smooth version of ranking with a softness knob, and never says what value it used.
- Our analysis, to be confirmed by unit test 7:
  - At knob values of **0.1 or more**, the smooth ranking becomes an exact rescaled copy of each agent's scores. At 0.01 that happens on almost all days.
  - Training then rewards ordinary correlation between the scores and the ranks of the returns. Only very small values (about 0.001-0.002) behave like real ranks, and in that range the most extreme stocks, the ones the portfolio buys, stop getting a training signal.

**Options:**
- **(a)** kappa_soft = 1.0. Any "normal-looking" value the authors might have used (0.1, 1) behaves identically, so this most likely matches what they ran.
- **(b)** kappa_soft = 0.0015, genuinely rank-like and closer to the paper's stated intent.
- **(c)** Train both once and keep the better one on validation data.

**My recommendation:** (a). The logs will show which regime we are in either way.

**Your answer:** a

### D5. lambda_mask (how much each agent learns from days outside its own regime)

**Plain version:**
- Each agent gets an "accuracy" training signal every day, and lambda_mask sets how much of it the agent receives on days outside its own regime: 0 = none, 1 = the same as on its own days.
- Every agent still gets the shared Sharpe and diversity signals every day.
- The paper uses 0.1 for CSI300, says it was picked on validation data, and gives no S&P 500 value.

**Options:**
- **(a)** 0.1 for both markets.
- **(b)** Pick it per market from {0, 0.05, 0.1, 0.2, 0.5} on validation data, as the paper describes. That is 5 extra training runs per market, about an hour of CPU.

**My recommendation:** (a).

**Your answer:** a

### D6. How to turn daily returns into yearly numbers

**Plain version:**
- **Standard recipe:** compound the daily returns, use the market's actual number of trading days per year (about 242 in China, 252 in the US), and measure the worst drop from a peak in percent.
- **Qlib's built-in tool:** adds up daily returns instead of compounding, always assumes 238 days a year, and measures drops in "summed return" units.
- The two can differ by a few percentage points on the same portfolio. We report both; the question is which is the headline used for the verdict.

**Options:**
- **(a)** Standard as the headline. The paper's words "annualized return" and "peak-to-trough decline" read most naturally this way.
- **(b)** Qlib's as the headline.

**My recommendation:** (a).

**Your answer:** a

### D7. How close counts as "successfully reproduced"

**Plain version:** Section 10 proposes "Reproduced" bands of IC 0.066-0.096 and Sharpe 1.45-2.25 for CSI300. Both are still very strong results by industry standards (Section 14). They are wide because 2 years of test data is noisy and about 45 details had to be guessed. There is also a floor: we must beat what the paper reports for a single plain network.

**Options:**
- **(a)** As written: IC +-0.015, Sharpe +-0.40.
- **(b)** Stricter: IC +-0.010, Sharpe +-0.25.
- **(c)** Looser: IC +-0.025, Sharpe +-0.60.

**My recommendation:** (a).

**Your answer:** a

### D8. Optional extra runs (each needs a separate yes or no)

None of these is in the paper. All of them are evaluated or chosen on validation data, and their results are reported separately from the reproduction.

1. **Alternative evaluations of the same trained model:** a 20-day-hold portfolio (D3's option b), and a version where features are allowed to use day t's prices (AlphaGen's timing, which removes the one-day gap). No retraining is needed except for the timing variant. Recommend: **only if results are far off**. Answer: **no for now**; revisit only if results are far off, and ask you first.
2. **Two quick sanity runs** that the paper calls a baseline and an ablation: a single plain network, and Det-MASO with the diversity penalty switched off. They would confirm the diversity mechanism actually does something on our data. Recommend: **no for now** (your scope excludes them); revisit if results are far off. Answer: **no**; you will run these as a separate step after the MVP.
3. **Re-runs with other guesses** for the biggest unknowns (agent layer sizes, softness, the 252-day Sharpe window), each chosen on validation data. Recommend: **only if results are far off**. Answer: **no for now**; revisit only if results are far off, and ask you first.
4. **Two extra plots** from the same run: how much the scheduler leans on each time window (the attention weights a_t, which the paper promises but never shows), and agent diversity measured over the test period. Recommend: **yes**, cheap. Answer: **yes**.

### D9. Your Mac's GPU

**Plain version:**
- Your brief says "use a GPU if available". An NVIDIA GPU, if one is present (for example on a lab machine), is used automatically.
- Your Mac's built-in GPU cannot do the 64-bit math our ranking and log-det steps need.
- The model is so small that moving data to the GPU would probably make it **slower** than the CPU.

**Options:**
- **(a)** Use the CPU on your Mac.
- **(b)** Try the Mac GPU for the neural-network parts only.

**My recommendation:** (a).

**Your answer:** a

### D10. Everything else

**Plain version:** Section 3 lists about 45 smaller defaults not covered above (layer sizes, feature formulas, suspension handling, numerical safeguards, and so on).

**Options:**
- **(a)** Accept all Section 3 defaults not covered by D1-D9.
- **(b)** Name the ones you want changed.

**My recommendation:** (a).

**Your answer:** a

---

## 14. Glossary (plain English)

**Markets and data**

| Term | Meaning |
|---|---|
| Stock universe / index members | The set of stocks we choose from. CSI300 = China's 300 largest listed companies; S&P 500 = about 500 large US companies. Membership changes over time (twice a year for CSI300). |
| Index level | One number per day that tracks the combined value of the index's members |
| Point-in-time membership | Using the member list as it was **on each past date**, not today's list |
| Delisted | A company that stopped trading (bought out or bankrupt). Different from merely leaving an index. |
| Survivorship bias | Testing only on companies that still exist today. It hides the losers and inflates results. |
| Adjusted prices | Prices corrected for stock splits and dividends, so a 2-for-1 split does not look like a 50% crash |
| Suspension (China) | A day a stock is not allowed to trade, so it has no price that day |
| Fundamental data | Company accounting numbers such as earnings or book value. The paper claims to use them, but no feature does. |
| WRDS / CRSP | A university research-data service and its US stock database (see D1) |
| Qlib / Alpha158 | Microsoft's open-source quant research platform, and its standard library of 158 stock features |

**Prediction and evaluation**

| Term | Meaning |
|---|---|
| Label | The "answer" the model learns to predict: here, each stock's return over the next 20 trading days |
| Forward return (20-day) | How much a stock's price changes over the next 20 trading days, about one calendar month |
| Cross-section / cross-sectional z-score | All stocks on one day. Z-scoring "cross-sectionally" means subtracting that day's average across stocks and dividing by that day's spread. |
| Factor / alpha / score | A number per stock per day that predicts which stocks will do relatively better. Det-MASO's final score is alpha_hat. |
| IC (information coefficient) | Each day, the rank correlation between our scores and the actual forward returns, averaged over days. 0 = useless, 1 = perfect. In practice 0.03-0.05 is decent and 0.08 is very strong. |
| Rank correlation (Spearman) | Correlation computed on ranks (1st, 2nd, 3rd ...) instead of raw values, so a few extreme stocks cannot dominate |
| ICIR | Average daily IC divided by how much it varies day to day. Higher = more consistent. |
| Backtest | Replaying history: pretend to trade on our scores day by day and record the gains and losses |
| Top-decile long-only portfolio | Each day, "buy" the 10% of stocks with the highest scores in equal amounts, and never bet against (short) any stock |
| Rebalance | Re-picking the portfolio's stocks |
| Long-short portfolio | Buy high-score stocks and short-sell low-score stocks. The paper's training loss uses one; its evaluation does not. |
| Annualized return (AR) | The yearly growth rate implied by the test-period returns |
| Sharpe ratio (SR) | Return divided by volatility, scaled to a year: roughly "reward per unit of bumpiness". Above 1 is good, above 2 is excellent. |
| Max drawdown (MDD) | The worst peak-to-bottom drop of the portfolio's value during the test period |
| Excess return | Our return minus a benchmark's return (here, the average member stock) |
| Points (pts) | Percentage points: going from 31.5% to 24.5% is a drop of 7 points |
| Look-ahead bias | Accidentally using information that would not have been available at the time, such as tomorrow's price. It makes backtests look far better than reality. |
| Purge | Leaving a gap between training and test data so labels that look into the future cannot leak across the boundary |
| Trading days per year | US markets trade about 252 days a year and China about 242. Qlib's tools use 238 by convention. |

**Regimes**

| Term | Meaning |
|---|---|
| Market regime | A recurring market "mood", defined here by volatility (how jumpy), momentum (recent trend) and dispersion (how differently stocks move) |
| The 5 regime names | Bull/Bear = rising/falling market; Low-Vol/High-Vol = calm/jumpy; Sideways = no clear trend; Tail-Risk = rare crash-like periods |

**Model and math**

| Term | Meaning |
|---|---|
| Soft rank | A smooth, differentiable approximation of "rank these numbers", so a network can be trained to optimize rankings |
| log det of a correlation matrix | One number for how different the 5 agents' outputs are: 0 = completely independent; very negative = some agents are near-copies. The penalty -log det pushes them apart. Its "smallest eigenvalue" is another such measure. |
| Shapley interaction index | A game-theory measure of whether two agents are worth more together than separately (positive = they complement each other) |
| Model knobs | c_IC, c_SR, c_div, c_ent = how much the loss cares about accuracy, Sharpe, diversity and scheduler spread. lambda_mask = how much an agent learns from other regimes' days (D5). kappa_soft = ranking softness (D4). T_train = the Sharpe window (A24). |
