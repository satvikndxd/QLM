"""Author the seed exemplar entries for the QLM 1 supervision corpus.

The embedded ``code_implementation`` blocks are kept here as real Python
triple-quoted strings (syntax-highlightable, reviewable, diffable) and
serialized to ``data/entries/*.json``. Regenerate with:

    python tools/author_entries.py

Every entry is then gated by ``python -m qlm.cli build`` which
schema-validates it and actually executes its code (execution-based
verification). Entries are deliberately self-contained and deterministic:
synthetic, seeded data generators stand in for market data so that the
*methodology* — not a data vendor — is what the student model learns.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "entries"

# =====================================================================
# Entry 1 — complexity 3: SMA crossover with honest frictions + bootstrap
# =====================================================================

CODE_SMA = '''"""Trend-following SMA crossover study with honest frictions.

Pipeline stages exercised: Construct -> Implement -> Simulate -> Measure
-> Validate (block bootstrap) -> Report.

The data generator produces a regime-switching geometric random walk so
the strategy faces both trending and mean-reverting regimes. All results
are deterministic (seeded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42
N_DAYS = 2520          # ~10 trading years
ANN_FACTOR = 252
COST_BPS = 5.0         # one-way cost per unit of turnover, in basis points
FAST, SLOW = 20, 100
N_BOOT = 2000
BLOCK_LEN = 21         # ~1 month blocks preserve short-range autocorrelation


def simulate_prices(n_days: int, seed: int) -> pd.Series:
    """Regime-switching GBM: alternating trend / chop regimes.

    Why: a single-drift GBM would make any trend filter look either
    trivially good or trivially bad; regime switching forces the test to
    say something about *conditional* performance.
    """
    rng = np.random.default_rng(seed)
    # Regime lengths ~ geometric with mean 126 days (half a year).
    drifts, vols = [], []
    trending = True
    remaining = n_days
    while remaining > 0:
        length = min(int(rng.geometric(1.0 / 126)), remaining)
        mu = (0.12 if trending else -0.02) / ANN_FACTOR      # daily drift
        sigma = (0.15 if trending else 0.25) / np.sqrt(ANN_FACTOR)
        drifts.append(np.full(length, mu))
        vols.append(np.full(length, sigma))
        trending = not trending
        remaining -= length
    mu_t = np.concatenate(drifts)
    sigma_t = np.concatenate(vols)
    log_rets = mu_t - 0.5 * sigma_t**2 + sigma_t * rng.standard_normal(n_days)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.Series(prices, index=idx, name="close")


def sma_signal(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """1.0 when fast SMA > slow SMA, else 0.0. NaN during warm-up -> flat."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    sig = (fast_ma > slow_ma).astype(float)
    sig[slow_ma.isna()] = 0.0  # no position before the slow window is full
    return sig


def backtest(prices: pd.Series, signal: pd.Series, cost_bps: float) -> pd.Series:
    """Vectorized long/flat backtest with a one-bar execution lag.

    The .shift(1) is the single most important line: the signal computed
    on bar t's close can only be traded on bar t+1. Removing it creates
    look-ahead bias that typically inflates Sharpe by 0.5-1.0.
    """
    position = signal.shift(1).fillna(0.0)
    simple_rets = prices.pct_change().fillna(0.0)
    gross = position * simple_rets
    turnover = position.diff().abs().fillna(position.abs())
    net = gross - turnover * cost_bps / 1e4
    return net


def annualized_sharpe(rets: np.ndarray) -> float:
    sd = rets.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(rets.mean() / sd * np.sqrt(ANN_FACTOR))


def max_drawdown(net_rets: pd.Series) -> float:
    equity = (1.0 + net_rets).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def block_bootstrap_pvalue(net: np.ndarray, n_boot: int, block: int, seed: int) -> float:
    """Moving-block bootstrap p-value for H0: true Sharpe <= 0.

    We resample under the null by demeaning, preserving the return
    autocorrelation structure inside blocks; the p-value is the fraction
    of null-world Sharpes that reach the observed one.
    """
    rng = np.random.default_rng(seed)
    observed = annualized_sharpe(net)
    demeaned = net - net.mean()  # impose H0: zero mean return
    n = len(demeaned)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    # Vectorized block assembly: (n_boot, n_blocks, block) -> (n_boot, n)
    offsets = np.arange(block)
    samples = demeaned[(starts[:, :, None] + offsets[None, None, :])]
    samples = samples.reshape(n_boot, -1)[:, :n]
    boot_sharpes = (
        samples.mean(axis=1) / samples.std(axis=1, ddof=1) * np.sqrt(ANN_FACTOR)
    )
    return float((boot_sharpes >= observed).mean())


def main() -> None:
    prices = simulate_prices(N_DAYS, RNG_SEED)
    signal = sma_signal(prices, FAST, SLOW)
    net = backtest(prices, signal, COST_BPS)
    net_np = net.to_numpy()

    sharpe = annualized_sharpe(net_np)
    mdd = max_drawdown(net)
    cagr = float((1.0 + net).prod() ** (ANN_FACTOR / len(net)) - 1.0)
    exposure = float(signal.shift(1).fillna(0.0).mean())
    ann_turnover = float(signal.shift(1).fillna(0.0).diff().abs().sum() / len(net) * ANN_FACTOR)
    pval = block_bootstrap_pvalue(net_np, N_BOOT, BLOCK_LEN, RNG_SEED + 1)

    print("RESULTS")
    print(f"annualized_sharpe_net={sharpe:.3f}")
    print(f"cagr_net={cagr:.4f}")
    print(f"max_drawdown={mdd:.4f}")
    print(f"time_in_market={exposure:.3f}")
    print(f"annualized_turnover={ann_turnover:.2f}")
    print(f"bootstrap_pvalue_sharpe_gt_0={pval:.4f}")
    verdict = "REJECT_H0" if pval < 0.05 else "FAIL_TO_REJECT_H0"
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_SMA = {
    "metadata": {
        "id": "qlm1-000001-sma-crossover",
        "domain": "Trend Following / Time-Series Momentum",
        "complexity": 3,
        "tags": [
            "sma-crossover",
            "backtesting",
            "transaction-costs",
            "look-ahead-bias",
            "block-bootstrap",
            "sharpe-ratio",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The user asks whether a 20/100 SMA crossover 'works'. Parse this as a testable "
            "statistical claim, not a yes/no question: does the net-of-cost return stream of the "
            "lagged crossover rule have a mean significantly above zero? The three failure modes "
            "to guard against from the start are (1) trading on the same bar the signal is "
            "computed (look-ahead), (2) ignoring turnover costs, and (3) quoting a Sharpe ratio "
            "with no significance test, since autocorrelated daily returns make naive t-stats "
            "invalid."
        ),
        "tool_selection": (
            "Python REPL only. numpy for vectorized simulation and the bootstrap, pandas for "
            "rolling windows and calendar indexing. No external data feed is needed because the "
            "question is methodological; a seeded regime-switching generator is sufficient and "
            "fully reproducible."
        ),
        "recursive_delegation": (
            "Not warranted at this complexity. A single agent completes the full pipeline. If "
            "the user later asks for a (fast, slow) parameter sweep, delegate the grid to "
            "parallel sub-agents and reserve the parent for multiple-testing correction of the "
            "pooled results (e.g. White's reality check), which must see all runs at once."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Let r_t be the strategy's net daily return: r_t = pos_{t} * R_t - c * |pos_t - "
            "pos_{t-1}|, with pos_t = 1{SMA_20(t-1) > SMA_100(t-1)}, R_t the asset's simple "
            "return, and c = 5 bps. H0: E[r_t] <= 0 (the rule has no positive edge net of "
            "costs). H1: E[r_t] > 0. Test statistic: annualized Sharpe = mean(r)/sd(r)*sqrt(252), "
            "with the null distribution obtained by moving-block bootstrap of the demeaned "
            "return series (block length 21 to preserve monthly-scale autocorrelation). "
            "Significance level alpha = 0.05, one-sided."
        ),
        "data_engineering": (
            "Data: 2520 business days of synthetic close prices from a regime-switching GBM "
            "(trend regimes: mu=12%/yr, vol=15%; chop regimes: mu=-2%/yr, vol=25%; regime "
            "durations geometric with mean 126 days), seeded for determinism. Construction "
            "rules that transfer directly to real data: use a business-day DatetimeIndex; "
            "compute rolling means with pandas .rolling so warm-up produces NaN, and map NaN "
            "signal to flat rather than backfilling (backfilling is a leakage bug); portfolio "
            "arithmetic uses simple returns because they aggregate across positions additively, "
            "while log returns would only be appropriate for single-asset compounding analysis."
        ),
        "methodology_justification": (
            "A vectorized backtest with an explicit one-bar execution lag is chosen over an "
            "event-driven engine: at daily frequency with a single instrument, vectorization is "
            "exact, faster, and easier to audit line-by-line. Costs are modeled proportional to "
            "turnover (|Δposition| * 5 bps) — the correct first-order friction for a liquid "
            "future/ETF. For inference, the moving-block bootstrap is preferred over (a) an IID "
            "bootstrap, which destroys the autocorrelation that trend rules feed on and thus "
            "understates the null variance, and (b) a Newey-West t-test, which is asymptotically "
            "valid but fragile for the heavily skewed, kurtotic return streams long/flat rules "
            "produce."
        ),
        "code_implementation": CODE_SMA,
        "statistical_validation": (
            "1) Bootstrap p-value for Sharpe > 0 under a demeaned moving-block null (reported as "
            "bootstrap_pvalue_sharpe_gt_0); reject H0 only if p < 0.05. 2) Sanity invariants "
            "that must hold before any p-value is read: time_in_market strictly between 0 and 1; "
            "annualized_turnover consistent with crossover frequency (order 2-8 round trips/yr, "
            "not hundreds — a huge value signals a signal-chatter bug); equity curve free of "
            "NaN. 3) Robustness: rerun with seeds 42..51 and with (fast, slow) in {10/50, "
            "20/100, 50/200}; a real effect degrades smoothly across neighbors, a coding "
            "artifact or overfit peak does not."
        ),
        "risk_and_backtest_audit": (
            "Frictions: 5 bps one-way is realistic for liquid index products but optimistic for "
            "single names; rerun at 10 and 25 bps and report the Sharpe decay curve — trend "
            "rules with ~4 trades/yr should degrade slowly, and rapid decay reveals the PnL "
            "lives in the microstructure, not the trend. Tail risk: report max drawdown and "
            "note that long/flat trend following typically has positive skew but concentrated "
            "drawdowns in whipsaw regimes; check the drawdown coincides with simulated chop "
            "regimes as an internal consistency test. Capacity/liquidity are out of scope for "
            "the synthetic asset and must be re-audited on real data."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The generator is rigged in favor of the strategy: regime-switching drift is "
            "exactly the structure SMA crossovers harvest, so a positive result here proves the "
            "machinery works, not that markets trend. (2) Removing the .shift(1) silently "
            "inflates Sharpe — the single most common backtest bug. (3) fillna choices: "
            "backfilling the signal or forward-filling prices across the warm-up window leaks "
            "future information. (4) The bootstrap resamples demeaned *strategy* returns; "
            "resampling *asset* returns instead and re-running the rule would be a stronger "
            "null but breaks the vectorized shortcut — know which null you are testing."
        ),
        "falsification_strategy": (
            "Run the identical pipeline on a pure GBM with constant drift equal to the "
            "regime-average: the rule's conditional edge should vanish (p-value ~uniform), and "
            "if it does not, the harness itself is biased. Second: permute the regime blocks "
            "(destroying trend persistence but keeping the marginal distribution) and confirm "
            "the Sharpe collapses. Third: set costs to 50 bps and confirm the verdict flips "
            "before believing any cost-sensitivity claims."
        ),
        "limitations": (
            "Synthetic single-asset daily data; no gaps, no dividends, no borrow costs, no "
            "regime estimation error. Conclusions transfer only as far as the data-generating "
            "assumptions do; on real data the same code requires corporate-action-adjusted "
            "prices and a multiple-testing correction if any parameter was tuned. The bootstrap "
            "assumes stationarity within the sample, which fails across major structural "
            "breaks."
        ),
    },
    "agent_instructions": (
        "1. Restate the request as H0/H1 with an explicit test statistic and alpha. "
        "2. Generate seeded regime-switching prices (simulate_prices) and eyeball min/max/NaN "
        "count before proceeding. 3. Build the signal with rolling windows; assert the first "
        "SLOW-1 positions are flat. 4. Backtest with .shift(1) lag and turnover costs; assert "
        "abs(position).max() <= 1. 5. Compute Sharpe, CAGR, max drawdown, exposure, turnover. "
        "6. Run the moving-block bootstrap (2000 draws, block 21) for the one-sided p-value. "
        "7. Audit: rerun without the lag and with costs=0; record how much each inflates "
        "Sharpe — if the lag matters more than 0.3 Sharpe, flag signal chatter. 8. Rerun across "
        "10 seeds and 3 parameter pairs; tabulate. 9. Conclude probabilistically: report the "
        "p-value, the cost-decay curve, and state that the synthetic generator favors trend "
        "structure. Never output a verdict without the audit table from steps 7-8."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": ["RESULTS", "annualized_sharpe_net=", "bootstrap_pvalue_sharpe_gt_0=", "verdict=FAIL_TO_REJECT_H0"],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "pandas"],
    },
}

# =====================================================================
# Entry 2 — complexity 6: pairs trading via Engle-Granger cointegration
# =====================================================================

CODE_PAIRS = '''"""Pairs trading study: Engle-Granger cointegration with a strict
formation/trading split (no look-ahead in hedge ratio or z-score bands).

Pipeline stages exercised: Hypothesize -> Construct -> Methodology ->
Implement -> Simulate -> Validate (ADF, half-life) -> Generalize (OOS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

RNG_SEED = 7
N_DAYS = 1500
FORMATION = 750        # first half: estimate hedge ratio, bands, half-life
ANN_FACTOR = 252
ENTRY_Z, EXIT_Z = 2.0, 0.0
COST_BPS = 5.0         # per leg, one-way
TRUE_GAMMA = 1.4
SPREAD_HALFLIFE = 15.0 # days, of the simulated OU spread


def simulate_pair(n: int, seed: int) -> pd.DataFrame:
    """Cointegrated pair: log_pb = alpha + gamma*log_pa + OU spread.

    The OU spread has a known half-life so the estimated half-life can be
    checked against ground truth — an execution-verifiable teaching hook.
    """
    rng = np.random.default_rng(seed)
    # Common stochastic trend (random walk with mild drift).
    log_pa = np.cumsum(0.0002 + 0.012 * rng.standard_normal(n)) + np.log(50.0)
    phi = np.exp(np.log(0.5) / SPREAD_HALFLIFE)   # AR(1) coeff for target half-life
    eps_sd = 0.010
    spread = np.empty(n)
    spread[0] = 0.0
    shocks = eps_sd * rng.standard_normal(n)
    for t in range(1, n):
        # O(n) scalar recursion: an AR(1) is inherently sequential; n=1500
        # makes the loop cost negligible and the code unambiguous.
        spread[t] = phi * spread[t - 1] + shocks[t]
    log_pb = 0.25 + TRUE_GAMMA * log_pa + spread
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"log_pa": log_pa, "log_pb": log_pb}, index=idx)


def engle_granger_formation(df: pd.DataFrame) -> dict:
    """Step 1 of Engle-Granger on the FORMATION window only.

    OLS of log_pb on log_pa; ADF on residuals. Note: the ADF p-value here
    uses standard Dickey-Fuller critical values, which are known to be
    mildly liberal for estimated residuals (MacKinnon adjustment); we
    therefore demand p < 0.01 rather than 0.05 as a partial correction.
    """
    x = df["log_pa"].to_numpy()
    y = df["log_pb"].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_hat, gamma_hat = float(beta[0]), float(beta[1])
    resid = y - (alpha_hat + gamma_hat * x)
    adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")

    # Half-life from AR(1) on residuals: resid_t = phi*resid_{t-1} + e_t.
    r0, r1 = resid[:-1], resid[1:]
    phi_hat = float(np.dot(r0, r1) / np.dot(r0, r0))
    halflife = float(np.log(0.5) / np.log(phi_hat)) if 0.0 < phi_hat < 1.0 else float(9999)

    return {
        "alpha": alpha_hat,
        "gamma": gamma_hat,
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "halflife": halflife,
        "mu": float(resid.mean()),
        "sd": float(resid.std(ddof=1)),
    }


def zscore_positions(z: pd.Series, entry: float, exit_: float) -> pd.Series:
    """Vectorized entry/exit with hysteresis.

    Mark +1 (long spread) where z < -entry, -1 where z > entry, 0 where z
    crosses through the exit level; forward-fill between events. This
    reproduces the stateful entry/exit logic without a Python loop.
    """
    raw = pd.Series(np.nan, index=z.index)
    raw[z > entry] = -1.0
    raw[z < -entry] = 1.0
    crossed = np.sign(z - exit_) != np.sign(z.shift(1) - exit_)
    raw[crossed & raw.isna()] = 0.0
    return raw.ffill().fillna(0.0)


def backtest_oos(df: pd.DataFrame, params: dict) -> pd.Series:
    """Trade the spread out-of-sample with FROZEN formation parameters.

    gamma, mu, sd all come from the formation window: re-estimating them
    inside the trading window with a full-sample fit is the classic pairs
    look-ahead bug.
    """
    spread = df["log_pb"] - (params["alpha"] + params["gamma"] * df["log_pa"])
    z = (spread - params["mu"]) / params["sd"]
    pos = zscore_positions(z, ENTRY_Z, EXIT_Z).shift(1).fillna(0.0)  # 1-bar lag

    ret_a = df["log_pa"].diff().fillna(0.0)
    ret_b = df["log_pb"].diff().fillna(0.0)
    # Long spread = long B, short gamma units of A (per unit of B notional).
    gross = pos * (ret_b - params["gamma"] * ret_a)
    turnover = pos.diff().abs().fillna(pos.abs()) * (1.0 + abs(params["gamma"]))
    net = gross - turnover * COST_BPS / 1e4
    return net


def main() -> None:
    df = simulate_pair(N_DAYS, RNG_SEED)
    formation, trading = df.iloc[:FORMATION], df.iloc[FORMATION:]

    params = engle_granger_formation(formation)
    net = backtest_oos(trading, params)

    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(ANN_FACTOR)) if sd > 0 else 0.0
    equity = (1.0 + net).cumprod()
    mdd = float((equity / equity.cummax() - 1.0).min())
    pos = zscore_positions(
        (df["log_pb"] - (params["alpha"] + params["gamma"] * df["log_pa"]) - params["mu"])
        / params["sd"],
        ENTRY_Z,
        EXIT_Z,
    ).iloc[FORMATION:]
    n_trades = int((pos.diff().abs() > 0).sum())

    print("RESULTS")
    print(f"gamma_hat={params['gamma']:.4f} (true={TRUE_GAMMA})")
    print(f"adf_stat={params['adf_stat']:.3f}")
    print(f"adf_pvalue={params['adf_p']:.6f}")
    print(f"halflife_days={params['halflife']:.1f} (true={SPREAD_HALFLIFE})")
    print(f"oos_sharpe_net={sharpe:.3f}")
    print(f"oos_max_drawdown={mdd:.4f}")
    print(f"oos_position_changes={n_trades}")
    tradeable = params["adf_p"] < 0.01 and params["halflife"] < 60.0
    verdict = "COINTEGRATED_TRADEABLE" if tradeable else "NOT_TRADEABLE"
    print(f"formation_verdict={verdict}")
    # Canonical verdict line: every corpus entry ends stdout with
    # verdict=<expected_verdict> so automated verdict matching is uniform.
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_PAIRS = {
    "metadata": {
        "id": "qlm1-000002-pairs-engle-granger",
        "domain": "Statistical Arbitrage",
        "complexity": 6,
        "tags": [
            "pairs-trading",
            "cointegration",
            "engle-granger",
            "adf-test",
            "ornstein-uhlenbeck",
            "half-life",
            "out-of-sample",
            "look-ahead-bias",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The request is to evaluate a pairs trade between two assets. Decompose into two "
            "separable questions that must NOT share data: (1) formation — are the log prices "
            "cointegrated, with a mean-reversion half-life short enough to trade? (2) trading — "
            "does a z-score rule on the spread earn net of costs *out of sample*, with every "
            "parameter (hedge ratio, band center, band width) frozen from the formation window? "
            "Most published pairs backtests fail exactly at this boundary by re-fitting the "
            "hedge ratio on the full sample."
        ),
        "tool_selection": (
            "Python REPL with numpy/pandas for construction and backtesting, statsmodels for "
            "the ADF unit-root test (hand-rolling ADF critical values is error-prone and not "
            "the lesson here). No SQL/arxiv retrieval needed: the study is self-contained on a "
            "simulated cointegrated pair with known ground-truth gamma and half-life, which "
            "lets the agent verify its estimators before trusting them."
        ),
        "recursive_delegation": (
            "For a single pair, one agent suffices. At universe scale (screening N*(N-1)/2 "
            "candidate pairs), delegate: sub-agents each ADF-test a shard of pairs; the parent "
            "then owns the multiple-comparisons problem — with 10,000 pairs tested at p<0.01, "
            "~100 spurious 'cointegrated' pairs are expected by chance, so the parent must "
            "apply FDR control and demand OOS confirmation before any pair is tradeable."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Formation hypothesis — H0: the residual e_t = log P_B(t) - alpha - gamma*log "
            "P_A(t) has a unit root (no cointegration); H1: e_t is stationary. Test: ADF on "
            "formation-window residuals, reject at p < 0.01 (stricter than 0.05 because "
            "Engle-Granger residual-based ADF uses estimated residuals, making standard "
            "critical values liberal). Trading hypothesis — H0: E[r_t] <= 0 for the net "
            "z-score strategy on the *disjoint* trading window with frozen parameters; H1: "
            "E[r_t] > 0. Auxiliary check: estimated AR(1) half-life of the spread must be "
            "< 60 days for the rule's holding periods to be economically viable."
        ),
        "data_engineering": (
            "Simulate 1500 business days: a common stochastic trend (random walk in log price "
            "of A) plus log P_B = 0.25 + 1.4*log P_A + s_t, where s_t is AR(1) with a "
            "ground-truth 15-day half-life. Split 750/750 into formation/trading with no "
            "overlap. Work in log prices throughout: the cointegrating regression is specified "
            "in logs, and log-return differences make the dollar-neutral spread PnL additive "
            "(r_spread = r_B - gamma*r_A). On real data this step also requires: identical "
            "trading calendars for both legs (inner-join, never forward-fill one leg), "
            "corporate-action adjustment before logging, and a liquidity screen so the "
            "short leg is actually borrowable."
        ),
        "methodology_justification": (
            "Engle-Granger (OLS + residual ADF) is chosen over the Johansen procedure because "
            "with exactly two assets and one candidate cointegrating vector, EG is simpler, "
            "near-equally powered, and its failure modes are better understood; Johansen "
            "becomes necessary only for 3+ asset baskets. The hedge ratio comes from formation "
            "OLS — acceptable here because simulated measurement noise is absent; on real data "
            "consider total-least-squares since OLS gamma is direction-dependent (regressing A "
            "on B gives 1/gamma only without noise). Half-life via AR(1) projection gives the "
            "OU speed-of-reversion estimate that gates economic viability. Position logic is "
            "vectorized hysteresis (enter |z|>2, exit at z=0) with a one-bar execution lag."
        ),
        "code_implementation": CODE_PAIRS,
        "statistical_validation": (
            "1) ADF on formation residuals: require p < 0.01; also report the test statistic so "
            "the margin over the critical value is visible, not just the binary verdict. 2) "
            "Estimator calibration against ground truth: gamma_hat should be within ~2% of 1.4 "
            "and halflife within ~30% of 15 days — if not, the estimator code is buggy, and "
            "this check is only possible because we simulate; it is the reason to always test "
            "estimators on synthetic data first. 3) OOS: all performance is computed on the "
            "disjoint trading window; a formation-window backtest is reported nowhere, by "
            "design. 4) Re-run with 20 seeds: the ADF rejection rate should be ~100% here "
            "(the pair IS cointegrated); then set the spread to a pure random walk and confirm "
            "the rejection rate falls to ~1%, calibrating the test's size."
        ),
        "risk_and_backtest_audit": (
            "Costs: 5 bps per leg scaled by (1+|gamma|) turnover — the short leg trades gamma "
            "units per unit of long leg. Missing frictions that must be added for real "
            "deployment: short borrow fees (often 25-300 bps/yr and spiking exactly when "
            "spreads blow out), margin on the short leg, and legging risk when the two legs "
            "cannot be executed simultaneously. Tail risk: the strategy is short "
            "divergence — its worst case is a structural break (merger, delisting, index "
            "reconstitution) where the spread never reverts; enforce a hard stop at |z| > 4 "
            "and a max holding period of 3*halflife in production. Report max drawdown and "
            "position-change count; a near-zero trade count means the bands never triggered "
            "and the Sharpe is meaningless."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) Look-ahead via parameter reuse: computing z-scores with full-sample mean/sd "
            "is the most common pairs bug and typically doubles reported Sharpe. (2) "
            "Cointegration mining: scanning many pairs and reporting the best ADF p-value "
            "without multiplicity correction guarantees spurious discoveries. (3) The "
            "simulated spread is exactly AR(1)-Gaussian, the best case for both the ADF test "
            "and the z-score rule; real spreads have heavy tails and regime breaks. (4) "
            "Non-negative-price subtlety: the simulation works in logs directly, so there is "
            "no np.log(0) hazard, but on real data zero/negative prices from bad ticks must "
            "be filtered before logging. (5) OLS hedge-ratio asymmetry: regressing B-on-A vs "
            "A-on-B gives materially different gammas when residual variance is high."
        ),
        "falsification_strategy": (
            "Break the cointegration deliberately: replace the OU spread with a random walk of "
            "matched innovation variance and rerun end-to-end — the ADF should fail to reject "
            "~99% of the time and OOS Sharpe should center on a *negative* value (costs on "
            "noise trades). If the pipeline still reports 'tradeable', it is broken. Second "
            "attack: inject a structural break (gamma shifts from 1.4 to 1.1 mid-trading-"
            "window) and verify the drawdown control catches it. Third: shuffle the trading-"
            "window returns; any remaining PnL is a mechanical artifact."
        ),
        "limitations": (
            "Engle-Granger assumes a single, constant cointegrating vector — it cannot detect "
            "time-varying gamma, and the method says nothing about *why* the pair co-moves, so "
            "economic-linkage screening (same sector, same underlying exposure) must precede "
            "statistical screening. The ADF has low power against slow mean reversion "
            "(half-life > ~1/5 of the sample), so a 750-day formation window cannot certify "
            "half-lives beyond roughly 150 days. Results are daily-frequency; intraday "
            "execution assumptions (fills at close, no legging risk) are simplifications."
        ),
    },
    "agent_instructions": (
        "1. Split data 50/50 into formation/trading BEFORE any estimation; treat the trading "
        "window as sealed. 2. On formation: OLS log_pb ~ log_pa; store alpha, gamma. 3. ADF "
        "the residuals (autolag=AIC); require p < 0.01, else STOP and report NOT_TRADEABLE. "
        "4. Estimate AR(1) half-life; require < 60 days, else STOP. 5. Freeze (alpha, gamma, "
        "mu, sd); compute trading-window z-scores from frozen params only. 6. Generate "
        "positions with entry |z|>2, exit z=0, one-bar lag; backtest with per-leg costs "
        "scaled by (1+|gamma|). 7. Verify estimator calibration against simulation ground "
        "truth (gamma within 2%, half-life within 30%); if violated, debug estimators before "
        "reporting anything. 8. Falsification pass: rerun with a random-walk spread and "
        "confirm the pipeline rejects it. 9. Report ADF stat and p, half-life, OOS Sharpe, "
        "max drawdown, trade count, and an explicit list of frictions NOT modeled (borrow, "
        "legging, breaks). Never report a formation-window Sharpe."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": ["RESULTS", "adf_pvalue=", "halflife_days=", "oos_sharpe_net=", "formation_verdict=", "verdict=COINTEGRATED_TRADEABLE"],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "pandas", "statsmodels"],
    },
}

# =====================================================================
# Entry 3 — complexity 8: GARCH(1,1) MLE + walk-forward vol forecasting
# =====================================================================

CODE_GARCH = '''"""GARCH(1,1) volatility forecasting: hand-rolled MLE, walk-forward
one-step-ahead forecasts, QLIKE evaluation, Diebold-Mariano test vs EWMA.

Pipeline stages exercised: Methodology -> Implement (numerically stable
MLE) -> Generalize (train/test split, frozen parameters) -> Measure
(QLIKE) -> Validate (DM test with HAC variance) -> Falsify (benchmark).

Returns are worked in PERCENT units: daily variances near 1e0 rather
than 1e-4 keep the optimizer's Hessian well-conditioned and avoid
float underflow in the likelihood — a deliberate numerical-stability
choice, not a cosmetic one.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize

RNG_SEED = 123
N_TOTAL, N_BURN, N_TRAIN = 3000, 500, 2000
TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 0.05, 0.08, 0.90   # percent^2 units
EWMA_LAMBDA = 0.94
VAR_FLOOR = 1e-8       # variance floor: guards log() and division
DM_LAG = 10            # HAC truncation lag for the DM test


def simulate_garch(n: int, burn: int, seed: int) -> np.ndarray:
    """Simulate GARCH(1,1) returns in percent, discarding a burn-in.

    Unconditional variance = omega/(1-alpha-beta) = 2.5 (%^2), i.e. ~25%
    annualized vol — a realistic equity-index level.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    sigma2 = np.empty(n + burn)
    r = np.empty(n + burn)
    sigma2[0] = TRUE_OMEGA / (1.0 - TRUE_ALPHA - TRUE_BETA)
    r[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n + burn):
        # The GARCH recursion is inherently sequential (sigma2[t] depends
        # on sigma2[t-1]); a loop is the correct implementation, and at
        # n=3500 its cost is irrelevant.
        sigma2[t] = TRUE_OMEGA + TRUE_ALPHA * r[t - 1] ** 2 + TRUE_BETA * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * z[t]
    return r[burn:]


def garch_filter(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Run the variance recursion; initialize at sample variance.

    Floors sigma2 at VAR_FLOOR so downstream log/division are safe even
    for adversarial parameter vectors visited by the optimizer.
    """
    n = len(r)
    sigma2 = np.empty(n)
    sigma2[0] = max(r.var(), VAR_FLOOR)
    for t in range(1, n):
        sigma2[t] = max(omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1], VAR_FLOOR)
    return sigma2


def neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    """Gaussian quasi-likelihood (QMLE): consistent for the variance
    dynamics even when true innovations are non-Gaussian."""
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
        return 1e10  # infeasible: reject without evaluating the recursion
    sigma2 = garch_filter(r, omega, alpha, beta)
    ll = -0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + r**2 / sigma2)
    return float(-ll.sum())


def fit_garch(r: np.ndarray) -> dict:
    """MLE via Nelder-Mead from a variance-targeted start.

    Nelder-Mead is chosen over L-BFGS-B because the finite-difference
    gradients of the recursive likelihood are noisy near the alpha+beta
    boundary; the simplex method is slower but markedly more robust here.
    """
    var_r = r.var()
    x0 = np.array([var_r * 0.05, 0.10, 0.85])  # variance targeting start
    res = minimize(neg_loglik, x0, args=(r,), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
    omega, alpha, beta = res.x
    return {"omega": float(omega), "alpha": float(alpha), "beta": float(beta),
            "converged": bool(res.success), "nll": float(res.fun)}


def forecast_oos(r: np.ndarray, n_train: int, params: dict) -> np.ndarray:
    """One-step-ahead OOS variance forecasts with FROZEN train parameters.

    The recursion runs over the full sample (it needs r[t-1]) but the
    parameters never see test data — this is walk-forward with a single
    estimation window, the honest minimal design.
    """
    sigma2 = garch_filter(r, params["omega"], params["alpha"], params["beta"])
    return sigma2[n_train:]


def ewma_forecast(r: np.ndarray, n_train: int, lam: float) -> np.ndarray:
    """RiskMetrics EWMA benchmark, same information set as the GARCH."""
    n = len(r)
    s2 = np.empty(n)
    s2[0] = max(r[:50].var(), VAR_FLOOR)
    for t in range(1, n):
        s2[t] = max(lam * s2[t - 1] + (1.0 - lam) * r[t - 1] ** 2, VAR_FLOOR)
    return s2[n_train:]


def qlike(r2: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """QLIKE loss: robust to noisy volatility proxies (Patton 2011).

    Preferred over MSE on r^2 because MSE is dominated by a handful of
    r^4-scale outliers, while QLIKE remains consistent for ranking
    forecasts when the proxy (squared return) is unbiased but noisy.
    """
    return np.log(s2) + r2 / s2


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int) -> tuple:
    """DM test with Newey-West (Bartlett) HAC variance.

    H0: equal predictive accuracy, E[d_t] = 0 where d = loss_a - loss_b.
    Negative DM stat => model A (GARCH) beats model B (EWMA).
    """
    d = loss_a - loss_b
    n = len(d)
    dbar = d.mean()
    u = d - dbar
    gamma0 = float(u @ u) / n
    var_hac = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)          # Bartlett kernel
        gk = float(u[k:] @ u[:-k]) / n
        var_hac += 2.0 * w * gk
    var_hac = max(var_hac, VAR_FLOOR)
    dm = dbar / np.sqrt(var_hac / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def main() -> None:
    r = simulate_garch(N_TOTAL, N_BURN, RNG_SEED)
    r_train = r[:N_TRAIN]

    fit = fit_garch(r_train)
    persistence = fit["alpha"] + fit["beta"]

    s2_garch = forecast_oos(r, N_TRAIN, fit)
    s2_ewma = ewma_forecast(r, N_TRAIN, EWMA_LAMBDA)
    r2_test = r[N_TRAIN:] ** 2

    ql_garch = qlike(r2_test, s2_garch)
    ql_ewma = qlike(r2_test, s2_ewma)
    dm_stat, dm_p = diebold_mariano(ql_garch, ql_ewma, DM_LAG)

    print("RESULTS")
    print(f"converged={fit['converged']}")
    print(f"omega_hat={fit['omega']:.4f} (true={TRUE_OMEGA})")
    print(f"alpha_hat={fit['alpha']:.4f} (true={TRUE_ALPHA})")
    print(f"beta_hat={fit['beta']:.4f} (true={TRUE_BETA})")
    print(f"persistence={persistence:.4f}")
    print(f"qlike_garch_mean={ql_garch.mean():.5f}")
    print(f"qlike_ewma_mean={ql_ewma.mean():.5f}")
    print(f"dm_stat={dm_stat:.3f}")
    print(f"dm_pvalue={dm_p:.5f}")
    garch_better = ql_garch.mean() < ql_ewma.mean() and dm_p < 0.05
    print(f"verdict={'GARCH_BEATS_EWMA' if garch_better else 'NO_SIGNIFICANT_EDGE'}")


if __name__ == "__main__":
    main()
'''

ENTRY_GARCH = {
    "metadata": {
        "id": "qlm1-000003-garch-qlike-dm",
        "domain": "Volatility Modeling / Risk Forecasting",
        "complexity": 8,
        "tags": [
            "garch",
            "quasi-mle",
            "numerical-stability",
            "volatility-forecasting",
            "qlike",
            "diebold-mariano",
            "newey-west",
            "walk-forward",
            "ewma-benchmark",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The question 'does GARCH forecast volatility well?' is ill-posed until anchored "
            "to (a) a benchmark — here RiskMetrics EWMA(0.94), the model any practitioner gets "
            "for free, (b) a loss function robust to the fact that true variance is latent — "
            "QLIKE against squared returns, and (c) a significance test that respects serial "
            "correlation in forecast-loss differentials — Diebold-Mariano with HAC variance. "
            "Beating no benchmark, or winning on MSE-of-r^2 (outlier-dominated), proves "
            "nothing. The simulation is done under a true GARCH DGP, so this entry also "
            "teaches the *upper bound* framing: if GARCH cannot beat EWMA significantly even "
            "when the world IS GARCH, claimed edges on real data deserve heavy skepticism."
        ),
        "tool_selection": (
            "Python REPL with numpy for the recursions and scipy.optimize for MLE. "
            "Deliberately NOT using the arch package: hand-rolling the likelihood teaches the "
            "numerical failure modes (variance floors, boundary behavior at alpha+beta -> 1, "
            "percent-unit conditioning) that libraries hide, and those failure modes are "
            "exactly what a small model must learn to anticipate. scipy.stats supplies the "
            "normal CDF for the DM p-value."
        ),
        "recursive_delegation": (
            "Fit is fast for one series; no delegation needed. For a cross-sectional study "
            "(500 equities), delegate per-asset fits to sub-agents, but the parent must own "
            "two things sub-agents cannot see: the cross-sectional distribution of persistence "
            "estimates (a cluster at alpha+beta=0.999 signals IGARCH misspecification or "
            "unmodeled structural breaks, not a discovery), and pooled DM inference with "
            "cross-asset correlation of losses accounted for."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "H0: E[d_t] = 0 where d_t = QLIKE_t(GARCH) - QLIKE_t(EWMA) on the out-of-sample "
            "window (equal predictive accuracy, Diebold-Mariano 1995). H1 (one-sided of "
            "interest): E[d_t] < 0, GARCH's conditional variance forecasts are more accurate. "
            "QLIKE_t(m) = log(sigma2_m,t) + r_t^2 / sigma2_m,t. Test statistic: DM = dbar / "
            "sqrt(HAC_var(d)/n) with Bartlett-kernel Newey-West variance (lag 10), asymptotically "
            "N(0,1) under H0. Secondary estimation hypotheses: the QMLE (omega, alpha, beta) "
            "should recover the true (0.05, 0.08, 0.90) within sampling error on n=2000."
        ),
        "data_engineering": (
            "Simulate 3500 days of GARCH(1,1) returns and discard 500 as burn-in so the "
            "variance process forgets its initialization; keep 3000, split 2000 train / 1000 "
            "test. Returns are kept in PERCENT units — a numerical-conditioning decision: "
            "daily variances near 2.5 rather than 2.5e-4 keep likelihood curvature "
            "well-scaled for the optimizer and avoid float underflow in r^2/sigma2 terms. On "
            "real data, precede this with: use log returns of adjusted closes, remove "
            "zero-volume stale-price days (they fake vol clustering), and winsorize genuine "
            "data errors but NEVER true crash returns — deleting 1987-style observations is "
            "how risk models die."
        ),
        "methodology_justification": (
            "GARCH(1,1) via Gaussian QMLE: consistent for variance dynamics even under "
            "non-Gaussian innovations (Bollerslev-Wooldridge), which is why the Gaussian "
            "likelihood is defensible without a normality claim. Nelder-Mead over L-BFGS-B "
            "because finite-difference gradients of a recursive likelihood are noisy near the "
            "alpha+beta stationarity boundary, where the optimizer must operate. QLIKE over "
            "MSE because squared returns are an unbiased but extremely noisy variance proxy; "
            "Patton (2011) shows QLIKE is robust to proxy noise for forecast ranking while "
            "MSE rankings can invert. DM with HAC variance because one-step QLIKE loss "
            "differentials are serially correlated through volatility clustering itself. The "
            "OOS design freezes parameters at the train boundary: refitting daily would be "
            "stronger but the single-split design is the honest minimum and keeps the lesson "
            "focused."
        ),
        "code_implementation": CODE_GARCH,
        "statistical_validation": (
            "1) Convergence flag from the optimizer must be True; a silent maxiter exit "
            "invalidates everything downstream. 2) Parameter recovery: hat(alpha)+hat(beta) "
            "must be < 0.999 and near 0.98; hat(omega) within ~50% of 0.05 (omega is the "
            "hardest to pin down — its sampling error is large at n=2000 and correlates "
            "strongly with persistence). 3) DM p-value with HAC lag 10; also rerun with lags "
            "5 and 20 to confirm the verdict is not a bandwidth artifact. 4) Size check: "
            "simulate under an EWMA-equivalent IGARCH DGP and confirm the DM test rejects at "
            "~5% — a test that always rejects is broken. 5) Sanity: mean QLIKE values must be "
            "finite and the GARCH/EWMA gap should be small (EWMA is a near-nested competitor); "
            "an enormous gap signals a bug, not brilliance."
        ),
        "risk_and_backtest_audit": (
            "This is a forecasting study, not a trading backtest, so the audit targets are "
            "different: (a) the variance floor (1e-8) must bind ~never on the equilibrium "
            "path — count floor activations and fail the run if > 0.1% of observations; (b) "
            "initialization sensitivity — re-run with sigma2[0] set to 0.5x and 2x sample "
            "variance and confirm OOS QLIKE is unchanged to 4 decimals (burn-in adequacy); "
            "(c) if these forecasts feed a VaR or vol-targeting layer, persistence near 1 "
            "means variance shocks half-life is ~34 days [ln(0.5)/ln(0.98)], so position "
            "sizes will trend — audit the downstream leverage path, not just forecast "
            "accuracy; (d) regime sensitivity: evaluate QLIKE separately on the calmest and "
            "stormiest OOS quintiles — EWMA typically loses most in the transition periods, "
            "and a model that only wins in calm markets is worthless."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The DGP is GARCH, so the deck is stacked for GARCH — this entry measures the "
            "machinery under the best case, and even then the edge over EWMA on 1000 OOS days "
            "can fail significance; treat this as calibration of expectations, not evidence "
            "about markets. (2) Variance floors can silently convert a diverging fit into a "
            "plausible-looking one; always count activations. (3) Using r^2 as the proxy "
            "inside MSE instead of QLIKE flips rankings often enough to change conclusions. "
            "(4) Optimizing in raw (non-percent) units makes L-BFGS-B report success at "
            "garbage parameters — a plausible-but-wrong result of the exact kind this corpus "
            "exists to teach against. (5) The single train/test split leaves the OOS verdict "
            "hostage to one draw; the seed-sweep in validation is mandatory, not optional."
        ),
        "falsification_strategy": (
            "Attack 1 — swap the DGP to constant variance: a correct pipeline should estimate "
            "alpha ~ 0 and the DM test should find no significant difference (both models "
            "collapse to the unconditional variance). Attack 2 — swap the DGP to IGARCH "
            "(alpha+beta=1, EWMA's home turf): GARCH's edge should vanish or invert; if GARCH "
            "still 'wins significantly', the loss computation or HAC variance is wrong. "
            "Attack 3 — shuffle test-set returns (destroying clustering): both models' QLIKE "
            "should degrade toward the unconditional forecast's and the DM stat toward 0. "
            "Attack 4 — rerun the whole study across 20 seeds and report the distribution of "
            "DM p-values rather than one cherry-pickable number."
        ),
        "limitations": (
            "GARCH(1,1) with Gaussian QMLE captures clustering but not leverage asymmetry "
            "(needs GJR/EGARCH), jumps, or long memory; on equity index data the asymmetric "
            "extensions usually dominate. The single fixed estimation window understates what "
            "rolling refits achieve and cannot adapt to parameter drift. Squared daily returns "
            "are a noisy proxy — with intraday data, realized variance shrinks proxy noise by "
            "an order of magnitude and can change DM verdicts. Conclusions are about 1-step "
            "forecasts; multi-horizon forecasts compound parameter uncertainty nonlinearly "
            "and require separate evaluation."
        ),
    },
    "agent_instructions": (
        "1. Simulate the GARCH DGP with burn-in; verify the realized unconditional variance "
        "is within 20% of omega/(1-alpha-beta) before trusting the generator. 2. Fit QMLE on "
        "the train window only (Nelder-Mead, variance-targeted start); assert converged=True "
        "and alpha+beta < 0.999. 3. Compare (omega, alpha, beta) estimates to ground truth; "
        "if recovery fails, debug the likelihood before proceeding — do not tune the "
        "optimizer to force a pass. 4. Produce one-step OOS variance forecasts with frozen "
        "parameters; produce EWMA(0.94) forecasts from the identical information set. 5. "
        "Compute per-day QLIKE for both; run DM with Newey-West lag 10, then lags 5 and 20. "
        "6. Count variance-floor activations; fail the run if any occur on the equilibrium "
        "path. 7. Falsification battery: constant-variance DGP, IGARCH DGP, shuffled test "
        "returns — each must produce its predicted null behavior. 8. Report: parameter table "
        "with true values, mean QLIKE per model, DM stat and p per lag choice, and an "
        "explicit statement that the DGP favors GARCH so this is an upper bound on real-data "
        "expectations."
    ),
    "verification": {
        "timeout_seconds": 240,
        "must_print": ["RESULTS", "converged=True", "alpha_hat=", "dm_stat=", "dm_pvalue=", "verdict=NO_SIGNIFICANT_EDGE"],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "scipy"],
    },
}

# =====================================================================
# Schema v1 -> v2 migration
# =====================================================================
#
# Migration rules (applied verbatim, research content untouched):
#   1. schema_version: 2
#   2. entry_kind: "correct" for all three v1 seed entries
#   3. expected_verdict: the honest verdict the entry's code actually prints
#   4. flaws: []  (correct entries declare no flaws)
#   5. minimal static_checks (loops declared only where mathematically necessary)
#   6. minimal rlm object (no environment; recursion depth 1)
#   7. minimal training_sequence (tool_use style, standard loss masking)

MINIMAL_RLM = {
    "environment_class": None,
    "actions": [],
    "max_recursion_depth": 1,
    "tool_timeout_seconds": 120,
}

MINIMAL_TRAINING_SEQUENCE = {
    "style": "tool_use",
    "include_adversarial_critique": True,
    "include_final_calibrated_answer": True,
    "loss_masking": {
        "user": 0.0,
        "environment_observation": 0.0,
        "assistant_thought": 1.0,
        "tool_call": 1.0,
        "final_answer": 1.0,
    },
}


def migrate_v1_to_v2(
    entry: dict,
    entry_kind: str,
    expected_verdict: str,
    static_checks: dict,
    flaws: list | None = None,
    rlm: dict | None = None,
    training_sequence: dict | None = None,
) -> dict:
    """Wrap a v1 entry into the v2 top-level structure.

    The six v1 sections pass through unchanged; only the new v2 header
    fields are prepended (dict order matches the documented v2 layout).
    """
    return {
        "schema_version": 2,
        "entry_kind": entry_kind,
        "expected_verdict": expected_verdict,
        "flaws": list(flaws or []),
        "static_checks": static_checks,
        "rlm": rlm or MINIMAL_RLM,
        "training_sequence": training_sequence or MINIMAL_TRAINING_SEQUENCE,
        **entry,
    }


# =====================================================================
# Entry 4 — complexity 4: ADVERSARIAL lookahead bias (v2-native)
# =====================================================================

CODE_LOOKAHEAD = '''"""ADVERSARIAL TEACHING ENTRY: same-bar execution lookahead bias.

Pedagogical arc: plausible strategy -> attractive naive result ->
adversarial audit -> flaw detected -> corrected conclusion.

The naive backtest trades the SMA(5/20) crossover on the SAME bar whose
close generated the signal. On a pure GBM with mild drift this "earns" a
Sharpe near 1.0 that does not exist. The audit exposes the flaw by (a)
counting bars where the naive position differs from the executable
(lagged) position, and (b) recomputing the backtest with the one-bar
execution lag. Deterministic (seeded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 11
N_DAYS = 2000
ANN_FACTOR = 252
FAST, SLOW = 5, 20
COST_BPS = 2.0
# Audit threshold: an execution-timing choice must never be worth this
# much Sharpe. If removing the lag adds > 0.5, the edge lives in the
# timestamp, not the signal.
INFLATION_THRESHOLD = 0.5


def simulate_prices(n: int, seed: int) -> pd.Series:
    """Plain GBM: mild drift (4%/yr), 18% vol. Deliberately trend-free
    beyond drift, so any large Sharpe MUST be an artifact."""
    rng = np.random.default_rng(seed)
    mu, sigma = 0.04 / ANN_FACTOR, 0.18 / np.sqrt(ANN_FACTOR)
    log_rets = mu - 0.5 * sigma**2 + sigma * rng.standard_normal(n)
    idx = pd.bdate_range("2017-01-01", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=idx, name="close")


def sma_signal(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    sig = (fast_ma > slow_ma).astype(float)
    sig[slow_ma.isna()] = 0.0  # flat during warm-up; never backfill
    return sig


def backtest(prices: pd.Series, signal: pd.Series, lag: int, cost_bps: float) -> pd.Series:
    """lag=0 is the PLANTED FLAW: position_t uses signal_t, which was
    computed from close_t — the trade sees the bar it is trading.
    lag=1 is the executable version."""
    position = signal.shift(lag).fillna(0.0) if lag > 0 else signal
    rets = prices.pct_change().fillna(0.0)
    gross = position * rets
    turnover = position.diff().abs().fillna(position.abs())
    return gross - turnover * cost_bps / 1e4


def annualized_sharpe(rets: pd.Series) -> float:
    sd = rets.std(ddof=1)
    return 0.0 if sd == 0 else float(rets.mean() / sd * np.sqrt(ANN_FACTOR))


def main() -> None:
    prices = simulate_prices(N_DAYS, RNG_SEED)
    signal = sma_signal(prices, FAST, SLOW)

    # --- The seductive naive result (flawed) -------------------------
    naive = backtest(prices, signal, lag=0, cost_bps=COST_BPS)
    naive_sharpe = annualized_sharpe(naive)

    # --- Adversarial audit -------------------------------------------
    # Detection 1: how often does the naive position differ from the
    # position that could actually have been held at the open of bar t?
    executable_pos = signal.shift(1).fillna(0.0)
    mismatch_days = int((signal != executable_pos).sum())

    # Detection 2: recompute with the honest one-bar lag.
    corrected = backtest(prices, signal, lag=1, cost_bps=COST_BPS)
    corrected_sharpe = annualized_sharpe(corrected)
    inflation = naive_sharpe - corrected_sharpe

    flawed = inflation > INFLATION_THRESHOLD
    audit_verdict = "REJECTED_LOOKAHEAD_BIAS" if flawed else "NO_MATERIAL_TIMING_EDGE"

    print("RESULTS")
    print(f"naive_sharpe={naive_sharpe:.3f}")
    print(f"corrected_sharpe={corrected_sharpe:.3f}")
    print(f"sharpe_inflation={inflation:.3f}")
    print(f"position_mismatch_days={mismatch_days}")
    print(f"inflation_threshold={INFLATION_THRESHOLD}")
    print("flaw_type=LOOKAHEAD_BIAS")
    print(f"audit_verdict={audit_verdict}")
    # Corrected conclusion: on drift-only GBM the honest rule has no
    # exploitable trend edge; the naive result was pure timestamp leakage.
    print(f"verdict={audit_verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_LOOKAHEAD = {
    "schema_version": 2,
    "entry_kind": "adversarial",
    "expected_verdict": "REJECTED_LOOKAHEAD_BIAS",
    "flaws": [
        {
            "type": "lookahead_bias",
            "severity": "fatal",
            "location": "code_implementation:backtest (lag=0 branch, naive position construction)",
            "description": (
                "The strategy trades on the same bar whose close generated the signal: "
                "position_t = signal_t, where signal_t is computed from close_t. The fast "
                "SMA(5) is dominated by the current bar's close, so the naive position "
                "systematically flips long on large up days it could not have known about, "
                "inflating Sharpe from ~0.27 to ~0.99 on drift-only GBM."
            ),
            "detection": (
                "Compare the naive position against the lagged executable position "
                "(signal.shift(1)) and count mismatch days (115 here); recompute Sharpe "
                "before and after applying .shift(1). An execution-timing choice worth more "
                "than 0.5 Sharpe is diagnostic of leakage, not alpha."
            ),
            "corrective_action": (
                "Use position = signal.shift(1).fillna(0.0) before multiplying by returns, "
                "so bar t's signal is only tradeable at bar t+1; re-report all metrics from "
                "the lagged backtest only."
            ),
        }
    ],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [],
    },
    "rlm": MINIMAL_RLM,
    "training_sequence": {
        "style": "multi_turn_audit",
        "include_adversarial_critique": True,
        "include_final_calibrated_answer": True,
        "loss_masking": {
            "user": 0.0,
            "environment_observation": 0.0,
            "assistant_thought": 1.0,
            "tool_call": 1.0,
            "final_answer": 1.0,
        },
    },
    "metadata": {
        "id": "qlm1-000004-adversarial-lookahead-sma",
        "domain": "Trend Following / Time-Series Momentum",
        "complexity": 4,
        "tags": [
            "adversarial",
            "lookahead-bias",
            "sma-crossover",
            "execution-lag",
            "backtest-audit",
            "sharpe-inflation",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "A colleague reports a Sharpe ~1.0 from a fast SMA(5/20) crossover on a liquid "
            "index. Before admiring the number, ask the auditor's first question: WHEN is the "
            "position established relative to the information that generated it? A fast SMA is "
            "dominated by the newest close, so same-bar execution quietly conditions the "
            "position on the very return being booked. The audit plan is fixed before looking "
            "at any PnL: reproduce the naive result, then difference the naive position "
            "against the only position that was physically executable, then re-price."
        ),
        "tool_selection": (
            "Python REPL with numpy/pandas only. The decisive tools are pandas .shift(1) for "
            "the executable position, a boolean position-mismatch count, and two identical "
            "backtests differing in nothing but the lag — an ablation, not an argument."
        ),
        "recursive_delegation": (
            "Not needed for one strategy. In a corpus-scale audit, spawn one sub-agent per "
            "submitted backtest to run the standardized lag-ablation, and have the parent "
            "flag any strategy whose Sharpe drops by more than 0.5 under a one-bar lag — "
            "those are timestamp leaks, not signals."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Naive claim under test: the SMA(5/20) crossover earns positive risk-adjusted "
            "returns (naive annualized Sharpe ~0.99, seed 11). Audit hypotheses — H0: the "
            "naive result is executable, i.e. Sharpe(naive) - Sharpe(lagged) <= 0.5 and the "
            "naive position equals the executable position almost everywhere. H1: the result "
            "depends on same-bar information (inflation > 0.5 with materially many mismatch "
            "days). Decision rule: if H1, reject the strategy claim entirely and report only "
            "the lagged result as the honest estimate."
        ),
        "data_engineering": (
            "2000 business days of seeded GBM with 4%/yr drift and 18% vol — deliberately "
            "trend-free beyond drift so ground truth is known: no fast-crossover edge exists "
            "to find. Signals built with pandas rolling means; warm-up NaN mapped to flat, "
            "never backfilled. Costs at 2 bps per unit turnover applied identically to both "
            "backtests so the lag is the ONLY difference between them — a controlled ablation "
            "requires changing exactly one thing."
        ),
        "methodology_justification": (
            "The audit is an ablation study, chosen over statistical testing as the primary "
            "instrument because the question is mechanical, not distributional: two runs "
            "identical except position_t = signal_t versus signal_{t-1}. The 0.5-Sharpe "
            "threshold encodes an economic prior — no plausible daily execution-timing skill "
            "is worth half a Sharpe on a 2-bps-cost instrument — so exceeding it identifies "
            "leakage without needing a p-value. The mismatch-day count (115/2000 bars) "
            "localizes WHERE the two backtests diverge: exactly the crossover bars, which is "
            "the fingerprint of same-bar signal use."
        ),
        "code_implementation": CODE_LOOKAHEAD,
        "statistical_validation": (
            "Deterministic checks: naive_sharpe ~0.99 and corrected_sharpe ~0.27 at seed 11, "
            "inflation ~0.72 > 0.5 threshold; position_mismatch_days = 115 and every mismatch "
            "bar must be a crossover bar. Robustness: across seeds 10-19 the naive Sharpe "
            "stays in ~[0.44, 1.41] while the corrected Sharpe straddles zero (some seeds "
            "negative) — i.e. the naive estimator is biased upward everywhere while the "
            "honest estimator is correctly centered near the no-edge truth. That systematic "
            "one-sided gap, not any single number, is the statistical signature of lookahead."
        ),
        "risk_and_backtest_audit": (
            "The relevant 'risk' here is epistemic: capital allocated to a leaked backtest "
            "realizes the corrected distribution (Sharpe ~0.27 gross of further real-world "
            "frictions, i.e. approximately nothing) while sized for the naive one — a direct "
            "path to oversized drawdowns. Frictions note: 2 bps is charitable; the naive "
            "rule's higher effective turnover makes real costs bite harder. Audit protocol "
            "for production: every backtest must ship with a lag-ablation table (lag 0/1/2) "
            "and any strategy whose PnL is concentrated in the lag-0 column is rejected "
            "without further review."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The trap is seductive precisely because the code LOOKS clean — one missing "
            ".shift(1) in an otherwise correct vectorized backtest; reviewers pattern-match "
            "on structure and miss timing. (2) A subtler variant plants the leak in feature "
            "construction (e.g. normalizing by the full-sample mean) where no shift call is "
            "visibly absent. (3) The 0.5 threshold is calibrated for daily bars and fast "
            "signals; slow signals leak less per bar, so a leaky SMA(50/200) could pass — "
            "threshold must scale with signal speed. (4) On real intraday data, using the "
            "close as both signal input and fill price is the same bug wearing different "
            "clothes."
        ),
        "falsification_strategy": (
            "Attempt to rescue the naive result: (a) rerun on 10 fresh seeds — if the naive "
            "edge were real it should persist under the lag, and it does not; (b) set drift "
            "to zero — the corrected Sharpe collapses to ~0 while the naive one stays large, "
            "proving the naive PnL is manufactured from contemporaneous returns, not drift "
            "capture; (c) invert the test: apply lag=2 — Sharpe should not materially drop "
            "further, confirming one bar of leakage was the entire effect."
        ),
        "limitations": (
            "The ablation detects timing leakage only; it cannot detect survivorship bias, "
            "parameter mining, or leaked features that survive lagging. The threshold is a "
            "calibrated heuristic, not a hypothesis test — borderline inflations (0.3-0.5) "
            "require the seed-sweep evidence instead. Synthetic GBM understates real "
            "autocorrelation structure, which can make same-bar leakage on real data even "
            "more flattering than shown here."
        ),
    },
    "agent_instructions": (
        "1. Reproduce the claimed backtest exactly as submitted; record naive_sharpe. "
        "2. Construct the executable position signal.shift(1).fillna(0.0); count bars where "
        "it differs from the naive position and verify the mismatches sit on crossover bars. "
        "3. Rerun the identical backtest changing ONLY the lag; record corrected_sharpe. "
        "4. Compute inflation = naive - corrected; compare to the 0.5 threshold. "
        "5. If exceeded: print flaw_type=LOOKAHEAD_BIAS and verdict=REJECTED_LOOKAHEAD_BIAS; "
        "report the corrected number as the only honest estimate. 6. Confirm with a "
        "10-seed sweep that the naive estimator is one-sidedly biased while the corrected "
        "one straddles zero. 7. In the final report, state the corrected conclusion first "
        "and the naive number only as the exhibit of the flaw — never the reverse."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": [
            "RESULTS",
            "naive_sharpe=",
            "corrected_sharpe=",
            "position_mismatch_days=",
            "flaw_type=LOOKAHEAD_BIAS",
            "audit_verdict=REJECTED_LOOKAHEAD_BIAS",
            "verdict=REJECTED_LOOKAHEAD_BIAS",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "pandas"],
    },
}

# =====================================================================
# Entry 5 — complexity 7: ADVERSARIAL p-hacking sweep (v2-native)
# =====================================================================

CODE_PHACK = '''"""ADVERSARIAL TEACHING ENTRY: p-hacking via uncorrected parameter sweep.

Pedagogical arc: plausible strategy family -> attractive best-in-sweep
result ("Sharpe 0.86, significant at 1%!") -> adversarial audit via a
max-statistic block-permutation test (simplified White reality check)
-> flaw detected -> corrected conclusion.

Ground truth is known by construction: prices are a DRIFTLESS seeded
random walk, so no SMA configuration has any true edge. The sweep still
"finds" one — that is the lesson. Deterministic (seeded).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

RNG_SEED = 24
N_DAYS = 2016              # 8 trading years
ANN_FACTOR = 252
COST_BPS = 2.0
FAST_GRID = [5, 10, 15, 20, 25, 30]
SLOW_GRID = [40, 60, 80, 100, 120, 140, 160, 180]
N_PERM = 200               # permutation draws for the max-statistic null
BLOCK = 21                 # block length preserves short-range dependence
ALPHA = 0.05


def simulate_log_returns(n: int, seed: int) -> np.ndarray:
    """Driftless log returns, 18% annualized vol: true edge = 0 for every
    configuration, by construction."""
    rng = np.random.default_rng(seed)
    sigma = 0.18 / np.sqrt(ANN_FACTOR)
    return sigma * rng.standard_normal(n) - 0.5 * sigma**2


def rolling_means(prices: np.ndarray, windows: list) -> dict:
    """O(n) rolling means via cumulative sums (vectorized; no pandas row
    loops). Warm-up cells hold +inf sentinels internally so warm-up
    comparisons are False; sentinels never reach stdout."""
    out = {}
    c = np.concatenate([[0.0], np.cumsum(prices)])
    for w in windows:
        m = np.full(len(prices), np.inf)
        m[w - 1:] = (c[w:] - c[:-w]) / w
        out[w] = m
    return out


def strategy_sharpes(prices: np.ndarray, cost_bps: float) -> np.ndarray:
    """Annualized net Sharpe for every (fast, slow) configuration.

    Every backtest here is individually HONEST: one-bar execution lag,
    turnover costs. The planted flaw is not in the backtests — it is in
    what is done with 48 of them afterwards.
    """
    n = len(prices)
    rets = np.empty(n)
    rets[0] = 0.0
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    smas = rolling_means(prices, sorted(set(FAST_GRID + SLOW_GRID)))
    sharpes = np.empty(len(FAST_GRID) * len(SLOW_GRID))
    k = 0
    for f in FAST_GRID:                # loop over configurations, not rows
        for s in SLOW_GRID:
            sig = (smas[f] > smas[s]).astype(float)
            sig[: s - 1] = 0.0         # flat during warm-up
            pos = np.empty(n)
            pos[0] = 0.0
            pos[1:] = sig[:-1]         # honest one-bar execution lag
            net = pos * rets - np.abs(np.diff(pos, prepend=0.0)) * cost_bps / 1e4
            sd = net.std(ddof=1)
            sharpes[k] = 0.0 if sd == 0 else net.mean() / sd * np.sqrt(ANN_FACTOR)
            k += 1
    return sharpes


def block_permute(log_rets: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    """Circular-style moving-block resample: destroys any (spurious)
    signal-return alignment while preserving local dependence."""
    n = len(log_rets)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return log_rets[idx]


def max_stat_permutation(log_rets: np.ndarray, observed_best: float,
                         n_perm: int, seed: int) -> tuple:
    """Simplified White reality check: the null distribution of the MAX
    Sharpe across the whole grid, not of a single strategy's Sharpe.

    THE key audit idea: 'best of 48' must be compared against 'best of
    48 under the null', never against 'one strategy under the null'.
    """
    rng = np.random.default_rng(seed)
    max_null = np.empty(n_perm)
    for b in range(n_perm):            # each permutation draw is independent
        perm = block_permute(log_rets, rng, BLOCK)
        p_prices = 100.0 * np.exp(np.cumsum(perm))
        max_null[b] = strategy_sharpes(p_prices, COST_BPS).max()
    p_corrected = float((1.0 + (max_null >= observed_best).sum()) / (n_perm + 1.0))
    return p_corrected, float(max_null.mean())


def main() -> None:
    log_rets = simulate_log_returns(N_DAYS, RNG_SEED)
    prices = 100.0 * np.exp(np.cumsum(log_rets))

    # --- The seductive naive procedure (flawed) ----------------------
    sharpes = strategy_sharpes(prices, COST_BPS)
    n_tested = len(sharpes)
    best_idx = int(sharpes.argmax())
    best_sharpe = float(sharpes.max())
    best_fast = FAST_GRID[best_idx // len(SLOW_GRID)]
    best_slow = SLOW_GRID[best_idx % len(SLOW_GRID)]
    # Naive inference treats the winner as if it were the only test run:
    t_stat = best_sharpe * np.sqrt(N_DAYS / ANN_FACTOR)
    naive_p = float(1.0 - stats.norm.cdf(t_stat))

    # --- Adversarial audit -------------------------------------------
    # Correction 1 (cheap): Bonferroni on the naive p-value.
    bonferroni_p = float(min(1.0, n_tested * naive_p))
    # Correction 2 (proper): max-statistic permutation null.
    corrected_p, null_max_mean = max_stat_permutation(
        log_rets, best_sharpe, N_PERM, RNG_SEED + 1
    )

    hacked = naive_p < ALPHA and corrected_p >= ALPHA
    audit_verdict = "REJECTED_P_HACKING" if hacked else (
        "SELECTION_SURVIVES_CORRECTION" if corrected_p < ALPHA else "NO_NAIVE_SIGNIFICANCE"
    )

    print("RESULTS")
    print(f"n_parameters_tested={n_tested}")
    print(f"best_config=sma_{best_fast}_{best_slow}")
    print(f"best_naive_sharpe={best_sharpe:.3f}")
    print(f"naive_pvalue={naive_p:.5f}")
    print(f"bonferroni_pvalue={bonferroni_p:.5f}")
    print(f"corrected_pvalue={corrected_p:.5f}")
    print(f"null_max_sharpe_mean={null_max_mean:.3f}")
    print("flaw_type=P_HACKING")
    print(f"audit_verdict={audit_verdict}")
    # Corrected conclusion: on a driftless random walk the expected max
    # Sharpe over 48 trials is ~0.57 — the "discovery" is the order
    # statistic of noise, exactly as the permutation null predicts.
    print(f"verdict={audit_verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_PHACK = {
    "schema_version": 2,
    "entry_kind": "adversarial",
    "expected_verdict": "REJECTED_P_HACKING",
    "flaws": [
        {
            "type": "p_hacking",
            "severity": "fatal",
            "location": "code_implementation:main (best-of-sweep selection and naive_p inference)",
            "description": (
                "The best configuration (SMA 15/120, Sharpe 0.86, 'p=0.008') was selected "
                "from 48 trials on a driftless random walk and its p-value computed as if it "
                "were the only test ever run; the expected MAX Sharpe over 48 null trials is "
                "~0.57, so the winner is an order statistic of noise."
            ),
            "detection": (
                "Apply a multiple-testing correction to the best observed statistic: compare "
                "it against the permutation null distribution of the grid-wide MAX Sharpe "
                "(simplified White reality check). Here the corrected p-value is ~0.19 vs a "
                "naive 0.008 — significance evaporates under the correct null."
            ),
            "corrective_action": (
                "Report max-statistic-corrected p-values (or at minimum Bonferroni, here "
                "~0.36) for any swept statistic, or confirm the selected configuration on a "
                "sealed holdout that played no role in selection before reporting it."
            ),
        },
        {
            "type": "multiple_testing_abuse",
            "severity": "high",
            "location": "research_corpus:data_engineering (grid design)",
            "description": (
                "The 6x8 grid contains heavily overlapping configurations (correlated "
                "strategies), which makes the naive 'winner looks stable across neighbors' "
                "defense misleading: neighboring cells share most of their trades, so an "
                "apparent plateau of good Sharpes is one lucky draw seen 48 times, not 48 "
                "confirmations."
            ),
            "detection": (
                "Compute the correlation matrix of the 48 strategy return streams; mean "
                "pairwise correlations far above zero mean the effective number of "
                "independent trials is far below 48, and 'neighborhood stability' arguments "
                "must be discarded in favor of the max-statistic null, which handles the "
                "correlation automatically."
            ),
            "corrective_action": (
                "Always use a correction that preserves the dependence structure "
                "(permutation/bootstrap of the full grid) rather than treating grid cells "
                "as independent evidence; never argue significance from parameter-plateau "
                "smoothness alone."
            ),
        },
    ],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [
            {
                "function": "strategy_sharpes",
                "reason": "loop over 48 strategy configurations (not over time/rows); each config's backtest is fully vectorized",
                "max_iterations": 48,
            },
            {
                "function": "max_stat_permutation",
                "reason": "each permutation draw must be generated and evaluated independently to build the null",
                "max_iterations": 200,
            },
        ],
    },
    "rlm": MINIMAL_RLM,
    "training_sequence": {
        "style": "multi_turn_audit",
        "include_adversarial_critique": True,
        "include_final_calibrated_answer": True,
        "loss_masking": {
            "user": 0.0,
            "environment_observation": 0.0,
            "assistant_thought": 1.0,
            "tool_call": 1.0,
            "final_answer": 1.0,
        },
    },
    "metadata": {
        "id": "qlm1-000005-adversarial-p-hacking-sweep",
        "domain": "Parameter Selection / Multiple Testing",
        "complexity": 7,
        "tags": [
            "adversarial",
            "p-hacking",
            "multiple-testing",
            "reality-check",
            "max-statistic",
            "block-permutation",
            "parameter-sweep",
            "order-statistics",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "A researcher swept 48 SMA configurations, found one with Sharpe 0.86 and "
            "p=0.008, and wants to trade it. The audit question is not 'is 0.86 good?' but "
            "'good compared to WHAT null?'. Selecting the maximum of 48 correlated trials "
            "changes the null distribution from 'one strategy's Sharpe' to 'the maximum of "
            "48 strategies' Sharpes' — and on pure noise that maximum averages ~0.57. The "
            "audit must rebuild the correct null with the selection step INSIDE it."
        ),
        "tool_selection": (
            "Python REPL with numpy for the vectorized grid backtests and the block-"
            "permutation engine, scipy.stats for the naive normal p-value (kept only as the "
            "exhibit of the flaw). No statsmodels needed — the permutation test builds its "
            "own null."
        ),
        "recursive_delegation": (
            "The 200 permutation draws are embarrassingly parallel: delegate shards of "
            "draws to sub-agents, each returning its vector of null max-Sharpes. The parent "
            "MUST own the selection step and the final quantile computation — delegating "
            "'find the best config' to one agent and 'test it' to another without sharing "
            "the trial count is exactly how p-hacking happens organizationally."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Naive claim: the best swept configuration (SMA 15/120) has true Sharpe > 0 "
            "(naive one-sided p=0.008 from SR*sqrt(T_years) ~ N(0,1)). Audit hypotheses — "
            "H0: all 48 configurations have zero true edge and the observed best is the "
            "expected order statistic of noise; H1: the best configuration's edge exceeds "
            "what maximal selection from 48 correlated null trials produces. Test: "
            "p_corrected = (1 + #{max-null draws >= observed best}) / (B+1) under a "
            "moving-block permutation null (B=200, block 21), plus Bonferroni as a cheap "
            "upper-bound cross-check. Ground truth is H0 by construction (driftless walk)."
        ),
        "data_engineering": (
            "2016 days (8 years) of seeded DRIFTLESS log returns at 18% vol — the "
            "generator guarantees every strategy's true edge is exactly zero, so any "
            "'discovery' is definitionally spurious and the audit's job is to say so. "
            "Grid: 6 fast x 8 slow = 48 configurations. Every individual backtest is "
            "honest (one-bar lag, 2 bps turnover costs, warm-up flat): the flaw is planted "
            "exclusively in the inference-after-selection step, which is what makes this "
            "entry seductive — reading any single backtest reveals nothing wrong."
        ),
        "methodology_justification": (
            "The max-statistic permutation test (simplified White 2000 reality check) is "
            "the appropriate correction because it (a) uses the null distribution of the "
            "SELECTED statistic — the grid maximum — rather than a single strategy's, and "
            "(b) preserves the correlation structure across the 48 strategies "
            "automatically, since each permutation re-runs the entire grid on the same "
            "resampled path. Bonferroni is reported alongside as the assumption-free upper "
            "bound: it over-corrects under correlation, so agreement between both (neither "
            "significant) is decisive. Block permutation (block 21) rather than IID "
            "shuffling preserves short-range dependence so the null is not artificially "
            "easy to beat."
        ),
        "code_implementation": CODE_PHACK,
        "statistical_validation": (
            "Deterministic values at seed 24: best_naive_sharpe ~0.86 (config sma_15_120), "
            "naive_pvalue ~0.008 (looks significant at 1%), bonferroni_pvalue ~0.36, "
            "corrected_pvalue ~0.19, null_max_sharpe_mean ~0.57. The audit verdict requires "
            "the conjunction naive_p < 0.05 AND corrected_p >= 0.05 — i.e. the entry "
            "certifies specifically that naive inference and corrected inference disagree, "
            "which is the operational definition of p-hacking. Cross-seed behavior "
            "(21-28): naive p is 'significant' in 5 of 8 seeds while corrected p is never "
            "below 0.05 — the naive procedure's false-discovery rate is the lesson."
        ),
        "risk_and_backtest_audit": (
            "Deploying the swept winner realizes zero expected edge minus costs: the "
            "certain outcome is negative carry with Sharpe ~0, sized as if it were 0.86 — "
            "and because the selection favored configurations that fit one path's noise, "
            "realized performance typically disappoints even the corrected estimate "
            "(regression to the mean past zero after costs). Institutional audit protocol: "
            "every reported statistic must carry its trial count n_parameters_tested; any "
            "research note quoting a swept result without it is returned unread; holdout "
            "windows are sealed BEFORE the sweep is designed, and one holdout look is "
            "budgeted per project, not per configuration."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The seduction is structural: every individual backtest in the sweep is "
            "clean, so no code review of any single cell finds the flaw — it exists only "
            "at the level of the procedure. (2) The naive defense 'neighboring parameters "
            "also look good' is invalid under correlation: overlapping SMA configs share "
            "trades, so a plateau is one lucky draw echoed 48 times. (3) 200 permutations "
            "bound the corrected p-value at ~0.005 resolution; claiming corrected "
            "significance near that floor requires more draws. (4) Bonferroni over-corrects "
            "under correlation and can be waved away by a motivated researcher — which is "
            "why the permutation test, not Bonferroni, must carry the verdict. (5) The "
            "same flaw recurs disguised as 'walk-forward optimization' when the walk-"
            "forward scheme itself was tuned across many variants."
        ),
        "falsification_strategy": (
            "Attempt to rescue the discovery: (a) plant a REAL edge (add 4%/yr drift so "
            "trend rules genuinely help) and verify the same pipeline then yields "
            "corrected_p < 0.05 — confirming the audit has power and is not a machine that "
            "rejects everything; (b) shrink the grid to the single pre-registered config "
            "sma_15_120 and rerun on 100 fresh seeds — its Sharpe distribution centers on "
            "zero, directly falsifying the claim the sweep 'found' something; (c) double "
            "N_PERM and confirm corrected_p is stable (~0.19), ruling out permutation-count "
            "artifacts."
        ),
        "limitations": (
            "The permutation null assumes block-stationarity; genuine long-memory "
            "structure beyond 21-day blocks would be partially destroyed, slightly "
            "cheapening the null. The simplified reality check tests only the single best "
            "configuration, not the full stepdown family (Romano-Wolf) needed to certify "
            "multiple survivors. The naive p-value's normal approximation for Sharpe "
            "ignores skew/kurtosis corrections (Lo 2002) — immaterial here since the audit "
            "rejects regardless, but material when verdicts are borderline. Conclusions "
            "transfer to real data only after adding the corrections this entry "
            "deliberately omits from the naive path."
        ),
    },
    "agent_instructions": (
        "1. Reproduce the sweep exactly: 48 configs, honest per-config backtests, record "
        "every Sharpe — never only the winner. 2. Record n_parameters_tested BEFORE any "
        "selection; this number is part of the result, not metadata. 3. Compute the naive "
        "p-value for the best config and label it explicitly as pre-correction. 4. Run the "
        "max-statistic block-permutation null (>=200 draws): re-run the FULL grid per draw "
        "and take the max. 5. Compute corrected_p with the +1 finite-sample adjustment; "
        "cross-check with Bonferroni. 6. Verdict rule: naive_p < 0.05 AND corrected_p >= "
        "0.05 => REJECTED_P_HACKING; report the corrected number first. 7. Power check: "
        "rerun the pipeline on a drifted DGP and confirm it CAN pass a real edge — an "
        "audit that rejects everything is as useless as one that rejects nothing. 8. In "
        "the final report, show the null-max distribution mean (~0.57) next to the "
        "'discovery' (0.86) so the order-statistic nature of the result is visible at a "
        "glance."
    ),
    "verification": {
        "timeout_seconds": 240,
        "must_print": [
            "RESULTS",
            "n_parameters_tested=48",
            "best_naive_sharpe=",
            "naive_pvalue=",
            "corrected_pvalue=",
            "flaw_type=P_HACKING",
            "audit_verdict=REJECTED_P_HACKING",
            "verdict=REJECTED_P_HACKING",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "scipy"],
    },
}

# =====================================================================
# Entry 6 — complexity 8: RLM environment interaction (v2-native)
# =====================================================================

CODE_RLM = '''"""RLM TEACHING ENTRY: quantitative research as ENVIRONMENT INTERACTION.

Instead of a monolithic script, the research workflow is exposed as a
stateful environment with a gym-like reset/step API. The agent must:

  inspect -> act -> read observation -> decide next action -> verify

Two actions in the scripted plan are deliberately out of order; the
environment refuses them with error observations instead of raising, and
the agent recovers by reordering — teaching observation-driven control
flow rather than script-following. All observations are JSON-serializable
dicts. Deterministic (seeded); replay is verified.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats
from scipy.optimize import minimize

RNG_SEED = 123
N_TOTAL, N_BURN, N_TRAIN = 3000, 500, 2000
TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 0.05, 0.08, 0.90   # percent^2 units
EWMA_LAMBDA = 0.94
VAR_FLOOR = 1e-8
DM_LAG = 10


def _simulate_garch(n: int, burn: int, seed: int) -> np.ndarray:
    """Seeded GARCH(1,1) returns in percent units; burn-in discarded."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    sigma2 = np.empty(n + burn)
    r = np.empty(n + burn)
    sigma2[0] = TRUE_OMEGA / (1.0 - TRUE_ALPHA - TRUE_BETA)
    r[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n + burn):       # GARCH recursion: inherently sequential
        sigma2[t] = TRUE_OMEGA + TRUE_ALPHA * r[t - 1] ** 2 + TRUE_BETA * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * z[t]
    return r[burn:]


def _garch_filter(r: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    n = len(r)
    s2 = np.empty(n)
    s2[0] = max(r.var(), VAR_FLOOR)
    for t in range(1, n):              # depends on s2[t-1]: sequential by nature
        s2[t] = max(omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1], VAR_FLOOR)
    return s2


def _neg_loglik(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.999:
        return 1e10                    # infeasible region: reject cheaply
    s2 = _garch_filter(r, omega, alpha, beta)
    return float(0.5 * (np.log(2.0 * np.pi) + np.log(s2) + r**2 / s2).sum())


class QuantEnvironment:
    """Stateful volatility-research environment.

    API: obs = env.step({"name": <action>}). Observations are plain
    dicts (JSON-serializable). Precondition violations return
    {"ok": False, "error": ...} — the environment NEVER raises for a
    bad action ordering, because the agent is supposed to read
    observations and re-plan, not crash.
    """

    ACTIONS = ("reset", "fit_train", "forecast_oos", "evaluate_qlike", "run_dm_test")

    def __init__(self, seed: int = RNG_SEED):
        self._seed = seed
        self._state = "uninitialized"
        self._r = None
        self._fit = None
        self._s2_garch = None
        self._s2_ewma = None
        self._ql = None

    def reset(self) -> dict:
        self._r = _simulate_garch(N_TOTAL, N_BURN, self._seed)
        self._fit = None
        self._s2_garch = None
        self._s2_ewma = None
        self._ql = None
        self._state = "reset"
        return {"ok": True, "state": self._state, "n_obs": int(len(self._r)),
                "n_train": N_TRAIN, "sample_var": round(float(self._r.var()), 4)}

    def step(self, action: dict) -> dict:
        name = action.get("name")
        if name not in self.ACTIONS:
            return {"ok": False, "error": f"unknown action: {name}", "state": self._state}
        if name == "reset":
            return self.reset()
        if self._state == "uninitialized":
            return {"ok": False, "error": "call reset first", "state": self._state}
        return getattr(self, f"_do_{name}")(action)

    def _do_fit_train(self, action: dict) -> dict:
        r_train = self._r[:N_TRAIN]
        x0 = np.array([r_train.var() * 0.05, 0.10, 0.85])   # variance-targeted start
        res = minimize(_neg_loglik, x0, args=(r_train,), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
        omega, alpha, beta = (float(v) for v in res.x)
        self._fit = {"omega": omega, "alpha": alpha, "beta": beta}
        self._state = "fitted"
        return {"ok": True, "state": self._state,
                "params": {"omega": round(omega, 4), "alpha": round(alpha, 4),
                           "beta": round(beta, 4)},
                "persistence": round(alpha + beta, 4), "converged": bool(res.success)}

    def _do_forecast_oos(self, action: dict) -> dict:
        if self._fit is None:
            return {"ok": False, "error": "fit_train before forecast_oos", "state": self._state}
        s2 = _garch_filter(self._r, self._fit["omega"], self._fit["alpha"], self._fit["beta"])
        self._s2_garch = s2[N_TRAIN:]                       # frozen-parameter OOS
        e2 = np.empty(len(self._r))
        e2[0] = max(self._r[:50].var(), VAR_FLOOR)
        for t in range(1, len(self._r)):                    # EWMA recursion: sequential
            e2[t] = max(EWMA_LAMBDA * e2[t - 1] + (1.0 - EWMA_LAMBDA) * self._r[t - 1] ** 2,
                        VAR_FLOOR)
        self._s2_ewma = e2[N_TRAIN:]
        self._state = "forecasted"
        return {"ok": True, "state": self._state, "n_forecasts": int(len(self._s2_garch)),
                "garch_mean_var": round(float(self._s2_garch.mean()), 4),
                "ewma_mean_var": round(float(self._s2_ewma.mean()), 4)}

    def _do_evaluate_qlike(self, action: dict) -> dict:
        if self._s2_garch is None:
            return {"ok": False, "error": "forecast_oos before evaluate_qlike",
                    "state": self._state}
        r2 = self._r[N_TRAIN:] ** 2
        ql_g = np.log(self._s2_garch) + r2 / self._s2_garch
        ql_e = np.log(self._s2_ewma) + r2 / self._s2_ewma
        self._ql = (ql_g, ql_e)
        self._state = "evaluated"
        return {"ok": True, "state": self._state,
                "qlike_garch": round(float(ql_g.mean()), 5),
                "qlike_ewma": round(float(ql_e.mean()), 5)}

    def _do_run_dm_test(self, action: dict) -> dict:
        if self._ql is None:
            return {"ok": False, "error": "evaluate_qlike before run_dm_test",
                    "state": self._state}
        d = self._ql[0] - self._ql[1]
        n = len(d)
        u = d - d.mean()
        var_hac = float(u @ u) / n
        for k in range(1, DM_LAG + 1):                      # HAC lag sum: bounded loop
            var_hac += 2.0 * (1.0 - k / (DM_LAG + 1.0)) * float(u[k:] @ u[:-k]) / n
        var_hac = max(var_hac, VAR_FLOOR)
        dm = float(d.mean() / np.sqrt(var_hac / n))
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(dm))))
        self._state = "tested"
        return {"ok": True, "state": self._state, "dm_stat": round(dm, 3),
                "dm_pvalue": round(p, 5),
                "conclusion": "GARCH_BEATS_EWMA" if (dm < 0 and p < 0.05)
                else "NO_SIGNIFICANT_EDGE"}


def main() -> None:
    env = QuantEnvironment(seed=RNG_SEED)
    # Scripted agent plan with TWO deliberate ordering mistakes: the agent
    # must read the error observations and recover, not crash.
    plan = [
        {"name": "fit_train"},        # invalid: environment not reset yet
        {"name": "reset"},
        {"name": "fit_train"},
        {"name": "evaluate_qlike"},   # invalid: nothing forecasted yet
        {"name": "forecast_oos"},
        {"name": "evaluate_qlike"},
        {"name": "run_dm_test"},
    ]
    executed = []
    guard_failures = 0
    obs = None
    for action in plan:
        obs = env.step(action)
        print(f"trace={json.dumps({'action': action['name'], 'observation': obs}, sort_keys=True)}")
        if obs["ok"]:
            executed.append(action["name"])
        else:
            guard_failures += 1

    # Environment verification battery.
    checks = {
        "guards_rejected_both_unordered_actions": guard_failures == 2,
        "recovered_full_action_sequence": executed == list(QuantEnvironment.ACTIONS),
        "final_observation_json_roundtrip": json.loads(json.dumps(obs)) == obs,
        "dm_stat_finite": bool(obs["ok"] and np.isfinite(obs["dm_stat"])),
        "honest_conclusion_preserved": obs.get("conclusion") == "NO_SIGNIFICANT_EDGE",
    }
    # Determinism: an identical fresh environment must replay identically.
    env2 = QuantEnvironment(seed=RNG_SEED)
    obs2 = None
    for action in plan:
        obs2 = env2.step(action)
    checks["deterministic_replay"] = obs == obs2

    print("RESULTS")
    print("environment_class=QuantEnvironment")
    print(f"actions_executed={','.join(executed)}")
    print(f"guard_failures_handled={guard_failures}")
    for name, passed in checks.items():
        print(f"check_{name}={passed}")
    print(f"final_observation={json.dumps(obs, sort_keys=True)}")
    verdict = "ENVIRONMENT_VERIFIED" if all(checks.values()) else "ENVIRONMENT_BROKEN"
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_RLM = {
    "schema_version": 2,
    "entry_kind": "rlm_environment",
    "expected_verdict": "ENVIRONMENT_VERIFIED",
    "flaws": [],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [
            {
                "function": "_simulate_garch",
                "reason": "GARCH variance recursion is inherently sequential",
                "max_iterations": 3500,
            },
            {
                "function": "_garch_filter",
                "reason": "conditional variance filter depends on s2[t-1]",
                "max_iterations": 3000,
            },
            {
                "function": "QuantEnvironment._do_forecast_oos",
                "reason": "EWMA benchmark recursion depends on e2[t-1]",
                "max_iterations": 3000,
            },
            {
                "function": "QuantEnvironment._do_run_dm_test",
                "reason": "HAC autocovariance sum over lags 1..DM_LAG",
                "max_iterations": 10,
            },
            {
                "function": "main",
                "reason": "agent action loop: each step depends on the previous observation",
                "max_iterations": 7,
            },
        ],
    },
    "rlm": {
        "environment_class": "QuantEnvironment",
        "actions": ["reset", "fit_train", "forecast_oos", "evaluate_qlike", "run_dm_test"],
        "max_recursion_depth": 3,
        "tool_timeout_seconds": 120,
    },
    "training_sequence": {
        "style": "environment_interaction",
        "include_adversarial_critique": True,
        "include_final_calibrated_answer": True,
        "loss_masking": {
            "user": 0.0,
            "environment_observation": 0.0,
            "assistant_thought": 1.0,
            "tool_call": 1.0,
            "final_answer": 1.0,
        },
    },
    "metadata": {
        "id": "qlm1-000006-rlm-environment-garch",
        "domain": "Volatility Modeling / Environment Interaction",
        "complexity": 8,
        "tags": [
            "rlm",
            "environment",
            "tool-use",
            "state-machine",
            "garch",
            "qlike",
            "diebold-mariano",
            "error-recovery",
            "deterministic-replay",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The task is not 'forecast volatility' but 'conduct the forecasting study "
            "THROUGH an environment': every research step is an action whose result "
            "arrives as an observation, and the next action must be chosen from the "
            "observation, not from a memorized script. The plan contains ordering "
            "mistakes on purpose — fit before reset, evaluate before forecast — because "
            "the behavior worth learning is reading {'ok': False, 'error': ...} and "
            "re-planning, not assuming success. State lives in the environment, not the "
            "agent: the agent's memory is the observation trace."
        ),
        "tool_selection": (
            "Python REPL hosting the environment class; numpy/scipy inside the "
            "environment for GARCH QMLE, forecasting, QLIKE, and the DM test; json for "
            "the observation protocol — every observation must round-trip through "
            "json.dumps/loads, because a real RLM agent consumes observations as "
            "serialized tool output, never as live Python objects."
        ),
        "recursive_delegation": (
            "The environment is the delegation boundary: a parent agent can hand a "
            "sub-agent the environment handle plus a goal ('produce a DM verdict') and "
            "judge it purely on the observation trace. Recursion depth 3 covers "
            "parent -> research sub-agent -> verification sub-agent replaying the trace "
            "against a fresh seeded environment to confirm determinism — the replay "
            "check in the code is exactly that verification step, performed inline."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Environment-level hypotheses (the research content — GARCH vs EWMA under "
            "QLIKE with a DM test — inherits its H0/H1 from entry qlm1-000003): H0-env: "
            "the environment is NOT a faithful research instrument — it permits invalid "
            "action orderings, returns non-serializable or non-finite observations, or "
            "replays non-deterministically. H1-env: all guards hold, all observations "
            "are JSON-round-trippable and finite, the full action sequence is reachable "
            "after error recovery, and an identical seeded environment replays the exact "
            "trace. Verdict ENVIRONMENT_VERIFIED requires every check to pass; a single "
            "failure yields ENVIRONMENT_BROKEN."
        ),
        "data_engineering": (
            "Data is generated INSIDE the environment on reset(): 3000 days of seeded "
            "GARCH(1,1) returns in percent units (burn-in 500 discarded), train/test "
            "split 2000/1000 fixed by the environment, not the agent — so no agent "
            "action can move the split boundary and leak test data into fitting. "
            "Observations expose only aggregates (sample variance, parameter estimates, "
            "mean forecasts, losses), never raw future returns, which is the "
            "information-hygiene property that makes the environment safe to hand to an "
            "untrusted optimizing agent."
        ),
        "methodology_justification": (
            "A state machine with explicit preconditions (reset -> fitted -> forecasted "
            "-> evaluated -> tested) is chosen over free function calls because ordering "
            "IS the methodology in walk-forward research: forecasting before fitting or "
            "evaluating before forecasting are not programming errors but research "
            "errors, and the environment encodes them as such. Guards return error "
            "observations instead of raising, because exception-crashing teaches the "
            "agent nothing — observation-driven recovery is the trainable behavior. The "
            "embedded quant methodology (Gaussian QMLE via Nelder-Mead, frozen-parameter "
            "OOS forecasts, QLIKE, DM with Bartlett HAC) is inherited unchanged from the "
            "verified correct entry qlm1-000003, so this entry adds interaction "
            "structure, not new statistical claims."
        ),
        "code_implementation": CODE_RLM,
        "statistical_validation": (
            "Six machine-checked environment invariants, all printed as check_* lines: "
            "(1) both deliberately unordered actions rejected by guards; (2) the full "
            "five-action sequence completes after recovery; (3) the final observation "
            "survives a json round-trip; (4) the DM statistic is finite; (5) the "
            "environment's conclusion equals NO_SIGNIFICANT_EDGE — the honest verdict "
            "inherited from the underlying study (dm_stat ~ -1.127, p ~ 0.26), so the "
            "interaction wrapper preserves calibrated uncertainty rather than "
            "manufacturing a win; (6) deterministic replay: a fresh environment with the "
            "same seed reproduces the final observation exactly. The verdict is the "
            "conjunction of all six."
        ),
        "risk_and_backtest_audit": (
            "Environment-specific risks replace market frictions here: (a) state leakage "
            "— a buggy environment that lets evaluate_qlike run before forecast_oos "
            "would silently score in-sample fits as OOS; the guard tests exist to catch "
            "exactly this class; (b) nondeterminism — an environment that draws fresh "
            "randomness per step makes agent trajectories unreproducible and RLM "
            "training data worthless, hence the replay check; (c) observation overflow — "
            "returning raw arrays would blow up serialized traces and leak future data; "
            "observations are capped to scalar aggregates; (d) the timeout contract "
            "(120s) bounds the cost of any single action so a stuck optimizer cannot "
            "hang a training-data generation run."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) A scripted plan that happens to be in the right order would pass "
            "without demonstrating recovery — the deliberate mistakes are load-bearing; "
            "removing them silently weakens the entry to a monolithic script with extra "
            "steps. (2) Guards that raise exceptions instead of returning error "
            "observations would still 'work' in this driver but teach crash-on-error to "
            "the student model. (3) Rounding observations to 4-5 decimals is required "
            "for stable replay comparison, but over-aggressive rounding could mask real "
            "nondeterminism — the replay check compares the rounded dicts, so keep "
            "rounding no coarser than shown. (4) The environment's honest "
            "NO_SIGNIFICANT_EDGE conclusion must never be 'fixed' to a win to make the "
            "trace look more satisfying — that would smuggle motivated reasoning into "
            "the RLM corpus."
        ),
        "falsification_strategy": (
            "Break the environment on purpose and confirm the battery catches it: (a) "
            "remove the fit_train guard — check 1 fails and the verdict flips to "
            "ENVIRONMENT_BROKEN; (b) inject a fresh random seed per reset — the replay "
            "check fails; (c) return the raw returns array in an observation — the json "
            "round-trip stays technically true but trace size explodes: add a size "
            "budget assertion when scaling; (d) flip the DM conclusion logic — check 5 "
            "fails, proving the wrapper cannot silently change the underlying study's "
            "verdict."
        ),
        "limitations": (
            "The scripted plan exercises one error-recovery path; a full RLM curriculum "
            "needs many plans (shuffled orders, repeated actions, unknown action names) "
            "over the same environment — this entry is the template, not the coverage. "
            "The environment exposes a single fixed DGP and split; parameterized resets "
            "(seed, split point as action arguments) are the natural extension but were "
            "kept out to keep the verification battery exact. Observations are scalar "
            "aggregates, so an agent cannot request diagnostics (residual plots, "
            "subsample losses) — richer read-only inspection actions are future work."
        ),
    },
    "agent_instructions": (
        "1. Instantiate QuantEnvironment with the fixed seed; do not generate data "
        "outside it. 2. Submit actions ONLY via env.step({'name': ...}); after every "
        "step, parse the observation and branch on obs['ok'] — on False, read "
        "obs['error'], do not retry blindly, and re-order the plan to satisfy the "
        "stated precondition. 3. Drive the sequence to completion: reset, fit_train, "
        "forecast_oos, evaluate_qlike, run_dm_test. 4. Log every (action, observation) "
        "pair as a json trace line — the trace IS the research record. 5. Run the "
        "verification battery: guard rejections counted, full sequence reached, json "
        "round-trip, finite statistics, honest conclusion preserved, deterministic "
        "replay on a fresh same-seed environment. 6. Emit verdict=ENVIRONMENT_VERIFIED "
        "only if every check passes; any failure is ENVIRONMENT_BROKEN with the failing "
        "check named. 7. Report the environment's statistical conclusion exactly as "
        "observed (NO_SIGNIFICANT_EDGE) — the wrapper must never editorialize the "
        "underlying result."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": [
            "RESULTS",
            "environment_class=QuantEnvironment",
            "actions_executed=reset,fit_train,forecast_oos,evaluate_qlike,run_dm_test",
            "guard_failures_handled=2",
            "check_deterministic_replay=True",
            "final_observation=",
            "verdict=ENVIRONMENT_VERIFIED",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "scipy"],
    },
}

# =====================================================================
# Entry 7 — complexity 7: BSM implied-vol calibration (v2-native, correct)
# =====================================================================

CODE_BSM = '''"""Black-Scholes implied-volatility calibration via safeguarded
Newton-Raphson on a synthetic options chain (5 strikes x 2 expiries).

Ground truth is known by construction: prices are generated from a
deterministic smile surface, so the calibrated IVs must recover it to
near machine precision. Key numerical lessons: (1) calibrate on OTM
instruments (puts below the forward, calls above) because deep-ITM
options carry almost no vol information (price ~ intrinsic, vega ~ 0);
(2) plain Newton-Raphson diverges from bad starts, so the solver brackets
the root and falls back to bisection whenever the Newton step leaves the
bracket — price is monotone in sigma, so the bracket always tightens.
Deterministic; no randomness needed at all.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

S0, R, Q = 100.0, 0.02, 0.0
STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
EXPIRIES = np.array([0.25, 1.0])
PRICE_TOL = 1e-10
IV_TOL = 1e-6
MAX_ITER = 100


def true_vol_surface(K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Deterministic smile: base vol + moneyness curvature + term slope."""
    m = K / S0 - 1.0
    return 0.20 + 0.30 * m**2 - 0.02 * (T - 0.5)


def bs_d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def bs_put(S, K, T, r, q, sigma):
    """Via put-call parity: same IV as the call by construction."""
    return bs_call(S, K, T, r, q, sigma) - S * np.exp(-q * T) + K * np.exp(-r * T)


def bs_vega(S, K, T, r, q, sigma):
    return S * np.exp(-q * T) * stats.norm.pdf(bs_d1(S, K, T, r, q, sigma)) * np.sqrt(T)


def implied_vol(target: np.ndarray, K: np.ndarray, T: np.ndarray,
                use_put: np.ndarray) -> tuple:
    """Vectorized safeguarded Newton-Raphson (Newton + bisection bracket).

    All 10 options are solved simultaneously; converged entries freeze.
    """
    def price(sig):
        return np.where(use_put, bs_put(S0, K, T, R, Q, sig),
                        bs_call(S0, K, T, R, Q, sig))

    lo = np.full(len(K), 1e-4)
    hi = np.full(len(K), 5.0)
    # Brenner-Subrahmanyam-style start, clamped into the bracket.
    sigma = np.clip(np.sqrt(2.0 * np.pi / T) * target / S0, 0.05, 2.0)
    converged = np.zeros(len(K), dtype=bool)
    iters = np.zeros(len(K))
    for _ in range(MAX_ITER):
        diff = price(sigma) - target
        converged |= np.abs(diff) < PRICE_TOL
        if converged.all():
            break
        # tighten the bracket (price is strictly increasing in sigma)
        hi = np.where(diff > 0, sigma, hi)
        lo = np.where(diff < 0, sigma, lo)
        vega = bs_vega(S0, K, T, R, Q, sigma)
        newton = sigma - diff / np.maximum(vega, 1e-12)
        ok = (newton > lo) & (newton < hi) & (vega > 1e-8)
        sigma = np.where(converged, sigma,
                         np.where(ok, newton, 0.5 * (lo + hi)))
        iters += ~converged
    return sigma, converged, iters


def main() -> None:
    K_grid, T_grid = np.meshgrid(STRIKES, EXPIRIES, indexing="ij")
    K, T = K_grid.ravel(), T_grid.ravel()
    true_iv = true_vol_surface(K, T)

    # Synthetic chain: exact BS prices from the true surface.
    call_prices = bs_call(S0, K, T, R, Q, true_iv)
    fwd = S0 * np.exp((R - Q) * T)
    use_put = K < fwd  # OTM instrument selection: puts below the forward
    target = np.where(use_put, bs_put(S0, K, T, R, Q, true_iv), call_prices)

    # No-arbitrage sanity on the generated chain before calibrating.
    intrinsic_ok = bool((call_prices >= np.maximum(
        S0 * np.exp(-Q * T) - K * np.exp(-R * T), 0.0) - 1e-12).all())

    iv_hat, converged, iters = implied_vol(target, K, T, use_put)
    max_err = float(np.abs(iv_hat - true_iv).max())
    reprice_err = float(np.abs(
        bs_call(S0, K, T, R, Q, iv_hat) - call_prices).max())

    ok = converged.all() and max_err < IV_TOL and intrinsic_ok
    print("RESULTS")
    print(f"n_options={len(K)}")
    print(f"n_otm_puts={int(use_put.sum())}")
    print(f"all_converged={bool(converged.all())}")
    print(f"max_abs_iv_error={max_err:.3e}")
    print(f"max_reprice_error={reprice_err:.3e}")
    print(f"mean_solver_iterations={iters.mean():.1f}")
    print(f"max_solver_iterations={int(iters.max())}")
    print(f"intrinsic_bounds_ok={intrinsic_ok}")
    print(f"verdict={'CALIBRATION_VERIFIED' if ok else 'CALIBRATION_FAILED'}")


if __name__ == "__main__":
    main()
'''

ENTRY_BSM = {
    "schema_version": 2,
    "entry_kind": "correct",
    "expected_verdict": "CALIBRATION_VERIFIED",
    "flaws": [],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [
            {
                "function": "implied_vol",
                "reason": "iterative root-finding: each Newton/bisection step depends on the previous iterate (all 10 options vectorized within each step)",
                "max_iterations": 100,
            }
        ],
    },
    "rlm": MINIMAL_RLM,
    "training_sequence": MINIMAL_TRAINING_SEQUENCE,
    "metadata": {
        "id": "qlm1-000007-options-bsm-calibration",
        "domain": "Options Pricing / Volatility Calibration",
        "complexity": 7,
        "tags": [
            "black-scholes",
            "implied-volatility",
            "newton-raphson",
            "bisection-safeguard",
            "otm-instruments",
            "vega",
            "put-call-parity",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The task is inverse-problem calibration with known ground truth: generate a "
            "synthetic chain from a deterministic smile, then recover the surface from prices "
            "alone. The two failure modes to anticipate before writing any code are (1) deep-"
            "ITM options whose price is almost all intrinsic value — vega collapses and any "
            "gradient method stalls, so calibration must use the OTM instrument at each "
            "strike (put below forward, call above, identical IV by parity); (2) raw Newton-"
            "Raphson divergence from poor starting points, solved by bracketing with a "
            "bisection fallback since option price is strictly monotone in volatility."
        ),
        "tool_selection": (
            "Python REPL with numpy for the vectorized solver and scipy.stats for the normal "
            "CDF/PDF. No optimizer library: hand-rolling the safeguarded Newton iteration IS "
            "the lesson — root-finding hygiene the student must internalize."
        ),
        "recursive_delegation": (
            "Not needed for 10 options. At production scale (thousands of quotes across an "
            "expiry grid), shard strikes to sub-agents but keep surface-level no-arbitrage "
            "checks (butterfly/calendar) in the parent, since they are cross-option "
            "constraints no single-quote solver can see."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "H0: the calibration pipeline is faithful — for every option i, the recovered "
            "sigma_i satisfies |sigma_i - sigma_true,i| < 1e-6 and the solver converges "
            "(|price(sigma_i) - target_i| < 1e-10) within 100 iterations. H1: at least one "
            "option fails convergence or tolerance, indicating a solver defect (typically "
            "vega collapse on ITM instruments). Verdict CALIBRATION_VERIFIED requires all "
            "options converged, max IV error < 1e-6, and intrinsic-value bounds satisfied on "
            "the generated chain."
        ),
        "data_engineering": (
            "Synthetic chain: 5 strikes (80-120) x 2 expiries (0.25y, 1y), S0=100, r=2%, "
            "q=0. True surface sigma(K,T) = 0.20 + 0.30*(K/S0-1)^2 - 0.02*(T-0.5): a smile "
            "with curvature and a mild term slope, so recovery is pointwise nontrivial. "
            "Prices are exact BS values — no noise — because the entry tests solver "
            "correctness, not statistical estimation; on real quotes, bid-ask midpoints, "
            "discrete dividends, and stale prices must be handled before this solver runs."
        ),
        "methodology_justification": (
            "Newton-Raphson with the analytic vega is quadratically convergent near the "
            "root, but naked NR fails on far-OTM starts where vega is tiny; the safeguard "
            "maintains a [lo, hi] bracket updated by the sign of the pricing error (valid "
            "because price is strictly increasing in sigma) and substitutes a bisection step "
            "whenever the Newton step exits the bracket. OTM instrument selection (puts for "
            "K < forward) is the standard practitioner fix for the ITM vega-collapse "
            "problem and is exact by put-call parity. The solver is vectorized across all "
            "options simultaneously with per-option convergence freezing."
        ),
        "code_implementation": CODE_BSM,
        "statistical_validation": (
            "Deterministic checks: max_abs_iv_error ~2e-12 (far below the 1e-6 tolerance), "
            "all 10 options converged, max 9 iterations (quadratic convergence visible in "
            "the low iteration count), max_reprice_error at machine precision, and "
            "intrinsic bounds hold on the generated chain. Robustness probes for the "
            "student: shift the surface level to 0.05 and 0.80 and confirm convergence "
            "still holds; drop the bisection safeguard and observe the deep-ITM failures "
            "reappear — the safeguard, not Newton, is what makes the solver production-"
            "grade."
        ),
        "risk_and_backtest_audit": (
            "No backtest here; the audit targets numerical risk. (a) Vega floor 1e-12 "
            "prevents division blow-ups but must never mask non-convergence — the "
            "convergence flag, not the floor, is the arbiter. (b) The bracket [1e-4, 5] "
            "bounds the vol space; quotes implying vols outside it are data errors, not "
            "solver failures, and must be rejected upstream. (c) On real chains, arbitrage "
            "violations (negative butterflies) make IVs non-existent — the intrinsic-bound "
            "check must run BEFORE calibration so bad quotes fail loudly. (d) IV errors "
            "propagate nonlinearly into greeks: a 1e-4 IV error is harmless for delta but "
            "material for short-dated gamma near expiry."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) Calibrating ITM calls directly looks fine ATM and silently fails deep ITM "
            "— the max error hides at the chain's edges, so per-option (not average) error "
            "must be checked. (2) A fixed-count Newton loop without a convergence flag "
            "reports plausible garbage on the stalled options. (3) The Brenner start "
            "formula is ATM-specific; using it far OTM without clamping can start Newton "
            "in a near-zero-vega region. (4) Noise-free synthetic prices overstate "
            "real-world precision: with 1-tick quote noise, IV error scales as tick/vega "
            "and explodes for short-dated wings."
        ),
        "falsification_strategy": (
            "Break the solver on purpose and confirm detection: (a) force calls-only "
            "calibration — the deep-ITM options must fail the 1e-6 tolerance and flip the "
            "verdict to CALIBRATION_FAILED; (b) remove the bracket fallback and confirm "
            "non-convergence is reported, not silently absorbed; (c) perturb one input "
            "price by 1% and verify only that option's IV moves materially (locality "
            "check); (d) set an expiry to 1 day and confirm wing IVs remain recoverable "
            "or fail loudly."
        ),
        "limitations": (
            "Black-Scholes IV is a quoting convention, not a model endorsement: constant-"
            "vol dynamics are contradicted by the very smile being calibrated. The entry "
            "covers European options without dividends; American exercise premia and "
            "discrete dividends require binomial/PDE de-Americanization before IV "
            "extraction. Noise-free prices make this an upper bound on real-chain "
            "precision, and no cross-strike arbitrage repair (butterfly smoothing) is "
            "performed."
        ),
    },
    "agent_instructions": (
        "1. Generate the synthetic chain from the declared true surface; verify intrinsic "
        "bounds before calibrating. 2. Select the OTM instrument per strike via the "
        "forward; compute targets with put-call parity. 3. Run the safeguarded Newton "
        "solver vectorized across all options; freeze converged entries. 4. Assert all "
        "converged and max |IV error| < 1e-6 against ground truth; report per-option "
        "errors, not just the mean. 5. Reprice calls from calibrated IVs and confirm "
        "machine-precision round-trip. 6. Falsification: rerun calls-only and confirm "
        "the deep-ITM failure is detected and the verdict flips. 7. Report iteration "
        "counts (quadratic convergence evidence) and state the noise-free caveat "
        "explicitly."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": [
            "RESULTS",
            "n_options=10",
            "all_converged=True",
            "max_abs_iv_error=",
            "intrinsic_bounds_ok=True",
            "verdict=CALIBRATION_VERIFIED",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "scipy"],
    },
}

# =====================================================================
# Entry 8 — complexity 5: ADVERSARIAL survivorship bias (v2-native)
# =====================================================================

CODE_SURV = '''"""ADVERSARIAL TEACHING ENTRY: survivorship bias in equity momentum.

Pedagogical arc: plausible strategy -> attractive naive result ->
adversarial audit -> flaw detected -> corrected conclusion.

Universe design makes the trap realistic: 140 'sound' assets (2%/yr
drift) plus 60 'bubble' assets (30%/yr drift, but a daily jump-to-default
hazard: a terminal -80% crash followed by delisting). Momentum loads on
exactly the assets that blow up. The NAIVE backtest is run on the
survivor-only universe with full clean histories — the classic database
mistake — and looks excellent. The AUDIT re-runs the identical strategy
on the full universe with delisting crashes included, and the Sharpe
collapses. Deterministic (seeded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 18
N_SOUND, N_BUBBLE = 140, 60
N_DAYS = 1008              # 4 trading years
ANN_FACTOR = 252
REBALANCE = 21             # monthly
LOOKBACK = 126             # 6-month momentum
TOP_Q = 0.2                # long top quintile
BLOWUP_RET = -0.80         # terminal crash on delisting day
HAZARD = 0.0012            # daily jump-to-default hazard (bubble assets)
COST_BPS = 10.0
INFLATION_THRESHOLD = 0.5  # Sharpe gap that convicts the naive backtest


def simulate_universe(seed: int) -> tuple:
    """Full panel with ground-truth delistings.

    Returns (clean_rets, audited_rets, alive, survivors): clean_rets is
    the counterfactual no-delisting panel (what a survivor-only database
    implicitly shows); audited_rets applies the -80% crash on the
    delisting day and zeros (untradeable) afterwards.
    """
    rng = np.random.default_rng(seed)
    n = N_SOUND + N_BUBBLE
    is_bubble = np.zeros(n, dtype=bool)
    is_bubble[N_SOUND:] = True
    drift = np.where(is_bubble, 0.30, 0.02) / ANN_FACTOR
    vol = np.where(is_bubble, 0.35, 0.28) / np.sqrt(ANN_FACTOR)
    rets = drift[None, :] + vol[None, :] * rng.standard_normal((N_DAYS, n))
    hit = (rng.random((N_DAYS, n)) < HAZARD) & is_bubble[None, :]
    delist_day = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
    alive = np.ones((N_DAYS, n), dtype=bool)
    audited = rets.copy()
    for j in range(n):     # per-asset delisting bookkeeping (n=200, cheap)
        d = delist_day[j]
        if d >= 0:
            audited[d, j] = BLOWUP_RET
            audited[d + 1:, j] = 0.0
            alive[d + 1:, j] = False
    survivors = delist_day < 0
    return rets, audited, alive, survivors


def momentum_backtest(rets: np.ndarray, eligible: np.ndarray,
                      cost_bps: float) -> np.ndarray:
    """Monthly-rebalanced long-only top-quintile 6m momentum.

    The strategy logic is IDENTICAL for naive and audited runs — only the
    universe and return panel differ. That isolation is what makes the
    audit an ablation of the data choice, not the strategy."""
    T, n = rets.shape
    port = np.zeros(T)
    weights = np.zeros(n)
    cum = np.cumsum(rets, axis=0)
    for t in range(LOOKBACK, T):    # sequential: positions carry state
        if (t - LOOKBACK) % REBALANCE == 0:
            mom = cum[t - 1] - cum[t - LOOKBACK - 1] if t - LOOKBACK - 1 >= 0 \
                else cum[t - 1]
            elig = eligible[t]
            mom_e = np.where(elig, mom, -np.inf)
            k = max(int(elig.sum() * TOP_Q), 1)
            top = np.argsort(mom_e)[-k:]
            new_w = np.zeros(n)
            new_w[top] = 1.0 / k
            port[t] -= np.abs(new_w - weights).sum() * cost_bps / 1e4
            weights = new_w
        port[t] += (weights * rets[t]).sum()
    return port[LOOKBACK:]


def annualized_sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return 0.0 if sd == 0 else float(x.mean() / sd * np.sqrt(ANN_FACTOR))


def main() -> None:
    rets, audited_rets, alive, survivors = simulate_universe(RNG_SEED)
    n_total = N_SOUND + N_BUBBLE
    n_surv = int(survivors.sum())

    # --- The seductive naive backtest (flawed universe) --------------
    naive_port = momentum_backtest(
        rets[:, survivors],
        np.ones((N_DAYS, n_surv), dtype=bool),   # everyone 'always tradeable'
        COST_BPS,
    )
    naive_sharpe = annualized_sharpe(naive_port)
    naive_ann_ret = float(naive_port.mean() * ANN_FACTOR)

    # --- Adversarial audit: full universe, deaths included -----------
    audited_port = momentum_backtest(audited_rets, alive, COST_BPS)
    audited_sharpe = annualized_sharpe(audited_port)
    audited_ann_ret = float(audited_port.mean() * ANN_FACTOR)

    inflation = naive_sharpe - audited_sharpe
    flawed = inflation > INFLATION_THRESHOLD
    audit_verdict = "REJECTED_SURVIVORSHIP_BIAS" if flawed else "NO_MATERIAL_SURVIVOR_EFFECT"

    print("RESULTS")
    print(f"n_assets_total={n_total}")
    print(f"n_survivors={n_surv}")
    print(f"n_delisted={n_total - n_surv}")
    print(f"naive_sharpe={naive_sharpe:.3f}")
    print(f"naive_ann_return={naive_ann_ret:.4f}")
    print(f"audited_sharpe={audited_sharpe:.3f}")
    print(f"audited_ann_return={audited_ann_ret:.4f}")
    print(f"sharpe_inflation={inflation:.3f}")
    print(f"inflation_threshold={INFLATION_THRESHOLD}")
    print("flaw_type=SURVIVORSHIP_BIAS")
    print(f"audit_verdict={audit_verdict}")
    print(f"verdict={audit_verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_SURV = {
    "schema_version": 2,
    "entry_kind": "adversarial",
    "expected_verdict": "REJECTED_SURVIVORSHIP_BIAS",
    "flaws": [
        {
            "type": "survivorship_bias",
            "severity": "fatal",
            "location": "code_implementation:main (naive backtest universe construction)",
            "description": (
                "The naive backtest is run on the survivor-only universe with full clean "
                "histories: the 43 delisted assets (bubble stocks that hit their -80% "
                "jump-to-default) are silently absent, and the surviving bubble stocks' "
                "spectacular runs dominate the momentum ranking, inflating Sharpe from "
                "~0.39 to ~1.74."
            ),
            "detection": (
                "Re-run the identical strategy on the full universe with delisting returns "
                "applied on the delisting day and dead assets marked untradeable "
                "thereafter; a Sharpe drop above 0.5 convicts the universe construction. "
                "On real data: compare the backtest universe's asset count per year "
                "against a point-in-time constituent file — a universe whose membership "
                "never shrinks is survivor-conditioned by definition."
            ),
            "corrective_action": (
                "Use point-in-time universe membership with delisting returns included "
                "(the standard CRSP-style treatment); apply the delisting-day return to "
                "any held position and remove the asset from the eligible set only "
                "afterwards. Report the audited number as the only honest estimate."
            ),
        }
    ],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [
            {
                "function": "simulate_universe",
                "reason": "per-asset delisting bookkeeping loop over 200 assets (not over time)",
                "max_iterations": 200,
            },
            {
                "function": "momentum_backtest",
                "reason": "portfolio state (weights, costs) carries across days; rebalancing is path-dependent",
                "max_iterations": 1008,
            },
        ],
    },
    "rlm": MINIMAL_RLM,
    "training_sequence": {
        "style": "multi_turn_audit",
        "include_adversarial_critique": True,
        "include_final_calibrated_answer": True,
        "loss_masking": {
            "user": 0.0,
            "environment_observation": 0.0,
            "assistant_thought": 1.0,
            "tool_call": 1.0,
            "final_answer": 1.0,
        },
    },
    "metadata": {
        "id": "qlm1-000008-adv-survivorship-bias",
        "domain": "Equity Momentum / Backtest Auditing",
        "complexity": 5,
        "tags": [
            "adversarial",
            "survivorship-bias",
            "momentum",
            "delisting-returns",
            "point-in-time-universe",
            "jump-to-default",
            "backtest-audit",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "A momentum backtest reports Sharpe ~1.7 on a 4-year equity panel. The "
            "auditor's first question for any cross-sectional equity result: WHERE did the "
            "universe come from, and can its membership shrink? A universe pulled from "
            "today's database contains only firms that survived to today — and momentum is "
            "maximally exposed to this bias because it deliberately buys the assets with "
            "the most spectacular trailing returns, which in the full universe are exactly "
            "the ones carrying blow-up risk. The audit plan: hold the strategy code fixed "
            "and ablate only the universe/return panel."
        ),
        "tool_selection": (
            "Python REPL with numpy for the panel simulation and vectorized momentum "
            "ranking, pandas available for tabulation. The decisive tool is the controlled "
            "ablation: two runs of the identical backtest function differing only in "
            "(universe, return panel, eligibility mask)."
        ),
        "recursive_delegation": (
            "Single-agent scale here. In a production audit across many submitted "
            "backtests, delegate one sub-agent per backtest to run the standardized "
            "survivor-ablation, with the parent maintaining the point-in-time constituent "
            "reference data that individual agents must never reconstruct from a current "
            "snapshot — centralizing the one data source whose corruption causes this "
            "bias."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Naive claim: long-only top-quintile 6m momentum earns Sharpe ~1.74 net of "
            "10 bps costs. Audit hypotheses — H0: the result is robust to universe "
            "construction, i.e. Sharpe(survivor-only) - Sharpe(full universe with "
            "delistings) <= 0.5. H1: the gap exceeds 0.5, convicting survivorship bias. "
            "Ground truth favors H1 by construction: the simulated economy contains "
            "bubble assets whose 30%/yr drift is compensation for a daily 0.12% "
            "jump-to-default hazard — in the full panel their unconditional edge is "
            "roughly zero, and only survivor-conditioning makes them look like alpha."
        ),
        "data_engineering": (
            "Panel: 200 assets x 1008 days, seeded. 140 sound assets (2%/yr drift, 28% "
            "vol) and 60 bubble assets (30%/yr drift, 35% vol, daily hazard 0.0012 of a "
            "terminal -80% crash then delisting; ~43 die over 4 years). Audited panel "
            "applies the crash return on the delisting day and zeros afterwards with an "
            "eligibility mask — mirroring correct CRSP-style delisting-return handling. "
            "The naive panel is the SAME simulation restricted to survivors with clean "
            "histories: precisely what querying a current-membership database produces."
        ),
        "methodology_justification": (
            "The audit is a controlled ablation rather than a statistical test because "
            "the question is about data construction, not sampling error: one function, "
            "two (universe, panel) inputs, all strategy parameters frozen. The 0.5-Sharpe "
            "threshold encodes the economic prior that universe bookkeeping should be "
            "performance-neutral for an honest backtest. The bubble/hazard design is the "
            "key teaching choice: it puts the delistings at the TOP of the momentum "
            "ranking (torpedo risk), where they maximally damage the strategy — a "
            "survivorship trap that random low-drift delistings would understate."
        ),
        "code_implementation": CODE_SURV,
        "statistical_validation": (
            "Deterministic values at seed 18: 157 survivors, 43 delisted; naive_sharpe "
            "~1.74 vs audited_sharpe ~0.39, inflation ~1.35 >> 0.5 threshold; annualized "
            "return drops from ~9.3% to ~2.7%. Cross-seed behavior (8-48): naive exceeds "
            "audited in every seed with gaps of 0.4-1.5 — the bias is one-sided by "
            "construction, never a wash. Sanity invariants: the two runs share identical "
            "strategy code (verified by the single backtest function), survivor count + "
            "delisted count = 200, and the audited panel's delisting days each realize "
            "exactly one -80% return."
        ),
        "risk_and_backtest_audit": (
            "Deploying the naive number means holding real bubble stocks at real hazard: "
            "expected blowups in the top-quintile portfolio are several per year, each "
            "costing weight x 80% — the audited Sharpe (~0.39) is the deployable "
            "estimate, and even it assumes delisting-day exits at the crash price rather "
            "than a halt/zero-recovery, so it remains optimistic. Frictions: 10 bps "
            "per-unit turnover is charged identically in both runs so costs cannot "
            "explain the gap. Institutional protocol: any equity backtest must ship with "
            "its universe-construction statement (point-in-time source, delisting-return "
            "treatment) before performance numbers are read."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The trap is seductive because the strategy code is genuinely honest — "
            "lagged signals, costs charged, no leakage; the corruption lives entirely in "
            "the data query, upstream of anything a code review inspects. (2) Partial "
            "fixes deceive: including dead assets' histories but omitting the delisting-"
            "day return (a common database gap) recovers only part of the damage and "
            "still inflates Sharpe. (3) The bias generalizes beyond delistings: "
            "backfilled histories of newly added index members create the same "
            "conditioning in reverse. (4) A mean-reversion strategy on the same corrupted "
            "universe would inflate even MORE (buying dips of stocks known to have "
            "recovered); momentum is not the worst case, just a realistic one."
        ),
        "falsification_strategy": (
            "Attempt to rescue the naive result: (a) set HAZARD=0 so no assets die — the "
            "naive and audited runs must converge to identical numbers, confirming the "
            "harness itself adds no gap; (b) rerun with delisting crashes included but "
            "universe still survivor-only — the residual gap isolates the ranking-"
            "contamination channel from the crash-return channel; (c) sweep the hazard "
            "over {0.0005, 0.0012, 0.003} and verify the Sharpe gap grows monotonically "
            "with the death rate, as survivorship theory predicts."
        ),
        "limitations": (
            "The simulation compresses survivorship into a single mechanism (jump-to-"
            "default on high-drift assets); real survivorship also flows through mergers, "
            "buyouts (which are often POSITIVE exits), and index reconstitution, so the "
            "real-world bias direction per exit type must be audited separately. The "
            "0.5-Sharpe conviction threshold is calibrated for this panel size; smaller "
            "universes need a bootstrap on the gap. Long-only quintile momentum is one "
            "strategy; the bias magnitude is strategy-dependent and must be re-measured "
            "per strategy family."
        ),
    },
    "agent_instructions": (
        "1. Reproduce the naive backtest exactly as submitted; record its Sharpe and "
        "universe size per rebalance date. 2. Interrogate the universe: does membership "
        "ever shrink? If not, presume survivor-conditioning and proceed to audit. "
        "3. Rebuild the panel point-in-time: delisting returns applied on the delisting "
        "day, assets ineligible afterwards. 4. Re-run the IDENTICAL strategy function on "
        "the audited panel; compute the Sharpe gap. 5. If gap > 0.5: print "
        "flaw_type=SURVIVORSHIP_BIAS and verdict=REJECTED_SURVIVORSHIP_BIAS; report the "
        "audited number first. 6. Run the HAZARD=0 control to certify the harness adds "
        "no artificial gap. 7. In the final report, name the universe-construction "
        "statement as a mandatory artifact for every future equity backtest."
    ),
    "verification": {
        "timeout_seconds": 240,
        "must_print": [
            "RESULTS",
            "n_assets_total=200",
            "naive_sharpe=",
            "audited_sharpe=",
            "sharpe_inflation=",
            "flaw_type=SURVIVORSHIP_BIAS",
            "audit_verdict=REJECTED_SURVIVORSHIP_BIAS",
            "verdict=REJECTED_SURVIVORSHIP_BIAS",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "pandas"],
    },
}

# =====================================================================
# Entry 9 — complexity 6: ES vs VaR under fat tails (v2-native, correct)
# =====================================================================

CODE_ES = '''"""Expected Shortfall vs Value-at-Risk under fat tails, via historical
simulation on Student-t returns, with a Cornish-Fisher comparison.

Lessons encoded: (1) VaR is a quantile and says nothing about loss
severity BEYOND the quantile; ES integrates the tail and captures what
VaR misses — the gap widens with tail heaviness and confidence level.
(2) Gaussian risk formulas understate the deep tail of t-distributed
returns and OVERSTATE moderate quantiles (the t/normal quantile
crossover). (3) Cornish-Fisher corrects the Gaussian quantile in the
right DIRECTION at both levels but overshoots in magnitude when excess
kurtosis is large — its validity domain is the honest caveat.
Deterministic (seeded).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

RNG_SEED = 9
N_DAYS = 5000
DF = 5                     # excess kurtosis 6/(df-4) = 6: finite but heavy
ANN_VOL = 0.15
MU_DAILY = 0.0002
LEVELS = (0.95, 0.99)


def simulate_returns(seed: int) -> np.ndarray:
    """Student-t returns scaled so the UNCONDITIONAL vol is 15% annualized.

    The t variance is df/(df-2), so the raw draws must be divided by its
    square root — forgetting this rescaling is a classic bug that
    silently inflates every risk number by ~29% at df=5.
    """
    rng = np.random.default_rng(seed)
    scale = (ANN_VOL / np.sqrt(252)) / np.sqrt(DF / (DF - 2))
    return MU_DAILY + scale * rng.standard_t(DF, N_DAYS)


def hs_var_es(r: np.ndarray, level: float) -> tuple:
    """Historical-simulation VaR and ES (losses reported as positive).

    ES averages ALL returns at or beyond the VaR quantile — the tail
    integral VaR ignores."""
    q = np.quantile(r, 1.0 - level)
    return -float(q), -float(r[r <= q].mean())


def gaussian_var_es(r: np.ndarray, level: float) -> tuple:
    """Parametric Gaussian benchmark with the same mean/sd."""
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1.0 - level)
    var = -(mu + z * sd)
    es = -(mu - sd * stats.norm.pdf(z) / (1.0 - level))
    return float(var), float(es)


def cornish_fisher_var(r: np.ndarray, level: float) -> float:
    """Cornish-Fisher expansion: skew/kurtosis-adjusted Gaussian quantile."""
    mu, sd = r.mean(), r.std(ddof=1)
    S = stats.skew(r)
    K = stats.kurtosis(r)          # excess kurtosis
    z = stats.norm.ppf(1.0 - level)
    z_cf = (z + (z**2 - 1.0) * S / 6.0 + (z**3 - 3.0 * z) * K / 24.0
            - (2.0 * z**3 - 5.0 * z) * S**2 / 36.0)
    return float(-(mu + z_cf * sd))


def main() -> None:
    r = simulate_returns(RNG_SEED)
    S = float(stats.skew(r))
    K = float(stats.kurtosis(r))

    print("RESULTS")
    print(f"n_days={N_DAYS}")
    print(f"sample_skew={S:.3f}")
    print(f"sample_excess_kurtosis={K:.3f}")

    checks = {}
    for level in LEVELS:
        tag = str(int(level * 100))
        var_hs, es_hs = hs_var_es(r, level)
        var_g, es_g = gaussian_var_es(r, level)
        cf = cornish_fisher_var(r, level)
        ratio_hs = es_hs / var_hs
        ratio_g = es_g / var_g
        print(f"hs_var_{tag}={var_hs * 100:.3f}pct")
        print(f"hs_es_{tag}={es_hs * 100:.3f}pct")
        print(f"gaussian_var_{tag}={var_g * 100:.3f}pct")
        print(f"gaussian_es_{tag}={es_g * 100:.3f}pct")
        print(f"cf_var_{tag}={cf * 100:.3f}pct")
        print(f"es_var_ratio_{tag}={ratio_hs:.3f}")
        print(f"gaussian_es_var_ratio_{tag}={ratio_g:.3f}")
        # ES must exceed VaR (tail severity), and the empirical ES/VaR
        # ratio must exceed the Gaussian one (fat tails).
        checks[f"es_exceeds_var_{tag}"] = es_hs > var_hs
        checks[f"tail_heavier_than_gaussian_{tag}"] = ratio_hs > ratio_g
        # CF must adjust the Gaussian quantile TOWARD the empirical one
        # (correct direction), even where it overshoots in magnitude.
        checks[f"cf_direction_correct_{tag}"] = \
            np.sign(cf - var_g) == np.sign(var_hs - var_g)

    # The t/normal crossover: Gaussian OVERSTATES the 95% quantile and
    # UNDERSTATES the 99% one — direction flips across the crossover.
    var95_hs, _ = hs_var_es(r, 0.95)
    var99_hs, _ = hs_var_es(r, 0.99)
    var95_g, _ = gaussian_var_es(r, 0.95)
    var99_g, _ = gaussian_var_es(r, 0.99)
    checks["gaussian_overstates_95"] = var95_g > var95_hs
    checks["gaussian_understates_99"] = var99_g < var99_hs

    for name, passed in checks.items():
        print(f"check_{name}={passed}")
    ok = all(checks.values())
    print(f"verdict={'ES_CAPTURES_TAIL_RISK' if ok else 'TAIL_CHECKS_FAILED'}")


if __name__ == "__main__":
    main()
'''

ENTRY_ES = {
    "schema_version": 2,
    "entry_kind": "correct",
    "expected_verdict": "ES_CAPTURES_TAIL_RISK",
    "flaws": [],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [],
    },
    "rlm": MINIMAL_RLM,
    "training_sequence": MINIMAL_TRAINING_SEQUENCE,
    "metadata": {
        "id": "qlm1-000009-risk-es-vs-var",
        "domain": "Risk Modeling / Tail Risk",
        "complexity": 6,
        "tags": [
            "expected-shortfall",
            "value-at-risk",
            "historical-simulation",
            "student-t",
            "fat-tails",
            "cornish-fisher",
            "quantile-crossover",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The question 'is ES better than VaR?' must be operationalized: on a return "
            "distribution with known heavy tails (Student-t, df=5, so excess kurtosis 6 "
            "by construction), quantify exactly what VaR misses — the severity of losses "
            "beyond the quantile — and verify the empirical ES/VaR ratio exceeds its "
            "Gaussian counterpart at both 95% and 99%. A second, subtler prediction "
            "disciplines the analysis: the t/normal quantile crossover means Gaussian VaR "
            "should OVERSTATE the 95% loss and UNDERSTATE the 99% one; finding both "
            "directions confirms the machinery rather than a one-sided bias."
        ),
        "tool_selection": (
            "Python REPL with numpy for simulation and quantile arithmetic, scipy.stats "
            "for the t generator, normal quantiles, and sample skew/kurtosis. Historical "
            "simulation needs no optimizer; the Cornish-Fisher expansion is closed-form."
        ),
        "recursive_delegation": (
            "Not needed for one series. For a book-level risk run (hundreds of desks), "
            "delegate per-desk ES computation to sub-agents but keep aggregation in the "
            "parent: ES is subadditive so the book ES is bounded by the sum, and the "
            "parent must own the copula/correlation assumptions that individual desk "
            "agents cannot see."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "H0 (all must hold for ES_CAPTURES_TAIL_RISK): (a) ES > VaR at 95% and 99%; "
            "(b) empirical ES/VaR ratio > Gaussian ES/VaR ratio at both levels (tail "
            "heaviness beyond Gaussian); (c) Cornish-Fisher adjusts the Gaussian quantile "
            "toward the empirical one (sign(CF - G_VaR) = sign(HS_VaR - G_VaR)) at both "
            "levels; (d) the quantile crossover holds: Gaussian VaR95 > HS VaR95 while "
            "Gaussian VaR99 < HS VaR99. Any failed check yields TAIL_CHECKS_FAILED — the "
            "checks are falsifiable predictions from t-distribution theory, not "
            "descriptive statistics."
        ),
        "data_engineering": (
            "5000 seeded daily returns from Student-t(5) scaled to 15% annualized "
            "unconditional vol — the scaling must divide by sqrt(df/(df-2)) or every risk "
            "number silently inflates ~29%, a bug the code documents explicitly. df=5 is "
            "chosen so excess kurtosis (6/(df-4)=6) is finite: df<=4 would make sample "
            "kurtosis non-convergent and the Cornish-Fisher input meaningless. On real "
            "data this step also requires cleaning stale prints and deciding whether to "
            "measure VaR on raw or vol-standardized (filtered HS) returns."
        ),
        "methodology_justification": (
            "Historical simulation is the reference method because it imposes no "
            "distributional assumption — the empirical quantile and tail mean are "
            "consistent estimators regardless of the true law — making it the fair judge "
            "between Gaussian and CF approximations. ES over VaR is motivated by theory "
            "(ES is coherent/subadditive; VaR is not) but the entry demonstrates the "
            "PRACTICAL gap: identical VaR can hide arbitrarily bad tails, which the "
            "ES/VaR ratio surfaces. Cornish-Fisher is included as the standard "
            "practitioner shortcut precisely so its failure mode — magnitude overshoot "
            "at high kurtosis, outside its validity domain — is taught alongside its "
            "correct directional behavior."
        ),
        "code_implementation": CODE_ES,
        "statistical_validation": (
            "Deterministic values at seed 9: sample excess kurtosis ~4.5; HS VaR95 "
            "~1.46% vs ES95 ~2.03% (ratio 1.39 vs Gaussian 1.26); HS VaR99 ~2.31% vs "
            "ES99 ~3.03% (ratio 1.31 vs Gaussian 1.15); crossover confirmed (Gaussian "
            "VaR95 1.55% > HS 1.46%; Gaussian VaR99 2.21% < HS 2.31%); CF direction "
            "correct at both levels while overshooting in magnitude (CF VaR99 ~2.85% vs "
            "HS 2.31%) — reported, not hidden. Robustness: rerun across 20 seeds and "
            "with df in {5, 8, 20}; all ratio gaps must shrink monotonically toward zero "
            "as df grows (Gaussian limit), a dose-response check on the entire pipeline."
        ),
        "risk_and_backtest_audit": (
            "Estimation risk concentrates in the tail: ES99 from 5000 days averages only "
            "~50 observations, so its sampling error is materially larger than VaR99's — "
            "production ES must ship with bootstrap confidence bands, and this is why "
            "Basel's FRTB pairs ES with quantile backtesting rather than direct ES "
            "backtesting (ES is not elicitable on its own). The unconditional HS window "
            "ignores volatility clustering: in a GARCH world, 5000-day HS under-responds "
            "to current regimes — filtered HS (vol-rescaled) is the deployment upgrade. "
            "CF-based capital at 99% with kurtosis ~4.5 would OVER-reserve here; the "
            "validity-domain caveat is a capital-efficiency issue, not pedantry."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The t-scaling bug (forgetting sqrt(df/(df-2))) inflates all numbers "
            "consistently, so internal ratio checks still pass while every level is "
            "wrong — absolute levels must be validated against the 15% vol target. "
            "(2) Quantile interpolation conventions shift VaR99 by several percent of "
            "its value at n=5000; fix the convention before comparing methods. "
            "(3) Sample skew of symmetric-t draws is nonzero in finite samples (~0.43 "
            "here); feeding it to CF is correct practice but means CF 'corrects' for "
            "skew that is sampling noise. (4) Reporting ES without its estimation error "
            "invites false precision — the deep-tail mean is the least stable statistic "
            "in this entire entry."
        ),
        "falsification_strategy": (
            "(a) Replace the t generator with Gaussian draws: every tail-heaviness and "
            "crossover check must FAIL (ratios converge, CF ≈ Gaussian), confirming the "
            "checks detect tails rather than always passing. (b) Increase df to 30 and "
            "verify all gaps shrink toward zero monotonically. (c) Negate the returns "
            "(flip skew): ES/VaR conclusions must be mirror-consistent. (d) Halve the "
            "sample to 2500 and confirm conclusions survive, establishing they are not "
            "sample-size artifacts."
        ),
        "limitations": (
            "Unconditional historical simulation ignores volatility clustering and "
            "regime shifts; conclusions apply to the marginal distribution, not to "
            "day-ahead conditional risk. The Cornish-Fisher validity domain excludes "
            "the very high-kurtosis cases where correction matters most — beyond it, "
            "fit a parametric tail (GPD/EVT) instead. ES lacks direct elicitability, "
            "complicating backtesting. All numbers are single-asset; portfolio ES "
            "additionally requires dependence modeling that this entry does not touch."
        ),
    },
    "agent_instructions": (
        "1. Simulate seeded t(5) returns; verify realized annualized vol is within 5% "
        "of the 15% target (catches the scaling bug). 2. Compute HS VaR and ES at 95% "
        "and 99%; assert ES > VaR at both. 3. Compute Gaussian VaR/ES from sample "
        "moments; compare ES/VaR ratios and record the tail-heaviness gap. 4. Test the "
        "quantile crossover: Gaussian must overstate the 95% loss and understate the "
        "99% one. 5. Compute Cornish-Fisher VaR at both levels; check direction "
        "against HS, and report the magnitude overshoot honestly with the validity-"
        "domain explanation. 6. Falsification: rerun with Gaussian draws and confirm "
        "the checks fail; rerun with df=30 and confirm monotone gap shrinkage. "
        "7. Conclude with ES_CAPTURES_TAIL_RISK only if every check passed, and state "
        "the estimation-error caveat on deep-tail ES."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": [
            "RESULTS",
            "hs_var_95=",
            "hs_es_95=",
            "hs_var_99=",
            "hs_es_99=",
            "cf_var_99=",
            "check_es_exceeds_var_95=True",
            "check_es_exceeds_var_99=True",
            "check_tail_heavier_than_gaussian_99=True",
            "verdict=ES_CAPTURES_TAIL_RISK",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "scipy"],
    },
}

# =====================================================================
# Entry 10 — complexity 8: RLM environment — risk-parity rebalancing
# =====================================================================

CODE_RP = '''"""RLM TEACHING ENTRY: multi-asset portfolio rebalancing as ENVIRONMENT
INTERACTION, with risk-parity optimization and constraint handling.

The rebalancing workflow is exposed as a stateful environment with a
gym-like reset/step API and a strict precondition chain:

  reset -> compute_covariance -> optimize_weights -> apply_constraints
        -> execute_rebalance -> evaluate_performance

Two actions in the scripted plan are deliberately out of order; guards
refuse them with error observations (never exceptions) and the driver
recovers by re-ordering. Constraint handling uses iterative waterfall
capping — the naive cap-then-renormalize approach pushes capped weights
back ABOVE the cap, a real bug this entry demonstrates the fix for.
All observations are JSON-serializable dicts; deterministic replay is
verified. Seeded and deterministic throughout.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

RNG_SEED = 10
N_ASSETS = 4
N_DAYS = 1250
N_TRAIN = 1000
ANN = 252
MAX_WEIGHT = 0.40
COST_BPS = 5.0
SHRINK = 0.10


def simulate_returns(seed: int) -> np.ndarray:
    """One common factor + heterogeneous idiosyncratic vols, so the
    risk-parity solution differs visibly from equal weight."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(N_DAYS)
    betas = np.array([1.2, 0.9, 0.5, 0.2])
    idio_vol = np.array([0.25, 0.18, 0.12, 0.07]) / np.sqrt(ANN)
    factor_vol = 0.10 / np.sqrt(ANN)
    rets = (betas[None, :] * factor[:, None] * factor_vol
            + idio_vol[None, :] * rng.standard_normal((N_DAYS, N_ASSETS)))
    return rets + 0.03 / ANN


def waterfall_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Iteratively cap weights and redistribute to uncapped assets.

    Naive min(w, cap)/sum violates the cap after renormalization; the
    waterfall repeats until no violation remains. Terminates because
    N_ASSETS * cap > 1."""
    w = w.copy()
    for _ in range(N_ASSETS):          # at most N iterations by construction
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        w[free] += excess * w[free] / w[free].sum()
    return w


class QuantEnvironment:
    """Stateful portfolio-rebalancing environment.

    Guards return {"ok": False, "error": ...} observations for invalid
    orderings — the agent must read observations and re-plan, not crash.
    """

    ACTIONS = ("reset", "compute_covariance", "optimize_weights",
               "apply_constraints", "execute_rebalance", "evaluate_performance")

    def __init__(self, seed: int = RNG_SEED):
        self._seed = seed
        self._state = "uninitialized"
        self._rets = None
        self._cov = None
        self._w_opt = None
        self._w_final = None
        self._prev_w = None
        self._cost = None

    def reset(self) -> dict:
        self._rets = simulate_returns(self._seed)
        self._cov = None
        self._w_opt = None
        self._w_final = None
        self._prev_w = np.full(N_ASSETS, 1.0 / N_ASSETS)   # incumbent book
        self._cost = None
        self._state = "reset"
        return {"ok": True, "state": self._state, "n_assets": N_ASSETS,
                "n_train": N_TRAIN, "n_eval": N_DAYS - N_TRAIN,
                "incumbent_weights": [round(float(x), 4) for x in self._prev_w]}

    def step(self, action: dict) -> dict:
        name = action.get("name")
        if name not in self.ACTIONS:
            return {"ok": False, "error": f"unknown action: {name}",
                    "state": self._state}
        if name == "reset":
            return self.reset()
        if self._state == "uninitialized":
            return {"ok": False, "error": "call reset first", "state": self._state}
        return getattr(self, f"_do_{name}")(action)

    def _do_compute_covariance(self, action: dict) -> dict:
        train = self._rets[:N_TRAIN]                  # train window only
        sample = np.cov(train, rowvar=False, ddof=1)
        target = np.diag(np.diag(sample))
        self._cov = (1.0 - SHRINK) * sample + SHRINK * target
        self._state = "covariance_ready"
        return {"ok": True, "state": self._state, "shrinkage": SHRINK,
                "ann_vols": [round(float(v), 4)
                             for v in np.sqrt(np.diag(self._cov) * ANN)],
                "condition_number": round(float(np.linalg.cond(self._cov)), 1)}

    def _do_optimize_weights(self, action: dict) -> dict:
        if self._cov is None:
            return {"ok": False, "error": "compute_covariance before optimize_weights",
                    "state": self._state}
        cov = self._cov

        def rc_dispersion(w):
            port_var = w @ cov @ w
            rc = w * (cov @ w) / port_var
            return float(((rc - 1.0 / N_ASSETS) ** 2).sum())

        res = minimize(rc_dispersion, np.full(N_ASSETS, 1.0 / N_ASSETS),
                       method="SLSQP", bounds=[(1e-4, 1.0)] * N_ASSETS,
                       constraints=[{"type": "eq",
                                     "fun": lambda w: w.sum() - 1.0}],
                       options={"ftol": 1e-14, "maxiter": 500})
        self._w_opt = res.x / res.x.sum()
        rc = self._w_opt * (cov @ self._w_opt) / (self._w_opt @ cov @ self._w_opt)
        self._state = "optimized"
        return {"ok": True, "state": self._state, "converged": bool(res.success),
                "weights": [round(float(x), 4) for x in self._w_opt],
                "risk_contributions": [round(float(x), 4) for x in rc],
                "rc_dispersion": float(f"{rc_dispersion(self._w_opt):.2e}")}

    def _do_apply_constraints(self, action: dict) -> dict:
        if self._w_opt is None:
            return {"ok": False, "error": "optimize_weights before apply_constraints",
                    "state": self._state}
        w = waterfall_cap(self._w_opt, MAX_WEIGHT)
        self._w_final = w
        self._state = "constrained"
        return {"ok": True, "state": self._state, "max_weight_cap": MAX_WEIGHT,
                "n_capped": int((self._w_opt > MAX_WEIGHT).sum()),
                "weights": [round(float(x), 4) for x in w],
                "sum_weights": round(float(w.sum()), 6),
                "cap_respected": bool((w <= MAX_WEIGHT + 1e-9).all())}

    def _do_execute_rebalance(self, action: dict) -> dict:
        if self._w_final is None:
            return {"ok": False, "error": "apply_constraints before execute_rebalance",
                    "state": self._state}
        turnover = float(np.abs(self._w_final - self._prev_w).sum())
        self._cost = turnover * COST_BPS / 1e4
        self._state = "rebalanced"
        return {"ok": True, "state": self._state, "turnover": round(turnover, 4),
                "cost_bps_paid": round(self._cost * 1e4, 3)}

    def _do_evaluate_performance(self, action: dict) -> dict:
        if self._cost is None:
            return {"ok": False, "error": "execute_rebalance before evaluate_performance",
                    "state": self._state}
        oos = self._rets[N_TRAIN:]                    # sealed evaluation window
        port = oos @ self._w_final
        port[0] -= self._cost
        sd = port.std(ddof=1)
        eq = oos @ np.full(N_ASSETS, 1.0 / N_ASSETS)
        self._state = "evaluated"
        return {"ok": True, "state": self._state,
                "oos_ann_vol": round(float(sd * np.sqrt(ANN)), 4),
                "oos_sharpe_net": round(0.0 if sd == 0
                                        else float(port.mean() / sd * np.sqrt(ANN)), 3),
                "equal_weight_ann_vol": round(float(eq.std(ddof=1) * np.sqrt(ANN)), 4)}


def main() -> None:
    env = QuantEnvironment(seed=RNG_SEED)
    # Scripted plan with TWO deliberate ordering mistakes.
    plan = [
        {"name": "optimize_weights"},      # invalid: environment not reset
        {"name": "reset"},
        {"name": "compute_covariance"},
        {"name": "optimize_weights"},
        {"name": "execute_rebalance"},     # invalid: constraints not applied
        {"name": "apply_constraints"},
        {"name": "execute_rebalance"},
        {"name": "evaluate_performance"},
    ]
    executed, guard_failures = [], 0
    obs = None
    final_by_action = {}
    for action in plan:
        obs = env.step(action)
        print(f"trace={json.dumps({'action': action['name'], 'observation': obs}, sort_keys=True)}")
        if obs["ok"]:
            executed.append(action["name"])
            final_by_action[action["name"]] = obs
        else:
            guard_failures += 1

    constrained = final_by_action["apply_constraints"]
    evaluated = final_by_action["evaluate_performance"]
    checks = {
        "guards_rejected_both_unordered_actions": guard_failures == 2,
        "recovered_full_action_sequence": executed == list(QuantEnvironment.ACTIONS),
        "final_observation_json_roundtrip": json.loads(json.dumps(obs)) == obs,
        "weights_sum_to_one": abs(constrained["sum_weights"] - 1.0) < 1e-6,
        "weight_cap_respected": constrained["cap_respected"],
        "risk_parity_achieved": final_by_action["optimize_weights"]["rc_dispersion"] < 1e-8,
        "rp_vol_below_equal_weight": evaluated["oos_ann_vol"] < evaluated["equal_weight_ann_vol"],
    }
    env2 = QuantEnvironment(seed=RNG_SEED)
    obs2 = None
    for action in plan:
        obs2 = env2.step(action)
    checks["deterministic_replay"] = obs == obs2

    print("RESULTS")
    print("environment_class=QuantEnvironment")
    print(f"actions_executed={','.join(executed)}")
    print(f"guard_failures_handled={guard_failures}")
    for name, passed in checks.items():
        print(f"check_{name}={passed}")
    print(f"final_observation={json.dumps(obs, sort_keys=True)}")
    verdict = "ENVIRONMENT_VERIFIED" if all(checks.values()) else "ENVIRONMENT_BROKEN"
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
'''

ENTRY_RP = {
    "schema_version": 2,
    "entry_kind": "rlm_environment",
    "expected_verdict": "ENVIRONMENT_VERIFIED",
    "flaws": [],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [
            {
                "function": "waterfall_cap",
                "reason": "iterative constraint projection: each capping round depends on the previous redistribution; bounded by N_ASSETS",
                "max_iterations": 4,
            },
            {
                "function": "main",
                "reason": "agent action loop: each step depends on the previous observation",
                "max_iterations": 8,
            },
        ],
    },
    "rlm": {
        "environment_class": "QuantEnvironment",
        "actions": [
            "reset",
            "compute_covariance",
            "optimize_weights",
            "apply_constraints",
            "execute_rebalance",
            "evaluate_performance",
        ],
        "max_recursion_depth": 3,
        "tool_timeout_seconds": 120,
    },
    "training_sequence": {
        "style": "environment_interaction",
        "include_adversarial_critique": True,
        "include_final_calibrated_answer": True,
        "loss_masking": {
            "user": 0.0,
            "environment_observation": 0.0,
            "assistant_thought": 1.0,
            "tool_call": 1.0,
            "final_answer": 1.0,
        },
    },
    "metadata": {
        "id": "qlm1-000010-rlm-portfolio-rebalance",
        "domain": "Portfolio Construction / Environment Interaction",
        "complexity": 8,
        "tags": [
            "rlm",
            "environment",
            "risk-parity",
            "covariance-shrinkage",
            "slsqp",
            "waterfall-capping",
            "turnover-costs",
            "error-recovery",
            "deterministic-replay",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "The task is not 'compute risk-parity weights' but 'conduct the rebalancing "
            "THROUGH an environment' whose precondition chain encodes the methodology: "
            "estimating covariance before optimizing, optimizing before constraining, "
            "constraining before trading, trading before evaluating. Skipping any link is "
            "a research error the environment must refuse with an error observation, and "
            "the plan's two deliberate ordering mistakes exist to train exactly that "
            "refusal-and-recovery loop. A second lesson rides along: constraint "
            "projection is nontrivial — the obvious cap-then-renormalize is WRONG "
            "(renormalization pushes capped weights back over the cap), and the "
            "environment's cap_respected observation exposes whether the waterfall fix "
            "actually holds."
        ),
        "tool_selection": (
            "Python REPL hosting the environment; numpy for the factor-model simulation "
            "and covariance algebra, scipy SLSQP for the risk-contribution-dispersion "
            "minimization, json for the observation protocol — every observation must "
            "survive a json round-trip because RLM agents consume serialized tool "
            "output, never live objects."
        ),
        "recursive_delegation": (
            "Depth 3 covers parent -> rebalancing sub-agent -> verification sub-agent "
            "replaying the trace against a fresh same-seed environment. The parent owns "
            "the plan; the sub-agent drives actions and must surface, not suppress, "
            "error observations; the verifier's replay check is performed inline here "
            "as the deterministic_replay invariant."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "Environment-level hypotheses — H0-env: the environment is NOT a faithful "
            "instrument (permits invalid orderings, emits non-serializable observations, "
            "violates the weight cap or budget constraint, fails risk parity, or replays "
            "non-deterministically). H1-env: all eight checks hold — both unordered "
            "actions rejected, full sequence recovered, json round-trip, weights sum to "
            "1, cap respected after waterfall projection, risk-contribution dispersion "
            "< 1e-8 on the optimized weights, OOS risk-parity vol below equal-weight "
            "vol, and exact deterministic replay. Verdict ENVIRONMENT_VERIFIED is the "
            "conjunction; any failure yields ENVIRONMENT_BROKEN."
        ),
        "data_engineering": (
            "Returns are generated INSIDE the environment on reset(): 4 assets x 1250 "
            "days from a one-factor model with betas (1.2, 0.9, 0.5, 0.2) and "
            "heterogeneous idiosyncratic vols (25%-7%), so the risk-parity solution is "
            "visibly far from equal weight and the 0.40 cap binds on the low-vol asset "
            "— making the constraint path non-trivial by construction. The 1000/250 "
            "train/eval split is fixed by the environment, not the agent, so no action "
            "can leak evaluation data into estimation. Covariance uses 10% shrinkage "
            "toward the diagonal — the minimal regularization gesture, with the "
            "condition number exposed in the observation."
        ),
        "methodology_justification": (
            "Risk parity via minimizing the dispersion of risk contributions RC_i = "
            "w_i (Σw)_i / (w'Σw) around 1/N is the standard formulation; SLSQP with a "
            "budget equality and positivity bounds solves the 4-asset problem to "
            "machine-level dispersion. Waterfall capping is chosen over naive "
            "cap-and-renormalize deliberately: the naive method returned a 0.4386 "
            "weight against a 0.40 cap during development — the environment now "
            "reports cap_respected so the defect class is machine-checked forever. "
            "Guards return error observations rather than raising because observation-"
            "driven re-planning, not crash handling, is the trainable RLM behavior. "
            "Costs are charged on turnover from the incumbent equal-weight book, so "
            "execute_rebalance has real economic content."
        ),
        "code_implementation": CODE_RP,
        "statistical_validation": (
            "Eight machine-checked invariants printed as check_* lines, including: "
            "risk-contribution dispersion < 1e-8 (achieved: ~5e-16, i.e. exact risk "
            "parity on the training covariance); cap respected after waterfall "
            "(naive renormalization provably violates it on this instance); OOS "
            "risk-parity vol (~8.7%) below equal-weight vol (~11.5%) — the economic "
            "point of risk parity realized out of sample; and exact deterministic "
            "replay of the final observation from a fresh same-seed environment. The "
            "guard tests are calibrated: exactly two invalid actions are planned and "
            "exactly two must be refused."
        ),
        "risk_and_backtest_audit": (
            "Environment-integrity risks: (a) state leakage — an environment that let "
            "evaluate_performance run before execute_rebalance would score a costless "
            "paper portfolio; the precondition chain plus guard test closes this; "
            "(b) constraint laundering — reporting capped weights while internally "
            "trading uncapped ones; cap_respected is computed from the weights actually "
            "used downstream; (c) nondeterminism — fresh randomness per reset would "
            "make RLM trajectories unreproducible; the replay check enforces bit-level "
            "stability. Financial caveats surfaced in observations: single-window "
            "covariance estimation, one rebalance (no multi-period turnover path), and "
            "5 bps linear costs are simplifications the evaluation observation makes "
            "no attempt to hide."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The naive cap-then-renormalize bug is the seductive trap for the "
            "student: it looks correct, sums to 1, and silently violates the cap — "
            "only the cap_respected check catches it. (2) A plan without deliberate "
            "mistakes would pass while demonstrating nothing about recovery; the "
            "mistakes are load-bearing. (3) SLSQP convergence is reported but not "
            "guaranteed for larger N or near-singular covariances — at scale, switch "
            "to the cyclical coordinate descent risk-parity algorithm with proven "
            "convergence. (4) Rounding observations to 4 decimals is required for "
            "replay comparison but could mask small nondeterminism; keep rounding no "
            "coarser. (5) Equal risk contribution on the TRAINING covariance does not "
            "guarantee equal realized OOS contributions — the evaluation observation "
            "reports vol, not a false claim of realized parity."
        ),
        "falsification_strategy": (
            "Break the environment and confirm the battery catches it: (a) replace "
            "waterfall_cap with naive cap-and-renormalize — cap_respected fails and "
            "the verdict flips to ENVIRONMENT_BROKEN; (b) remove the execute_rebalance "
            "guard — the guard-count check fails; (c) inject fresh randomness per "
            "reset — deterministic_replay fails; (d) swap the factor model for equal "
            "vols — risk parity converges to equal weight and the rp_vol_below_equal_"
            "weight check correctly FAILS, proving the battery rejects environments "
            "where the optimization is vacuous."
        ),
        "limitations": (
            "One scripted plan exercises one recovery path; full RLM coverage needs "
            "many plans (shuffled orders, repeated actions, unknown actions) over this "
            "environment. Single rebalance date: multi-period paths with turnover "
            "budgets and drift-triggered rebalancing are the natural extension. The "
            "covariance is estimated once; rolling re-estimation with the attendant "
            "estimation-noise-vs-turnover tradeoff is out of scope. Long-only unit-"
            "leverage risk parity only — levered risk parity introduces financing "
            "costs and a different risk profile entirely."
        ),
    },
    "agent_instructions": (
        "1. Instantiate QuantEnvironment with the fixed seed; never generate data "
        "outside it. 2. Drive actions exclusively via env.step({'name': ...}); after "
        "each step branch on obs['ok'] — on False, read obs['error'] and re-order the "
        "plan to satisfy the stated precondition, never retry blindly. 3. Complete the "
        "chain: reset, compute_covariance, optimize_weights, apply_constraints, "
        "execute_rebalance, evaluate_performance. 4. Log every (action, observation) "
        "pair as a json trace line. 5. Run the verification battery: guard counts, "
        "sequence recovery, json round-trip, budget and cap invariants, risk-parity "
        "dispersion < 1e-8, OOS vol comparison, deterministic replay. 6. Emit "
        "verdict=ENVIRONMENT_VERIFIED only if every check passes; otherwise name the "
        "failing check in ENVIRONMENT_BROKEN. 7. Report the OOS numbers exactly as "
        "observed — the wrapper never editorializes the underlying result."
    ),
    "verification": {
        "timeout_seconds": 120,
        "must_print": [
            "RESULTS",
            "environment_class=QuantEnvironment",
            "actions_executed=reset,compute_covariance,optimize_weights,apply_constraints,execute_rebalance,evaluate_performance",
            "guard_failures_handled=2",
            "check_weight_cap_respected=True",
            "check_risk_parity_achieved=True",
            "check_deterministic_replay=True",
            "final_observation=",
            "verdict=ENVIRONMENT_VERIFIED",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy", "scipy"],
    },
}

# =====================================================================
# Entry 11 — complexity 8: limit order book microstructure (v2-native)
# =====================================================================

CODE_LOB = '''"""Limit order book dynamics: bid-ask spread vs order-flow imbalance,
queue position for a resting limit order, and adverse-selection cost.

Model: best-quote-level LOB with Poisson-thinned event types (market
orders, limit joins, inside-quote improvements, cancels) and MARKOV-
PERSISTENT market-order direction (rho=0.75) — persistence is essential:
with IID flow, resting-order fills carry no information and adverse
selection vanishes (verified during development). Three measurements:
(1) mean spread as a function of |OFI| terciles + OLS slope;
(2) fill probability for a tagged order joining the back of the bid
queue, with correct front-of-queue mechanics (the order becomes the
level when everyone ahead leaves — it is NOT canceled);
(3) adverse selection via markouts: mid at fill+H versus fill price,
compared against the unconditional immediate-fill counterfactual.
numpy only; deterministic (seeded).
"""

from __future__ import annotations

import numpy as np

RNG_SEED = 11
N_EVENTS = 60000
TICK = 0.01
MID0 = 100.00
P_MKT = 0.15           # market order prob per event (each side via direction)
P_LIMIT = 0.20         # limit join prob per side
P_CANCEL = 0.15        # cancel prob per side
DEPTH_MEAN = 6.0       # mean refill depth after a level breaks
GAP2_PROB = 0.35       # prob the refreshed best sits 2 ticks away
RHO = 0.75             # market-order direction persistence
N_TAGGED = 2000
EP_EVENTS = 600
HORIZON = 40


class LOB:
    """Best-quote-level order book. The event loop is inherently
    sequential: every event conditions on current book state."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.bid, self.ask = MID0 - TICK / 2, MID0 + TICK / 2
        self.Qb, self.Qa = 8, 8
        self.last_dir = -1 if rng.random() < 0.5 else 1

    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    def spread_ticks(self) -> int:
        return round((self.ask - self.bid) / TICK)

    def ofi(self) -> float:
        return (self.Qb - self.Qa) / (self.Qb + self.Qa)

    def refill(self) -> int:
        return 1 + int(-DEPTH_MEAN * np.log(max(self.rng.random(), 1e-12)))

    def step(self) -> str:
        x = self.rng.random()
        if x < 2 * P_MKT:
            if self.rng.random() >= RHO:
                self.last_dir = -self.last_dir
            if self.last_dir == -1:            # market sell hits the bid
                self.Qb -= 1
                if self.Qb <= 0:
                    gap = 2 if self.rng.random() < GAP2_PROB else 1
                    self.bid -= gap * TICK
                    self.Qb = self.refill()
                    return "sell_break"
                return "sell"
            self.Qa -= 1                       # market buy lifts the ask
            if self.Qa <= 0:
                gap = 2 if self.rng.random() < GAP2_PROB else 1
                self.ask += gap * TICK
                self.Qa = self.refill()
                return "buy_break"
            return "buy"
        if x < 2 * P_MKT + P_LIMIT:            # bid-side limit order
            if self.spread_ticks() > 1 and self.rng.random() < 0.5:
                self.bid += TICK               # improve inside the spread
                self.Qb = 1 + int(self.rng.random() * 4)
                return "jb_inside"
            self.Qb += 1
            return "jb"
        if x < 2 * P_MKT + 2 * P_LIMIT:        # ask-side limit order
            if self.spread_ticks() > 1 and self.rng.random() < 0.5:
                self.ask -= TICK
                self.Qa = 1 + int(self.rng.random() * 4)
                return "ja_inside"
            self.Qa += 1
            return "ja"
        if x < 2 * P_MKT + 2 * P_LIMIT + P_CANCEL:
            if self.Qb > 1:
                self.Qb -= 1
            return "cb"
        if self.Qa > 1:
            self.Qa -= 1
        return "ca"


def spread_vs_ofi(seed: int) -> tuple:
    """Long-path measurement of spread conditional on |OFI|."""
    rng = np.random.default_rng(seed)
    lob = LOB(rng)
    spread = np.empty(N_EVENTS)
    ofi = np.empty(N_EVENTS)
    mids = np.empty(N_EVENTS)
    for t in range(N_EVENTS):
        spread[t] = lob.spread_ticks()
        ofi[t] = lob.ofi()
        mids[t] = lob.mid()
        lob.step()
    a = np.abs(ofi)
    e1, e2 = np.quantile(a, [1 / 3, 2 / 3])
    terciles = (float(spread[a <= e1].mean()),
                float(spread[(a > e1) & (a <= e2)].mean()),
                float(spread[a > e2].mean()))
    slope = float(np.polyfit(a, spread, 1)[0])
    return terciles, slope, mids, spread


def tagged_order_study(seed: int) -> dict:
    """Episodes: a tagged order joins the back of the bid queue at t=0.

    Front-of-queue mechanics: market sells consume the units AHEAD first;
    when ahead==0 the next market sell fills US (the level cannot break
    past a resting order). Cancels remove ahead-units with probability
    ahead/(ahead+behind). Inside-quote improvements strand the order
    (treated as unfilled). Markout = mid(fill+H) - fill_price; benchmark
    is the immediate-fill counterfactual mid(H) - bid(0) over ALL
    episodes, so both legs share the same dynamics.
    """
    rng = np.random.default_rng(seed)
    fills = 0
    adverse = 0
    markouts = []
    uncond_markouts = []
    uncond_down = 0
    for _ in range(N_TAGGED):
        lob = LOB(rng)
        bid0 = lob.bid
        ahead, behind = lob.Qb, 0
        fill_price, fill_t = None, None
        path = []
        alive = True
        for t in range(EP_EVENTS):
            path.append(lob.mid())
            if fill_price is not None and t >= fill_t + HORIZON:
                break
            if alive and fill_price is None:
                b0 = lob.bid
                ev = lob.step()
                if ev in ("sell", "sell_break"):
                    if ahead > 0:
                        ahead -= 1
                        if ev == "sell_break":
                            alive = False      # book/tag inconsistency: stale
                    else:
                        fill_price = b0        # front of queue: we are filled
                        fill_t = t + 1
                elif ev == "jb":
                    behind += 1
                elif ev == "jb_inside":
                    alive = False              # price improved past our level
                elif ev == "cb":
                    tot = ahead + behind
                    if tot > 0 and rng.random() < ahead / tot:
                        ahead = max(ahead - 1, 0)
                    else:
                        behind = max(behind - 1, 0)
            else:
                lob.step()
        h_end = min(HORIZON, len(path) - 1)
        uncond_markouts.append(path[h_end] - bid0)
        if path[h_end] < bid0:
            uncond_down += 1
        if fill_price is not None:
            idx = min(fill_t + HORIZON, len(path) - 1)
            mo = path[idx] - fill_price
            markouts.append(mo)
            fills += 1
            if mo < 0:
                adverse += 1
    return {
        "fill_prob": fills / N_TAGGED,
        "filled_markout_ticks": float(np.mean(markouts)) / TICK,
        "uncond_markout_ticks": float(np.mean(uncond_markouts)) / TICK,
        "adverse_prob": adverse / max(fills, 1),
        "uncond_down_prob": uncond_down / N_TAGGED,
    }


def main() -> None:
    terciles, slope, mids, spread = spread_vs_ofi(RNG_SEED)
    study = tagged_order_study(RNG_SEED + 1)

    checks = {
        "spread_increases_with_ofi": terciles[0] < terciles[2] and slope > 0.0,
        "fill_prob_interior": 0.05 < study["fill_prob"] < 0.99,
        "fills_adversely_selected":
            study["filled_markout_ticks"] < study["uncond_markout_ticks"],
        "adverse_prob_exceeds_unconditional":
            study["adverse_prob"] > study["uncond_down_prob"],
        "mid_moves_exist": float(np.std(mids)) > 0.0,
    }

    print("RESULTS")
    print(f"n_events={N_EVENTS}")
    print(f"spread_ticks_low_ofi={terciles[0]:.3f}")
    print(f"spread_ticks_mid_ofi={terciles[1]:.3f}")
    print(f"spread_ticks_high_ofi={terciles[2]:.3f}")
    print(f"spread_ofi_ols_slope={slope:.3f}")
    print(f"fill_prob={study['fill_prob']:.3f}")
    print(f"filled_markout_ticks={study['filled_markout_ticks']:.3f}")
    print(f"uncond_markout_ticks={study['uncond_markout_ticks']:.3f}")
    print(f"adverse_selection_cost_ticks="
          f"{study['uncond_markout_ticks'] - study['filled_markout_ticks']:.3f}")
    print(f"adverse_prob_given_fill={study['adverse_prob']:.3f}")
    print(f"uncond_down_prob={study['uncond_down_prob']:.3f}")
    for name, passed in checks.items():
        print(f"check_{name}={passed}")
    ok = all(checks.values())
    print(f"verdict={'MICROSTRUCTURE_MODEL_VALIDATED' if ok else 'MODEL_CHECKS_FAILED'}")


if __name__ == "__main__":
    main()
'''

ENTRY_LOB = {
    "schema_version": 2,
    "entry_kind": "correct",
    "expected_verdict": "MICROSTRUCTURE_MODEL_VALIDATED",
    "flaws": [],
    "static_checks": {
        "require_vectorized": True,
        "forbid_pandas_row_loops": True,
        "allowed_sequential_loops": [
            {
                "function": "spread_vs_ofi",
                "reason": "event-driven LOB simulation: every event conditions on current book state; inherently sequential",
                "max_iterations": 60000,
            },
            {
                "function": "tagged_order_study",
                "reason": "episode loop (2000) x event loop (600): queue position and fill state are path-dependent",
                "max_iterations": 1200000,
            },
        ],
    },
    "rlm": MINIMAL_RLM,
    "training_sequence": MINIMAL_TRAINING_SEQUENCE,
    "metadata": {
        "id": "qlm1-000011-microstructure-lob",
        "domain": "Market Microstructure / Limit Order Books",
        "complexity": 8,
        "tags": [
            "limit-order-book",
            "order-flow-imbalance",
            "queue-position",
            "adverse-selection",
            "markout",
            "poisson-arrivals",
            "flow-persistence",
            "event-driven-simulation",
        ],
    },
    "agent_thought_process": {
        "initial_analysis": (
            "Three microstructure claims must each be made falsifiable: (1) spreads widen "
            "when the book is imbalanced — measured as mean spread by |OFI| tercile plus "
            "an OLS slope, both of which must be positive; (2) a resting order's fill "
            "probability is governed by queue mechanics — requiring correct front-of-"
            "queue bookkeeping, since the naive implementation that cancels the order "
            "when the queue ahead empties deletes exactly the paths where fills happen; "
            "(3) fills are adversely selected — which is NOT automatic: with IID order "
            "flow, fills carry no information and the markout gap vanishes (verified "
            "during development), so Markov-persistent flow direction is a structural "
            "requirement of the model, not decoration."
        ),
        "tool_selection": (
            "Python REPL with numpy only, per the constraint. The event loop is a "
            "deliberate exception to vectorization: LOB state transitions are inherently "
            "sequential, declared in static_checks with iteration bounds. Measurement "
            "layers (terciles, OLS slope, markout means) are vectorized numpy."
        ),
        "recursive_delegation": (
            "The 2000 tagged-order episodes are independent given separate RNG streams "
            "and shard naturally to sub-agents; the parent must own the benchmark "
            "definition (immediate-fill counterfactual) so conditional and unconditional "
            "markouts are computed under identical dynamics — a benchmark mismatch "
            "across shards would silently manufacture or destroy the adverse-selection "
            "finding."
        ),
    },
    "research_corpus": {
        "hypothesis_formulation": (
            "H0 (all must hold for MICROSTRUCTURE_MODEL_VALIDATED): (a) spread is "
            "increasing in |OFI| (tercile monotonicity ends and positive OLS slope); "
            "(b) tagged-order fill probability is interior (0.05, 0.99) — neither "
            "degenerate certainty; (c) conditional-on-fill markout < unconditional "
            "immediate-fill markout (fills earn less than the naive half-spread "
            "expectation: adverse selection); (d) P(mid down at horizon | filled) > "
            "P(mid down | unconditional); (e) the mid actually moves (non-degenerate "
            "dynamics — the check that caught a dead parameterization during "
            "development). Failure of any check yields MODEL_CHECKS_FAILED."
        ),
        "data_engineering": (
            "No external data: the book IS the data-generating process. Event "
            "probabilities (market 0.30, limit 0.40, cancel 0.30 across sides) are "
            "balanced so queues deplete and prices move — an earlier draft with limit "
            "arrivals dominating produced ever-growing queues, a frozen mid, and a "
            "constant 1-tick spread, silently degenerating every downstream statistic; "
            "the mid_moves_exist check now guards that failure class permanently. "
            "Refill depth after a level break is geometric (mean 6); refreshed bests "
            "sit 2 ticks away with prob 0.35, generating the spread variation that the "
            "OFI regression explains."
        ),
        "methodology_justification": (
            "A best-quote-level model (prices, queue sizes, persistent flow) is the "
            "minimal structure that produces all three phenomena; full depth-ladder "
            "simulation would add realism but no additional lesson at this complexity. "
            "Markov flow persistence (rho=0.75) implements the empirical fact that "
            "market-order signs are strongly autocorrelated, and is the causal "
            "ingredient of adverse selection here — fills cluster in sell runs that "
            "continue. Adverse selection is measured by markouts (the industry-standard "
            "execution diagnostic) against an immediate-fill counterfactual from the "
            "SAME episode ensemble, so the comparison isolates conditioning-on-fill "
            "rather than differing dynamics. Queue-position mechanics follow "
            "price-time priority: sells consume ahead-units first, cancels thin the "
            "ahead-count proportionally, and a resting order becomes the level when "
            "alone — it cannot be broken past."
        ),
        "code_implementation": CODE_LOB,
        "statistical_validation": (
            "Deterministic values at seed 11: spread terciles ~1.86/1.93/2.05 ticks "
            "with OLS slope ~0.29 (spread widens with imbalance); fill_prob ~0.79; "
            "filled markout ~+0.30 ticks vs unconditional ~+0.48 ticks — an adverse-"
            "selection cost of ~0.18 ticks, with P(down|fill) ~0.34 vs "
            "unconditional ~0.14. Dose-response probes for the student: set RHO=0.5 "
            "(IID flow) and confirm the adverse-selection checks FAIL — the model "
            "correctly predicts no adverse selection without informed/persistent flow; "
            "raise RHO to 0.9 and confirm the markout gap widens monotonically."
        ),
        "risk_and_backtest_audit": (
            "For a market-making or execution strategy built on this model, the audit "
            "targets are: (a) the half-spread earned by passive fills is NOT the "
            "realized edge — subtract the markout-measured adverse-selection cost, "
            "which here consumes a large fraction of the half-spread; (b) queue-"
            "position value decays nonlinearly — back-of-queue fills concentrate in "
            "toxic sell runs, so fill quantity and fill quality anticorrelate; "
            "(c) simulation-to-market gaps: real books have hidden liquidity, "
            "cross-venue queues, and latency races the model omits, so any parameter "
            "fitted here must be re-estimated on real messages before deployment; "
            "(d) the episode design measures one order size (1 unit) — market impact "
            "of larger orders is absent by construction."
        ),
    },
    "adversarial_critique": {
        "potential_pitfalls": (
            "(1) The naive tagged-order implementation (cancel when the queue ahead "
            "empties) silently deletes the adverse-selection paths and reports "
            "near-zero adverse selection with high confidence — the subtlest bug class: "
            "wrong bookkeeping producing a plausible null. (2) Unbalanced event rates "
            "freeze the book and degenerate every statistic to a constant; the "
            "mid_moves_exist check exists because this happened. (3) Measuring "
            "adverse selection against a benchmark from a DIFFERENT path or parameter "
            "set manufactures spurious gaps. (4) With IID flow the model predicts no "
            "adverse selection — a student who finds adverse selection under IID flow "
            "has a bug, not a discovery. (5) OFI at the best level only is a noisy "
            "proxy for book pressure; deeper-book imbalance measures behave "
            "differently."
        ),
        "falsification_strategy": (
            "(a) Set RHO=0.5 (IID flow): the fills_adversely_selected and adverse_prob "
            "checks must fail while spread-OFI structure survives — cleanly separating "
            "which phenomena need flow persistence. (b) Set GAP2_PROB=0 and inside-"
            "improvement prob to 1.0 so the spread pins at 1 tick: the spread-OFI "
            "checks must fail while fill mechanics survive. (c) Double DEPTH_MEAN and "
            "confirm fill_prob falls (deeper refills mean longer queues ahead). "
            "(d) Rerun across 10 seeds and confirm every check verdict is stable — "
            "the checks are about structural properties, not seed luck."
        ),
        "limitations": (
            "Best-quote-level abstraction: no depth ladder, hidden orders, icebergs, "
            "or cross-venue fragmentation; queue-position estimates are exact only "
            "under strict price-time priority with visible depth. Unit-size orders "
            "throughout — no market impact or order-splitting. Event time, not "
            "calendar time: intensities are stationary, so intraday seasonality "
            "(open/close auctions, news bursts) is absent. Parameters are stylized, "
            "not fitted to message data; all magnitudes (markouts, fill rates) are "
            "model-relative, and only the qualitative structure transfers."
        ),
    },
    "agent_instructions": (
        "1. Run the long-path simulation; FIRST verify non-degeneracy (mid variance "
        "> 0, spread not constant) before reading any statistic. 2. Measure spread "
        "by |OFI| tercile and the OLS slope; require monotone ends and positive "
        "slope. 3. Run the tagged-order episodes with front-of-queue mechanics: "
        "sells consume ahead-units, the order becomes the level when alone, inside-"
        "quote improvements strand it. 4. Compute fill probability and per-fill "
        "markouts at the fixed horizon; compute the unconditional immediate-fill "
        "benchmark from the SAME episode ensemble. 5. Assert the adverse-selection "
        "ordering: filled markout < unconditional markout and P(down|fill) > "
        "P(down). 6. Falsification battery: RHO=0.5 must kill adverse selection; "
        "spread-pinning must kill the OFI relation; each check must fail exactly "
        "when its mechanism is removed. 7. Emit MICROSTRUCTURE_MODEL_VALIDATED only "
        "if all checks pass, and state the model-relative-magnitudes caveat."
    ),
    "verification": {
        "timeout_seconds": 240,
        "must_print": [
            "RESULTS",
            "spread_ofi_ols_slope=",
            "fill_prob=",
            "adverse_selection_cost_ticks=",
            "check_spread_increases_with_ofi=True",
            "check_fills_adversely_selected=True",
            "check_adverse_prob_exceeds_unconditional=True",
            "verdict=MICROSTRUCTURE_MODEL_VALIDATED",
        ],
        "forbid_nan_in_stdout": True,
        "required_packages": ["numpy"],
    },
}

ENTRIES = {
    "001_sma_crossover.json": migrate_v1_to_v2(
        ENTRY_SMA,
        entry_kind="correct",
        expected_verdict="FAIL_TO_REJECT_H0",
        static_checks={
            "require_vectorized": True,
            "forbid_pandas_row_loops": True,
            "allowed_sequential_loops": [
                {
                    "function": "simulate_prices",
                    "reason": "regime-length draws are sequential until the day budget is exhausted",
                    "max_iterations": 2520,
                }
            ],
        },
    ),
    "002_pairs_engle_granger.json": migrate_v1_to_v2(
        ENTRY_PAIRS,
        entry_kind="correct",
        expected_verdict="COINTEGRATED_TRADEABLE",
        static_checks={
            "require_vectorized": True,
            "forbid_pandas_row_loops": True,
            "allowed_sequential_loops": [
                {
                    "function": "simulate_pair",
                    "reason": "AR(1) spread recursion is inherently sequential",
                    "max_iterations": 1500,
                }
            ],
        },
    ),
    "003_garch_qlike_dm.json": migrate_v1_to_v2(
        ENTRY_GARCH,
        entry_kind="correct",
        expected_verdict="NO_SIGNIFICANT_EDGE",
        static_checks={
            "require_vectorized": True,
            "forbid_pandas_row_loops": True,
            "allowed_sequential_loops": [
                {
                    "function": "simulate_garch",
                    "reason": "GARCH variance recursion is inherently sequential",
                    "max_iterations": 3500,
                },
                {
                    "function": "garch_filter",
                    "reason": "conditional variance filter depends on sigma2[t-1]",
                    "max_iterations": 3000,
                },
                {
                    "function": "ewma_forecast",
                    "reason": "EWMA recursion depends on s2[t-1]",
                    "max_iterations": 3000,
                },
                {
                    "function": "diebold_mariano",
                    "reason": "HAC autocovariance sum over lags 1..DM_LAG",
                    "max_iterations": 20,
                },
            ],
        },
    ),
    "004_adversarial_lookahead_sma.json": ENTRY_LOOKAHEAD,
    "005_adversarial_p_hacking_sweep.json": ENTRY_PHACK,
    "006_rlm_environment_garch.json": ENTRY_RLM,
    "007_options_bsm_calibration.json": ENTRY_BSM,
    "008_adv_survivorship_bias.json": ENTRY_SURV,
    "009_risk_es_vs_var.json": ENTRY_ES,
    "010_rlm_portfolio_rebalance.json": ENTRY_RP,
    "011_microstructure_lob.json": ENTRY_LOB,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, entry in ENTRIES.items():
        path = OUT_DIR / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
