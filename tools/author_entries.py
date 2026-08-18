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
