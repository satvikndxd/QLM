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
        "must_print": ["RESULTS", "annualized_sharpe_net=", "bootstrap_pvalue_sharpe_gt_0=", "verdict="],
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
    print(f"formation_verdict={'COINTEGRATED_TRADEABLE' if tradeable else 'NOT_TRADEABLE'}")


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
        "must_print": ["RESULTS", "adf_pvalue=", "halflife_days=", "oos_sharpe_net=", "formation_verdict="],
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
        "must_print": ["RESULTS", "converged=True", "alpha_hat=", "dm_stat=", "dm_pvalue=", "verdict="],
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
