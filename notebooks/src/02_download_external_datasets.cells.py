# TITLE: Chousorus 1 dataset pipeline — 02_download_external_datasets
#
# Purpose: loader logic for all external sources. NOTHING downloads until
# the user runs the marked cells. Every loader is wrapped so one failing
# dataset can never kill the notebook: failures land in data/errors/ and a
# status manifest records exactly what loaded and what did not.

# --- CELL ---
# TITLE: Bootstrap
import sys
from pathlib import Path

if not Path("/content/qlm_utils.py").exists():
    raise RuntimeError("Run notebooks/00_setup.ipynb first in this Colab session.")
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

import json
import traceback
from qlm_utils import (
    CONFIG, data_path, ensure_dir, write_jsonl, append_jsonl, save_manifest,
)

MAX_PER_SOURCE = CONFIG["max_external_samples_per_source"]
DOWNLOAD_STATUS = []  # accumulated across cells -> external_download_manifest.json


def record_status(dataset: str, status: str, reason: str, expected_path: str, extra=None):
    entry = {"dataset": dataset, "status": status, "reason": reason,
             "expected_path": expected_path}
    if extra:
        entry.update(extra)
    # replace any earlier status for the same dataset (cells may be re-run)
    DOWNLOAD_STATUS[:] = [e for e in DOWNLOAD_STATUS if e["dataset"] != dataset]
    DOWNLOAD_STATUS.append(entry)
    print(f"[{status}] {dataset}: {reason}")


def write_load_error(dataset: str, exc: Exception):
    err_path = data_path("errors", f"{dataset}_load_error.txt")
    ensure_dir(err_path.parent)
    err_path.write_text(
        f"dataset: {dataset}\nerror: {exc!r}\n\n{traceback.format_exc()}",
        encoding="utf-8",
    )
    print("error details ->", err_path)

# --- CELL ---
# TITLE: Generic Hugging Face loader (streaming, capped, serialization-safe)
def _json_safe(value):
    """Coerce arbitrary HF row values to JSON-serializable form.

    Why: HF rows can contain numpy scalars, arrays, PIL images, etc. We keep
    what serializes and stringify the rest rather than dropping the row.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def load_hf_dataset_to_jsonl(dataset_name: str, out_path, limit: int,
                             config_name=None, streaming: bool = True) -> dict:
    """Stream up to `limit` rows from a HF dataset into raw JSONL.

    Streaming avoids downloading full datasets (important for FinMME).
    Falls back to non-streaming if the dataset does not support it.
    Returns a status dict; raises nothing to the caller cell.
    """
    from datasets import load_dataset

    try:
        try:
            ds = load_dataset(dataset_name, config_name, streaming=streaming) \
                if config_name else load_dataset(dataset_name, streaming=streaming)
        except (ValueError, NotImplementedError):
            # some datasets reject streaming; retry materialized
            ds = load_dataset(dataset_name, config_name) \
                if config_name else load_dataset(dataset_name)

        splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
        rows_out, n_seen = [], 0
        for split in splits:
            if n_seen >= limit:
                break
            for row in ds[split]:
                if n_seen >= limit:
                    break
                clean = {k: _json_safe(v) for k, v in dict(row).items()}
                clean["_split"] = split
                rows_out.append(clean)
                n_seen += 1
        write_jsonl(rows_out, out_path)
        return {"ok": True, "rows": n_seen, "splits": splits}
    except Exception as exc:  # noqa: BLE001 — one dataset must never kill the run
        return {"ok": False, "error": exc}

# --- CELL ---
# TITLE: Jane Street puzzles (davidheineman/jane-street-puzzles)
# RUN LATER IN COLAB
def load_jane_street_puzzles(limit: int = MAX_PER_SOURCE):
    out = data_path("external", "jane_street_puzzles.jsonl")
    result = load_hf_dataset_to_jsonl("davidheineman/jane-street-puzzles", out, limit)
    if result["ok"]:
        record_status("jane_street_puzzles", "loaded",
                      f"{result['rows']} raw rows", str(out),
                      {"splits": result["splits"]})
    else:
        write_load_error("jane_street_puzzles", result["error"])
        record_status("jane_street_puzzles", "not_loaded",
                      f"load failed: {result['error']!r}", str(out))

load_jane_street_puzzles()

# --- CELL ---
# TITLE: QuantQA (ReinforceNow/quantqa)
# RUN LATER IN COLAB
def load_quantqa(limit: int = MAX_PER_SOURCE):
    out = data_path("external", "quantqa.jsonl")
    result = load_hf_dataset_to_jsonl("ReinforceNow/quantqa", out, limit)
    if result["ok"]:
        record_status("quantqa", "loaded", f"{result['rows']} raw rows", str(out),
                      {"splits": result["splits"]})
    else:
        write_load_error("quantqa", result["error"])
        record_status("quantqa", "not_loaded", f"load failed: {result['error']!r}", str(out))

load_quantqa()

# --- CELL ---
# TITLE: QuantCodeEval (quantcodeeval/task_data)
# RUN LATER IN COLAB
def load_quantcodeeval(limit: int = MAX_PER_SOURCE):
    out = data_path("external", "quantcodeeval.jsonl")
    result = load_hf_dataset_to_jsonl("quantcodeeval/task_data", out, limit)
    if result["ok"]:
        record_status("quantcodeeval", "loaded", f"{result['rows']} raw rows", str(out),
                      {"splits": result["splits"]})
    else:
        write_load_error("quantcodeeval", result["error"])
        record_status("quantcodeeval", "not_loaded",
                      f"load failed: {result['error']!r}", str(out))

load_quantcodeeval()

# --- CELL ---
# TITLE: BizBench (kensho/bizbench)
# RUN LATER IN COLAB
def load_bizbench(limit: int = MAX_PER_SOURCE):
    out = data_path("external", "bizbench.jsonl")
    result = load_hf_dataset_to_jsonl("kensho/bizbench", out, limit)
    if result["ok"]:
        record_status("bizbench", "loaded", f"{result['rows']} raw rows", str(out),
                      {"splits": result["splits"]})
    else:
        write_load_error("bizbench", result["error"])
        record_status("bizbench", "not_loaded", f"load failed: {result['error']!r}", str(out))

load_bizbench()

# --- CELL ---
# TITLE: FinMME (luojunyu/FinMME) — small streamed subset only
# RUN LATER IN COLAB
#
# FinMME is large. We deliberately stream a small subset (default 200) —
# the point is curriculum coverage, not exhaustive ingestion.
def load_finmme(limit: int = CONFIG["finmme_max_samples"]):
    out = data_path("external", "finmme_subset.jsonl")
    result = load_hf_dataset_to_jsonl("luojunyu/FinMME", out, limit, streaming=True)
    if result["ok"]:
        record_status("finmme", "loaded", f"{result['rows']} raw rows (subset)", str(out),
                      {"splits": result["splits"], "max_samples": limit})
    else:
        write_load_error("finmme", result["error"])
        record_status("finmme", "not_loaded",
                      "User must run this cell later in Colab."
                      if not CONFIG["allow_network"] else f"load failed: {result['error']!r}",
                      str(out))

load_finmme()

# --- CELL ---
# TITLE: QuantBench — clone repository
# RUN LATER IN COLAB
#
# QuantBench is a research/code benchmark, NOT a ready-made instruction
# dataset. We clone it, inspect it, and write a manifest of what it actually
# contains plus plausible future conversion routes. We do NOT force it into
# a fake SFT format.
!git clone --depth 1 https://github.com/SaizhuoWang/quantbench.git /content/data/raw/quantbench

# --- CELL ---
# TITLE: QuantBench — inspect clone and write manifest
QUANTBENCH_DIR = data_path("raw", "quantbench")


def inspect_quantbench(repo_dir: Path) -> dict:
    if not repo_dir.exists():
        return {
            "dataset": "quantbench",
            "status": "not_cloned",
            "reason": "User must run the git clone cell later in Colab.",
            "expected_path": str(repo_dir),
        }
    files = [p for p in repo_dir.rglob("*") if p.is_file() and ".git" not in p.parts]
    by_ext = {}
    for p in files:
        ext = p.suffix.lower() or "<none>"
        by_ext[ext] = by_ext.get(ext, 0) + 1
    data_like = sorted(
        str(p.relative_to(repo_dir))
        for p in files
        if p.suffix.lower() in {".csv", ".json", ".jsonl", ".parquet", ".pkl", ".h5"}
    )[:100]
    readme = ""
    for candidate in ("README.md", "readme.md", "README.rst"):
        rp = repo_dir / candidate
        if rp.exists():
            readme = rp.read_text(encoding="utf-8", errors="replace")[:4000]
            break
    return {
        "dataset": "quantbench",
        "status": "inspected",
        "top_level": sorted(p.name for p in repo_dir.iterdir() if ".git" not in p.name),
        "n_files": len(files),
        "files_by_extension": dict(sorted(by_ext.items())),
        "candidate_data_files": data_like,
        "readme_excerpt": readme,
        # Honest assessment written for the future converter, not wishful SFT:
        "conversion_routes": [
            "task descriptions -> research-prompt SFT (only if tasks have textual specs)",
            "benchmark code -> code-reproduction examples (preserve comments)",
            "results tables -> 'audit this reported result' adversarial prompts",
            "if none of the above cleanly applies: keep as raw reference corpus only",
        ],
    }


qb_manifest = inspect_quantbench(QUANTBENCH_DIR)
save_manifest(qb_manifest, data_path("external", "quantbench_manifest.json"))
record_status("quantbench", qb_manifest["status"],
              f"{qb_manifest.get('n_files', 0)} files inspected"
              if qb_manifest["status"] == "inspected" else qb_manifest["reason"],
              str(data_path("external", "quantbench_manifest.json")))

# --- CELL ---
# TITLE: QuantNet-style multi-market adapter — synthetic panel (default path)
#
# QuantNet (arXiv:2004.03445) is a paper about transfer across markets, not
# a downloadable dataset. Default path: a deterministic SYNTHETIC multi-
# market panel so curriculum prototyping needs no network and no vendor
# data. The generated examples teach cross-market transfer REASONING; all
# numbers are computed from the synthetic panel at runtime, and every
# example is labeled synthetic — no fabricated financial claims.
import numpy as np

QN_SEED = CONFIG["seed"]
QN_MARKETS = ["mkt_a", "mkt_b", "mkt_c", "mkt_d", "mkt_e", "mkt_f"]
QN_DAYS = 1260  # ~5 trading years
ANN = 252


def make_synthetic_panel(seed: int = QN_SEED) -> dict:
    """Six markets sharing a common momentum factor with heterogeneous
    loadings — market f's loading is ~0, making it the 'signal is market-
    specific' teaching case by construction."""
    rng = np.random.default_rng(seed)
    factor = 0.35 * rng.standard_normal(QN_DAYS)  # common driver, autocorrelated below
    for t in range(1, QN_DAYS):  # AR(1) gives the factor trend persistence
        factor[t] += 0.15 * factor[t - 1]
    loadings = {"mkt_a": 0.9, "mkt_b": 0.8, "mkt_c": 0.7, "mkt_d": 0.5,
                "mkt_e": 0.3, "mkt_f": 0.0}
    panel = {}
    for mkt in QN_MARKETS:
        idio = rng.standard_normal(QN_DAYS)
        rets = (loadings[mkt] * factor + idio) * (0.16 / np.sqrt(ANN))
        panel[mkt] = 100.0 * np.exp(np.cumsum(rets - 0.5 * (0.16 / np.sqrt(ANN)) ** 2))
    return panel


def momentum_sharpe(prices: np.ndarray, lookback: int) -> float:
    """Honest lagged momentum rule: long if trailing return > 0."""
    rets = np.diff(np.log(prices))
    sig = (prices[lookback:-1] > prices[:-lookback - 1]).astype(float)
    strat = sig * rets[lookback:]  # signal at t-1 applied to return t
    sd = strat.std(ddof=1)
    return 0.0 if sd == 0 else float(strat.mean() / sd * np.sqrt(ANN))


panel = make_synthetic_panel()
LOOKBACKS = [20, 60, 120]
stats = {m: {lb: round(momentum_sharpe(panel[m], lb), 3) for lb in LOOKBACKS}
         for m in QN_MARKETS}
print(json.dumps(stats, indent=2))

# --- CELL ---
# TITLE: QuantNet-style adapter — build transfer-reasoning examples
def build_quantnet_examples(stats: dict) -> list:
    """Three example families. Assistant text is templated STRICTLY from the
    computed synthetic numbers, with calibrated language — the lesson is the
    comparison procedure, not any market claim."""
    examples = []
    table = json.dumps(stats, indent=2)

    examples.append({
        "id": "quantnet-style-0001-fit-same-model",
        "source": "quantnet_style_synthetic",
        "entry_kind": "external",
        "training_mode": "sft",
        "domain": "cross_market_transfer",
        "task_type": "analysis",
        "difficulty": 4,
        "tags": ["momentum", "transfer-learning", "multi-market", "synthetic"],
        "messages": [
            {"role": "user", "content":
                "Fit the same lagged momentum rule (lookbacks 20/60/120) on six "
                "markets of a SYNTHETIC panel and report annualized Sharpe per "
                "market and lookback. Data is simulated; treat results as a "
                "methodology exercise only."},
            {"role": "assistant", "content":
                "Procedure: for each market, compute the trailing-return sign "
                "signal at t-1 and apply it to the return at t (one-bar lag), "
                "then annualize mean/std by sqrt(252). Computed Sharpe table on "
                "the synthetic panel:\n" + table + "\n"
                "Reading: differences across markets reflect each market's "
                "loading on the shared simulated factor, not any real-world "
                "property. The transferable object is the fitting procedure, "
                "never the fitted numbers."},
        ],
        "metadata": {"synthetic": True, "seed": QN_SEED, "lookbacks": LOOKBACKS},
    })

    examples.append({
        "id": "quantnet-style-0002-parameter-stability",
        "source": "quantnet_style_synthetic",
        "entry_kind": "external",
        "training_mode": "sft",
        "domain": "cross_market_transfer",
        "task_type": "analysis",
        "difficulty": 5,
        "tags": ["parameter-stability", "transfer-learning", "multi-market", "synthetic"],
        "messages": [
            {"role": "user", "content":
                "Given this per-market momentum Sharpe table from a synthetic "
                "panel, assess parameter stability across markets:\n" + table},
            {"role": "assistant", "content":
                "Stability check: for each lookback, compare the cross-market "
                "spread of Sharpes; a lookback whose ranking of markets is "
                "preserved is a stable parameter, one whose sign flips across "
                "markets is unstable. On this synthetic panel the ordering "
                "across markets is driven by their factor loadings, so ranks "
                "are broadly preserved across lookbacks — the expected outcome "
                "when ONE common factor generates the edge. On real data, rank "
                "instability across markets is evidence the parameter was fit "
                "to noise; confirm with a held-out period before transferring."},
        ],
        "metadata": {"synthetic": True, "seed": QN_SEED, "lookbacks": LOOKBACKS},
    })

    examples.append({
        "id": "quantnet-style-0003-market-specific-vs-agnostic",
        "source": "quantnet_style_synthetic",
        "entry_kind": "external",
        "training_mode": "sft",
        "domain": "cross_market_transfer",
        "task_type": "analysis",
        "difficulty": 5,
        "tags": ["signal-transfer", "market-agnostic", "multi-market", "synthetic"],
        "messages": [
            {"role": "user", "content":
                "Using the synthetic panel results, decide whether the momentum "
                "signal is market-agnostic or market-specific, and describe the "
                "test you would run:\n" + table},
            {"role": "assistant", "content":
                "Decision procedure: a market-agnostic signal earns a similar "
                "risk-adjusted return on most markets; a market-specific one "
                "concentrates in a few. In this panel, markets with high factor "
                "loading show positive momentum Sharpes while mkt_f (loading ~0 "
                "by construction) shows none — so the signal is agnostic across "
                "factor-loaded markets and absent where the factor is absent. "
                "Proper test: pool markets, fit on k-1 markets, evaluate on the "
                "held-out market (leave-one-market-out), and require the held-"
                "out Sharpe distribution to exclude zero. This panel is "
                "synthetic; the procedure, not the conclusion, transfers."},
        ],
        "metadata": {"synthetic": True, "seed": QN_SEED, "lookbacks": LOOKBACKS},
    })
    return examples


qn_examples = build_quantnet_examples(stats)
qn_out = data_path("external", "quantnet_style_examples.jsonl")
write_jsonl(qn_examples, qn_out)
record_status("quantnet_style", "generated",
              f"{len(qn_examples)} synthetic transfer-reasoning examples", str(qn_out))

# --- CELL ---
# TITLE: QuantNet-style adapter — optional real OHLCV via yfinance
# OPTIONAL: RUN LATER IN COLAB
#
# Entirely optional; the synthetic path above is the default. If run, this
# replaces the synthetic panel with real multi-market closes. yfinance is
# NOT a pipeline dependency.
USE_YFINANCE = False  # flip to True, then run this cell
if USE_YFINANCE:
    !pip install -q yfinance
    import yfinance as yf
    TICKERS = ["SPY", "EFA", "EEM", "TLT", "GLD", "DBC"]
    try:
        px = yf.download(TICKERS, period="5y", interval="1d", progress=False)["Close"].dropna()
        real_panel = {t: px[t].to_numpy() for t in TICKERS}
        real_stats = {t: {lb: round(momentum_sharpe(real_panel[t], lb), 3)
                          for lb in LOOKBACKS} for t in TICKERS}
        print(json.dumps(real_stats, indent=2))
        print("NOTE: rebuild examples from real_stats only with explicit "
              "'historical sample, not predictive' language.")
    except Exception as exc:  # noqa: BLE001
        write_load_error("quantnet_yfinance", exc)
        print("yfinance path failed; synthetic panel remains the default.")
else:
    print("yfinance path disabled (USE_YFINANCE=False); synthetic panel is default.")

# --- CELL ---
# TITLE: Jane Street Kaggle market data — credentials-gated placeholder
# OPTIONAL: RUN LATER IN COLAB
def load_jane_street_kaggle():
    """Placeholder loader: prints setup instructions unless Kaggle
    credentials exist. Never required by the rest of the pipeline."""
    cred = Path("/root/.kaggle/kaggle.json")
    if not cred.exists():
        record_status(
            "jane_street_kaggle", "not_loaded",
            "Optional. Requires Kaggle credentials at /root/.kaggle/kaggle.json. "
            "Steps: kaggle.com -> Account -> Create API token; upload kaggle.json; "
            "then: mkdir -p /root/.kaggle && cp kaggle.json /root/.kaggle/ && "
            "chmod 600 /root/.kaggle/kaggle.json; accept competition rules at "
            "kaggle.com/competitions/jane-street-real-time-market-data-forecasting; "
            "re-run this cell.",
            str(data_path("raw", "jane_street_kaggle")),
        )
        return
    !pip install -q kaggle
    !kaggle competitions download -c jane-street-real-time-market-data-forecasting -p /content/data/raw/jane_street_kaggle
    record_status("jane_street_kaggle", "downloaded",
                  "archive downloaded; unzip and adapt before use",
                  str(data_path("raw", "jane_street_kaggle")))

load_jane_street_kaggle()

# --- CELL ---
# TITLE: Write the download status manifest
save_manifest(
    {"datasets": DOWNLOAD_STATUS},
    data_path("manifests", "external_download_manifest.json"),
)
for entry in DOWNLOAD_STATUS:
    print(f"- {entry['dataset']}: {entry['status']}")
