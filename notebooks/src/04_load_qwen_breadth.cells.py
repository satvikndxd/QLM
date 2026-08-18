# TITLE: Chousorus 1 dataset pipeline — 04_load_qwen_breadth
#
# Purpose: load the Qwen 3.8 Max breadth dataset from EITHER an uploaded
# file (preferred, if present) OR the pasted JSONL string below, normalize
# it into the breadth schema, and write processed output + manifest.
#
# Tolerant by design: a few bad rows go to the errors file; they never kill
# the notebook.

# --- CELL ---
# TITLE: Bootstrap
import sys
from pathlib import Path

if not Path("/content/qlm_utils.py").exists():
    raise RuntimeError("Run notebooks/00_setup.ipynb first in this Colab session.")
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

import json
import pandas as pd
from qlm_utils import (
    data_path, write_jsonl, append_jsonl, safe_load_json, save_manifest,
    stable_id, count_by,
)

QWEN_BREADTH_PATH = Path("/content/data/breadth/qwen38_breadth.jsonl")
ERRORS_PATH = data_path("errors", "qwen_breadth_errors.jsonl")

# --- CELL ---
# TITLE: Pasted JSONL fallback (used only when the file above is absent)
#
# The sample below is the user-provided Qwen 3.8 Max seed batch (25 rows).
# Paste additional/replacement rows between the triple quotes, one JSON
# object per line. Raw string (r-prefix) keeps \n escapes inside JSON
# strings intact.
QWEN_BREADTH_JSONL = r"""
{"id":"breadth-0001","source":"qwen3.8-max","domain":"python","task_type":"implement","difficulty":2,"tags":["python","dictionaries","edge-cases"],"messages":[{"role":"user","content":"Write a Python function get_path(data, keys, default=None) that safely gets a nested dictionary value."},{"role":"assistant","content":"def get_path(data, keys, default=None):\n    cur = data\n    for k in keys:\n        if not isinstance(cur, dict) or k not in cur:\n            return default\n        cur = cur[k]\n    return cur\n\nEdge case: if an intermediate value is not a dict, return default instead of raising."}]}
{"id":"breadth-0002","source":"qwen3.8-max","domain":"python","task_type":"debug","difficulty":2,"tags":["python","mutable-defaults","functions"],"messages":[{"role":"user","content":"Why is def append_item(item, bucket=[]) dangerous in Python?"},{"role":"assistant","content":"A default list is created once when the function is defined, so calls share the same list. Use bucket=None and create a new list inside:\n\ndef append_item(item, bucket=None):\n    if bucket is None:\n        bucket = []\n    bucket.append(item)\n    return bucket\n\nEdge case: default dicts and sets have the same shared-state problem."}]}
{"id":"breadth-0003","source":"qwen3.8-max","domain":"python","task_type":"implement","difficulty":3,"tags":["python","binary-search","algorithms"],"messages":[{"role":"user","content":"Implement binary search on a sorted list in Python without recursion."},{"role":"assistant","content":"def binary_search(items, target):\n    lo, hi = 0, len(items) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if items[mid] == target:\n            return mid\n        if items[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\nEdge case: use lo + (hi - lo) // 2 in languages where lo + hi can overflow; Python integers avoid this."}]}
{"id":"breadth-0004","source":"qwen3.8-max","domain":"python","task_type":"explain","difficulty":2,"tags":["python","generators","memory"],"messages":[{"role":"user","content":"When should you prefer a generator over a list in Python?"},{"role":"assistant","content":"Use a generator when you can produce items one at a time and do not need to store all of them in memory. Generators are lazy, so they can stream large sequences, but they can usually be iterated only once. Edge case: if you need random access, len, or multiple passes, materialize a list."}]}
{"id":"breadth-0005","source":"qwen3.8-max","domain":"python","task_type":"implement","difficulty":2,"tags":["python","chunking","lists"],"messages":[{"role":"user","content":"Write a Python function chunked(items, size) that splits a list into chunks of a given size."},{"role":"assistant","content":"def chunked(items, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [items[i:i + size] for i in range(0, len(items), size)]\n\nEdge case: the final chunk may be shorter than size."}]}
{"id":"breadth-0006","source":"qwen3.8-max","domain":"python","task_type":"implement","difficulty":3,"tags":["python","context-managers","cleanup"],"messages":[{"role":"user","content":"How do you create a context manager with a Python class?"},{"role":"assistant","content":"Implement __enter__ and __exit__. __enter__ returns the resource and __exit__ handles cleanup. Example:\n\nclass FileLike:\n    def __enter__(self):\n        self.resource = open('data.txt')\n        return self.resource\n\n    def __exit__(self, exc_type, exc, tb):\n        self.resource.close()\n\nEdge case: returning True from __exit__ suppresses exceptions; usually return None or False."}]}
{"id":"breadth-0007","source":"qwen3.8-max","domain":"numpy","task_type":"implement","difficulty":2,"tags":["numpy","percentiles","outliers"],"messages":[{"role":"user","content":"How do you clip values in a NumPy array to the 1st and 99th percentiles without loops?"},{"role":"assistant","content":"import numpy as np\n\nlow, high = np.percentile(x, [1, 99])\nx_clipped = np.clip(x, low, high)\n\nEdge case: if all values are identical, low and high are equal and the output is constant."}]}
{"id":"breadth-0008","source":"qwen3.8-max","domain":"pandas","task_type":"implement","difficulty":3,"tags":["pandas","time-series","reindex"],"messages":[{"role":"user","content":"How can you ensure a daily pandas time series has every calendar day present?"},{"role":"assistant","content":"import pandas as pd\n\ns = s.sort_index()\nfull_idx = pd.date_range(s.index.min(), s.index.max(), freq='D')\ns = s.reindex(full_idx)\n\nThis inserts missing days with NaN. Edge case: use freq='B' for business days if weekends should be excluded."}]}
{"id":"breadth-0009","source":"qwen3.8-max","domain":"data_engineering","task_type":"design","difficulty":3,"tags":["data-cleaning","winsorization","outliers"],"messages":[{"role":"user","content":"Design a robust rule for capping numeric outliers in a dataset without deleting rows."},{"role":"assistant","content":"Use winsorization: compute lower = quantile(alpha) and upper = quantile(1 - alpha), then cap values with np.clip or pandas clip. A common alpha is 0.01. Keep an indicator column if values were capped. Edge case: do not winsorize categorical variables or genuine extreme events that carry risk information."}]}
{"id":"breadth-0010","source":"qwen3.8-max","domain":"pandas","task_type":"explain","difficulty":3,"tags":["pandas","merge","fanout"],"messages":[{"role":"user","content":"Why can a pandas merge create more rows than either input table?"},{"role":"assistant","content":"A merge creates more rows when the join keys are not unique in at least one table. Each matching pair produces a row, so one-to-many or many-to-many joins fan out. Check duplicates with df[key].duplicated().sum() before merging. Edge case: duplicate keys plus NaNs can behave unexpectedly depending on the pandas version and join type."}]}
{"id":"breadth-0011","source":"qwen3.8-max","domain":"sql","task_type":"implement","difficulty":3,"tags":["sql","window-functions","top-n"],"messages":[{"role":"user","content":"Write SQL to return the top 3 orders by amount for each customer."},{"role":"assistant","content":"WITH ranked AS (\n  SELECT customer_id, order_id, amount,\n         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rn\n  FROM orders\n)\nSELECT customer_id, order_id, amount\nFROM ranked\nWHERE rn <= 3;\n\nUse DENSE_RANK if ties should all be included. Edge case: ROW_NUMBER drops ties arbitrarily."}]}
{"id":"breadth-0012","source":"qwen3.8-max","domain":"sql","task_type":"implement","difficulty":4,"tags":["sql","rolling-average","window-functions"],"messages":[{"role":"user","content":"Write SQL to compute a 7-day rolling average of daily revenue per product."},{"role":"assistant","content":"SELECT product_id, day, revenue,\n       AVG(revenue) OVER (\n         PARTITION BY product_id\n         ORDER BY day\n         ROWS BETWEEN 6 PRECEDING AND CURRENT ROW\n       ) AS rolling_7d_avg\nFROM daily_revenue;\n\nEdge case: if days are missing, ROWS counts existing rows, not calendar days; use RANGE or a calendar table for true elapsed days."}]}
{"id":"breadth-0013","source":"qwen3.8-max","domain":"sql","task_type":"implement","difficulty":3,"tags":["sql","deduplication","window-functions"],"messages":[{"role":"user","content":"Write SQL to keep only the most recent email event per user."},{"role":"assistant","content":"WITH ranked AS (\n  SELECT user_id, email_event, occurred_at,\n         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY occurred_at DESC) AS rn\n  FROM email_events\n)\nSELECT user_id, email_event, occurred_at\nFROM ranked\nWHERE rn = 1;\n\nEdge case: if two events have the same timestamp, add a tie-breaker such as event_id DESC."}]}
{"id":"breadth-0014","source":"qwen3.8-max","domain":"probability","task_type":"calculate","difficulty":4,"tags":["probability","bayes-theorem","base-rate"],"messages":[{"role":"user","content":"A disease affects 1% of people. A test has 95% sensitivity and 90% specificity. What is P(disease | positive)?"},{"role":"assistant","content":"Let D be disease and + be positive. P(D)=0.01, P(+|D)=0.95, P(+|not D)=0.10. Bayes: P(D|+) = 0.95*0.01 / (0.95*0.01 + 0.10*0.99) = 0.0095 / 0.1085, about 0.0876. Even with a positive test, probability is about 8.8% because the disease is rare."}]}
{"id":"breadth-0015","source":"qwen3.8-max","domain":"probability","task_type":"calculate","difficulty":2,"tags":["probability","expected-value","games"],"messages":[{"role":"user","content":"You roll a fair die. You win $10 if the result is 6, $2 if it is 5, and nothing otherwise. Entry costs $3. What is expected net profit?"},{"role":"assistant","content":"Gross EV = (1/6)*10 + (1/6)*2 + (4/6)*0 = 12/6 = 2. Net EV = 2 - 3 = -1. The game loses $1 on average per play. Edge case: expected value does not describe risk or short-run outcomes."}]}
{"id":"breadth-0016","source":"qwen3.8-max","domain":"statistics","task_type":"explain","difficulty":3,"tags":["statistics","confidence-intervals","inference"],"messages":[{"role":"user","content":"Explain what a 95% confidence interval means."},{"role":"assistant","content":"If the same sampling and estimation procedure were repeated many times, about 95% of the computed intervals would contain the true parameter. It is not correct to say there is a 95% probability that the fixed true parameter lies inside one observed interval under frequentist inference. Edge case: coverage can fail if assumptions such as independence or correct variance estimation are violated."}]}
{"id":"breadth-0017","source":"qwen3.8-max","domain":"statistics","task_type":"calculate","difficulty":3,"tags":["statistics","variance","linear-combinations"],"messages":[{"role":"user","content":"If X and Y are independent with Var(X)=4 and Var(Y)=9, what is Var(2X + 3Y)?"},{"role":"assistant","content":"For independent variables, Var(aX + bY) = a^2 Var(X) + b^2 Var(Y). Thus Var(2X + 3Y) = 4*4 + 9*9 = 16 + 81 = 97. If X and Y are correlated, add 2ab Cov(X,Y)."}]}
{"id":"breadth-0018","source":"qwen3.8-max","domain":"finance","task_type":"compare","difficulty":3,"tags":["finance","returns","time-series"],"messages":[{"role":"user","content":"Compare simple and log returns for portfolio analysis."},{"role":"assistant","content":"Simple return is P_t/P_{t-1} - 1 and aggregates correctly across assets in a portfolio. Log return is log(P_t/P_{t-1}) and adds over time, making it convenient for single-asset time-series modeling. Limitation: log returns are not additive across assets, so portfolio-level arithmetic usually uses simple returns. For small returns they are close."}]}
{"id":"breadth-0019","source":"qwen3.8-max","domain":"finance","task_type":"explain","difficulty":3,"tags":["finance","sharpe-ratio","performance"],"messages":[{"role":"user","content":"What are the main limitations of the Sharpe ratio?"},{"role":"assistant","content":"Sharpe = mean excess return divided by standard deviation. It assumes mean-variance preferences, treats upside and downside volatility equally, and can be misleading for skewed or fat-tailed returns. It is also sensitive to autocorrelation and sampling window. Edge case: annualization by sqrt(252) assumes IID daily returns."}]}
{"id":"breadth-0020","source":"qwen3.8-max","domain":"finance","task_type":"explain","difficulty":3,"tags":["finance","beta","capm"],"messages":[{"role":"user","content":"Define beta in the CAPM and explain its interpretation."},{"role":"assistant","content":"Beta = Cov(R_i, R_m) / Var(R_m). It measures an asset's sensitivity to market returns. Beta 1 means the asset tends to move with the market, beta greater than 1 means higher market sensitivity, and beta less than 1 means lower sensitivity. Limitation: beta is estimated from historical data and may not be stable across regimes."}]}
{"id":"breadth-0021","source":"qwen3.8-max","domain":"options","task_type":"explain","difficulty":4,"tags":["options","no-arbitrage","lower-bound"],"messages":[{"role":"user","content":"What is the lower bound for a European call on a non-dividend-paying stock?"},{"role":"assistant","content":"For a European call with spot S, strike K, rate r, maturity T, and no dividends, the lower bound is max(0, S - K*exp(-rT)). If the call trades below this bound, arbitrageurs could buy the call and short the stock to lock in riskless profit after financing. Limitation: real markets have borrowing costs, discrete dividends, and early exercise considerations."}]}
{"id":"breadth-0022","source":"qwen3.8-max","domain":"machine_learning","task_type":"explain","difficulty":2,"tags":["machine-learning","validation","data-leakage"],"messages":[{"role":"user","content":"Why should hyperparameter tuning use a validation set rather than the test set?"},{"role":"assistant","content":"The validation set guides model selection, while the test set estimates final generalization only once. Repeatedly evaluating on the test set leaks information and can produce optimistic performance estimates. Edge case: if data is time-dependent, use temporal splits instead of random splits."}]}
{"id":"breadth-0023","source":"qwen3.8-max","domain":"llm","task_type":"explain","difficulty":3,"tags":["llm","json","parsing"],"messages":[{"role":"user","content":"How can you make an LLM return reliable JSON for downstream parsing?"},{"role":"assistant","content":"Ask for a strict schema, provide an example, set low temperature, and instruct the model to output only JSON without markdown. After generation, parse with a JSON parser and retry or repair if parsing fails. Edge case: models can emit trailing commas or unescaped quotes, so validate with a schema before trusting fields."}]}
{"id":"breadth-0024","source":"qwen3.8-max","domain":"system_design","task_type":"design","difficulty":4,"tags":["system-design","idempotency","payments"],"messages":[{"role":"user","content":"Design an idempotent payment-processing API endpoint."},{"role":"assistant","content":"Require a client-generated idempotency key. Store key, request hash, status, and response. If the same key arrives again, return the stored response instead of processing again. If the request hash differs for the same key, reject the call. Edge case: use a database transaction or lock to prevent concurrent duplicate processing."}]}
{"id":"breadth-0025","source":"qwen3.8-max","domain":"general_reasoning","task_type":"calculate","difficulty":3,"tags":["estimation","fermi-problem","assumptions"],"messages":[{"role":"user","content":"Estimate how many piano tuners are needed in a city of 5 million people."},{"role":"assistant","content":"Assume 1 in 50 people owns a piano, giving 100,000 pianos. If each piano is tuned once per year, demand is 100,000 tunings per year. If one tuner can do 1,000 tunings per year, the city needs about 100 tuners. The key is stating assumptions and checking sensitivity, not the exact number."}]}
"""

# --- CELL ---
# TITLE: Choose the input: file preferred, pasted string as fallback
if QWEN_BREADTH_PATH.exists():
    raw_text = QWEN_BREADTH_PATH.read_text(encoding="utf-8")
    input_source = f"file:{QWEN_BREADTH_PATH}"
else:
    raw_text = QWEN_BREADTH_JSONL
    input_source = "pasted_string"
raw_lines = [ln for ln in raw_text.splitlines() if ln.strip()]
print(f"input: {input_source} ({len(raw_lines)} non-empty lines)")

# --- CELL ---
# TITLE: Normalize into the breadth schema
#
# Tolerance rules:
#  - messages present and well-formed -> use as-is
#  - else convert from prompt/response, input/output, question/answer
#  - neither -> row goes to the errors file; the notebook continues
FALLBACK_PAIRS = [("prompt", "response"), ("input", "output"), ("question", "answer")]


def normalize_breadth_row(row: dict, index: int):
    """Return (record, error). Exactly one is None."""
    msgs = row.get("messages")
    ok_msgs = (
        isinstance(msgs, list) and len(msgs) >= 2
        and all(isinstance(m, dict) and m.get("role") and str(m.get("content", "")).strip()
                for m in msgs)
    )
    if not ok_msgs:
        msgs = None
        for q_key, a_key in FALLBACK_PAIRS:
            q = str(row.get(q_key, "") or "").strip()
            a = str(row.get(a_key, "") or "").strip()
            if q and a:
                msgs = [{"role": "user", "content": q},
                        {"role": "assistant", "content": a}]
                break
    if msgs is None:
        return None, {"row_index": index, "error": "no messages and no known field pair",
                      "keys": sorted(row.keys())}

    rec_id = str(row.get("id", "") or "").strip() or f"breadth-{index + 1:04d}"
    record = {
        "id": rec_id,
        "source": "qwen38_breadth",          # canonical corpus source name
        "entry_kind": "breadth",
        "training_mode": "sft",
        "domain": str(row.get("domain", "") or "general").strip() or "general",
        "task_type": str(row.get("task_type", "") or "qa").strip() or "qa",
        "difficulty": int(row["difficulty"]) if isinstance(row.get("difficulty"), (int, float)) else 1,
        "tags": row.get("tags", []) if isinstance(row.get("tags"), list) else [],
        "messages": [{"role": m["role"], "content": str(m["content"])} for m in msgs],
        # provenance: keep the original source string from the row
        "metadata": {"original_source": str(row.get("source", "") or ""),
                     "input_source": input_source},
    }
    return record, None


records, n_errors = [], 0
seen_ids = set()
for i, line in enumerate(raw_lines):
    row = safe_load_json(line)
    if row is None:
        n_errors += 1
        append_jsonl({"row_index": i, "error": "invalid json", "raw": line[:2000]},
                     ERRORS_PATH)
        continue
    record, error = normalize_breadth_row(row, i)
    if error is not None:
        n_errors += 1
        append_jsonl(error, ERRORS_PATH)
        continue
    if record["id"] in seen_ids:  # duplicate ids get a deterministic suffix
        record["id"] = stable_id(record["id"], row)
    seen_ids.add(record["id"])
    records.append(record)

print(f"normalized: {len(records)}  errors: {n_errors}")

# --- CELL ---
# TITLE: Write processed output and manifest
out_path = data_path("processed", "qwen_breadth.jsonl")
write_jsonl(records, out_path)
print("wrote", out_path)

manifest = {
    "total_records": len(records),
    "total_errors": n_errors,
    "input_source": input_source,
    "by_domain": count_by(records, lambda r: r["domain"]),
    "by_task_type": count_by(records, lambda r: r["task_type"]),
    "by_difficulty": count_by(records, lambda r: r["difficulty"]),
}
save_manifest(manifest, data_path("manifests", "qwen_breadth_manifest.json"))

pd.DataFrame([{k: r[k] for k in ("id", "domain", "task_type", "difficulty")}
              for r in records]).groupby(["domain"]).agg(
    n=("id", "count"), mean_difficulty=("difficulty", "mean"))
