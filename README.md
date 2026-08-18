# QLM 1 — Supervision Dataset Curation Pipeline for Chousorus 1

Teacher-side tooling that builds the training / evaluation / agent-behavior
corpus for **Chousorus 1**, a 100–150M parameter subquadratic quantitative
reasoning model trained from scratch.

**The Golden Rule:** everything in this repository runs at *dataset build
time* only. Chousorus 1 must never depend on a frontier model — or on this
pipeline — at inference time.

## Pipeline

```
author entries ──► gate 1: schema validation ──► gate 2: EXECUTION verification ──► curriculum builder
(tools/)           (qlm/schema.py)               (qlm/verify.py)                    (qlm/curriculum.py)
```

An entry reaches the training corpus only if:

1. **Schema gate** — it validates against `schema/entry.schema.json`
   (all five sections present, complexity ∈ [1,10], minimum-substance
   lengths on every narrative field).
2. **Execution gate** — its `research_corpus.code_implementation` is run in
   an isolated subprocess and must: exit 0 within the timeout, print every
   substring in `verification.must_print`, and emit **no NaN/Inf tokens**
   in stdout. Code that "looks right" but crashes or silently produces NaNs
   (e.g. `np.log(0)` in return calculations) is rejected — execution is the
   only ground truth accepted.

Admitted entries are then stable-sorted by `metadata.complexity` into an
ordered `curriculum.jsonl` plus three stage shards for curriculum learning:

| Shard | Complexity | Content |
|---|---|---|
| `stage1_foundational.jsonl` | 1–3 | backtest mechanics, frictions, bias hygiene |
| `stage2_intermediate.jsonl` | 4–7 | cointegration, OOS design, hypothesis testing |
| `stage3_research.jsonl` | 8–10 | MLE, forecast evaluation, numerical stability |

## Usage

```bash
python3 -m pip install -r requirements.txt

python3 tools/author_entries.py     # regenerate seed entry JSONs from source
python3 -m qlm.cli validate         # gate 1 only
python3 -m qlm.cli verify           # gate 2 only (executes all entry code)
python3 -m qlm.cli build            # full gate + write dist/ curriculum
python3 tests/test_pipeline.py      # smoke tests (also: python -m pytest tests/)
```

## Seed corpus (`data/entries/`)

Three exemplar entries spanning the curriculum, each fully self-contained
(seeded synthetic data generators — no network, no vendor data — so the
*methodology* is what the student learns) and each reaching a **different
calibrated verdict**, so the model never learns "always conclude success":

| Entry | Complexity | Study | Executed verdict |
|---|---|---|---|
| `001_sma_crossover` | 3 | SMA 20/100 trend rule, turnover costs, moving-block bootstrap | `FAIL_TO_REJECT_H0` — honest null result (bootstrap p ≈ 0.71) |
| `002_pairs_engle_granger` | 6 | Engle-Granger cointegration, frozen formation params, OOS z-score trading | `COINTEGRATED_TRADEABLE` — ADF p ≈ 0.006, γ̂ = 1.4145 vs true 1.4, half-life 17.5d vs true 15d |
| `003_garch_qlike_dm` | 8 | Hand-rolled GARCH(1,1) QMLE, walk-forward forecasts, QLIKE + Diebold-Mariano vs EWMA | `NO_SIGNIFICANT_EDGE` — even under a true GARCH DGP (DM p ≈ 0.26): an upper-bound calibration lesson |

Every entry follows the 16-stage cognitive pipeline (Deconstruct →
Hypothesize → … → Falsify → Conclude → Report) and carries all five
schema sections: `metadata`, `agent_thought_process` (thought-trace /
attention-sink CoT), `research_corpus`, `adversarial_critique`, and
`agent_instructions`, plus a machine-checked `verification` contract.

Entries are authored in `tools/author_entries.py` (code blocks kept as real
Python for reviewability) and serialized to JSON — edit there, regenerate,
re-gate.

## Generation notes for scaling the corpus

- **Temperature splitting** (API generation): low temp (0.2–0.4) for
  `code_implementation` / `statistical_validation`; higher (0.7–0.9) for
  `adversarial_critique` / `agent_thought_process` to diversify failure
  modes and reasoning paths.
- **Self-correction loop:** on an execution-gate failure, feed the
  traceback back to the teacher model for repair; keep only code that runs
  flawlessly (the gate in `qlm/verify.py` is the arbiter).
- **Verdict diversity:** monitor the distribution of conclusions; a corpus
  that always "finds alpha" teaches motivated reasoning.
- **Curriculum:** train on stage 1 first, introduce stages 2–3 as steps
  progress; `dist/manifest.json` records the exact ordering.

## Colab notebooks (`notebooks/`)

Seven unexecuted Colab notebooks for dataset collection, validation,
normalization, curriculum building, and RLM-trajectory export (run in order;
`00_setup` first in every fresh session):

| Notebook | Purpose |
|---|---|
| `00_setup` | deps, `/content/data` tree, CONFIG, shared `qlm_utils.py` |
| `01_load_qlm_v2_entries` | load + validate the six QLM v2 entries (no code execution) |
| `02_download_external_datasets` | loaders: Jane Street puzzles, QuantQA, QuantCodeEval, BizBench, FinMME (streamed subset), QuantBench (clone+inspect manifest), QuantNet-style synthetic multi-market adapter, Kaggle placeholder |
| `03_normalize_external_datasets` | unified SFT schema, robust field detection, unpaired/error routing |
| `04_load_qwen_breadth` | Qwen 3.8 Max breadth (file preferred, pasted JSONL fallback; sample embedded) |
| `05_build_unified_corpus_and_curriculum` | unified corpus + 3 curriculum stages + manifests |
| `06_export_rlm_trajectories` | QLM entries -> RLM trajectories; real stdout logs if present, marked templates otherwise |

Notebook sources live in `notebooks/src/*.cells.py` (cells separated by
`# --- CELL ---`); `python tools/build_notebooks.py` serializes them to
`.ipynb` with a static syntax check and never executes any cell.
The 25-row Qwen breadth seed batch is at `data/breadth/qwen38_breadth.jsonl`.
