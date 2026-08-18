# Fable 5 — generation instruction for schema v2 entries

1. Start from `schema/entry.v2.skeleton.json`; keep all five pedagogical sections.
2. Set `schema_version: 2` and exactly one `entry_kind`.
3. `correct`: honest methodology; `flaws: []`; `expected_verdict` = what the code's
   audit actually prints (nulls like FAIL_TO_REJECT_H0 are first-class outcomes).
4. `adversarial`: code must run cleanly and look attractive; plant 1-3 subtle flaws,
   declare EVERY one in `flaws` (type/severity/location/detection/corrective_action);
   `expected_verdict` starts with REJECTED_ or FLAWED_ — the audit, not the naive
   result, is the lesson.
5. `rlm_environment`: non-null `rlm.environment_class`, >=1 action; teach
   inspect -> act -> observe -> verify loops, not final answers.
6. Code: seeded, deterministic, self-contained, no network, exits 0, no NaN/Inf in
   stdout, prints a RESULTS block ending in `verdict=<expected_verdict>`.
7. Declare every mathematically necessary sequential loop in
   `static_checks.allowed_sequential_loops` (function, reason, max_iterations).
8. Give every entry a stable `metadata.id` (`qlm1-NNNNNN-slug`) and honest complexity.
9. Keep verdict diversity across the corpus; never teach "always finds alpha".
10. Gate everything through `python -m qlm.cli build` before admission.
