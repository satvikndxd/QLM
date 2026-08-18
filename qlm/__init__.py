"""QLM 1 — dataset curation pipeline for training Chousorus 1.

This package is the *teacher-side* tooling. It never runs at Chousorus 1
inference time (the Golden Rule): it validates, execution-verifies, and
curriculum-orders the supervision corpus before training.
"""

__version__ = "0.1.0"
