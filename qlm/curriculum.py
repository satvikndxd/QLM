"""Curriculum builder: order verified entries for staged training.

Chousorus 1 is trained with curriculum learning: entries are sorted by
``metadata.complexity`` (stable-sorted, ties broken by entry id) and
emitted both as a single ordered JSONL file and as per-stage shards:

  stage 1: complexity 1-3   (foundational mechanics)
  stage 2: complexity 4-7   (intermediate research workflows)
  stage 3: complexity 8-10  (research-grade modeling)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGES: dict[str, tuple[int, int]] = {
    "stage1_foundational": (1, 3),
    "stage2_intermediate": (4, 7),
    "stage3_research": (8, 10),
}


@dataclass
class CurriculumReport:
    total: int
    per_stage: dict[str, int]
    output_dir: Path


def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    meta = entry["metadata"]
    return (meta["complexity"], meta.get("id", meta["domain"]))


def build_curriculum(entries: list[dict[str, Any]], output_dir: Path) -> CurriculumReport:
    """Write curriculum.jsonl (globally ordered) plus per-stage shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=_entry_sort_key)

    with (output_dir / "curriculum.jsonl").open("w", encoding="utf-8") as fh:
        for entry in ordered:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    per_stage: dict[str, int] = {}
    for stage, (lo, hi) in STAGES.items():
        stage_entries = [e for e in ordered if lo <= e["metadata"]["complexity"] <= hi]
        per_stage[stage] = len(stage_entries)
        with (output_dir / f"{stage}.jsonl").open("w", encoding="utf-8") as fh:
            for entry in stage_entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    manifest = {
        "total_entries": len(ordered),
        "stages": {
            stage: {"complexity_range": list(STAGES[stage]), "count": per_stage[stage]}
            for stage in STAGES
        },
        "ordering": [
            {"id": e["metadata"].get("id"), "complexity": e["metadata"]["complexity"]}
            for e in ordered
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return CurriculumReport(total=len(ordered), per_stage=per_stage, output_dir=output_dir)
