from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Union
from src.models.results import ExperimentRun


class RunStore:
    """Append-only JSONL journal for a single experiment session."""

    def __init__(self, journal_path: Union[str, Path]) -> None:
        self._path = Path(journal_path)
        self._entries: List[ExperimentRun] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._entries.append(ExperimentRun.model_validate_json(line))

    def append(self, entry: ExperimentRun) -> None:
        self._entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get_history(self) -> List[ExperimentRun]:
        return list(self._entries)

    def get_incumbent(self, higher_is_better: bool = True) -> Optional[ExperimentRun]:
        successful = [e for e in self._entries if e.result.status == "success"
                      and e.result.primary_metric is not None]
        if not successful:
            return None
        return max(successful, key=lambda e: (
            e.result.primary_metric if higher_is_better else -e.result.primary_metric
        ))

    def get_failed(self) -> List[ExperimentRun]:
        return [e for e in self._entries if e.result.status == "failed"]
