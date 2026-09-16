from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogger:
    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str = "run",
        console: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.console = console

        self.csv_path = self.log_dir / f"{experiment_name}_history.csv"
        self.json_path = self.log_dir / f"{experiment_name}_history.json"

        self.history: List[Dict[str, Any]] = []
        self._fieldnames: Optional[List[str]] = None

        self._csv_file = None
        self._csv_writer = None

    def log(self, epoch: int, metrics: Dict[str, Any]) -> None:
        row = {"epoch": epoch, **metrics}
        self.history.append(row)

        # Console
        if self.console:
            parts = [
                f"{key}: {value:.4f}"
                if isinstance(value, float)
                else f"{key}: {value}"
                for key, value in row.items()
            ]
            print(" | ".join(parts))

        # Initialize CSV writer on first log
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

            self._csv_file = open(
                self.csv_path,
                "w",
                newline="",
                encoding="utf-8",
            )

            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )

            self._csv_writer.writeheader()

        # Write row
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def save_json(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get_best(
        self,
        key: str = "val_accuracy",
        mode: str = "max",
    ) -> Dict[str, Any]:
        if not self.history:
            return {}

        if mode == "max":
            return max(
                self.history,
                key=lambda row: row.get(key, float("-inf")),
            )

        if mode == "min":
            return min(
                self.history,
                key=lambda row: row.get(key, float("inf")),
            )

        raise ValueError("mode must be 'min' or 'max'")

    def close(self) -> None:
        self.save_json()

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None