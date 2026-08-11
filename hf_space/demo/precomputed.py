"""
precomputed.py
~~~~~~~~~~~~~~
Loads and indexes pre-computed pipeline results for instant gallery browsing.
Supports multiple datasets via an index.json registry.
"""

import json
from pathlib import Path


class DemoDataRegistry:
    """Loads and manages pre-computed evidence datasets from multiple sources."""

    def __init__(self, data_dir: str | Path | None = None):
        if data_dir is None:
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.index = {}
        self.datasets = {}  # Cache of loaded datasets
        self._load_index()

    def _load_index(self):
        """Load the dataset registry index."""
        index_path = self.data_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            # Fallback if index.json is missing but sample_results.json exists
            sample_path = self.data_dir / "sample_results.json"
            if sample_path.exists():
                self.index = {
                    "sample": {
                        "label": "Sample Data",
                        "file": "sample_results.json",
                        "n": 20
                    }
                }

    def get_available_datasets(self) -> list[tuple[str, str]]:
        """Return list of (dataset_id, label)."""
        return [(k, v["label"]) for k, v in self.index.items()]

    def load_dataset(self, dataset_id: str) -> list[dict]:
        """Lazy load a dataset by ID."""
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]

        if dataset_id not in self.index:
            return []

        file_path = self.data_dir / self.index[dataset_id]["file"]
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.datasets[dataset_id] = data
                return data
        return []

    def get_queries(self, dataset_id: str) -> list[tuple[int, str, str]]:
        """
        Get list of available queries for a dataset.
        Returns: list of (index, query_entity, target)
        """
        data = self.load_dataset(dataset_id)
        queries = []
        for i, item in enumerate(data):
            q = item.get("query_entity", "")
            t = item.get("target", "")
            if not t:
                # Fallback if target missing but triple exists
                triple = item.get("triple", [])
                if len(triple) == 3:
                    t = triple[2] if item.get("type") == "predicted_tail" else triple[0]
            queries.append((i, q, t))
        return queries

    def get_by_index(self, dataset_id: str, idx: int) -> dict | None:
        """Get a specific result by index within a dataset."""
        data = self.load_dataset(dataset_id)
        if 0 <= idx < len(data):
            return data[idx]
        return None
