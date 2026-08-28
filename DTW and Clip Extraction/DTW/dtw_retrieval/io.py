from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from .columns import Columns


def list_parquet_files(root: Path) -> List[Path]:
    """Featured Parquet files are returned in name order."""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() == ".parquet" else []
    files = [
        path
        for path in root.rglob("*.parquet")
        if path.is_file() and not path.name.startswith(".")
    ]
    return sorted(files)


def build_manifest(files: Sequence[Path], cols: Columns) -> pd.DataFrame:
    """File paths and match identifiers are listed in one table."""
    rows = []
    for path in files:
        try:
            frame = pd.read_parquet(path, columns=[cols.match_id])
            if cols.match_id in frame.columns and not frame[cols.match_id].isna().all():
                match_id = str(frame[cols.match_id].iloc[0])
            else:
                match_id = path.stem
        except Exception:
            match_id = path.stem

        rows.append({"file": str(path), cols.match_id: match_id})

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class MatchInfo:
    match_id: str
    path: Path


class MatchStore:
    """Recently used match tables are retained in memory."""

    def __init__(self, max_items: int):
        self.max_items = max_items
        self._cache: "OrderedDict[str, pd.DataFrame]" = OrderedDict()

    def get(
        self,
        match_id: str,
        path: Path,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if match_id in self._cache:
            df = self._cache.pop(match_id)
            self._cache[match_id] = df
            return df

        df = pd.read_parquet(
            path,
            columns=list(columns) if columns is not None else None,
        )
        self._cache[match_id] = df

        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)

        return df
