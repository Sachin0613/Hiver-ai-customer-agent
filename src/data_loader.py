"""Load and inspect the local Kaggle Twitter customer-support files.

Column names are read from the CSV header. We do not hard-require a schema;
helpers only *map* well-known names when they actually appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Names used by the official Kaggle dataset page. Used only as lookup keys.
DOCUMENTED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


@dataclass(frozen=True)
class RawFileInfo:
    path: Path
    size_bytes: int


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative_or_absolute: str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(relative_or_absolute)
    if not path.is_absolute():
        path = root / path
    return path


def discover_data_files(raw_dir: Path) -> list[RawFileInfo]:
    """Find CSV / zip files under data/raw, including nested folders like twcs/."""
    if not raw_dir.exists():
        return []

    files: list[RawFileInfo] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".csv", ".zip", ".gz"}:
            files.append(RawFileInfo(path=path, size_bytes=path.stat().st_size))
    return files


def choose_main_csv(files: list[RawFileInfo]) -> Path | None:
    """Prefer the largest CSV (full corpus) over sample.csv."""
    csv_files = [item for item in files if item.path.suffix.lower() == ".csv"]
    if not csv_files:
        return None
    csv_files.sort(key=lambda item: item.size_bytes, reverse=True)
    return csv_files[0].path


def load_tweets_csv(csv_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load tweets. IDs stay as strings so leading zeros / large ints are safe."""
    return pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=True,
        nrows=nrows,
        low_memory=False,
    )


def column_map(columns: list[str]) -> dict[str, str]:
    """Map documented names to the actual header spelling (case-insensitive)."""
    lower_to_actual = {col.lower(): col for col in columns}
    mapping: dict[str, str] = {}
    for name in DOCUMENTED_COLUMNS:
        if name.lower() in lower_to_actual:
            mapping[name] = lower_to_actual[name.lower()]
    return mapping


def parse_inbound_series(series: pd.Series) -> pd.Series:
    """Normalize inbound flags that may be bool, 'True'/'False', or 0/1."""
    normalized = series.astype(str).str.strip().str.lower()
    true_like = {"true", "1", "yes", "t"}
    false_like = {"false", "0", "no", "f"}
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result = result.mask(normalized.isin(true_like), True)
    result = result.mask(normalized.isin(false_like), False)
    return result
