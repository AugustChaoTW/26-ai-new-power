"""AI4I 2020 Predictive Maintenance Dataset acquisition and persistence.

Source: UCI Machine Learning Repository
  https://archive.ics.uci.edu/dataset/601
DOI: 10.24432/C5HS5C
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
Reference: S. Matzka, "Explainable Artificial Intelligence for Predictive
Maintenance Applications," AI4I 2020.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
COLUMNS = REQUIRED_COLUMNS + [
    "Machine failure",
    # Failure-mode flags (components of `Machine failure`): EDA only, never features.
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
INT_COLUMNS = [
    "UDI",
    "Rotational speed [rpm]",
    "Tool wear [min]",
    TARGET,
    *FAILURE_MODES,
]
FLOAT_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Torque [Nm]",
]
CATEGORICAL_COLUMNS = ["Product ID", "Type"]

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
DEFAULT_ZIP_MEMBER = "ai4i2020.csv"
DEFAULT_CSV_PATH = Path("data/ai4i2020.csv")


def _validate_columns(columns: pd.Index) -> None:
    normalized = [column.strip() for column in columns]
    missing = [column for column in COLUMNS if column not in normalized]
    if missing:
        raise ValueError(f"AI4I dataset is missing columns: {missing}")


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and enforce the documented dtypes."""
    frame = frame.rename(columns=lambda column: column.strip())
    _validate_columns(frame.columns)
    frame = frame[COLUMNS]
    for column in INT_COLUMNS:
        frame[column] = frame[column].astype(int)
    for column in FLOAT_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def load_ai4i_dataframe(path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the AI4I 2020 CSV with schema validation and dtype normalization."""
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return _clean(frame)


def save_dataset(df: pd.DataFrame, path: Path | str) -> Path:
    """Persist the dataset as CSV and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load a previously saved AI4I CSV with schema validation."""
    return load_ai4i_dataframe(path)


def download_dataset(
    url: str = UCI_ZIP_URL, dest_dir: Path | str = Path("data")
) -> Path:
    """Download the official UCI zip into ``dest_dir`` and return the CSV path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "ai4i2020.zip"
    urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.namelist()
        csv_members = [name for name in members if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No CSV file found in {url}: {members}")
        member = DEFAULT_ZIP_MEMBER if DEFAULT_ZIP_MEMBER in members else csv_members[0]
        csv_path = dest_dir / member
        csv_path.write_bytes(zf.read(member))
    archive_path.unlink(missing_ok=True)
    return csv_path


def ensure_dataset(
    path: Path | str = DEFAULT_CSV_PATH, dest_dir: Path | str = Path("data")
) -> Path:
    """Return ``path`` if the dataset exists locally; download it otherwise."""
    if Path(path).exists():
        return Path(path)
    return download_dataset(dest_dir=dest_dir)


if __name__ == "__main__":  # pragma: no cover - manual convenience entry point
    written = ensure_dataset()
    df = load_ai4i_dataframe(written)
    print(f"AI4I dataset: {written} shape={df.shape}")
    print("Source: UCI ML Repository (DOI 10.24432/C5HS5C, license CC BY 4.0)")
    print(f"Missing values: {df.isna().sum().sum()}")
