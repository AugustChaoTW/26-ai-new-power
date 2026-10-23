"""TDD tests for the AI4I 2020 data acquisition module (written before implementation)."""

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ai4i_analysis import data as ai4i_data
from ai4i_analysis.data import (
    COLUMNS,
    FAILURE_MODES,
    TARGET,
    download_dataset,
    load_ai4i_dataframe,
    load_dataset,
    save_dataset,
)

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]


def build_raw_frame(rows: int = 5, seed: int = 0) -> pd.DataFrame:
    """Build a small dataset that follows the real AI4I schema."""
    import numpy as np

    rng = np.random.RandomState(seed)
    types = ["L", "M", "H"]
    frame = pd.DataFrame(
        {
            "UDI": np.arange(1, rows + 1),
            "Product ID": [
                f"{types[i % 3]}{10000 + i}" for i in range(rows)
            ],
            "Type": [types[i % 3] for i in range(rows)],
            "Air temperature [K]": rng.normal(300, 2, rows).round(1),
            "Process temperature [K]": rng.normal(310, 1, rows).round(1),
            "Rotational speed [rpm]": rng.normal(1500, 150, rows).round().astype(int),
            "Torque [Nm]": np.clip(rng.normal(40, 10, rows), 0, None).round(1),
            "Tool wear [min]": rng.randint(0, 250, rows),
            "Machine failure": rng.randint(0, 2, rows),
            "TWF": rng.randint(0, 2, rows),
            "HDF": rng.randint(0, 2, rows),
            "PWF": rng.randint(0, 2, rows),
            "OSF": rng.randint(0, 2, rows),
            "RNF": rng.randint(0, 2, rows),
        }
    )
    return frame


# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------


def test_module_exposes_expected_schema():
    assert COLUMNS == EXPECTED_COLUMNS + list(FAILURE_MODES)
    assert len(COLUMNS) == 14
    assert TARGET == "Machine failure"
    assert list(FAILURE_MODES) == ["TWF", "HDF", "PWF", "OSF", "RNF"]
    assert "601" in ai4i_data.UCI_ZIP_URL


# ---------------------------------------------------------------------------
# Real dataset (downloaded during implementation)
# ---------------------------------------------------------------------------


def test_real_dataset_shape_and_columns():
    df = load_ai4i_dataframe()

    assert df.shape == (10000, 14)
    assert list(df.columns) == EXPECTED_COLUMNS + list(FAILURE_MODES)
    assert df.isna().sum().sum() == 0


def test_real_dataset_dtypes_and_value_ranges():
    df = load_ai4i_dataframe()

    for column in ["UDI", "Rotational speed [rpm]", "Tool wear [min]", TARGET] + list(
        FAILURE_MODES
    ):
        assert pd.api.types.is_integer_dtype(df[column]), column
    for column in [
        "Air temperature [K]",
        "Process temperature [K]",
        "Torque [Nm]",
    ]:
        assert pd.api.types.is_float_dtype(df[column]), column
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df["Air temperature [K]"].between(290, 310).all()
    assert df["Torque [Nm]"].min() >= 0.0
    assert df["UDI"].tolist() == list(range(1, 10001))
    assert set(df["Type"].unique()) == {"L", "M", "H"}


# ---------------------------------------------------------------------------
# Loading / saving behaviour
# ---------------------------------------------------------------------------


def test_missing_column_raises_value_error(tmp_path: Path):
    broken = build_raw_frame().drop(columns=["Torque [Nm]", "HDF"])
    broken_path = tmp_path / "broken.csv"
    broken.to_csv(broken_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        load_dataset(broken_path)
    assert "Torque [Nm]" in str(excinfo.value)


def test_save_and_load_roundtrip(tmp_path: Path):
    frame = build_raw_frame()
    target = tmp_path / "ai4i.csv"

    returned = save_dataset(frame, target)

    assert returned == target
    assert target.exists()
    pd.testing.assert_frame_equal(load_dataset(target), frame)


def test_load_ai4i_dataframe_accepts_explicit_path(tmp_path: Path):
    frame = build_raw_frame(rows=3)
    path = save_dataset(frame, tmp_path / "subset.csv")

    loaded = load_ai4i_dataframe(path)

    assert loaded.shape == (3, 14)


# ---------------------------------------------------------------------------
# Download + extraction
# ---------------------------------------------------------------------------


def make_zip_bytes(member_name: str = "ai4i2020.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(member_name, build_raw_frame().to_csv(index=False))
    return buffer.getvalue()


def test_download_dataset_extracts_csv(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_urlretrieve(url, filename):
        calls["url"] = url
        Path(filename).write_bytes(make_zip_bytes())
        return filename, None  # type: ignore[return-value]

    monkeypatch.setattr(
        "ai4i_analysis.data.urllib.request.urlretrieve", fake_urlretrieve
    )
    calls["url"] = ""

    result = download_dataset(dest_dir=tmp_path)

    assert result.name == "ai4i2020.csv"
    assert result.exists()
    assert "601" in calls["url"]
    assert load_dataset(result).shape == (5, 14)


def test_download_dataset_fails_when_zip_has_no_csv(monkeypatch, tmp_path: Path):
    def fake_urlretrieve(url, filename):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("readme.txt", "no data here")
        Path(filename).write_bytes(buffer.getvalue())
        return filename, None  # type: ignore[return-value]

    monkeypatch.setattr(
        "ai4i_analysis.data.urllib.request.urlretrieve", fake_urlretrieve
    )

    with pytest.raises(ValueError):
        download_dataset(dest_dir=tmp_path)
