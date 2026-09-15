"""DataPulse dataset ingestion and loading utilities.

The loader accepts analysis-ready CSV/XLSX/XLS files and returns a
standardized pandas DataFrame plus lightweight metadata. It intentionally
DOES NOT perform EDA, feature engineering, schema mapping, validation,
or machine learning. Those stages belong to downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
SUPPORTED_EXCEL_EXTENSIONS = {".xlsx", ".xls"}


class DataLoadError(Exception):
    """Raised when a supported dataset cannot be loaded safely."""


@dataclass(frozen=True)
class DatasetMetadata:
    """Basic metadata produced immediately after ingestion."""

    file_name: str
    file_type: str
    file_size_mb: float
    rows: int
    columns: int
    column_names: tuple[str, ...]


def _validate_extension(file_name: str | Path) -> str:
    """Return a normalized extension or raise DataLoadError."""
    suffix = Path(str(file_name)).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
        raise DataLoadError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Supported formats: {supported}."
        )

    return suffix


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ingestion-level checks without performing EDA."""
    if df is None:
        raise DataLoadError("The dataset could not be loaded.")

    if df.empty:
        raise DataLoadError("The uploaded dataset is empty.")

    if len(df.columns) == 0:
        raise DataLoadError("The uploaded dataset contains no columns.")

    # Normalize column labels only enough to give downstream modules a stable
    # interface. We do not infer business meaning or map columns here.
    normalized_columns = [str(column).strip() for column in df.columns]

    if any(not column for column in normalized_columns):
        raise DataLoadError("The dataset contains one or more blank column names.")

    if len(set(normalized_columns)) != len(normalized_columns):
        duplicates = sorted(
            {
                column
                for column in normalized_columns
                if normalized_columns.count(column) > 1
            }
        )
        raise DataLoadError(
            "The dataset contains duplicate column names: "
            + ", ".join(duplicates)
        )

    df = df.copy()
    df.columns = normalized_columns
    return df


def _read_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read CSV while keeping the loader tolerant of common CSV encodings."""
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError:
        # Common fallback for exports containing non-UTF-8 characters.
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="latin-1")
    except Exception as exc:
        raise DataLoadError(f"Could not read the CSV dataset: {exc}") from exc


def _read_excel(source: str | Path | BinaryIO, suffix: str) -> pd.DataFrame:
    """Read XLSX/XLS with the appropriate pandas engine."""
    try:
        if suffix == ".xlsx":
            return pd.read_excel(source, engine="openpyxl")

        # .xls requires xlrd. Keeping the engine explicit makes failures clear
        # rather than silently choosing an incompatible reader.
        return pd.read_excel(source, engine="xlrd")
    except ImportError as exc:
        if suffix == ".xls":
            raise DataLoadError(
                "XLS files require the 'xlrd' package. Install it with "
                "'pip install xlrd'."
            ) from exc
        raise DataLoadError(
            "XLSX files require the 'openpyxl' package. Install it with "
            "'pip install openpyxl'."
        ) from exc
    except Exception as exc:
        raise DataLoadError(f"Could not read the Excel dataset: {exc}") from exc


def load_dataset(
    source: str | Path | BinaryIO,
    *,
    file_name: str | None = None,
    max_size_mb: int = 200,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load one analysis-ready CSV/XLSX/XLS dataset.

    Parameters
    ----------
    source:
        Local path or binary file-like object.
    file_name:
        Original upload name. Required when ``source`` is a file-like object.
    max_size_mb:
        Maximum accepted file size. DataPulse currently uses 200 MB.

    Returns
    -------
    tuple[pandas.DataFrame, DatasetMetadata]
        The loaded DataFrame and ingestion metadata.
    """
    source_name = file_name or getattr(source, "name", None)
    if not source_name:
        raise DataLoadError("A dataset file name is required.")

    suffix = _validate_extension(source_name)

    # Validate local-file size before loading. For uploaded file-like objects,
    # Streamlit already enforces the configured upload limit; their byte size
    # can still be checked when the object exposes getvalue().
    size_bytes: int | None = None

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"Dataset file not found: {path}")
        if not path.is_file():
            raise DataLoadError(f"Dataset path is not a file: {path}")
        size_bytes = path.stat().st_size
    elif hasattr(source, "getvalue"):
        try:
            size_bytes = len(source.getvalue())
        except Exception:
            size_bytes = None

    if size_bytes is not None and size_bytes > max_size_mb * 1024 * 1024:
        raise DataLoadError(
            f"Dataset exceeds the {max_size_mb} MB upload limit. "
            f"Received approximately {size_bytes / (1024 * 1024):.1f} MB."
        )

    if suffix == ".csv":
        df = _read_csv(source)
    elif suffix in SUPPORTED_EXCEL_EXTENSIONS:
        df = _read_excel(source, suffix)
    else:  # Defensive; extension validation already catches this.
        raise DataLoadError(f"Unsupported dataset format: {suffix}")

    df = _validate_dataframe(df)

    metadata = DatasetMetadata(
        file_name=Path(str(source_name)).name,
        file_type=suffix.lstrip("."),
        file_size_mb=(size_bytes / (1024 * 1024)) if size_bytes is not None else 0.0,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=tuple(str(column) for column in df.columns),
    )

    return df, metadata
