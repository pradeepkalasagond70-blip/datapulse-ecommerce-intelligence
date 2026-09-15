from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ============================================================
# GENERAL VALUE HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:
        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        return int(result)

    except (TypeError, ValueError):
        return default


def safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Safely divide two numeric values."""

    num = safe_float(numerator)
    den = safe_float(denominator)

    if den == 0:
        return default

    return num / den


def clamp(
    value: Any,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Clamp a numeric value to a defined range."""

    number = safe_float(value)

    return max(
        minimum,
        min(maximum, number),
    )


# ============================================================
# PERCENTAGE HELPERS
# ============================================================

def to_percent(
    value: Any,
    decimals: int = 1,
) -> float:
    """
    Convert a ratio or percentage-like value to percentage.

    Examples:
        0.25 -> 25.0
        25   -> 25.0
    """

    number = safe_float(value)

    if abs(number) <= 1:
        number *= 100

    return round(
        number,
        decimals,
    )


def percent_change(
    current: Any,
    previous: Any,
    default: float = 0.0,
) -> float:
    """Calculate percentage change."""

    current_value = safe_float(current)
    previous_value = safe_float(previous)

    if previous_value == 0:
        return default

    return (
        (current_value - previous_value)
        / abs(previous_value)
    ) * 100


# ============================================================
# CURRENCY FORMATTING
# ============================================================

def format_currency(
    value: Any,
    currency_symbol: str = "₹",
    decimals: int = 0,
) -> str:
    """Format a number as Indian-style currency."""

    number = safe_float(value)

    if abs(number) >= 10_000_000:

        return (
            f"{currency_symbol}"
            f"{number / 10_000_000:.2f} Cr"
        )

    if abs(number) >= 100_000:

        return (
            f"{currency_symbol}"
            f"{number / 100_000:.2f} L"
        )

    if abs(number) >= 1_000:

        return (
            f"{currency_symbol}"
            f"{number / 1_000:.1f}K"
        )

    return (
        f"{currency_symbol}"
        f"{number:,.{decimals}f}"
    )


def format_number(
    value: Any,
    decimals: int = 0,
) -> str:
    """Format a numeric value with thousands separators."""

    number = safe_float(value)

    return f"{number:,.{decimals}f}"


def format_percent(
    value: Any,
    decimals: int = 1,
) -> str:
    """Format a ratio/percentage as a readable percentage."""

    return f"{to_percent(value, decimals):.{decimals}f}%"


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(
    value: Any,
    default: str = "",
) -> str:
    """Clean arbitrary values into normalized text."""

    if value is None:
        return default

    try:
        text = str(value).strip()
    except Exception:
        return default

    if text.lower() in {
        "nan",
        "none",
        "null",
        "nat",
    }:
        return default

    return text


def normalize_column_name(
    column_name: Any,
) -> str:
    """Normalize a column name for matching."""

    text = clean_text(column_name)

    text = text.strip().lower()

    replacements = {
        "-": "_",
        " ": "_",
        "/": "_",
        "\\": "_",
        ".": "_",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    while "__" in text:
        text = text.replace(
            "__",
            "_",
        )

    return text.strip("_")


def title_case_column(
    column_name: Any,
) -> str:
    """Convert snake_case into presentation-friendly text."""

    text = normalize_column_name(
        column_name
    )

    return text.replace(
        "_",
        " ",
    ).title()


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """
    Find a DataFrame column using exact and normalized matching.
    """

    if dataframe is None or dataframe.empty:
        return None

    columns = list(
        dataframe.columns
    )

    # Exact match
    for candidate in candidates:

        if candidate in columns:
            return candidate

    # Normalized match
    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:

        normalized_candidate = (
            normalize_column_name(candidate)
        )

        if normalized_candidate in normalized_columns:

            return normalized_columns[
                normalized_candidate
            ]

    return None


def find_columns(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> list[str]:
    """Return all matching columns."""

    if dataframe is None or dataframe.empty:
        return []

    matches = []

    normalized_columns = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:

        if candidate in dataframe.columns:

            if candidate not in matches:
                matches.append(candidate)

            continue

        normalized_candidate = (
            normalize_column_name(candidate)
        )

        column = normalized_columns.get(
            normalized_candidate
        )

        if column is not None and column not in matches:
            matches.append(column)

    return matches


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def ensure_dataframe(
    value: Any,
) -> pd.DataFrame:
    """Safely convert supported objects into a DataFrame."""

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):

        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, dict):

        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric version of a DataFrame column."""

    if (
        dataframe is None
        or column not in dataframe.columns
    ):
        return pd.Series(
            dtype="float64"
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def clean_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Convert a column to numeric and remove invalid rows."""

    result = dataframe.copy()

    if column not in result.columns:
        return result

    result[column] = pd.to_numeric(
        result[column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[column]
    )

    return result


def clean_dataframe_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize DataFrame column names."""

    result = dataframe.copy()

    result.columns = [
        normalize_column_name(column)
        for column in result.columns
    ]

    return result


# ============================================================
# DATE / TIME HELPERS
# ============================================================

def safe_datetime(
    value: Any,
    default: Any = None,
):
    """Safely parse a datetime value."""

    if value is None:
        return default

    try:
        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(parsed):
            return default

        return parsed

    except Exception:
        return default


def ensure_datetime_column(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Convert a DataFrame column into datetime values."""

    result = dataframe.copy()

    if column not in result.columns:
        return result

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
    )

    return result


def date_range_days(
    start: Any,
    end: Any,
) -> int:
    """Return number of days between two dates."""

    start_date = safe_datetime(start)
    end_date = safe_datetime(end)

    if start_date is None or end_date is None:
        return 0

    try:
        return max(
            0,
            int(
                (
                    end_date - start_date
                ).days
            ),
        )

    except Exception:
        return 0


def now_timestamp() -> str:
    """Return a stable human-readable timestamp."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# DATA QUALITY HELPERS
# ============================================================

def missing_value_count(
    dataframe: pd.DataFrame,
) -> int:
    """Return total missing cells."""

    if dataframe is None or dataframe.empty:
        return 0

    return int(
        dataframe.isna().sum().sum()
    )


def missing_value_rate(
    dataframe: pd.DataFrame,
) -> float:
    """Return percentage of cells that are missing."""

    if dataframe is None or dataframe.empty:
        return 0.0

    total_cells = (
        dataframe.shape[0]
        * dataframe.shape[1]
    )

    if total_cells == 0:
        return 0.0

    return (
        missing_value_count(dataframe)
        / total_cells
    ) * 100


def duplicate_row_count(
    dataframe: pd.DataFrame,
) -> int:
    """Return number of duplicate rows."""

    if dataframe is None or dataframe.empty:
        return 0

    return int(
        dataframe.duplicated().sum()
    )


def duplicate_row_rate(
    dataframe: pd.DataFrame,
) -> float:
    """Return percentage of duplicate rows."""

    if dataframe is None or dataframe.empty:
        return 0.0

    return (
        duplicate_row_count(dataframe)
        / len(dataframe)
    ) * 100


# ============================================================
# FILE HELPERS
# ============================================================

def safe_filename(
    filename: Any,
    default: str = "dataset",
) -> str:
    """Return a safe basename without directory traversal."""

    if filename is None:
        return default

    try:
        name = Path(
            str(filename)
        ).name

    except Exception:
        return default

    if not name:
        return default

    return name


def file_extension(
    filename: Any,
) -> str:
    """Return a lowercase file extension."""

    name = safe_filename(filename)

    return Path(name).suffix.lower()


def file_size_mb(
    size_bytes: Any,
) -> float:
    """Convert bytes to megabytes."""

    size = safe_float(
        size_bytes
    )

    return size / (
        1024 * 1024
    )


# ============================================================
# DICTIONARY HELPERS
# ============================================================

def first_available(
    data: dict[str, Any],
    keys: Iterable[str],
    default: Any = None,
) -> Any:
    """Return the first non-null value."""

    if not isinstance(data, dict):
        return default

    for key in keys:

        if key not in data:
            continue

        value = data[key]

        if value is None:
            continue

        if isinstance(value, float):

            if np.isnan(value):
                continue

        return value

    return default


def get_nested(
    data: dict[str, Any],
    path: Iterable[str],
    default: Any = None,
) -> Any:
    """Safely retrieve a nested dictionary value."""

    current: Any = data

    for key in path:

        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


# ============================================================
# LIST HELPERS
# ============================================================

def safe_list(
    value: Any,
) -> list:
    """Convert common iterable values to a list."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, pd.Series):
        return value.tolist()

    return []


def unique_preserve_order(
    values: Iterable[Any],
) -> list:
    """Return unique values while preserving order."""

    result = []

    seen = set()

    for value in values:

        try:
            key = value

            if key in seen:
                continue

            seen.add(key)
            result.append(value)

        except TypeError:
            text = str(value)

            if text in seen:
                continue

            seen.add(text)
            result.append(value)

    return result


# ============================================================
# DATASET SUMMARY HELPERS
# ============================================================

def dataframe_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a lightweight structural summary."""

    if dataframe is None:
        return {
            "rows": 0,
            "columns": 0,
            "missing_cells": 0,
            "missing_rate_pct": 0.0,
            "duplicate_rows": 0,
            "duplicate_rate_pct": 0.0,
        }

    return {
        "rows": int(
            dataframe.shape[0]
        ),
        "columns": int(
            dataframe.shape[1]
        ),
        "missing_cells": missing_value_count(
            dataframe
        ),
        "missing_rate_pct": round(
            missing_value_rate(dataframe),
            2,
        ),
        "duplicate_rows": duplicate_row_count(
            dataframe
        ),
        "duplicate_rate_pct": round(
            duplicate_row_rate(dataframe),
            2,
        ),
    }


# ============================================================
# MODEL / ANALYTICS HELPERS
# ============================================================

def dataframe_is_model_ready(
    dataframe: pd.DataFrame,
    minimum_rows: int = 20,
) -> bool:
    """
    Basic readiness check for model inputs.

    This does not replace model-specific validation.
    """

    if dataframe is None:
        return False

    if dataframe.empty:
        return False

    if len(dataframe) < minimum_rows:
        return False

    return dataframe.shape[1] > 0


def numeric_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return numeric columns."""

    if dataframe is None or dataframe.empty:
        return []

    return dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return categorical/object columns."""

    if dataframe is None or dataframe.empty:
        return []

    return dataframe.select_dtypes(
        include=[
            "object",
            "category",
            "string",
        ]
    ).columns.tolist()


# ============================================================
# SORTING HELPERS
# ============================================================

def top_n(
    dataframe: pd.DataFrame,
    column: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return top/bottom N rows safely."""

    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    if column not in dataframe.columns:
        return dataframe.head(n).copy()

    return (
        dataframe
        .sort_values(
            by=column,
            ascending=ascending,
        )
        .head(n)
        .copy()
    )


# ============================================================
# EXPORT HELPERS
# ============================================================

def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into JSON-friendly records."""

    if dataframe is None or dataframe.empty:
        return []

    result = dataframe.copy()

    result = result.replace(
        {
            np.nan: None,
            np.inf: None,
            -np.inf: None,
        }
    )

    return result.to_dict(
        orient="records"
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # General
    "safe_float",
    "safe_int",
    "safe_divide",
    "clamp",

    # Percentages
    "to_percent",
    "percent_change",

    # Formatting
    "format_currency",
    "format_number",
    "format_percent",

    # Text
    "clean_text",
    "normalize_column_name",
    "title_case_column",

    # Columns
    "find_column",
    "find_columns",

    # DataFrames
    "ensure_dataframe",
    "numeric_series",
    "clean_numeric_column",
    "clean_dataframe_columns",

    # Date/time
    "safe_datetime",
    "ensure_datetime_column",
    "date_range_days",
    "now_timestamp",

    # Data quality
    "missing_value_count",
    "missing_value_rate",
    "duplicate_row_count",
    "duplicate_row_rate",

    # Files
    "safe_filename",
    "file_extension",
    "file_size_mb",

    # Dictionaries
    "first_available",
    "get_nested",

    # Lists
    "safe_list",
    "unique_preserve_order",

    # Dataset
    "dataframe_summary",

    # Model/analytics
    "dataframe_is_model_ready",
    "numeric_columns",
    "categorical_columns",

    # Sorting
    "top_n",

    # Export
    "dataframe_to_records",
]