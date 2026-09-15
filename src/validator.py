"""
DataPulse — Dataset Validator

Purpose
-------
Validates an analysis-ready e-commerce dataset after schema mapping.

Responsibilities
----------------
- Validate structural integrity
- Validate required DataPulse fields
- Check data types
- Check missing values
- Check date usability
- Check numeric-field usability
- Detect suspicious values
- Determine which DataPulse intelligence modules are available

This module does NOT:
- perform EDA
- silently modify the dataframe
- train ML models
- calculate business KPIs
- clean the dataset
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.schema_mapper import (
    CORE_REQUIRED_FIELDS,
    FIELD_GROUPS,
    get_source_column,
)


# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

MAX_MISSING_RATIO_WARNING = 0.40
MAX_MISSING_RATIO_ERROR = 0.95

MIN_ROWS_FOR_ANALYTICS = 10
MIN_ROWS_FOR_ML = 50

NUMERIC_FIELDS = {
    "sales_amount",
    "quantity",
    "unit_price",
    "cost",
    "profit",
    "customer_age",
    "campaign_cost",
    "discount",
    "review_rating",
}

DATE_FIELDS = {
    "order_date",
    "customer_since",
    "campaign_start_date",
    "campaign_end_date",
    "shipping_date",
    "delivery_date",
    "promised_delivery_date",
    "return_date",
    "review_date",
}

ID_FIELDS = {
    "order_id",
    "customer_id",
    "product_id",
    "campaign_id",
    "seller_id",
    "review_id",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def _column_exists(
    df: pd.DataFrame,
    column: Optional[str],
) -> bool:
    return bool(
        column
        and column in df.columns
    )


def _missing_ratio(
    series: pd.Series,
) -> float:
    if len(series) == 0:
        return 1.0

    return float(
        series.isna().mean()
    )


def _unique_ratio(
    series: pd.Series,
) -> float:
    if len(series) == 0:
        return 0.0

    return float(
        series.nunique(dropna=True) / len(series)
    )


def _numeric_conversion_ratio(
    series: pd.Series,
) -> float:
    """
    Measures how much of a series can be interpreted as numeric
    without modifying the original dataframe.
    """

    if len(series) == 0:
        return 0.0

    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    return float(
        converted.notna().mean()
    )


def _datetime_conversion_ratio(
    series: pd.Series,
) -> float:
    """
    Measures how much of a series can be interpreted as dates.
    """

    if len(series) == 0:
        return 0.0

    converted = pd.to_datetime(
        series,
        errors="coerce",
    )

    return float(
        converted.notna().mean()
    )


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

def validate_structure(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Validate basic dataframe structure."""

    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(df, pd.DataFrame):
        errors.append(
            "Dataset must be a pandas DataFrame."
        )
        return errors, warnings

    if df.empty:
        errors.append(
            "Dataset is empty."
        )
        return errors, warnings

    if len(df.columns) == 0:
        errors.append(
            "Dataset contains no columns."
        )

    if df.columns.duplicated().any():
        duplicates = (
            df.columns[
                df.columns.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        errors.append(
            "Duplicate column names detected: "
            + ", ".join(duplicates)
        )

    blank_columns = [
        str(column)
        for column in df.columns
        if not str(column).strip()
    ]

    if blank_columns:
        errors.append(
            "Blank column names are not allowed."
        )

    if len(df) < MIN_ROWS_FOR_ANALYTICS:
        warnings.append(
            f"Dataset contains only {len(df)} rows. "
            f"Analytics may be statistically limited."
        )

    if len(df) < MIN_ROWS_FOR_ML:
        warnings.append(
            f"Dataset contains fewer than {MIN_ROWS_FOR_ML} rows. "
            "ML predictions may be unavailable or unreliable."
        )

    return errors, warnings


# ============================================================
# CORE FIELD VALIDATION
# ============================================================

def validate_core_fields(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """Validate DataPulse's core fields."""

    errors: List[str] = []
    warnings: List[str] = []

    mapped_columns = mapping.get(
        "mapped_columns",
        {},
    )

    missing_core_fields = [
        field
        for field in CORE_REQUIRED_FIELDS
        if field not in mapped_columns
    ]

    if missing_core_fields:
        errors.append(
            "Missing required DataPulse fields: "
            + ", ".join(missing_core_fields)
        )

    for field in CORE_REQUIRED_FIELDS:

        column = get_source_column(
            mapping,
            field,
        )

        if not _column_exists(
            df,
            column,
        ):
            continue

        ratio = _missing_ratio(
            df[column]
        )

        if ratio >= MAX_MISSING_RATIO_ERROR:
            errors.append(
                f"Core field '{field}' "
                f"({column}) has "
                f"{ratio:.1%} missing values."
            )

        elif ratio >= MAX_MISSING_RATIO_WARNING:
            warnings.append(
                f"Core field '{field}' "
                f"({column}) has "
                f"{ratio:.1%} missing values."
            )

    return errors, warnings


# ============================================================
# DATA TYPE VALIDATION
# ============================================================

def validate_numeric_fields(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validate numeric business fields.

    We don't convert values here.
    Conversion/feature preparation belongs to the feature
    engineering layer.
    """

    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    for field in NUMERIC_FIELDS:

        column = get_source_column(
            mapping,
            field,
        )

        if not _column_exists(
            df,
            column,
        ):
            continue

        series = df[column]

        is_numeric = pd.api.types.is_numeric_dtype(
            series
        )

        conversion_ratio = (
            1.0
            if is_numeric
            else _numeric_conversion_ratio(series)
        )

        details[field] = {
            "source_column": column,
            "is_numeric_dtype": bool(is_numeric),
            "numeric_conversion_ratio": round(
                conversion_ratio,
                4,
            ),
        }

        if is_numeric:
            numeric_values = pd.to_numeric(
                series,
                errors="coerce",
            )

            if np.isinf(
                numeric_values
            ).any():
                warnings.append(
                    f"'{column}' mapped to '{field}' "
                    "contains infinite numeric values."
                )

            if field in {
                "quantity",
                "unit_price",
                "cost",
                "sales_amount",
            }:

                negative_count = int(
                    (numeric_values < 0)
                    .sum()
                )

                if negative_count > 0:
                    warnings.append(
                        f"'{column}' mapped to '{field}' "
                        f"contains {negative_count} negative values."
                    )

        else:

            if conversion_ratio >= 0.90:
                warnings.append(
                    f"'{column}' mapped to '{field}' "
                    "is stored as text but appears mostly numeric. "
                    "Feature engineering may convert it."
                )

            elif conversion_ratio >= 0.60:
                warnings.append(
                    f"'{column}' mapped to '{field}' "
                    "contains mixed/non-numeric values."
                )

            else:
                warnings.append(
                    f"'{column}' mapped to '{field}' "
                    "is not sufficiently numeric for reliable "
                    "numeric analytics."
                )

    return errors, warnings, details


def validate_date_fields(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Validate date-like fields without modifying the dataframe."""

    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    for field in DATE_FIELDS:

        column = get_source_column(
            mapping,
            field,
        )

        if not _column_exists(
            df,
            column,
        ):
            continue

        series = df[column]

        is_datetime = pd.api.types.is_datetime64_any_dtype(
            series
        )

        conversion_ratio = (
            1.0
            if is_datetime
            else _datetime_conversion_ratio(series)
        )

        details[field] = {
            "source_column": column,
            "is_datetime_dtype": bool(is_datetime),
            "datetime_conversion_ratio": round(
                conversion_ratio,
                4,
            ),
        }

        if conversion_ratio < 0.60:
            warnings.append(
                f"'{column}' mapped to '{field}' "
                "does not contain enough recognizable dates."
            )

        elif conversion_ratio < 0.90:
            warnings.append(
                f"'{column}' mapped to '{field}' "
                "contains some invalid/unparseable dates."
            )

    return errors, warnings, details


# ============================================================
# MISSING VALUE VALIDATION
# ============================================================

def validate_missing_values(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Profile missingness only for mapped business fields."""

    warnings: List[str] = []
    details: Dict[str, Any] = {}

    for field in mapping.get(
        "available_fields",
        [],
    ):

        column = get_source_column(
            mapping,
            field,
        )

        if not _column_exists(
            df,
            column,
        ):
            continue

        ratio = _missing_ratio(
            df[column]
        )

        details[field] = {
            "source_column": column,
            "missing_count": int(
                df[column].isna().sum()
            ),
            "missing_ratio": round(
                ratio,
                4,
            ),
        }

        if ratio >= MAX_MISSING_RATIO_WARNING:

            warnings.append(
                f"'{column}' mapped to '{field}' "
                f"has {ratio:.1%} missing values."
            )

    return warnings, details


# ============================================================
# ID VALIDATION
# ============================================================

def validate_identifier_fields(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Validate identifier usability."""

    warnings: List[str] = []
    details: Dict[str, Any] = {}

    for field in ID_FIELDS:

        column = get_source_column(
            mapping,
            field,
        )

        if not _column_exists(
            df,
            column,
        ):
            continue

        series = df[column]

        unique_ratio = _unique_ratio(
            series
        )

        missing_ratio = _missing_ratio(
            series
        )

        details[field] = {
            "source_column": column,
            "unique_count": int(
                series.nunique(dropna=True)
            ),
            "unique_ratio": round(
                unique_ratio,
                4,
            ),
            "missing_ratio": round(
                missing_ratio,
                4,
            ),
        }

        if missing_ratio >= 0.50:

            warnings.append(
                f"Identifier '{column}' mapped to '{field}' "
                f"is missing in {missing_ratio:.1%} of rows."
            )

        if field.endswith("_id") and unique_ratio >= 0.99:

            # Not necessarily wrong — some transaction-level IDs
            # are expected to be unique.
            if field not in {
                "order_id",
                "review_id",
            }:
                warnings.append(
                    f"'{column}' mapped to '{field}' is almost "
                    "unique for every row; customer/product "
                    "level aggregation may be limited."
                )

    return warnings, details


# ============================================================
# BUSINESS LOGIC CHECKS
# ============================================================

def validate_business_relationships(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Perform lightweight sanity checks between mapped fields.

    These are warnings rather than hard failures because real-world
    e-commerce datasets can legitimately contain unusual values.
    """

    errors: List[str] = []
    warnings: List[str] = []

    sales_column = get_source_column(
        mapping,
        "sales_amount",
    )

    quantity_column = get_source_column(
        mapping,
        "quantity",
    )

    unit_price_column = get_source_column(
        mapping,
        "unit_price",
    )

    profit_column = get_source_column(
        mapping,
        "profit",
    )

    # --------------------------------------------------------
    # Sales amount
    # --------------------------------------------------------

    if _column_exists(
        df,
        sales_column,
    ):

        numeric_sales = pd.to_numeric(
            df[sales_column],
            errors="coerce",
        )

        valid_sales = numeric_sales.dropna()

        if not valid_sales.empty:

            if (valid_sales == 0).mean() > 0.50:
                warnings.append(
                    f"More than 50% of '{sales_column}' "
                    "contains zero sales values."
                )

    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    if _column_exists(
        df,
        quantity_column,
    ):

        numeric_quantity = pd.to_numeric(
            df[quantity_column],
            errors="coerce",
        )

        negative_quantity = (
            numeric_quantity < 0
        ).sum()

        if negative_quantity > 0:

            warnings.append(
                f"'{quantity_column}' contains "
                f"{int(negative_quantity)} negative quantities."
            )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    if _column_exists(
        df,
        unit_price_column,
    ):

        numeric_price = pd.to_numeric(
            df[unit_price_column],
            errors="coerce",
        )

        negative_price = (
            numeric_price < 0
        ).sum()

        if negative_price > 0:

            warnings.append(
                f"'{unit_price_column}' contains negative prices."
            )

    # --------------------------------------------------------
    # Profit
    # --------------------------------------------------------

    if (
        _column_exists(
            df,
            profit_column,
        )
        and _column_exists(
            df,
            sales_column,
        )
    ):

        profit = pd.to_numeric(
            df[profit_column],
            errors="coerce",
        )

        sales = pd.to_numeric(
            df[sales_column],
            errors="coerce",
        )

        comparable = pd.concat(
            [profit, sales],
            axis=1,
        ).dropna()

        if not comparable.empty:

            invalid_margin = (
                comparable.iloc[:, 0]
                > comparable.iloc[:, 1]
            )

            invalid_count = int(
                invalid_margin.sum()
            )

            if invalid_count > 0:

                warnings.append(
                    f"{invalid_count} rows contain profit greater "
                    "than sales amount. Verify the business definition "
                    "of these columns."
                )

    return errors, warnings


# ============================================================
# MODULE AVAILABILITY
# ============================================================

def _calculate_module_availability(
    mapping: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Determine which DataPulse modules have enough mapped fields.

    This is a capability assessment, not an analysis result.
    """

    available = set(
        mapping.get(
            "available_fields",
            [],
        )
    )

    modules: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------------
    # Sales Intelligence
    # --------------------------------------------------------

    sales_fields = {
        "order_id",
        "order_date",
        "sales_amount",
    }

    sales_count = len(
        available.intersection(
            sales_fields
        )
    )

    modules["sales_intelligence"] = {
        "available": sales_count >= 2,
        "field_count": sales_count,
        "required_for_basic": sorted(
            sales_fields
        ),
    }

    # --------------------------------------------------------
    # Customer Intelligence
    # --------------------------------------------------------

    customer_available = (
        "customer_id" in available
        or "customer_name" in available
    )

    modules["customer_intelligence"] = {
        "available": customer_available,
        "field_count": sum(
            field in available
            for field in FIELD_GROUPS["customer"]
        ),
        "required_for_basic": [
            "customer_id"
        ],
    }

    # --------------------------------------------------------
    # Product Intelligence
    # --------------------------------------------------------

    product_available = (
        "product_id" in available
        or "product_name" in available
        or "category" in available
    )

    modules["product_intelligence"] = {
        "available": product_available,
        "field_count": sum(
            field in available
            for field in FIELD_GROUPS["product"]
        ),
        "required_for_basic": [
            "product_name"
        ],
    }

    # --------------------------------------------------------
    # Campaign Impact
    # --------------------------------------------------------

    campaign_available = (
        (
            "campaign_id" in available
            or "campaign_name" in available
        )
        and "sales_amount" in available
    )

    modules["campaign_impact"] = {
        "available": campaign_available,
        "field_count": sum(
            field in available
            for field in FIELD_GROUPS["campaign"]
        ),
        "required_for_basic": [
            "campaign_name",
            "sales_amount",
        ],
    }

    # --------------------------------------------------------
    # Market / Seller
    # --------------------------------------------------------

    market_available = (
        any(
            field in available
            for field in (
                "seller_id",
                "seller_name",
                "market",
                "region",
                "customer_city",
                "customer_state",
            )
        )
        and "sales_amount" in available
    )

    modules["market_seller_intelligence"] = {
        "available": market_available,
        "field_count": sum(
            field in available
            for field in FIELD_GROUPS["market_seller"]
        ),
        "required_for_basic": [
            "sales_amount"
        ],
    }

    # --------------------------------------------------------
    # Delivery Intelligence
    # --------------------------------------------------------

    delivery_available = (
        "delivery_date" in available
        or "delivery_status" in available
        or "shipping_date" in available
    )

    modules["delivery_intelligence"] = {
        "available": delivery_available,
        "field_count": sum(
            field in available
            for field in FIELD_GROUPS["delivery"]
        ),
        "required_for_basic": [
            "delivery_status"
        ],
    }

    # --------------------------------------------------------
    # Review / NLP
    # --------------------------------------------------------

    review_available = (
        "review_text" in available
        or "review_rating" in available
    )

    modules["review_nlp"] = {
        "available": review_available,
        "field_count": sum(
            field in available
            for field in FIELD_GROUPS["reviews"]
        ),
        "required_for_basic": [
            "review_text"
        ],
    }

    # --------------------------------------------------------
    # Pricing
    # --------------------------------------------------------

    pricing_available = (
        (
            "unit_price" in available
            or "discount" in available
        )
        and "sales_amount" in available
    )

    modules["pricing_intelligence"] = {
        "available": pricing_available,
        "field_count": sum(
            field in available
            for field in (
                "unit_price",
                "discount",
                "sales_amount",
                "profit",
                "quantity",
            )
        ),
        "required_for_basic": [
            "sales_amount"
        ],
    }

    # --------------------------------------------------------
    # Churn
    # --------------------------------------------------------

    churn_available = (
        "customer_id" in available
        and "order_date" in available
    )

    modules["churn_prediction"] = {
        "available": churn_available,
        "field_count": sum(
            field in available
            for field in (
                "customer_id",
                "order_date",
                "sales_amount",
                "quantity",
                "profit",
            )
        ),
        "required_for_basic": [
            "customer_id",
            "order_date",
        ],
    }

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    segmentation_available = (
        "customer_id" in available
        and (
            "sales_amount" in available
            or "order_date" in available
        )
    )

    modules["customer_segmentation"] = {
        "available": segmentation_available,
        "field_count": sum(
            field in available
            for field in (
                "customer_id",
                "sales_amount",
                "order_date",
                "quantity",
                "profit",
            )
        ),
        "required_for_basic": [
            "customer_id"
        ],
    }

    # --------------------------------------------------------
    # Sales Forecast
    # --------------------------------------------------------

    forecasting_available = (
        "order_date" in available
        and "sales_amount" in available
    )

    modules["sales_forecasting"] = {
        "available": forecasting_available,
        "field_count": sum(
            field in available
            for field in (
                "order_date",
                "sales_amount",
                "quantity",
            )
        ),
        "required_for_basic": [
            "order_date",
            "sales_amount",
        ],
    }

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    anomaly_available = (
        "sales_amount" in available
        or "quantity" in available
        or "profit" in available
    )

    modules["anomaly_detection"] = {
        "available": anomaly_available,
        "field_count": sum(
            field in available
            for field in (
                "sales_amount",
                "quantity",
                "profit",
                "discount",
                "unit_price",
            )
        ),
        "required_for_basic": [
            "sales_amount"
        ],
    }

    return modules


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    warnings: List[str],
    errors: List[str],
) -> float:
    """
    Calculate a simple dataset readiness score.

    Score is intended for UI communication, not statistical
    certification of data quality.
    """

    if errors:
        base_score = 40.0
    else:
        base_score = 80.0

    # Mapping coverage
    total_fields = len(
        mapping.get(
            "mappings",
            {},
        )
    )

    mapped_fields = len(
        mapping.get(
            "available_fields",
            [],
        )
    )

    if total_fields:
        coverage = mapped_fields / total_fields
        base_score += coverage * 10.0

    # Missingness
    mapped_columns = mapping.get(
        "mapped_columns",
        {},
    )

    if mapped_columns:

        ratios = []

        for column in mapped_columns.values():

            if column in df.columns:
                ratios.append(
                    _missing_ratio(
                        df[column]
                    )
                )

        if ratios:

            average_missing = float(
                np.mean(ratios)
            )

            base_score -= (
                average_missing * 25.0
            )

    # Warning penalty
    base_score -= min(
        len(warnings) * 0.5,
        10.0,
    )

    return round(
        float(
            np.clip(
                base_score,
                0,
                100,
            )
        ),
        1,
    )


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Complete DataPulse dataset validation.

    Returns
    -------
    dict
        Stable validation contract consumed by app.py and
        downstream DataPulse modules.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "validate_dataset() expects a pandas DataFrame."
        )

    if not isinstance(mapping, dict):
        raise TypeError(
            "mapping must be a dictionary returned by map_schema()."
        )

    errors: List[str] = []
    warnings: List[str] = []

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    structure_errors, structure_warnings = (
        validate_structure(df)
    )

    errors.extend(
        structure_errors
    )

    warnings.extend(
        structure_warnings
    )

    # --------------------------------------------------------
    # Core fields
    # --------------------------------------------------------

    core_errors, core_warnings = (
        validate_core_fields(
            df,
            mapping,
        )
    )

    errors.extend(
        core_errors
    )

    warnings.extend(
        core_warnings
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    (
        numeric_errors,
        numeric_warnings,
        numeric_details,
    ) = validate_numeric_fields(
        df,
        mapping,
    )

    errors.extend(
        numeric_errors
    )

    warnings.extend(
        numeric_warnings
    )

    # --------------------------------------------------------
    # Date fields
    # --------------------------------------------------------

    (
        date_errors,
        date_warnings,
        date_details,
    ) = validate_date_fields(
        df,
        mapping,
    )

    errors.extend(
        date_errors
    )

    warnings.extend(
        date_warnings
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    (
        missing_warnings,
        missing_details,
    ) = validate_missing_values(
        df,
        mapping,
    )

    warnings.extend(
        missing_warnings
    )

    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------

    (
        identifier_warnings,
        identifier_details,
    ) = validate_identifier_fields(
        df,
        mapping,
    )

    warnings.extend(
        identifier_warnings
    )

    # --------------------------------------------------------
    # Business relationships
    # --------------------------------------------------------

    (
        relationship_errors,
        relationship_warnings,
    ) = validate_business_relationships(
        df,
        mapping,
    )

    errors.extend(
        relationship_errors
    )

    warnings.extend(
        relationship_warnings
    )

    # --------------------------------------------------------
    # Module availability
    # --------------------------------------------------------

    modules = _calculate_module_availability(
        mapping
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    valid = len(errors) == 0

    quality_score = calculate_quality_score(
        df,
        mapping,
        warnings,
        errors,
    )

    if errors:
        status = "blocked"

    elif quality_score >= 80:
        status = "ready"

    elif quality_score >= 60:
        status = "ready_with_warnings"

    else:
        status = "limited"

    # --------------------------------------------------------
    # Return stable contract
    # --------------------------------------------------------

    return {
        "valid": valid,
        "status": status,
        "quality_score": quality_score,

        "errors": errors,
        "warnings": warnings,

        "row_count": int(
            len(df)
        ),

        "column_count": int(
            len(df.columns)
        ),

        "mapped_field_count": len(
            mapping.get(
                "available_fields",
                [],
            )
        ),

        "missing_core_fields": list(
            mapping.get(
                "missing_core_fields",
                [],
            )
        ),

        "numeric_fields": numeric_details,

        "date_fields": date_details,

        "missing_values": missing_details,

        "identifiers": identifier_details,

        "modules": modules,

        "module_count": int(
            sum(
                1
                for module in modules.values()
                if module["available"]
            )
        ),

        "available_modules": [
            name
            for name, module in modules.items()
            if module["available"]
        ],
    }


# ============================================================
# CONVENIENCE HELPERS
# ============================================================

def is_dataset_ready(
    validation: Dict[str, Any],
) -> bool:
    """Return whether the dataset passed hard validation."""

    return bool(
        validation.get(
            "valid",
            False,
        )
    )


def get_validation_errors(
    validation: Dict[str, Any],
) -> List[str]:
    """Return validation errors."""

    return list(
        validation.get(
            "errors",
            [],
        )
    )


def get_validation_warnings(
    validation: Dict[str, Any],
) -> List[str]:
    """Return validation warnings."""

    return list(
        validation.get(
            "warnings",
            [],
        )
    )


def get_available_modules(
    validation: Dict[str, Any],
) -> List[str]:
    """Return modules that can operate on the uploaded dataset."""

    return list(
        validation.get(
            "available_modules",
            [],
        )
    )


def module_is_available(
    validation: Dict[str, Any],
    module_name: str,
) -> bool:
    """Check whether a specific DataPulse module is available."""

    return bool(
        validation.get(
            "modules",
            {},
        )
        .get(
            module_name,
            {},
        )
        .get(
            "available",
            False,
        )
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "validate_dataset",
    "validate_structure",
    "validate_core_fields",
    "validate_numeric_fields",
    "validate_date_fields",
    "validate_missing_values",
    "validate_identifier_fields",
    "validate_business_relationships",
    "calculate_quality_score",
    "is_dataset_ready",
    "get_validation_errors",
    "get_validation_warnings",
    "get_available_modules",
    "module_is_available",
]