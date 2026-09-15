"""
DataPulse — Core Analytics Engine

Responsibilities
----------------
- Executive KPIs
- Sales performance
- Revenue trends
- Order trends
- Quantity analysis
- Profitability
- Average Order Value
- Growth metrics
- Daily / monthly aggregation
- Top products/categories when available

This module works from the schema mapping produced by
src.schema_mapper.map_schema().

It does NOT:
- perform EDA
- modify the source dataframe
- perform schema mapping
- train ML models
- generate natural-language recommendations

Those responsibilities belong to other DataPulse modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.schema_mapper import get_source_column


# ============================================================
# HELPERS
# ============================================================

def _column(
    mapping: Dict[str, Any],
    field: str,
) -> Optional[str]:
    """Return source column for a canonical DataPulse field."""

    return get_source_column(
        mapping,
        field,
    )


def _has(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    field: str,
) -> bool:
    """Check whether a mapped field exists in the dataframe."""

    column = _column(
        mapping,
        field,
    )

    return bool(
        column
        and column in df.columns
    )


def _numeric(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    field: str,
) -> Optional[pd.Series]:
    """Return a numeric representation of a mapped field."""

    if not _has(
        df,
        mapping,
        field,
    ):
        return None

    column = _column(
        mapping,
        field,
    )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def _date(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    field: str,
) -> Optional[pd.Series]:
    """Return a datetime representation of a mapped field."""

    if not _has(
        df,
        mapping,
        field,
    ):
        return None

    column = _column(
        mapping,
        field,
    )

    return pd.to_datetime(
        df[column],
        errors="coerce",
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert numeric values safely."""

    try:

        if pd.isna(value):
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


def _round(
    value: Any,
    digits: int = 2,
) -> float:
    return round(
        _safe_float(value),
        digits,
    )


def _growth(
    current: float,
    previous: float,
) -> Optional[float]:
    """
    Percentage growth.

    Returns None when the previous period is zero because
    percentage growth would be undefined.
    """

    current = _safe_float(current)
    previous = _safe_float(previous)

    if previous == 0:
        return None

    return round(
        ((current - previous) / abs(previous)) * 100,
        2,
    )


# ============================================================
# BASIC TRANSACTION METRICS
# ============================================================

def calculate_basic_metrics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate core executive KPIs."""

    metrics: Dict[str, Any] = {
        "rows": int(len(df)),
        "orders": 0,
        "customers": 0,
        "products": 0,
        "revenue": 0.0,
        "quantity": 0.0,
        "profit": None,
        "average_order_value": 0.0,
        "average_unit_price": None,
        "profit_margin": None,
    }

    # --------------------------------------------------------
    # Orders
    # --------------------------------------------------------

    if _has(
        df,
        mapping,
        "order_id",
    ):

        order_column = _column(
            mapping,
            "order_id",
        )

        metrics["orders"] = int(
            df[order_column]
            .dropna()
            .nunique()
        )

    else:
        metrics["orders"] = int(
            len(df)
        )

    # --------------------------------------------------------
    # Customers
    # --------------------------------------------------------

    if _has(
        df,
        mapping,
        "customer_id",
    ):

        customer_column = _column(
            mapping,
            "customer_id",
        )

        metrics["customers"] = int(
            df[customer_column]
            .dropna()
            .nunique()
        )

    # --------------------------------------------------------
    # Products
    # --------------------------------------------------------

    if _has(
        df,
        mapping,
        "product_id",
    ):

        product_column = _column(
            mapping,
            "product_id",
        )

        metrics["products"] = int(
            df[product_column]
            .dropna()
            .nunique()
        )

    elif _has(
        df,
        mapping,
        "product_name",
    ):

        product_column = _column(
            mapping,
            "product_name",
        )

        metrics["products"] = int(
            df[product_column]
            .dropna()
            .nunique()
        )

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is not None:

        metrics["revenue"] = _round(
            revenue.sum()
        )

    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    if quantity is not None:

        metrics["quantity"] = _round(
            quantity.sum()
        )

    # --------------------------------------------------------
    # Profit
    # --------------------------------------------------------

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    if profit is not None:

        metrics["profit"] = _round(
            profit.sum()
        )

        if metrics["revenue"] != 0:

            metrics["profit_margin"] = _round(
                (
                    metrics["profit"]
                    / metrics["revenue"]
                )
                * 100
            )

    # --------------------------------------------------------
    # Average Order Value
    # --------------------------------------------------------

    if metrics["orders"] > 0:

        metrics["average_order_value"] = _round(
            metrics["revenue"]
            / metrics["orders"]
        )

    # --------------------------------------------------------
    # Average Unit Price
    # --------------------------------------------------------

    unit_price = _numeric(
        df,
        mapping,
        "unit_price",
    )

    if unit_price is not None:

        metrics["average_unit_price"] = _round(
            unit_price.mean()
        )

    return metrics


# ============================================================
# DATE RANGE
# ============================================================

def calculate_date_range(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine the usable transaction date range."""

    dates = _date(
        df,
        mapping,
        "order_date",
    )

    if dates is None:

        return {
            "available": False,
            "start_date": None,
            "end_date": None,
            "days": None,
        }

    valid_dates = dates.dropna()

    if valid_dates.empty:

        return {
            "available": False,
            "start_date": None,
            "end_date": None,
            "days": None,
        }

    start = valid_dates.min()
    end = valid_dates.max()

    days = int(
        (end - start).days
    ) + 1

    return {
        "available": True,
        "start_date": start.strftime(
            "%Y-%m-%d"
        ),
        "end_date": end.strftime(
            "%Y-%m-%d"
        ),
        "days": days,
    }


# ============================================================
# DAILY SALES TREND
# ============================================================

def calculate_daily_sales(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Create daily sales aggregation."""

    dates = _date(
        df,
        mapping,
        "order_date",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if dates is None or revenue is None:

        return pd.DataFrame(
            columns=[
                "date",
                "revenue",
                "orders",
                "quantity",
                "profit",
            ]
        )

    working = pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue,
        }
    )

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    if quantity is not None:
        working["quantity"] = quantity
    else:
        working["quantity"] = 0.0

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    if profit is not None:
        working["profit"] = profit
    else:
        working["profit"] = np.nan

    order_column = _column(
        mapping,
        "order_id",
    )

    if order_column and order_column in df.columns:

        working["order_id"] = df[
            order_column
        ].values

        result = (
            working.dropna(
                subset=["date"]
            )
            .groupby("date")
            .agg(
                revenue=("revenue", "sum"),
                orders=(
                    "order_id",
                    "nunique",
                ),
                quantity=(
                    "quantity",
                    "sum",
                ),
                profit=(
                    "profit",
                    "sum",
                ),
            )
            .reset_index()
        )

    else:

        result = (
            working.dropna(
                subset=["date"]
            )
            .groupby("date")
            .agg(
                revenue=("revenue", "sum"),
                orders=("revenue", "size"),
                quantity=(
                    "quantity",
                    "sum",
                ),
                profit=(
                    "profit",
                    "sum",
                ),
            )
            .reset_index()
        )

    result["revenue"] = result[
        "revenue"
    ].round(2)

    result["quantity"] = result[
        "quantity"
    ].round(2)

    result["profit"] = result[
        "profit"
    ].round(2)

    return result.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


# ============================================================
# MONTHLY SALES TREND
# ============================================================

def calculate_monthly_sales(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Create monthly sales aggregation."""

    dates = _date(
        df,
        mapping,
        "order_date",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if dates is None or revenue is None:

        return pd.DataFrame(
            columns=[
                "month",
                "revenue",
                "orders",
                "quantity",
                "profit",
            ]
        )

    working = pd.DataFrame(
        {
            "date": dates,
            "revenue": revenue,
        }
    )

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    working["quantity"] = (
        quantity
        if quantity is not None
        else 0.0
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working["profit"] = (
        profit
        if profit is not None
        else np.nan
    )

    order_column = _column(
        mapping,
        "order_id",
    )

    if order_column and order_column in df.columns:

        working["order_id"] = df[
            order_column
        ].values

    working = working.dropna(
        subset=["date"]
    )

    working["month"] = (
        working["date"]
        .dt.to_period("M")
        .astype(str)
    )

    if "order_id" in working.columns:

        result = (
            working.groupby("month")
            .agg(
                revenue=("revenue", "sum"),
                orders=(
                    "order_id",
                    "nunique",
                ),
                quantity=(
                    "quantity",
                    "sum",
                ),
                profit=(
                    "profit",
                    "sum",
                ),
            )
            .reset_index()
        )

    else:

        result = (
            working.groupby("month")
            .agg(
                revenue=("revenue", "sum"),
                orders=("revenue", "size"),
                quantity=(
                    "quantity",
                    "sum",
                ),
                profit=(
                    "profit",
                    "sum",
                ),
            )
            .reset_index()
        )

    for column in (
        "revenue",
        "quantity",
        "profit",
    ):

        result[column] = result[
            column
        ].round(2)

    return result


# ============================================================
# PERIOD GROWTH
# ============================================================

def calculate_period_growth(
    monthly_sales: pd.DataFrame,
) -> Dict[str, Any]:
    """Calculate latest-month growth versus previous month."""

    if monthly_sales.empty:

        return {
            "available": False,
            "revenue_growth": None,
            "orders_growth": None,
            "quantity_growth": None,
            "current_month": None,
            "previous_month": None,
        }

    if len(monthly_sales) < 2:

        return {
            "available": False,
            "revenue_growth": None,
            "orders_growth": None,
            "quantity_growth": None,
            "current_month": monthly_sales.iloc[
                -1
            ]["month"],
            "previous_month": None,
        }

    current = monthly_sales.iloc[
        -1
    ]

    previous = monthly_sales.iloc[
        -2
    ]

    return {
        "available": True,

        "current_month": current["month"],
        "previous_month": previous["month"],

        "revenue_growth": _growth(
            current["revenue"],
            previous["revenue"],
        ),

        "orders_growth": _growth(
            current["orders"],
            previous["orders"],
        ),

        "quantity_growth": _growth(
            current["quantity"],
            previous["quantity"],
        ),
    }


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def calculate_category_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate revenue/profit performance by category."""

    if not _has(
        df,
        mapping,
        "category",
    ):
        return pd.DataFrame()

    category_column = _column(
        mapping,
        "category",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        {
            "category": df[
                category_column
            ],
            "revenue": revenue,
        }
    )

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    working["quantity"] = (
        quantity
        if quantity is not None
        else 0.0
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working["profit"] = (
        profit
        if profit is not None
        else np.nan
    )

    result = (
        working.dropna(
            subset=["category"]
        )
        .groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["revenue"] = result[
        "revenue"
    ].round(2)

    result["quantity"] = result[
        "quantity"
    ].round(2)

    result["profit"] = result[
        "profit"
    ].round(2)

    if "profit" in result.columns:

        result["profit_margin"] = np.where(
            result["revenue"] != 0,
            (
                result["profit"]
                / result["revenue"]
            )
            * 100,
            np.nan,
        )

        result["profit_margin"] = result[
            "profit_margin"
        ].round(2)

    return result


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

def calculate_product_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    top_n: int = 20,
) -> pd.DataFrame:
    """Calculate top product performance."""

    product_field = None

    if _has(
        df,
        mapping,
        "product_name",
    ):
        product_field = "product_name"

    elif _has(
        df,
        mapping,
        "product_id",
    ):
        product_field = "product_id"

    if product_field is None:
        return pd.DataFrame()

    product_column = _column(
        mapping,
        product_field,
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    working = pd.DataFrame(
        {
            "product": df[
                product_column
            ],
            "revenue": revenue,
        }
    )

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    working["quantity"] = (
        quantity
        if quantity is not None
        else 0.0
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working["profit"] = (
        profit
        if profit is not None
        else np.nan
    )

    result = (
        working.dropna(
            subset=["product"]
        )
        .groupby("product")
        .agg(
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )

    for column in (
        "revenue",
        "quantity",
        "profit",
    ):
        result[column] = result[
            column
        ].round(2)

    return result


# ============================================================
# SELLER PERFORMANCE
# ============================================================

def calculate_seller_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    top_n: int = 20,
) -> pd.DataFrame:
    """Basic seller performance used by Market & Seller Intelligence."""

    seller_field = None

    if _has(
        df,
        mapping,
        "seller_name",
    ):
        seller_field = "seller_name"

    elif _has(
        df,
        mapping,
        "seller_id",
    ):
        seller_field = "seller_id"

    if seller_field is None:
        return pd.DataFrame()

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    seller_column = _column(
        mapping,
        seller_field,
    )

    working = pd.DataFrame(
        {
            "seller": df[
                seller_column
            ],
            "revenue": revenue,
        }
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working["profit"] = (
        profit
        if profit is not None
        else np.nan
    )

    result = (
        working.dropna(
            subset=["seller"]
        )
        .groupby("seller")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )

    result["revenue"] = result[
        "revenue"
    ].round(2)

    result["profit"] = result[
        "profit"
    ].round(2)

    return result


# ============================================================
# GEOGRAPHIC PERFORMANCE
# ============================================================

def calculate_geographic_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    level: str = "customer_state",
) -> pd.DataFrame:
    """Calculate sales performance by geographic field."""

    allowed = {
        "customer_city",
        "customer_state",
        "customer_country",
        "region",
        "market",
    }

    if level not in allowed:
        raise ValueError(
            f"Unsupported geographic level: {level}"
        )

    if not _has(
        df,
        mapping,
        level,
    ):
        return pd.DataFrame()

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    location_column = _column(
        mapping,
        level,
    )

    working = pd.DataFrame(
        {
            "location": df[
                location_column
            ],
            "revenue": revenue,
        }
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working["profit"] = (
        profit
        if profit is not None
        else np.nan
    )

    result = (
        working.dropna(
            subset=["location"]
        )
        .groupby("location")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["revenue"] = result[
        "revenue"
    ].round(2)

    result["profit"] = result[
        "profit"
    ].round(2)

    return result


# ============================================================
# SALES STATUS
# ============================================================

def calculate_order_status(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate order distribution by status."""

    if not _has(
        df,
        mapping,
        "order_status",
    ):
        return pd.DataFrame()

    status_column = _column(
        mapping,
        "order_status",
    )

    result = (
        df[status_column]
        .value_counts(
            dropna=False
        )
        .rename_axis("status")
        .reset_index(
            name="orders"
        )
    )

    result["percentage"] = (
        result["orders"]
        / max(len(df), 1)
        * 100
    ).round(2)

    return result


# ============================================================
# PROFITABILITY ANALYSIS
# ============================================================

def calculate_profitability(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculate profitability metrics."""

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    cost = _numeric(
        df,
        mapping,
        "cost",
    )

    result = {
        "available": False,
        "revenue": None,
        "profit": None,
        "cost": None,
        "profit_margin": None,
        "cost_ratio": None,
    }

    if revenue is None:
        return result

    revenue_total = _safe_float(
        revenue.sum()
    )

    result["revenue"] = _round(
        revenue_total
    )

    if profit is not None:

        profit_total = _safe_float(
            profit.sum()
        )

        result["profit"] = _round(
            profit_total
        )

        if revenue_total != 0:

            result["profit_margin"] = _round(
                (
                    profit_total
                    / revenue_total
                )
                * 100
            )

        result["available"] = True

    if cost is not None:

        cost_total = _safe_float(
            cost.sum()
        )

        result["cost"] = _round(
            cost_total
        )

        if revenue_total != 0:

            result["cost_ratio"] = _round(
                (
                    cost_total
                    / revenue_total
                )
                * 100
            )

    return result


# ============================================================
# DATASET SUMMARY
# ============================================================

def build_dataset_summary(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a compact summary for the Executive Intelligence layer."""

    basic = calculate_basic_metrics(
        df,
        mapping,
    )

    date_range = calculate_date_range(
        df,
        mapping,
    )

    monthly = calculate_monthly_sales(
        df,
        mapping,
    )

    growth = calculate_period_growth(
        monthly
    )

    profitability = calculate_profitability(
        df,
        mapping,
    )

    return {
        "basic": basic,
        "date_range": date_range,
        "growth": growth,
        "profitability": profitability,
    }


# ============================================================
# MAIN CORE ANALYTICS FUNCTION
# ============================================================

def run_core_analytics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the complete core DataPulse analytics engine.

    This is the main public interface consumed by app.py.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "run_core_analytics() expects a pandas DataFrame."
        )

    if not isinstance(
        mapping,
        dict,
    ):
        raise TypeError(
            "mapping must be the dictionary returned by map_schema()."
        )

    # --------------------------------------------------------
    # Core calculations
    # --------------------------------------------------------

    basic = calculate_basic_metrics(
        df,
        mapping,
    )

    date_range = calculate_date_range(
        df,
        mapping,
    )

    daily_sales = calculate_daily_sales(
        df,
        mapping,
    )

    monthly_sales = calculate_monthly_sales(
        df,
        mapping,
    )

    growth = calculate_period_growth(
        monthly_sales
    )

    profitability = calculate_profitability(
        df,
        mapping,
    )

    # --------------------------------------------------------
    # Dimension analysis
    # --------------------------------------------------------

    category_performance = (
        calculate_category_performance(
            df,
            mapping,
        )
    )

    product_performance = (
        calculate_product_performance(
            df,
            mapping,
        )
    )

    seller_performance = (
        calculate_seller_performance(
            df,
            mapping,
        )
    )

    geographic_performance = (
        calculate_geographic_performance(
            df,
            mapping,
            level="customer_state",
        )
    )

    order_status = (
        calculate_order_status(
            df,
            mapping,
        )
    )

    # --------------------------------------------------------
    # Executive summary metrics
    # --------------------------------------------------------

    latest_month_revenue = None
    previous_month_revenue = None

    if not monthly_sales.empty:

        latest_month_revenue = _round(
            monthly_sales.iloc[-1][
                "revenue"
            ]
        )

        if len(monthly_sales) >= 2:

            previous_month_revenue = _round(
                monthly_sales.iloc[-2][
                    "revenue"
                ]
            )

    # --------------------------------------------------------
    # Return stable contract
    # --------------------------------------------------------

    return {
        "summary": {
            "rows": basic["rows"],
            "orders": basic["orders"],
            "customers": basic["customers"],
            "products": basic["products"],
            "revenue": basic["revenue"],
            "quantity": basic["quantity"],
            "profit": basic["profit"],
            "average_order_value": basic[
                "average_order_value"
            ],
            "average_unit_price": basic[
                "average_unit_price"
            ],
            "profit_margin": basic[
                "profit_margin"
            ],
        },

        "date_range": date_range,

        "growth": growth,

        "profitability": profitability,

        "latest_period": {
            "revenue": latest_month_revenue,
            "previous_revenue": previous_month_revenue,
        },

        "trends": {
            "daily_sales": daily_sales,
            "monthly_sales": monthly_sales,
        },

        "dimensions": {
            "category_performance": category_performance,
            "product_performance": product_performance,
            "seller_performance": seller_performance,
            "geographic_performance": geographic_performance,
            "order_status": order_status,
        },

        "capabilities": {
            "time_series": not monthly_sales.empty,
            "profitability": profitability[
                "available"
            ],
            "category_analysis": not category_performance.empty,
            "product_analysis": not product_performance.empty,
            "seller_analysis": not seller_performance.empty,
            "geographic_analysis": not geographic_performance.empty,
            "order_status_analysis": not order_status.empty,
        },
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "calculate_basic_metrics",
    "calculate_date_range",
    "calculate_daily_sales",
    "calculate_monthly_sales",
    "calculate_period_growth",
    "calculate_category_performance",
    "calculate_product_performance",
    "calculate_seller_performance",
    "calculate_geographic_performance",
    "calculate_order_status",
    "calculate_profitability",
    "build_dataset_summary",
    "run_core_analytics",
]