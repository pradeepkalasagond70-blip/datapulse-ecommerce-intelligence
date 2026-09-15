"""
DataPulse — Product Intelligence Engine

Responsibilities
----------------
- Product-level performance
- Category-level performance
- Revenue contribution
- Quantity contribution
- Profitability
- Product margins
- Average selling price
- Top / bottom products
- Product concentration
- Category comparison
- Product decision signals

This module consumes the stable DataPulse schema mapping.

It does NOT:
- modify the original dataframe
- perform EDA
- perform schema mapping
- train ML models
- generate final natural-language recommendations
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.schema_mapper import get_source_column


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_N = 20


# ============================================================
# HELPERS
# ============================================================

def _column(
    mapping: Dict[str, Any],
    field: str,
) -> Optional[str]:
    return get_source_column(
        mapping,
        field,
    )


def _has(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    field: str,
) -> bool:

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


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

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


# ============================================================
# PRODUCT FIELD SELECTION
# ============================================================

def _get_product_field(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Optional[str]:

    if _has(
        df,
        mapping,
        "product_name",
    ):
        return "product_name"

    if _has(
        df,
        mapping,
        "product_id",
    ):
        return "product_id"

    return None


# ============================================================
# PRODUCT BASE
# ============================================================

def build_product_base(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build an internal transaction-level product dataframe.

    The original dataframe is never modified.
    """

    product_field = _get_product_field(
        df,
        mapping,
    )

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

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    unit_price = _numeric(
        df,
        mapping,
        "unit_price",
    )

    cost = _numeric(
        df,
        mapping,
        "cost",
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    result = pd.DataFrame(
        {
            "product": df[
                product_column
            ].astype("string"),

            "revenue": revenue,

            "quantity": (
                quantity
                if quantity is not None
                else 0.0
            ),

            "unit_price": (
                unit_price
                if unit_price is not None
                else np.nan
            ),

            "cost": (
                cost
                if cost is not None
                else np.nan
            ),

            "profit": (
                profit
                if profit is not None
                else np.nan
            ),
        }
    )

    # Optional product dimensions
    for field in (
        "category",
        "subcategory",
        "brand",
    ):

        source_column = _column(
            mapping,
            field,
        )

        if (
            source_column
            and source_column in df.columns
        ):

            result[field] = df[
                source_column
            ].values

    result = result[
        result["product"].notna()
        & (
            result["product"]
            .astype(str)
            .str.strip()
            .ne("")
        )
    ].copy()

    return result.reset_index(
        drop=True
    )


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

def calculate_product_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Calculate detailed product-level performance.
    """

    base = build_product_base(
        df,
        mapping,
    )

    if base.empty:
        return pd.DataFrame()

    aggregation = {
        "revenue": (
            "revenue",
            "sum",
        ),

        "quantity": (
            "quantity",
            "sum",
        ),

        "average_unit_price": (
            "unit_price",
            "mean",
        ),

        "cost": (
            "cost",
            "sum",
        ),

        "profit": (
            "profit",
            "sum",
        ),
    }

    result = (
        base.groupby(
            "product",
            dropna=True,
        )
        .agg(**aggregation)
        .reset_index()
    )

    # --------------------------------------------------------
    # Order count
    # --------------------------------------------------------

    result["transactions"] = (
        base.groupby(
            "product",
            dropna=True,
        )
        .size()
        .values
    )

    # --------------------------------------------------------
    # Revenue contribution
    # --------------------------------------------------------

    total_revenue = _safe_float(
        result["revenue"].sum()
    )

    if total_revenue != 0:

        result["revenue_contribution_pct"] = (
            result["revenue"]
            / total_revenue
            * 100
        )

    else:

        result["revenue_contribution_pct"] = 0.0

    # --------------------------------------------------------
    # Profit margin
    # --------------------------------------------------------

    result["profit_margin_pct"] = np.where(
        result["revenue"] != 0,
        (
            result["profit"]
            / result["revenue"]
        )
        * 100,
        np.nan,
    )

    # --------------------------------------------------------
    # Revenue per transaction
    # --------------------------------------------------------

    result["revenue_per_transaction"] = np.where(
        result["transactions"] != 0,
        result["revenue"]
        / result["transactions"],
        0.0,
    )

    # --------------------------------------------------------
    # Clean output
    # --------------------------------------------------------

    for column in (
        "revenue",
        "quantity",
        "average_unit_price",
        "cost",
        "profit",
        "revenue_contribution_pct",
        "profit_margin_pct",
        "revenue_per_transaction",
    ):

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).round(2)

    return (
        result
        .sort_values(
            "revenue",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# COMPLETE PRODUCT PERFORMANCE
# ============================================================

def calculate_all_product_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate performance for every product.

    Unlike calculate_product_performance(), this does not limit
    the result to the top N products.
    """

    base = build_product_base(
        df,
        mapping,
    )

    if base.empty:
        return pd.DataFrame()

    result = (
        base.groupby(
            "product",
            dropna=True,
        )
        .agg(
            revenue=(
                "revenue",
                "sum",
            ),
            quantity=(
                "quantity",
                "sum",
            ),
            average_unit_price=(
                "unit_price",
                "mean",
            ),
            cost=(
                "cost",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            transactions=(
                "product",
                "size",
            ),
        )
        .reset_index()
    )

    total_revenue = _safe_float(
        result["revenue"].sum()
    )

    result["revenue_contribution_pct"] = np.where(
        total_revenue != 0,
        (
            result["revenue"]
            / total_revenue
        )
        * 100,
        0.0,
    )

    result["profit_margin_pct"] = np.where(
        result["revenue"] != 0,
        (
            result["profit"]
            / result["revenue"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "quantity",
        "average_unit_price",
        "cost",
        "profit",
        "revenue_contribution_pct",
        "profit_margin_pct",
    ):

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).round(2)

    return result.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def calculate_category_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate revenue and profitability by category."""

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

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    result = pd.DataFrame(
        {
            "category": df[
                category_column
            ],
            "revenue": revenue,
            "quantity": (
                quantity
                if quantity is not None
                else 0.0
            ),
            "profit": (
                profit
                if profit is not None
                else np.nan
            ),
        }
    )

    result = (
        result.dropna(
            subset=["category"]
        )
        .groupby(
            "category",
            dropna=False,
        )
        .agg(
            revenue=(
                "revenue",
                "sum",
            ),
            quantity=(
                "quantity",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            transactions=(
                "revenue",
                "size",
            ),
        )
        .reset_index()
    )

    total_revenue = _safe_float(
        result["revenue"].sum()
    )

    result["revenue_contribution_pct"] = np.where(
        total_revenue != 0,
        (
            result["revenue"]
            / total_revenue
        )
        * 100,
        0.0,
    )

    result["profit_margin_pct"] = np.where(
        result["revenue"] != 0,
        (
            result["profit"]
            / result["revenue"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "quantity",
        "profit",
        "revenue_contribution_pct",
        "profit_margin_pct",
    ):

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).round(2)

    return result.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# BRAND PERFORMANCE
# ============================================================

def calculate_brand_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate performance by brand."""

    if not _has(
        df,
        mapping,
        "brand",
    ):
        return pd.DataFrame()

    brand_column = _column(
        mapping,
        "brand",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working = pd.DataFrame(
        {
            "brand": df[
                brand_column
            ],
            "revenue": revenue,
            "quantity": (
                quantity
                if quantity is not None
                else 0.0
            ),
            "profit": (
                profit
                if profit is not None
                else np.nan
            ),
        }
    )

    result = (
        working.dropna(
            subset=["brand"]
        )
        .groupby(
            "brand",
            dropna=False,
        )
        .agg(
            revenue=(
                "revenue",
                "sum",
            ),
            quantity=(
                "quantity",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            transactions=(
                "revenue",
                "size",
            ),
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

    result["profit_margin_pct"] = np.where(
        result["revenue"] != 0,
        (
            result["profit"]
            / result["revenue"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "quantity",
        "profit",
        "profit_margin_pct",
    ):

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).round(2)

    return result


# ============================================================
# TOP / BOTTOM PRODUCTS
# ============================================================

def get_top_products(
    product_performance: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Return highest-revenue products."""

    if product_performance.empty:
        return pd.DataFrame()

    return (
        product_performance
        .sort_values(
            "revenue",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


def get_bottom_products(
    product_performance: pd.DataFrame,
    bottom_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """
    Return lowest-revenue products.

    This should be interpreted together with product volume and
    profitability rather than as an automatic recommendation to
    discontinue a product.
    """

    if product_performance.empty:
        return pd.DataFrame()

    return (
        product_performance
        .sort_values(
            "revenue",
            ascending=True,
        )
        .head(bottom_n)
        .reset_index(
            drop=True
        )
    )


def get_most_profitable_products(
    product_performance: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Return products ranked by absolute profit."""

    if (
        product_performance.empty
        or "profit" not in product_performance.columns
    ):
        return pd.DataFrame()

    return (
        product_performance
        .sort_values(
            "profit",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRODUCT CONCENTRATION
# ============================================================

def calculate_product_concentration(
    product_performance: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Measure the percentage of revenue generated by the top products.
    """

    if product_performance.empty:

        return {
            "available": False,
            "top_1_share": None,
            "top_5_share": None,
            "top_10_share": None,
            "total_products": 0,
        }

    revenue = (
        product_performance[
            "revenue"
        ]
        .fillna(0)
        .sort_values(
            ascending=False
        )
    )

    total_revenue = _safe_float(
        revenue.sum()
    )

    if total_revenue == 0:

        return {
            "available": False,
            "top_1_share": None,
            "top_5_share": None,
            "top_10_share": None,
            "total_products": int(
                len(revenue)
            ),
        }

    return {
        "available": True,

        "total_products": int(
            len(revenue)
        ),

        "top_1_share": round(
            revenue.head(1).sum()
            / total_revenue
            * 100,
            2,
        ),

        "top_5_share": round(
            revenue.head(5).sum()
            / total_revenue
            * 100,
            2,
        ),

        "top_10_share": round(
            revenue.head(10).sum()
            / total_revenue
            * 100,
            2,
        ),
    }


# ============================================================
# PRODUCT PROFITABILITY
# ============================================================

def calculate_product_profitability(
    product_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add business-oriented profitability classifications.

    These are decision signals, not automatic business decisions.
    """

    if product_performance.empty:
        return product_performance.copy()

    result = product_performance.copy()

    # --------------------------------------------------------
    # Margin classification
    # --------------------------------------------------------

    def classify_margin(
        margin: Any,
    ) -> str:

        value = _safe_float(
            margin,
            default=np.nan,
        )

        if not np.isfinite(value):
            return "Unknown"

        if value >= 30:
            return "High Margin"

        if value >= 15:
            return "Healthy Margin"

        if value >= 0:
            return "Low Margin"

        return "Negative Margin"

    result["margin_class"] = (
        result[
            "profit_margin_pct"
        ]
        .apply(classify_margin)
    )

    # --------------------------------------------------------
    # Profitability signal
    # --------------------------------------------------------

    def profitability_signal(
        row: pd.Series,
    ) -> str:

        revenue = _safe_float(
            row.get("revenue")
        )

        profit = _safe_float(
            row.get("profit"),
            default=np.nan,
        )

        margin = _safe_float(
            row.get("profit_margin_pct"),
            default=np.nan,
        )

        if not np.isfinite(profit):
            return "Profit data unavailable"

        if profit < 0:
            return "Loss Making"

        if np.isfinite(margin) and margin < 10:
            return "Review Pricing"

        if revenue > 0 and profit > 0:
            return "Profitable"

        return "Needs Review"

    result["profitability_signal"] = result.apply(
        profitability_signal,
        axis=1,
    )

    return result


# ============================================================
# PRODUCT DECISION MATRIX
# ============================================================

def calculate_product_decision_matrix(
    product_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify products using relative revenue and profitability.

    Matrix:
        High Revenue + High Margin
        High Revenue + Low Margin
        Low Revenue + High Margin
        Low Revenue + Low Margin
    """

    if product_performance.empty:
        return product_performance.copy()

    result = product_performance.copy()

    revenue_median = _safe_float(
        result["revenue"].median()
    )

    margin_series = result[
        "profit_margin_pct"
    ]

    margin_median = _safe_float(
        margin_series.median(),
        default=np.nan,
    )

    def classify(
        row: pd.Series,
    ) -> str:

        revenue = _safe_float(
            row.get("revenue")
        )

        margin = _safe_float(
            row.get(
                "profit_margin_pct"
            ),
            default=np.nan,
        )

        high_revenue = (
            revenue >= revenue_median
        )

        high_margin = (
            np.isfinite(margin)
            and np.isfinite(margin_median)
            and margin >= margin_median
        )

        if high_revenue and high_margin:
            return "Scale & Protect"

        if high_revenue and not high_margin:
            return "Revenue Driver — Margin Review"

        if not high_revenue and high_margin:
            return "Growth Opportunity"

        if not high_revenue and not high_margin:
            return "Low Priority"

        return "Needs Review"

    result["decision_class"] = result.apply(
        classify,
        axis=1,
    )

    return result


# ============================================================
# CATEGORY DECISION MATRIX
# ============================================================

def calculate_category_decision_matrix(
    category_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the same revenue/margin decision framework to categories."""

    if category_performance.empty:
        return category_performance.copy()

    result = category_performance.copy()

    revenue_median = _safe_float(
        result["revenue"].median()
    )

    margin_median = _safe_float(
        result[
            "profit_margin_pct"
        ].median(),
        default=np.nan,
    )

    def classify(
        row: pd.Series,
    ) -> str:

        revenue = _safe_float(
            row.get("revenue")
        )

        margin = _safe_float(
            row.get(
                "profit_margin_pct"
            ),
            default=np.nan,
        )

        high_revenue = (
            revenue >= revenue_median
        )

        high_margin = (
            np.isfinite(margin)
            and np.isfinite(margin_median)
            and margin >= margin_median
        )

        if high_revenue and high_margin:
            return "Core Category"

        if high_revenue and not high_margin:
            return "Margin Optimization"

        if not high_revenue and high_margin:
            return "Growth Opportunity"

        return "Low Priority"

    result["decision_class"] = result.apply(
        classify,
        axis=1,
    )

    return result


# ============================================================
# SUBCATEGORY PERFORMANCE
# ============================================================

def calculate_subcategory_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate performance by subcategory."""

    if not _has(
        df,
        mapping,
        "subcategory",
    ):
        return pd.DataFrame()

    subcategory_column = _column(
        mapping,
        "subcategory",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    quantity = _numeric(
        df,
        mapping,
        "quantity",
    )

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    working = pd.DataFrame(
        {
            "subcategory": df[
                subcategory_column
            ],
            "revenue": revenue,
            "quantity": (
                quantity
                if quantity is not None
                else 0.0
            ),
            "profit": (
                profit
                if profit is not None
                else np.nan
            ),
        }
    )

    result = (
        working.dropna(
            subset=["subcategory"]
        )
        .groupby(
            "subcategory",
            dropna=False,
        )
        .agg(
            revenue=(
                "revenue",
                "sum",
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
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["profit_margin_pct"] = np.where(
        result["revenue"] != 0,
        (
            result["profit"]
            / result["revenue"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "quantity",
        "profit",
        "profit_margin_pct",
    ):

        result[column] = result[
            column
        ].round(2)

    return result


# ============================================================
# MAIN PRODUCT ANALYTICS
# ============================================================

def run_product_analytics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the complete DataPulse Product Intelligence engine.

    Public interface consumed by app.py.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "run_product_analytics() expects a pandas DataFrame."
        )

    if not isinstance(
        mapping,
        dict,
    ):
        raise TypeError(
            "mapping must be the dictionary returned by map_schema()."
        )

    # --------------------------------------------------------
    # Product availability
    # --------------------------------------------------------

    product_field = _get_product_field(
        df,
        mapping,
    )

    if product_field is None:

        return {
            "available": False,

            "summary": {
                "products": 0,
                "categories": 0,
                "total_product_revenue": None,
                "average_product_revenue": None,
                "top_product": None,
            },

            "product_performance": pd.DataFrame(),
            "all_products": pd.DataFrame(),
            "category_performance": pd.DataFrame(),
            "subcategory_performance": pd.DataFrame(),
            "brand_performance": pd.DataFrame(),
            "top_products": pd.DataFrame(),
            "bottom_products": pd.DataFrame(),
            "most_profitable_products": pd.DataFrame(),
            "decision_matrix": pd.DataFrame(),
            "category_decision_matrix": pd.DataFrame(),

            "concentration": {
                "available": False,
            },

            "capabilities": {
                "product_intelligence": False,
                "category_analysis": False,
                "subcategory_analysis": False,
                "brand_analysis": False,
                "profitability_analysis": False,
                "decision_matrix": False,
            },
        }

    # --------------------------------------------------------
    # Main product calculations
    # --------------------------------------------------------

    all_products = calculate_all_product_performance(
        df,
        mapping,
    )

    product_performance = calculate_product_profitability(
        all_products
    )

    product_performance = calculate_product_decision_matrix(
        product_performance
    )

    category_performance = (
        calculate_category_performance(
            df,
            mapping,
        )
    )

    category_performance = (
        calculate_category_decision_matrix(
            category_performance
        )
    )

    subcategory_performance = (
        calculate_subcategory_performance(
            df,
            mapping,
        )
    )

    brand_performance = (
        calculate_brand_performance(
            df,
            mapping,
        )
    )

    # --------------------------------------------------------
    # Top / bottom
    # --------------------------------------------------------

    top_products = get_top_products(
        product_performance,
        DEFAULT_TOP_N,
    )

    bottom_products = get_bottom_products(
        product_performance,
        DEFAULT_TOP_N,
    )

    most_profitable_products = (
        get_most_profitable_products(
            product_performance,
            DEFAULT_TOP_N,
        )
    )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    concentration = (
        calculate_product_concentration(
            all_products
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    product_count = int(
        len(all_products)
    )

    category_count = int(
        len(category_performance)
    )

    total_product_revenue = _safe_float(
        all_products["revenue"].sum()
    )

    average_product_revenue = (
        total_product_revenue
        / product_count
        if product_count
        else 0.0
    )

    top_product = None

    if not all_products.empty:

        top_product = {
            "product": all_products.iloc[
                0
            ]["product"],

            "revenue": _round(
                all_products.iloc[
                    0
                ]["revenue"]
            ),
        }

    # --------------------------------------------------------
    # Final contract
    # --------------------------------------------------------

    return {
        "available": True,

        "summary": {
            "products": product_count,
            "categories": category_count,

            "total_product_revenue": _round(
                total_product_revenue
            ),

            "average_product_revenue": _round(
                average_product_revenue
            ),

            "top_product": top_product,
        },

        "product_performance": product_performance,

        "all_products": all_products,

        "category_performance": (
            category_performance
        ),

        "subcategory_performance": (
            subcategory_performance
        ),

        "brand_performance": (
            brand_performance
        ),

        "top_products": top_products,

        "bottom_products": bottom_products,

        "most_profitable_products": (
            most_profitable_products
        ),

        "decision_matrix": (
            product_performance
        ),

        "category_decision_matrix": (
            category_performance
        ),

        "concentration": concentration,

        "capabilities": {
            "product_intelligence": True,

            "category_analysis": (
                not category_performance.empty
            ),

            "subcategory_analysis": (
                not subcategory_performance.empty
            ),

            "brand_analysis": (
                not brand_performance.empty
            ),

            "profitability_analysis": (
                "profit"
                in all_products.columns
                and all_products["profit"]
                .notna()
                .any()
            ),

            "decision_matrix": (
                not product_performance.empty
            ),
        },
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "build_product_base",
    "calculate_product_performance",
    "calculate_all_product_performance",
    "calculate_category_performance",
    "calculate_brand_performance",
    "get_top_products",
    "get_bottom_products",
    "get_most_profitable_products",
    "calculate_product_concentration",
    "calculate_product_profitability",
    "calculate_product_decision_matrix",
    "calculate_category_decision_matrix",
    "calculate_subcategory_performance",
    "run_product_analytics",
]