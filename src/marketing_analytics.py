"""
DataPulse — Campaign Impact Analytics Engine

Internal filename:
    marketing_analytics.py

Product/UI name:
    Campaign Impact

Purpose
-------
Answers:

    "Did our campaigns actually impact the business?"

Capabilities
------------
- Campaign revenue
- Campaign orders
- Campaign customers
- Campaign quantity
- Campaign profit
- Campaign cost/spend
- ROI
- ROAS
- Average order value
- Discount analysis
- Campaign ranking
- Campaign contribution
- Campaign profitability
- Customer response by campaign
- Channel performance
- Campaign decision classification
- Before/after campaign comparison when dates are available

This module does NOT:
- perform EDA
- modify the uploaded dataframe
- perform schema mapping
- train ML models
- generate final natural-language recommendations

It produces structured outputs for:
    - Campaign Impact page
    - Insight Engine
    - Business Decision Center
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
    """Return source column mapped to a canonical field."""

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


def _date(
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

    return pd.to_datetime(
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
# CAMPAIGN FIELD SELECTION
# ============================================================

def _get_campaign_field(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Optional[str]:
    """
    Select the strongest campaign identifier.

    Priority:
        campaign_name
        campaign_id
    """

    if _has(
        df,
        mapping,
        "campaign_name",
    ):
        return "campaign_name"

    if _has(
        df,
        mapping,
        "campaign_id",
    ):
        return "campaign_id"

    return None


# ============================================================
# CAMPAIGN BASE
# ============================================================

def build_campaign_base(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build internal transaction-level campaign data.

    The source dataframe is never modified.
    """

    campaign_field = _get_campaign_field(
        df,
        mapping,
    )

    if campaign_field is None:
        return pd.DataFrame()

    campaign_column = _column(
        mapping,
        campaign_field,
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

    campaign_cost = _numeric(
        df,
        mapping,
        "campaign_cost",
    )

    baseline_sales = _numeric(
        df,
        mapping,
        "campaign_baseline_sales",
    )

    discount = _numeric(
        df,
        mapping,
        "discount",
    )

    order_date = _date(
        df,
        mapping,
        "order_date",
    )

    result = pd.DataFrame(
        {
            "campaign": df[
                campaign_column
            ].astype("string"),

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

            "campaign_cost": (
                campaign_cost
                if campaign_cost is not None
                else np.nan
            ),

            "baseline_sales": (
                baseline_sales
                if baseline_sales is not None
                else np.nan
            ),

            "discount": (
                discount
                if discount is not None
                else np.nan
            ),

            "order_date": (
                order_date
                if order_date is not None
                else pd.NaT
            ),
        }
    )

    # --------------------------------------------------------
    # Optional identifiers
    # --------------------------------------------------------

    for field, output_name in (
        ("order_id", "order_id"),
        ("customer_id", "customer_id"),
        ("product_id", "product_id"),
    ):

        source_column = _column(
            mapping,
            field,
        )

        if (
            source_column
            and source_column in df.columns
        ):

            result[output_name] = df[
                source_column
            ].values

    # --------------------------------------------------------
    # Optional dimensions
    # --------------------------------------------------------

    for field in (
        "campaign_type",
        "channel",
        "category",
        "customer_segment",
        "customer_state",
        "customer_city",
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

    # --------------------------------------------------------
    # Remove missing campaign identifiers
    # --------------------------------------------------------

    result = result[
        result["campaign"].notna()
        & (
            result["campaign"]
            .astype(str)
            .str.strip()
            .ne("")
        )
    ].copy()

    return result.reset_index(
        drop=True
    )


# ============================================================
# CAMPAIGN PERFORMANCE
# ============================================================

def calculate_campaign_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate complete campaign-level performance.
    """

    base = build_campaign_base(
        df,
        mapping,
    )

    if base.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    aggregation = {
        "revenue": (
            "revenue",
            "sum",
        ),

        "quantity": (
            "quantity",
            "sum",
        ),

        "profit": (
            "profit",
            "sum",
        ),

        "campaign_cost": (
            "campaign_cost",
            "sum",
        ),

        "baseline_sales": (
            "baseline_sales",
            "sum",
        ),

        "average_discount": (
            "discount",
            "mean",
        ),
    }

    result = (
        base.groupby(
            "campaign",
            dropna=True,
        )
        .agg(**aggregation)
        .reset_index()
    )

    # --------------------------------------------------------
    # Orders
    # --------------------------------------------------------

    if "order_id" in base.columns:

        orders = (
            base.groupby(
                "campaign"
            )["order_id"]
            .nunique()
            .rename("orders")
        )

    else:

        orders = (
            base.groupby(
                "campaign"
            )
            .size()
            .rename("orders")
        )

    result = result.merge(
        orders.reset_index(),
        on="campaign",
        how="left",
    )

    # --------------------------------------------------------
    # Customers
    # --------------------------------------------------------

    if "customer_id" in base.columns:

        customers = (
            base.groupby(
                "campaign"
            )["customer_id"]
            .nunique()
            .rename("customers")
        )

        result = result.merge(
            customers.reset_index(),
            on="campaign",
            how="left",
        )

    else:

        result["customers"] = np.nan

    # --------------------------------------------------------
    # Product count
    # --------------------------------------------------------

    if "product_id" in base.columns:

        products = (
            base.groupby(
                "campaign"
            )["product_id"]
            .nunique()
            .rename("products")
        )

        result = result.merge(
            products.reset_index(),
            on="campaign",
            how="left",
        )

    else:

        result["products"] = np.nan

    # --------------------------------------------------------
    # AOV
    # --------------------------------------------------------

    result["average_order_value"] = np.where(
        result["orders"] != 0,
        result["revenue"]
        / result["orders"],
        np.nan,
    )

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
    # ROAS
    # --------------------------------------------------------

    result["roas"] = np.where(
        result["campaign_cost"] > 0,
        result["revenue"]
        / result["campaign_cost"],
        np.nan,
    )

    # --------------------------------------------------------
    # ROI
    #
    # ROI = (Profit - Campaign Cost) / Campaign Cost × 100
    # --------------------------------------------------------

    result["roi_pct"] = np.where(
        result["campaign_cost"] > 0,
        (
            (
                result["profit"]
                - result["campaign_cost"]
            )
            / result["campaign_cost"]
        )
        * 100,
        np.nan,
    )

    # --------------------------------------------------------
    # Revenue contribution
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clean numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "revenue",
        "quantity",
        "profit",
        "campaign_cost",
        "baseline_sales",
        "average_discount",
        "orders",
        "customers",
        "products",
        "average_order_value",
        "profit_margin_pct",
        "roas",
        "roi_pct",
        "revenue_contribution_pct",
    ]

    for column in numeric_columns:

        if column in result.columns:

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
# CAMPAIGN TYPE PERFORMANCE
# ============================================================

def calculate_campaign_type_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate performance by campaign type."""

    if not _has(
        df,
        mapping,
        "campaign_type",
    ):
        return pd.DataFrame()

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    campaign_cost = _numeric(
        df,
        mapping,
        "campaign_cost",
    )

    campaign_type_column = _column(
        mapping,
        "campaign_type",
    )

    working = pd.DataFrame(
        {
            "campaign_type": df[
                campaign_type_column
            ],
            "revenue": revenue,
            "profit": (
                profit
                if profit is not None
                else np.nan
            ),
            "campaign_cost": (
                campaign_cost
                if campaign_cost is not None
                else np.nan
            ),
        }
    )

    result = (
        working.dropna(
            subset=["campaign_type"]
        )
        .groupby(
            "campaign_type",
            dropna=False,
        )
        .agg(
            revenue=(
                "revenue",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            campaign_cost=(
                "campaign_cost",
                "sum",
            ),
            campaigns=(
                "campaign_type",
                "size",
            ),
        )
        .reset_index()
    )

    result["roas"] = np.where(
        result["campaign_cost"] > 0,
        result["revenue"]
        / result["campaign_cost"],
        np.nan,
    )

    result["roi_pct"] = np.where(
        result["campaign_cost"] > 0,
        (
            (
                result["profit"]
                - result["campaign_cost"]
            )
            / result["campaign_cost"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "profit",
        "campaign_cost",
        "roas",
        "roi_pct",
    ):

        result[column] = result[
            column
        ].round(2)

    return result.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# CHANNEL PERFORMANCE
# ============================================================

def calculate_channel_performance(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """Calculate campaign/sales performance by channel."""

    if not _has(
        df,
        mapping,
        "channel",
    ):
        return pd.DataFrame()

    channel_column = _column(
        mapping,
        "channel",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    profit = _numeric(
        df,
        mapping,
        "profit",
    )

    campaign_cost = _numeric(
        df,
        mapping,
        "campaign_cost",
    )

    working = pd.DataFrame(
        {
            "channel": df[
                channel_column
            ],
            "revenue": revenue,
            "profit": (
                profit
                if profit is not None
                else np.nan
            ),
            "campaign_cost": (
                campaign_cost
                if campaign_cost is not None
                else np.nan
            ),
        }
    )

    result = (
        working.dropna(
            subset=["channel"]
        )
        .groupby(
            "channel",
            dropna=False,
        )
        .agg(
            revenue=(
                "revenue",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            campaign_cost=(
                "campaign_cost",
                "sum",
            ),
            transactions=(
                "revenue",
                "size",
            ),
        )
        .reset_index()
    )

    result["roas"] = np.where(
        result["campaign_cost"] > 0,
        result["revenue"]
        / result["campaign_cost"],
        np.nan,
    )

    result["roi_pct"] = np.where(
        result["campaign_cost"] > 0,
        (
            (
                result["profit"]
                - result["campaign_cost"]
            )
            / result["campaign_cost"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "profit",
        "campaign_cost",
        "roas",
        "roi_pct",
    ):

        result[column] = result[
            column
        ].round(2)

    return result.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# CAMPAIGN CUSTOMER RESPONSE
# ============================================================

def calculate_campaign_customer_response(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Measure customer response across campaigns.

    Useful for understanding whether campaigns attract:
    - more customers
    - higher-value customers
    - repeat customers
    """

    base = build_campaign_base(
        df,
        mapping,
    )

    if (
        base.empty
        or "customer_id" not in base.columns
    ):
        return pd.DataFrame()

    result = (
        base.groupby(
            "campaign",
            dropna=True,
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),
            revenue=(
                "revenue",
                "sum",
            ),
        )
        .reset_index()
    )

    result["revenue_per_customer"] = np.where(
        result["customers"] != 0,
        result["revenue"]
        / result["customers"],
        np.nan,
    )

    # --------------------------------------------------------
    # Repeat-customer indicator
    # --------------------------------------------------------

    customer_campaign_counts = (
        base.groupby(
            [
                "campaign",
                "customer_id",
            ]
        )
        .size()
        .reset_index(
            name="transactions"
        )
    )

    repeat_counts = (
        customer_campaign_counts[
            customer_campaign_counts[
                "transactions"
            ] > 1
        ]
        .groupby("campaign")
        .size()
        .rename(
            "repeat_customers"
        )
    )

    result = result.merge(
        repeat_counts.reset_index(),
        on="campaign",
        how="left",
    )

    result["repeat_customers"] = (
        result["repeat_customers"]
        .fillna(0)
        .astype(int)
    )

    result["repeat_customer_rate_pct"] = np.where(
        result["customers"] != 0,
        (
            result["repeat_customers"]
            / result["customers"]
        )
        * 100,
        np.nan,
    )

    for column in (
        "revenue",
        "revenue_per_customer",
        "repeat_customer_rate_pct",
    ):

        result[column] = result[
            column
        ].round(2)

    return result.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# CAMPAIGN PRODUCT IMPACT
# ============================================================

def calculate_campaign_product_impact(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate which products/categories generated campaign revenue.
    """

    base = build_campaign_base(
        df,
        mapping,
    )

    if base.empty:
        return pd.DataFrame()

    if "product_id" in base.columns:

        product_column = "product_id"

    elif _has(
        df,
        mapping,
        "product_name",
    ):

        source_column = _column(
            mapping,
            "product_name",
        )

        base["product"] = df[
            source_column
        ].values

        product_column = "product"

    else:

        product_column = None

    if product_column is None:
        return pd.DataFrame()

    result = (
        base.groupby(
            [
                "campaign",
                product_column,
            ],
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
            profit=(
                "profit",
                "sum",
            ),
        )
        .reset_index()
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

    result = result.sort_values(
        [
            "campaign",
            "revenue",
        ],
        ascending=[
            True,
            False,
        ],
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

    return result.reset_index(
        drop=True
    )


# ============================================================
# DISCOUNT IMPACT
# ============================================================

def calculate_discount_impact(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Estimate relationship between discount and sales.

    This is descriptive analytics, not causal inference.
    """

    discount = _numeric(
        df,
        mapping,
        "discount",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if (
        discount is None
        or revenue is None
    ):

        return {
            "available": False,
            "average_discount": None,
            "discounted_transactions": 0,
            "discounted_revenue": None,
            "non_discounted_revenue": None,
            "revenue_difference_pct": None,
        }

    working = pd.DataFrame(
        {
            "discount": discount,
            "revenue": revenue,
        }
    ).dropna(
        subset=[
            "discount",
            "revenue",
        ]
    )

    if working.empty:

        return {
            "available": False,
            "average_discount": None,
            "discounted_transactions": 0,
            "discounted_revenue": None,
            "non_discounted_revenue": None,
            "revenue_difference_pct": None,
        }

    discounted = working[
        working["discount"] > 0
    ]

    non_discounted = working[
        working["discount"] <= 0
    ]

    discounted_revenue = _safe_float(
        discounted["revenue"].sum()
    )

    non_discounted_revenue = _safe_float(
        non_discounted["revenue"].sum()
    )

    discounted_average = (
        discounted["revenue"].mean()
        if not discounted.empty
        else np.nan
    )

    non_discounted_average = (
        non_discounted["revenue"].mean()
        if not non_discounted.empty
        else np.nan
    )

    difference = None

    if (
        np.isfinite(discounted_average)
        and np.isfinite(non_discounted_average)
        and non_discounted_average != 0
    ):

        difference = round(
            (
                (
                    discounted_average
                    - non_discounted_average
                )
                / abs(non_discounted_average)
            )
            * 100,
            2,
        )

    return {
        "available": True,

        "average_discount": round(
            _safe_float(
                working["discount"].mean()
            ),
            2,
        ),

        "discounted_transactions": int(
            len(discounted)
        ),

        "discounted_revenue": round(
            discounted_revenue,
            2,
        ),

        "non_discounted_revenue": round(
            non_discounted_revenue,
            2,
        ),

        "discounted_average_transaction_value": (
            round(
                _safe_float(
                    discounted_average
                ),
                2,
            )
            if np.isfinite(
                discounted_average
            )
            else None
        ),

        "non_discounted_average_transaction_value": (
            round(
                _safe_float(
                    non_discounted_average
                ),
                2,
            )
            if np.isfinite(
                non_discounted_average
            )
            else None
        ),

        "revenue_difference_pct": difference,
    }


# ============================================================
# CAMPAIGN DECISION CLASSIFICATION
# ============================================================

def classify_campaigns(
    campaign_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify campaigns into actionable business groups.

    The classification is descriptive and relative.
    """

    if campaign_performance.empty:
        return campaign_performance.copy()

    result = campaign_performance.copy()

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

        roas = _safe_float(
            row.get(
                "roas"
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

        positive_roas = (
            np.isfinite(roas)
            and roas >= 1
        )

        if (
            high_revenue
            and high_margin
            and positive_roas
        ):
            return "Scale"

        if (
            high_revenue
            and not high_margin
        ):
            return "Optimize Margin"

        if (
            not high_revenue
            and high_margin
        ):
            return "Growth Potential"

        if (
            np.isfinite(roas)
            and roas < 1
        ):
            return "Review Spend"

        return "Monitor"

    result["decision_class"] = result.apply(
        classify,
        axis=1,
    )

    return result


# ============================================================
# TOP / BOTTOM CAMPAIGNS
# ============================================================

def get_top_campaigns(
    campaign_performance: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:

    if campaign_performance.empty:
        return pd.DataFrame()

    return (
        campaign_performance
        .sort_values(
            "revenue",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


def get_bottom_campaigns(
    campaign_performance: pd.DataFrame,
    bottom_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:

    if campaign_performance.empty:
        return pd.DataFrame()

    return (
        campaign_performance
        .sort_values(
            "revenue",
            ascending=True,
        )
        .head(bottom_n)
        .reset_index(
            drop=True
        )
    )


def get_best_roi_campaigns(
    campaign_performance: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:

    if campaign_performance.empty:
        return pd.DataFrame()

    if "roi_pct" not in campaign_performance.columns:
        return pd.DataFrame()

    return (
        campaign_performance
        .dropna(
            subset=["roi_pct"]
        )
        .sort_values(
            "roi_pct",
            ascending=False,
        )
        .head(top_n)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CAMPAIGN CONCENTRATION
# ============================================================

def calculate_campaign_concentration(
    campaign_performance: pd.DataFrame,
) -> Dict[str, Any]:
    """Measure dependence on top campaigns."""

    if campaign_performance.empty:

        return {
            "available": False,
            "total_campaigns": 0,
            "top_1_share": None,
            "top_5_share": None,
            "top_10_share": None,
        }

    revenue = (
        campaign_performance[
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
            "total_campaigns": int(
                len(revenue)
            ),
            "top_1_share": None,
            "top_5_share": None,
            "top_10_share": None,
        }

    return {
        "available": True,

        "total_campaigns": int(
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
# BEFORE / AFTER CAMPAIGN ANALYSIS
# ============================================================

def calculate_campaign_period_impact(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compare campaign-period revenue against the immediately
    preceding equivalent period when campaign start/end dates
    are available.

    This is a descriptive before/after comparison.
    It is NOT a causal estimate of campaign uplift.
    """

    campaign_field = _get_campaign_field(
        df,
        mapping,
    )

    if campaign_field is None:
        return pd.DataFrame()

    campaign_start = _date(
        df,
        mapping,
        "campaign_start_date",
    )

    campaign_end = _date(
        df,
        mapping,
        "campaign_end_date",
    )

    order_date = _date(
        df,
        mapping,
        "order_date",
    )

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if (
        campaign_start is None
        or campaign_end is None
        or order_date is None
        or revenue is None
    ):
        return pd.DataFrame()

    campaign_column = _column(
        mapping,
        campaign_field,
    )

    working = pd.DataFrame(
        {
            "campaign": df[
                campaign_column
            ],
            "campaign_start": campaign_start,
            "campaign_end": campaign_end,
            "order_date": order_date,
            "revenue": revenue,
        }
    )

    working = working.dropna(
        subset=[
            "campaign",
            "campaign_start",
            "campaign_end",
            "order_date",
            "revenue",
        ]
    )

    if working.empty:
        return pd.DataFrame()

    rows = []

    for campaign, group in working.groupby(
        "campaign"
    ):

        start = group[
            "campaign_start"
        ].iloc[0]

        end = group[
            "campaign_end"
        ].iloc[0]

        if end < start:
            continue

        duration = (
            end - start
        ).days + 1

        before_start = (
            start
            - pd.Timedelta(
                days=duration
            )
        )

        before_end = (
            start
            - pd.Timedelta(
                days=1
            )
        )

        campaign_period = group[
            (
                group["order_date"]
                >= start
            )
            & (
                group["order_date"]
                <= end
            )
        ]

        before_period = group[
            (
                group["order_date"]
                >= before_start
            )
            & (
                group["order_date"]
                <= before_end
            )
        ]

        campaign_revenue = _safe_float(
            campaign_period[
                "revenue"
            ].sum()
        )

        before_revenue = _safe_float(
            before_period[
                "revenue"
            ].sum()
        )

        uplift = None

        if before_revenue != 0:

            uplift = (
                (
                    campaign_revenue
                    - before_revenue
                )
                / abs(before_revenue)
            ) * 100

        rows.append(
            {
                "campaign": campaign,

                "campaign_start": start,

                "campaign_end": end,

                "campaign_revenue": round(
                    campaign_revenue,
                    2,
                ),

                "before_period_revenue": round(
                    before_revenue,
                    2,
                ),

                "revenue_uplift_pct": (
                    round(
                        uplift,
                        2,
                    )
                    if uplift is not None
                    else None
                ),

                "comparison_days": duration,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN CAMPAIGN IMPACT ENGINE
# ============================================================

def run_campaign_impact(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the complete Campaign Impact engine.

    Public interface consumed by app.py.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "run_campaign_impact() expects a pandas DataFrame."
        )

    if not isinstance(
        mapping,
        dict,
    ):
        raise TypeError(
            "mapping must be the dictionary returned by map_schema()."
        )

    campaign_field = _get_campaign_field(
        df,
        mapping,
    )

    # --------------------------------------------------------
    # Campaign availability
    # --------------------------------------------------------

    if campaign_field is None:

        return {
            "available": False,

            "summary": {
                "campaigns": 0,
                "campaign_revenue": None,
                "campaign_profit": None,
                "campaign_cost": None,
                "average_campaign_roas": None,
                "average_campaign_roi": None,
            },

            "campaign_performance": pd.DataFrame(),
            "campaign_type_performance": pd.DataFrame(),
            "channel_performance": pd.DataFrame(),
            "customer_response": pd.DataFrame(),
            "product_impact": pd.DataFrame(),
            "period_impact": pd.DataFrame(),
            "top_campaigns": pd.DataFrame(),
            "bottom_campaigns": pd.DataFrame(),
            "best_roi_campaigns": pd.DataFrame(),

            "discount_impact": {
                "available": False,
            },

            "concentration": {
                "available": False,
            },

            "capabilities": {
                "campaign_impact": False,
                "roi_analysis": False,
                "roas_analysis": False,
                "customer_response": False,
                "channel_analysis": False,
                "product_impact": False,
                "before_after_analysis": False,
                "discount_analysis": False,
            },
        }

    # --------------------------------------------------------
    # Main performance
    # --------------------------------------------------------

    campaign_performance = (
        calculate_campaign_performance(
            df,
            mapping,
        )
    )

    if campaign_performance.empty:

        return {
            "available": False,
            "summary": {},
            "campaign_performance": pd.DataFrame(),
            "campaign_type_performance": pd.DataFrame(),
            "channel_performance": pd.DataFrame(),
            "customer_response": pd.DataFrame(),
            "product_impact": pd.DataFrame(),
            "period_impact": pd.DataFrame(),
            "top_campaigns": pd.DataFrame(),
            "bottom_campaigns": pd.DataFrame(),
            "best_roi_campaigns": pd.DataFrame(),
            "discount_impact": {
                "available": False,
            },
            "concentration": {
                "available": False,
            },
            "capabilities": {
                "campaign_impact": False,
                "roi_analysis": False,
                "roas_analysis": False,
                "customer_response": False,
                "channel_analysis": False,
                "product_impact": False,
                "before_after_analysis": False,
                "discount_analysis": False,
            },
        }

    # --------------------------------------------------------
    # Decision classifications
    # --------------------------------------------------------

    campaign_performance = classify_campaigns(
        campaign_performance
    )

    # --------------------------------------------------------
    # Supporting analyses
    # --------------------------------------------------------

    campaign_type_performance = (
        calculate_campaign_type_performance(
            df,
            mapping,
        )
    )

    channel_performance = (
        calculate_channel_performance(
            df,
            mapping,
        )
    )

    customer_response = (
        calculate_campaign_customer_response(
            df,
            mapping,
        )
    )

    product_impact = (
        calculate_campaign_product_impact(
            df,
            mapping,
        )
    )

    period_impact = (
        calculate_campaign_period_impact(
            df,
            mapping,
        )
    )

    discount_impact = (
        calculate_discount_impact(
            df,
            mapping,
        )
    )

    # --------------------------------------------------------
    # Rankings
    # --------------------------------------------------------

    top_campaigns = get_top_campaigns(
        campaign_performance
    )

    bottom_campaigns = get_bottom_campaigns(
        campaign_performance
    )

    best_roi_campaigns = get_best_roi_campaigns(
        campaign_performance
    )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    concentration = (
        calculate_campaign_concentration(
            campaign_performance
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    campaigns = int(
        len(campaign_performance)
    )

    campaign_revenue = _safe_float(
        campaign_performance[
            "revenue"
        ].sum()
    )

    campaign_profit = _safe_float(
        campaign_performance[
            "profit"
        ].sum()
    )

    campaign_orders = int(
        campaign_performance["orders"].sum()
    )

    campaign_cost = _safe_float(
        campaign_performance[
            "campaign_cost"
        ].sum()
    )

    baseline_sales = _safe_float(
        campaign_performance[
            "baseline_sales"
        ].sum()
    )

    valid_roas = campaign_performance[
        "roas"
    ].dropna()

    valid_roi = campaign_performance[
        "roi_pct"
    ].dropna()

    average_campaign_roas = (
        _safe_float(
            valid_roas.mean(),
            default=np.nan,
        )
        if not valid_roas.empty
        else None
    )

    average_campaign_roi = (
        _safe_float(
            valid_roi.mean(),
            default=np.nan,
        )
        if not valid_roi.empty
        else None
    )

    revenue_uplift_pct = (
        round(
            (
                (campaign_revenue - baseline_sales)
                / abs(baseline_sales)
            )
            * 100,
            2,
        )
        if baseline_sales != 0
        else None
    )

    # --------------------------------------------------------
    # Final contract
    # --------------------------------------------------------

    return {
        "available": True,

        "summary": {
            "campaigns": campaigns,

            "campaign_orders": campaign_orders,

            "campaign_revenue": _round(
                campaign_revenue
            ),

            "campaign_profit": _round(
                campaign_profit
            ),

            "campaign_cost": _round(
                campaign_cost
            ),

            "baseline_sales": _round(
                baseline_sales
            ),

            "revenue_uplift_pct": revenue_uplift_pct,

            "average_campaign_roas": (
                round(
                    average_campaign_roas,
                    2,
                )
                if (
                    average_campaign_roas is not None
                    and np.isfinite(
                        average_campaign_roas
                    )
                )
                else None
            ),

            "average_campaign_roi": (
                round(
                    average_campaign_roi,
                    2,
                )
                if (
                    average_campaign_roi is not None
                    and np.isfinite(
                        average_campaign_roi
                    )
                )
                else None
            ),
        },

        "campaign_performance": (
            campaign_performance
        ),

        "campaign_type_performance": (
            campaign_type_performance
        ),

        "channel_performance": (
            channel_performance
        ),

        "customer_response": (
            customer_response
        ),

        "product_impact": (
            product_impact
        ),

        "period_impact": (
            period_impact
        ),

        "top_campaigns": (
            top_campaigns
        ),

        "bottom_campaigns": (
            bottom_campaigns
        ),

        "best_roi_campaigns": (
            best_roi_campaigns
        ),

        "discount_impact": (
            discount_impact
        ),

        "concentration": (
            concentration
        ),

        "capabilities": {
            "campaign_impact": True,

            "roi_analysis": (
                not campaign_performance[
                    "roi_pct"
                ].dropna().empty
            ),

            "roas_analysis": (
                not campaign_performance[
                    "roas"
                ].dropna().empty
            ),

            "customer_response": (
                not customer_response.empty
            ),

            "channel_analysis": (
                not channel_performance.empty
            ),

            "product_impact": (
                not product_impact.empty
            ),

            "before_after_analysis": (
                not period_impact.empty
            ),

            "discount_analysis": (
                discount_impact[
                    "available"
                ]
            ),
        },
    }


# ============================================================
# BACKWARD-COMPATIBILITY ALIAS
# ============================================================

# Keep this alias so any existing code referring to the old
# marketing terminology does not break.
#
# Product/UI terminology remains:
#       Campaign Impact
#
# Internal compatibility:
#       run_marketing_analytics()

run_marketing_analytics = run_campaign_impact


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "build_campaign_base",
    "calculate_campaign_performance",
    "calculate_campaign_type_performance",
    "calculate_channel_performance",
    "calculate_campaign_customer_response",
    "calculate_campaign_product_impact",
    "calculate_discount_impact",
    "classify_campaigns",
    "get_top_campaigns",
    "get_bottom_campaigns",
    "get_best_roi_campaigns",
    "calculate_campaign_concentration",
    "calculate_campaign_period_impact",
    "run_campaign_impact",
    "run_marketing_analytics",
]