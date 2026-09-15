"""
DataPulse — Customer Intelligence Engine

Responsibilities
----------------
- Customer-level KPIs
- Customer revenue
- Customer orders
- Customer quantity
- Customer profit
- Average order value
- First / last purchase
- Recency
- Purchase frequency
- Customer lifetime value proxy
- Repeat customer analysis
- Customer concentration
- RFM-style customer features
- Customer segment summaries
- Churn-ready feature foundation

Important
---------
This module does NOT:
- modify the original dataframe
- perform schema mapping
- perform EDA
- train ML models
- predict churn
- perform ML clustering

The outputs from this module are intentionally designed to feed:
    1. Churn Prediction
    2. Customer Segmentation
    3. Business Decision Center
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.schema_mapper import get_source_column


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHURN_DAYS = 90

RFM_QUANTILES = 5

TOP_CUSTOMERS_DEFAULT = 20


# ============================================================
# HELPERS
# ============================================================

def _column(
    mapping: Dict[str, Any],
    field: str,
) -> Optional[str]:
    """Return the source column mapped to a DataPulse field."""

    return get_source_column(
        mapping,
        field,
    )


def _has(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    field: str,
) -> bool:
    """Check whether a mapped field exists."""

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
# CUSTOMER BASE DATAFRAME
# ============================================================

def build_customer_base(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build the normalized transaction-level customer dataframe.

    This dataframe is internal to the customer analytics layer.
    It does not modify the uploaded dataset.
    """

    if not _has(
        df,
        mapping,
        "customer_id",
    ):

        return pd.DataFrame()

    customer_column = _column(
        mapping,
        "customer_id",
    )

    order_date = _date(
        df,
        mapping,
        "order_date",
    )

    sales = _numeric(
        df,
        mapping,
        "sales_amount",
    )

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

    customer = df[
        customer_column
    ].astype("string")

    result = pd.DataFrame(
        {
            "customer_id": customer,
        }
    )

    result["order_date"] = (
        order_date
        if order_date is not None
        else pd.NaT
    )

    result["sales_amount"] = (
        sales
        if sales is not None
        else 0.0
    )

    result["quantity"] = (
        quantity
        if quantity is not None
        else 0.0
    )

    result["profit"] = (
        profit
        if profit is not None
        else np.nan
    )

    # Optional customer attributes
    optional_fields = {
        "customer_name": "customer_name",
        "customer_age": "customer_age",
        "customer_gender": "customer_gender",
        "customer_segment": "customer_segment",
        "customer_city": "customer_city",
        "customer_state": "customer_state",
        "customer_country": "customer_country",
    }

    for output_name, canonical_field in optional_fields.items():

        source_column = _column(
            mapping,
            canonical_field,
        )

        if (
            source_column
            and source_column in df.columns
        ):

            result[output_name] = df[
                source_column
            ].values

    # Remove rows without a customer identifier.
    result = result[
        result["customer_id"].notna()
        & (
            result["customer_id"]
            .astype(str)
            .str.strip()
            .ne("")
        )
    ].copy()

    return result.reset_index(
        drop=True
    )


# ============================================================
# CUSTOMER METRICS
# ============================================================

def calculate_customer_metrics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Calculate customer-level business metrics.

    Output is intentionally ML-friendly.
    """

    base = build_customer_base(
        df,
        mapping,
    )

    if base.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    aggregation = (
        base.groupby(
            "customer_id",
            dropna=True,
        )
        .agg(
            revenue=(
                "sales_amount",
                "sum",
            ),
            orders=(
                "order_date",
                "count",
            ),
            quantity=(
                "quantity",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            first_purchase=(
                "order_date",
                "min",
            ),
            last_purchase=(
                "order_date",
                "max",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Average Order Value
    # --------------------------------------------------------

    aggregation["average_order_value"] = np.where(
        aggregation["orders"] > 0,
        aggregation["revenue"]
        / aggregation["orders"],
        0.0,
    )

    # --------------------------------------------------------
    # Purchase frequency
    # --------------------------------------------------------

    aggregation["purchase_frequency"] = (
        aggregation["orders"]
    )

    # --------------------------------------------------------
    # Customer lifetime days
    # --------------------------------------------------------

    aggregation["lifetime_days"] = (
        aggregation["last_purchase"]
        - aggregation["first_purchase"]
    ).dt.days

    aggregation["lifetime_days"] = (
        aggregation["lifetime_days"]
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # Average days between purchases
    # --------------------------------------------------------

    aggregation["avg_days_between_orders"] = np.where(
        aggregation["orders"] > 1,
        aggregation["lifetime_days"]
        / (
            aggregation["orders"] - 1
        ),
        np.nan,
    )

    # --------------------------------------------------------
    # Profit margin
    # --------------------------------------------------------

    aggregation["profit_margin"] = np.where(
        aggregation["revenue"] != 0,
        (
            aggregation["profit"]
            / aggregation["revenue"]
        )
        * 100,
        np.nan,
    )

    # --------------------------------------------------------
    # Repeat customer flag
    # --------------------------------------------------------

    aggregation["is_repeat_customer"] = (
        aggregation["orders"] > 1
    )

    # --------------------------------------------------------
    # Clean numeric values
    # --------------------------------------------------------

    numeric_columns = [
        "revenue",
        "quantity",
        "profit",
        "average_order_value",
        "lifetime_days",
        "avg_days_between_orders",
        "profit_margin",
    ]

    for column in numeric_columns:

        aggregation[column] = pd.to_numeric(
            aggregation[column],
            errors="coerce",
        ).round(2)

    return aggregation


# ============================================================
# RECENCY
# ============================================================

def calculate_customer_recency(
    customer_metrics: pd.DataFrame,
    analysis_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Calculate customer recency.

    Recency is the number of days since the customer's latest
    purchase.

    If analysis_date is not provided, the latest dataset purchase
    date is used.
    """

    if customer_metrics.empty:
        return customer_metrics.copy()

    result = customer_metrics.copy()

    if analysis_date is None:

        valid_dates = result[
            "last_purchase"
        ].dropna()

        if valid_dates.empty:

            result["recency_days"] = np.nan

            return result

        analysis_date = valid_dates.max()

    analysis_date = pd.Timestamp(
        analysis_date
    )

    result["recency_days"] = (
        analysis_date
        - result["last_purchase"]
    ).dt.days

    result["recency_days"] = (
        result["recency_days"]
        .clip(lower=0)
    )

    return result


# ============================================================
# RFM FEATURES
# ============================================================

def _safe_qcut_score(
    series: pd.Series,
    reverse: bool = False,
    bins: int = RFM_QUANTILES,
) -> pd.Series:
    """
    Create robust 1–5 quantile scores.

    Handles duplicate values and small datasets safely.
    """

    if series.empty:
        return pd.Series(
            index=series.index,
            dtype=float,
        )

    ranks = series.rank(
        method="first"
    )

    unique_count = ranks.nunique()

    if unique_count < 2:

        scores = pd.Series(
            3,
            index=series.index,
            dtype=float,
        )

    else:

        effective_bins = min(
            bins,
            int(unique_count),
        )

        scores = pd.qcut(
            ranks,
            q=effective_bins,
            labels=False,
            duplicates="drop",
        )

        scores = (
            scores
            .astype(float)
            + 1
        )

    if reverse:
        scores = (
            scores.max()
            + 1
            - scores
        )

    return scores


def calculate_rfm_features(
    customer_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create RFM-style customer features.

    R = Recency
    F = Frequency
    M = Monetary

    These features become the foundation for:
    - Customer segmentation
    - Churn prediction
    - Customer value analysis
    """

    if customer_metrics.empty:
        return customer_metrics.copy()

    result = customer_metrics.copy()

    # --------------------------------------------------------
    # RFM scores
    # --------------------------------------------------------

    result["recency_score"] = _safe_qcut_score(
        result["recency_days"],
        reverse=True,
    )

    result["frequency_score"] = _safe_qcut_score(
        result["orders"],
        reverse=False,
    )

    result["monetary_score"] = _safe_qcut_score(
        result["revenue"],
        reverse=False,
    )

    # --------------------------------------------------------
    # Combined RFM score
    # --------------------------------------------------------

    result["rfm_score"] = (
        result["recency_score"]
        + result["frequency_score"]
        + result["monetary_score"]
    )

    result["rfm_score"] = (
        result["rfm_score"]
        .round(0)
        .astype("Int64")
    )

    # --------------------------------------------------------
    # RFM code
    # --------------------------------------------------------

    result["rfm_code"] = (
        result["recency_score"]
        .fillna(0)
        .astype(int)
        .astype(str)
        + result["frequency_score"]
        .fillna(0)
        .astype(int)
        .astype(str)
        + result["monetary_score"]
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    return result


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

def assign_customer_segments(
    customer_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign business-friendly RFM segments.

    This is a rule-based business segmentation layer.
    ML clustering will be handled separately.
    """

    if customer_metrics.empty:
        return customer_metrics.copy()

    result = customer_metrics.copy()

    def classify(row: pd.Series) -> str:

        r = _safe_float(
            row.get("recency_score")
        )

        f = _safe_float(
            row.get("frequency_score")
        )

        m = _safe_float(
            row.get("monetary_score")
        )

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        if r >= 4 and f >= 3:
            return "Loyal Customers"

        if r >= 4 and m >= 3:
            return "High Value New"

        if r >= 3 and f >= 3 and m >= 3:
            return "Potential Loyalists"

        if r >= 3 and f <= 2:
            return "New / Developing"

        if r <= 2 and f >= 4 and m >= 3:
            return "At Risk High Value"

        if r <= 2 and f >= 3:
            return "At Risk"

        if r <= 2 and m >= 3:
            return "High Value At Risk"

        return "Needs Attention"

    result["business_segment"] = result.apply(
        classify,
        axis=1,
    )

    return result


# ============================================================
# CHURN SIGNALS
# ============================================================

def add_churn_signals(
    customer_metrics: pd.DataFrame,
    churn_days: int = DEFAULT_CHURN_DAYS,
) -> pd.DataFrame:
    """
    Add rule-based churn signals.

    This is NOT the ML churn prediction.

    It creates a transparent baseline that the later ML model
    can improve upon.
    """

    if customer_metrics.empty:
        return customer_metrics.copy()

    result = customer_metrics.copy()

    result["churn_threshold_days"] = int(
        churn_days
    )

    result["churn_risk_flag"] = (
        result["recency_days"]
        >= churn_days
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    def risk_level(
        recency: Any,
    ) -> str:

        value = _safe_float(
            recency,
            default=np.nan,
        )

        if not np.isfinite(value):
            return "Unknown"

        if value >= churn_days * 1.5:
            return "High"

        if value >= churn_days:
            return "Medium"

        if value >= churn_days * 0.67:
            return "Watch"

        return "Low"

    result["churn_risk_level"] = (
        result["recency_days"]
        .apply(risk_level)
    )

    return result


# ============================================================
# CUSTOMER CONCENTRATION
# ============================================================

def calculate_customer_concentration(
    customer_metrics: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Measure how concentrated revenue is among customers.

    Useful for executive decision-making and risk analysis.
    """

    if customer_metrics.empty:

        return {
            "available": False,
            "top_1_share": None,
            "top_5_share": None,
            "top_10_share": None,
        }

    revenue = (
        customer_metrics["revenue"]
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
        }

    return {
        "available": True,

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
# TOP CUSTOMERS
# ============================================================

def get_top_customers(
    customer_metrics: pd.DataFrame,
    top_n: int = TOP_CUSTOMERS_DEFAULT,
) -> pd.DataFrame:
    """Return highest-value customers."""

    if customer_metrics.empty:
        return pd.DataFrame()

    columns = [
        "customer_id",
        "revenue",
        "orders",
        "quantity",
        "profit",
        "average_order_value",
        "recency_days",
        "business_segment",
        "churn_risk_level",
    ]

    available_columns = [
        column
        for column in columns
        if column in customer_metrics.columns
    ]

    return (
        customer_metrics[
            available_columns
        ]
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
# REPEAT CUSTOMER ANALYSIS
# ============================================================

def calculate_repeat_customer_metrics(
    customer_metrics: pd.DataFrame,
) -> Dict[str, Any]:
    """Calculate repeat customer KPIs."""

    if customer_metrics.empty:

        return {
            "available": False,
            "total_customers": 0,
            "repeat_customers": 0,
            "repeat_customer_rate": None,
            "one_time_customers": 0,
        }

    total_customers = int(
        len(customer_metrics)
    )

    repeat_customers = int(
        (
            customer_metrics["orders"]
            > 1
        ).sum()
    )

    one_time_customers = (
        total_customers
        - repeat_customers
    )

    repeat_rate = (
        repeat_customers
        / total_customers
        * 100
        if total_customers
        else None
    )

    return {
        "available": True,
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "one_time_customers": one_time_customers,
        "repeat_customer_rate": (
            round(
                repeat_rate,
                2,
            )
            if repeat_rate is not None
            else None
        ),
    }


# ============================================================
# CUSTOMER SEGMENT SUMMARY
# ============================================================

def calculate_segment_summary(
    customer_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize customers by business segment."""

    if (
        customer_metrics.empty
        or "business_segment"
        not in customer_metrics.columns
    ):
        return pd.DataFrame()

    result = (
        customer_metrics
        .groupby(
            "business_segment",
            dropna=False,
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
            average_revenue=(
                "revenue",
                "mean",
            ),
            orders=(
                "orders",
                "sum",
            ),
            profit=(
                "profit",
                "sum",
            ),
            average_recency=(
                "recency_days",
                "mean",
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

    for column in (
        "revenue",
        "average_revenue",
        "orders",
        "profit",
        "average_recency",
    ):

        result[column] = result[
            column
        ].round(2)

    return result


# ============================================================
# CUSTOMER DEMOGRAPHIC SUMMARY
# ============================================================

def calculate_customer_dimension(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    field: str,
) -> pd.DataFrame:
    """
    Calculate customer revenue by a customer dimension.

    Supported:
    - customer_gender
    - customer_age
    - customer_city
    - customer_state
    - customer_country
    """

    allowed_fields = {
        "customer_gender",
        "customer_age",
        "customer_city",
        "customer_state",
        "customer_country",
    }

    if field not in allowed_fields:
        raise ValueError(
            f"Unsupported customer dimension: {field}"
        )

    if not _has(
        df,
        mapping,
        field,
    ):
        return pd.DataFrame()

    revenue = _numeric(
        df,
        mapping,
        "sales_amount",
    )

    if revenue is None:
        return pd.DataFrame()

    source_column = _column(
        mapping,
        field,
    )

    working = pd.DataFrame(
        {
            "dimension": df[
                source_column
            ],
            "revenue": revenue,
        }
    )

    result = (
        working.dropna(
            subset=["dimension"]
        )
        .groupby(
            "dimension",
            dropna=False,
        )
        .agg(
            revenue=(
                "revenue",
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

    result["revenue"] = result[
        "revenue"
    ].round(2)

    return result


# ============================================================
# CUSTOMER LIFETIME VALUE PROXY
# ============================================================

def calculate_customer_ltv_proxy(
    customer_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a transparent customer lifetime value proxy.

    This is NOT a probabilistic CLV model.

    Formula:
        Average Order Value × Purchase Frequency

    For customers with observed lifetime duration, the metric
    is additionally exposed as annualized revenue proxy.
    """

    if customer_metrics.empty:
        return customer_metrics.copy()

    result = customer_metrics.copy()

    result["ltv_proxy"] = (
        result["average_order_value"]
        * result["purchase_frequency"]
    )

    # Annualized proxy where lifetime information exists.
    lifetime_years = (
        result["lifetime_days"]
        / 365.25
    )

    result["annualized_revenue_proxy"] = np.where(
        lifetime_years > 0,
        result["revenue"]
        / lifetime_years,
        result["revenue"],
    )

    result["ltv_proxy"] = result[
        "ltv_proxy"
    ].round(2)

    result["annualized_revenue_proxy"] = (
        result[
            "annualized_revenue_proxy"
        ]
        .round(2)
    )

    return result


# ============================================================
# MAIN CUSTOMER ANALYTICS
# ============================================================

def run_customer_analytics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the complete DataPulse Customer Intelligence engine.

    Public interface consumed by app.py.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "run_customer_analytics() expects a pandas DataFrame."
        )

    if not isinstance(
        mapping,
        dict,
    ):
        raise TypeError(
            "mapping must be the dictionary returned by map_schema()."
        )

    # --------------------------------------------------------
    # Customer availability
    # --------------------------------------------------------

    if not _has(
        df,
        mapping,
        "customer_id",
    ):

        return {
            "available": False,

            "summary": {
                "customers": 0,
                "repeat_customers": 0,
                "repeat_customer_rate": None,
                "average_customer_revenue": None,
                "average_customer_order_value": None,
                "total_customer_revenue": None,
            },

            "customer_metrics": pd.DataFrame(),

            "rfm": pd.DataFrame(),

            "segments": pd.DataFrame(),

            "top_customers": pd.DataFrame(),

            "segment_summary": pd.DataFrame(),

            "concentration": {
                "available": False,
            },

            "repeat_customer": {
                "available": False,
            },

            "dimensions": {},

            "churn_ready_features": pd.DataFrame(),

            "capabilities": {
                "customer_intelligence": False,
                "rfm": False,
                "customer_segmentation": False,
                "churn_foundation": False,
            },
        }

    # --------------------------------------------------------
    # Base customer metrics
    # --------------------------------------------------------

    customer_metrics = calculate_customer_metrics(
        df,
        mapping,
    )

    if customer_metrics.empty:

        return {
            "available": False,
            "summary": {},
            "customer_metrics": pd.DataFrame(),
            "rfm": pd.DataFrame(),
            "segments": pd.DataFrame(),
            "top_customers": pd.DataFrame(),
            "segment_summary": pd.DataFrame(),
            "concentration": {
                "available": False,
            },
            "repeat_customer": {
                "available": False,
            },
            "dimensions": {},
            "churn_ready_features": pd.DataFrame(),
            "capabilities": {
                "customer_intelligence": False,
                "rfm": False,
                "customer_segmentation": False,
                "churn_foundation": False,
            },
        }

    # --------------------------------------------------------
    # Recency
    # --------------------------------------------------------

    customer_metrics = calculate_customer_recency(
        customer_metrics
    )

    # --------------------------------------------------------
    # RFM
    # --------------------------------------------------------

    customer_metrics = calculate_rfm_features(
        customer_metrics
    )

    # --------------------------------------------------------
    # Business segments
    # --------------------------------------------------------

    customer_metrics = assign_customer_segments(
        customer_metrics
    )

    # --------------------------------------------------------
    # Churn baseline
    # --------------------------------------------------------

    customer_metrics = add_churn_signals(
        customer_metrics
    )

    # --------------------------------------------------------
    # LTV proxy
    # --------------------------------------------------------

    customer_metrics = calculate_customer_ltv_proxy(
        customer_metrics
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    repeat_customer = (
        calculate_repeat_customer_metrics(
            customer_metrics
        )
    )

    concentration = (
        calculate_customer_concentration(
            customer_metrics
        )
    )

    top_customers = get_top_customers(
        customer_metrics
    )

    segment_summary = calculate_segment_summary(
        customer_metrics
    )

    # --------------------------------------------------------
    # Customer dimensions
    # --------------------------------------------------------

    dimensions: Dict[
        str,
        pd.DataFrame,
    ] = {}

    for field in (
        "customer_gender",
        "customer_city",
        "customer_state",
        "customer_country",
    ):

        dimension_result = calculate_customer_dimension(
            df,
            mapping,
            field,
        )

        if not dimension_result.empty:

            dimensions[field] = (
                dimension_result
            )

    # --------------------------------------------------------
    # Executive summary
    # --------------------------------------------------------

    total_customers = int(
        len(customer_metrics)
    )

    total_revenue = _safe_float(
        customer_metrics["revenue"].sum()
    )

    average_customer_revenue = (
        total_revenue
        / total_customers
        if total_customers
        else 0.0
    )

    average_customer_aov = _safe_float(
        customer_metrics[
            "average_order_value"
        ].mean()
    )

    total_profit = _safe_float(
        customer_metrics["profit"].sum()
    )

    high_risk_customers = int(
        (
            customer_metrics[
                "churn_risk_level"
            ]
            .isin(
                [
                    "High",
                    "Medium",
                ]
            )
        ).sum()
    )

    # --------------------------------------------------------
    # ML-ready customer features
    # --------------------------------------------------------

    ml_feature_columns = [
        "customer_id",
        "revenue",
        "orders",
        "quantity",
        "profit",
        "average_order_value",
        "recency_days",
        "lifetime_days",
        "avg_days_between_orders",
        "profit_margin",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score",
        "ltv_proxy",
        "annualized_revenue_proxy",
    ]

    churn_ready_features = customer_metrics[
        [
            column
            for column in ml_feature_columns
            if column in customer_metrics.columns
        ]
    ].copy()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "available": True,

        "summary": {
            "customers": total_customers,
            "repeat_customers": repeat_customer[
                "repeat_customers"
            ],
            "total_customer_revenue": _round(
                total_revenue
            ),
            "total_customer_profit": _round(
                total_profit
            ),
            "average_customer_value": _round(
                average_customer_revenue
            ),
            "average_customer_revenue": _round(
                average_customer_revenue
            ),
            "average_customer_order_value": _round(
                average_customer_aov
            ),
            "high_risk_customers": high_risk_customers,
            "repeat_customer_rate": repeat_customer[
                "repeat_customer_rate"
            ],
        },

        "customer_metrics": customer_metrics,

        "rfm": customer_metrics[
            [
                column
                for column in [
                    "customer_id",
                    "recency_days",
                    "orders",
                    "revenue",
                    "recency_score",
                    "frequency_score",
                    "monetary_score",
                    "rfm_score",
                    "rfm_code",
                ]
                if column in customer_metrics.columns
            ]
        ].copy(),

        "segments": customer_metrics[
            [
                column
                for column in [
                    "customer_id",
                    "business_segment",
                    "revenue",
                    "orders",
                    "recency_days",
                    "profit",
                    "ltv_proxy",
                    "churn_risk_level",
                ]
                if column in customer_metrics.columns
            ]
        ].copy(),

        "top_customers": top_customers,

        "segment_summary": segment_summary,

        "concentration": concentration,

        "repeat_customer": repeat_customer,

        "dimensions": dimensions,

        "churn_ready_features": churn_ready_features,

        "capabilities": {
            "customer_intelligence": True,
            "rfm": True,
            "customer_segmentation": True,
            "churn_foundation": True,
            "customer_ltv_proxy": True,
            "repeat_customer_analysis": True,
            "customer_concentration": True,
        },
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "build_customer_base",
    "calculate_customer_metrics",
    "calculate_customer_recency",
    "calculate_rfm_features",
    "assign_customer_segments",
    "add_churn_signals",
    "calculate_customer_concentration",
    "get_top_customers",
    "calculate_repeat_customer_metrics",
    "calculate_segment_summary",
    "calculate_customer_dimension",
    "calculate_customer_ltv_proxy",
    "run_customer_analytics",
]