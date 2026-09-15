"""
DataPulse - Market & Seller Intelligence

Purpose
-------
Analyzes where the business is performing and who is driving that performance.

This module covers:
- Seller performance
- Market performance
- Regional / geographic performance
- Seller profitability
- Market comparison
- Product performance by region
- Regional opportunities
- Seller concentration
- Decision-oriented signals

Stable interface
----------------
run_market_seller_analytics(df, mapping) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_CORE_FIELDS = (
    "order_id",
    "order_date",
    "sales_amount",
)

OPTIONAL_FIELDS = (
    "quantity",
    "unit_price",
    "cost",
    "profit",
    "seller_id",
    "seller_name",
    "market",
    "region",
    "customer_id",
    "product_id",
    "product_name",
    "category",
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _get_source_column(
    mapping: Dict[str, Any],
    canonical_field: str,
) -> Optional[str]:
    """Return the source dataframe column mapped to a canonical field."""

    if not mapping:
        return None

    mappings = mapping.get("mappings", {})

    value = mappings.get(canonical_field)

    if isinstance(value, dict):
        return value.get("source_column") or value.get("column")

    if isinstance(value, str):
        return value

    return None


def _has_field(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    canonical_field: str,
) -> bool:
    """Check whether a mapped canonical field exists in the dataframe."""

    source_column = _get_source_column(mapping, canonical_field)

    return (
        source_column is not None
        and source_column in df.columns
    )


def _safe_numeric(
    series: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    """Convert a series to numeric safely."""

    result = pd.to_numeric(series, errors="coerce")

    if default is not None:
        result = result.fillna(default)

    return result


def _safe_text(
    series: pd.Series,
    default: str = "Unknown",
) -> pd.Series:
    """Normalize text dimensions."""

    result = series.astype("string").str.strip()

    result = result.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "null": pd.NA,
        }
    )

    return result.fillna(default)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Safe vectorized division."""

    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator = pd.to_numeric(denominator, errors="coerce")

    return (
        numerator
        .div(denominator.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def _percentage(
    values: pd.Series,
) -> pd.Series:
    """Convert values into percentage contribution."""

    total = values.sum()

    if total == 0:
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    return values / total * 100


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty dataframe with predictable columns."""

    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------
# Prepared dataframe
# ---------------------------------------------------------------------

def _prepare_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create an internal standardized dataframe.

    Only mapped fields are added.
    The original dataframe is never modified.
    """

    working = pd.DataFrame(index=df.index)

    for field in OPTIONAL_FIELDS:
        source_column = _get_source_column(mapping, field)

        if source_column and source_column in df.columns:
            working[field] = df[source_column]

    # Core financial fields
    if "sales_amount" in working.columns:
        working["sales_amount"] = _safe_numeric(
            working["sales_amount"]
        )
    else:
        working["sales_amount"] = 0.0

    if "quantity" in working.columns:
        working["quantity"] = _safe_numeric(
            working["quantity"]
        )
    else:
        working["quantity"] = 0.0

    if "profit" in working.columns:
        working["profit"] = _safe_numeric(
            working["profit"]
        )
    elif "cost" in working.columns:
        working["cost"] = _safe_numeric(
            working["cost"]
        )
        working["profit"] = (
            working["sales_amount"] - working["cost"]
        )
    else:
        working["profit"] = 0.0

    if "cost" in working.columns:
        working["cost"] = _safe_numeric(
            working["cost"]
        )
    else:
        working["cost"] = (
            working["sales_amount"] - working["profit"]
        )

    if "unit_price" in working.columns:
        working["unit_price"] = _safe_numeric(
            working["unit_price"]
        )

    # IDs / dimensions
    text_fields = [
        "order_id",
        "seller_id",
        "seller_name",
        "market",
        "region",
        "customer_id",
        "product_id",
        "product_name",
        "category",
    ]

    for field in text_fields:
        if field in working.columns:
            working[field] = _safe_text(
                working[field]
            )

    return working


# ---------------------------------------------------------------------
# Seller intelligence
# ---------------------------------------------------------------------

def _build_seller_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Build seller-level performance analytics."""

    if "seller_id" not in working.columns and "seller_name" not in working.columns:
        return _empty_frame(
            [
                "seller",
                "revenue",
                "orders",
                "customers",
                "quantity",
                "profit",
                "cost",
                "average_order_value",
                "profit_margin",
                "revenue_contribution_pct",
                "profit_contribution_pct",
                "decision_class",
            ]
        )

    if "seller_name" in working.columns:
        seller_key = working["seller_name"]
    else:
        seller_key = working["seller_id"]

    temp = working.copy()
    temp["_seller"] = seller_key

    grouped = (
        temp.groupby("_seller", dropna=False)
        .agg(
            revenue=("sales_amount", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in temp.columns
            else ("sales_amount", "size"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
            cost=("cost", "sum"),
        )
        .reset_index()
        .rename(columns={"_seller": "seller"})
    )

    if "customer_id" in temp.columns:
        customers = (
            temp.groupby("_seller")["customer_id"]
            .nunique()
            .reset_index(name="customers")
        )

        grouped = grouped.merge(
            customers,
            on="_seller" if "_seller" in grouped.columns else "seller",
            how="left",
        ) if "_seller" in grouped.columns else grouped

        if "customers" not in grouped.columns:
            customer_counts = (
                temp.groupby("_seller")["customer_id"]
                .nunique()
                .to_dict()
            )

            grouped["customers"] = grouped["seller"].map(
                customer_counts
            )

    if "customers" not in grouped.columns:
        grouped["customers"] = 0

    grouped["average_order_value"] = _safe_divide(
        grouped["revenue"],
        grouped["orders"],
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    grouped["revenue_contribution_pct"] = _percentage(
        grouped["revenue"]
    )

    grouped["profit_contribution_pct"] = _percentage(
        grouped["profit"]
    )

    def classify(row: pd.Series) -> str:
        revenue = row["revenue"]
        margin = row["profit_margin"]

        if revenue <= 0:
            return "Negative / No Revenue"

        if margin >= 25 and revenue > grouped["revenue"].median():
            return "Strategic Seller"

        if margin >= 15:
            return "Strong Performer"

        if margin >= 5:
            return "Monitor Margin"

        if margin > 0:
            return "Low Margin"

        return "Loss Making"

    grouped["decision_class"] = grouped.apply(
        classify,
        axis=1,
    )

    return grouped.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Market intelligence
# ---------------------------------------------------------------------

def _build_market_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Build market-level performance analytics."""

    if "market" not in working.columns:
        return _empty_frame(
            [
                "market",
                "revenue",
                "orders",
                "customers",
                "quantity",
                "profit",
                "average_order_value",
                "profit_margin",
                "revenue_contribution_pct",
                "decision_class",
            ]
        )

    grouped = (
        working.groupby("market", dropna=False)
        .agg(
            revenue=("sales_amount", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )

    if "customer_id" in working.columns:
        customer_counts = (
            working.groupby("market")["customer_id"]
            .nunique()
            .reset_index(name="customers")
        )

        grouped = grouped.merge(
            customer_counts,
            on="market",
            how="left",
        )

    if "customers" not in grouped.columns:
        grouped["customers"] = 0

    grouped["average_order_value"] = _safe_divide(
        grouped["revenue"],
        grouped["orders"],
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    grouped["revenue_contribution_pct"] = _percentage(
        grouped["revenue"]
    )

    median_revenue = grouped["revenue"].median()
    median_margin = grouped["profit_margin"].median()

    def classify(row: pd.Series) -> str:
        if row["revenue"] >= median_revenue and row["profit_margin"] >= median_margin:
            return "Priority Market"

        if row["revenue"] >= median_revenue:
            return "High Revenue / Margin Review"

        if row["profit_margin"] >= median_margin:
            return "Growth Opportunity"

        return "Developing Market"

    grouped["decision_class"] = grouped.apply(
        classify,
        axis=1,
    )

    return grouped.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Regional intelligence
# ---------------------------------------------------------------------

def _build_geographic_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Build region-level performance."""

    if "region" not in working.columns:
        return _empty_frame(
            [
                "region",
                "revenue",
                "orders",
                "customers",
                "quantity",
                "profit",
                "average_order_value",
                "profit_margin",
                "revenue_contribution_pct",
                "decision_class",
            ]
        )

    grouped = (
        working.groupby("region", dropna=False)
        .agg(
            revenue=("sales_amount", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )

    if "customer_id" in working.columns:
        customer_counts = (
            working.groupby("region")["customer_id"]
            .nunique()
            .reset_index(name="customers")
        )

        grouped = grouped.merge(
            customer_counts,
            on="region",
            how="left",
        )

    if "customers" not in grouped.columns:
        grouped["customers"] = 0

    grouped["average_order_value"] = _safe_divide(
        grouped["revenue"],
        grouped["orders"],
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    grouped["revenue_contribution_pct"] = _percentage(
        grouped["revenue"]
    )

    median_revenue = grouped["revenue"].median()
    median_margin = grouped["profit_margin"].median()

    def classify(row: pd.Series) -> str:
        if (
            row["revenue"] >= median_revenue
            and row["profit_margin"] >= median_margin
        ):
            return "Scale Region"

        if row["revenue"] >= median_revenue:
            return "Revenue Strong / Margin Review"

        if row["profit_margin"] >= median_margin:
            return "Growth Region"

        return "Develop Region"

    grouped["decision_class"] = grouped.apply(
        classify,
        axis=1,
    )

    return grouped.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Product by region
# ---------------------------------------------------------------------

def _build_product_region_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze which products perform best across regions.

    This is useful for identifying regional product-market fit.
    """

    if (
        "region" not in working.columns
        or "product_name" not in working.columns
    ):
        return _empty_frame(
            [
                "region",
                "product",
                "revenue",
                "orders",
                "quantity",
                "profit",
                "profit_margin",
                "revenue_contribution_pct",
            ]
        )

    grouped = (
        working.groupby(
            ["region", "product_name"],
            dropna=False,
        )
        .agg(
            revenue=("sales_amount", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            quantity=("quantity", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "product_name": "product",
            }
        )
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    grouped["revenue_contribution_pct"] = (
        grouped.groupby("region")["revenue"]
        .transform(_percentage)
    )

    return grouped.sort_values(
        ["region", "revenue"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Seller by region
# ---------------------------------------------------------------------

def _build_seller_region_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze seller performance within each region."""

    if (
        "region" not in working.columns
        or (
            "seller_name" not in working.columns
            and "seller_id" not in working.columns
        )
    ):
        return _empty_frame(
            [
                "region",
                "seller",
                "revenue",
                "orders",
                "profit",
                "profit_margin",
            ]
        )

    seller_field = (
        "seller_name"
        if "seller_name" in working.columns
        else "seller_id"
    )

    grouped = (
        working.groupby(
            ["region", seller_field],
            dropna=False,
        )
        .agg(
            revenue=("sales_amount", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                seller_field: "seller",
            }
        )
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    return grouped.sort_values(
        ["region", "revenue"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Regional opportunities
# ---------------------------------------------------------------------

def _build_regional_opportunities(
    geographic_performance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate opportunity signals from regional performance.

    This is decision support, not causal inference.
    """

    if geographic_performance.empty:
        return _empty_frame(
            [
                "region",
                "opportunity_type",
                "revenue",
                "profit_margin",
                "priority",
                "recommendation",
            ]
        )

    data = geographic_performance.copy()

    revenue_median = data["revenue"].median()
    margin_median = data["profit_margin"].median()

    rows = []

    for _, row in data.iterrows():
        revenue = float(row["revenue"])
        margin = float(row["profit_margin"])

        if revenue < revenue_median and margin >= margin_median:
            opportunity = "High Margin / Low Revenue"
            priority = "High"
            recommendation = (
                "Increase acquisition and sales coverage in this region "
                "while protecting current margins."
            )

        elif revenue >= revenue_median and margin < margin_median:
            opportunity = "High Revenue / Low Margin"
            priority = "High"
            recommendation = (
                "Review pricing, discounting, product mix and fulfillment "
                "economics before increasing volume."
            )

        elif revenue >= revenue_median and margin >= margin_median:
            opportunity = "Scale Opportunity"
            priority = "High"
            recommendation = (
                "Prioritize this region for growth while maintaining "
                "current profitability."
            )

        else:
            opportunity = "Developing Region"
            priority = "Medium"
            recommendation = (
                "Test targeted products, sellers or campaigns before "
                "committing significant additional resources."
            )

        rows.append(
            {
                "region": row["region"],
                "opportunity_type": opportunity,
                "revenue": revenue,
                "profit_margin": margin,
                "priority": priority,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["priority", "revenue"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------

def _build_concentration(
    performance: pd.DataFrame,
    dimension_name: str,
) -> Dict[str, Any]:
    """Calculate concentration of revenue across a dimension."""

    if performance.empty or "revenue" not in performance.columns:
        return {
            "dimension": dimension_name,
            "entities": 0,
            "top_1_revenue_share_pct": 0.0,
            "top_3_revenue_share_pct": 0.0,
            "top_5_revenue_share_pct": 0.0,
            "concentration_level": "Unavailable",
        }

    revenue = (
        performance["revenue"]
        .clip(lower=0)
        .sort_values(ascending=False)
    )

    total = revenue.sum()

    if total <= 0:
        return {
            "dimension": dimension_name,
            "entities": int(len(revenue)),
            "top_1_revenue_share_pct": 0.0,
            "top_3_revenue_share_pct": 0.0,
            "top_5_revenue_share_pct": 0.0,
            "concentration_level": "No Revenue",
        }

    top_1 = revenue.head(1).sum() / total * 100
    top_3 = revenue.head(3).sum() / total * 100
    top_5 = revenue.head(5).sum() / total * 100

    if top_1 >= 50:
        level = "Very High"
    elif top_3 >= 70:
        level = "High"
    elif top_5 >= 80:
        level = "Moderate"
    else:
        level = "Distributed"

    return {
        "dimension": dimension_name,
        "entities": int(len(revenue)),
        "top_1_revenue_share_pct": round(float(top_1), 2),
        "top_3_revenue_share_pct": round(float(top_3), 2),
        "top_5_revenue_share_pct": round(float(top_5), 2),
        "concentration_level": level,
    }


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def _build_summary(
    working: pd.DataFrame,
    seller_performance: pd.DataFrame,
    market_performance: pd.DataFrame,
    geographic_performance: pd.DataFrame,
) -> Dict[str, Any]:
    """Build high-level market/seller intelligence summary."""

    revenue = float(working["sales_amount"].sum())
    profit = float(working["profit"].sum())

    orders = (
        int(working["order_id"].nunique())
        if "order_id" in working.columns
        else int(len(working))
    )

    sellers = (
        int(
            working[
                "seller_name"
                if "seller_name" in working.columns
                else "seller_id"
            ].nunique()
        )
        if (
            "seller_name" in working.columns
            or "seller_id" in working.columns
        )
        else 0
    )

    markets = (
        int(working["market"].nunique())
        if "market" in working.columns
        else 0
    )

    regions = (
        int(working["region"].nunique())
        if "region" in working.columns
        else 0
    )

    return {
        "revenue": revenue,
        "profit": profit,
        "orders": orders,
        "sellers": sellers,
        "markets": markets,
        "regions": regions,
        "average_order_value": (
            revenue / orders
            if orders > 0
            else 0.0
        ),
        "profit_margin": (
            profit / revenue * 100
            if revenue != 0
            else 0.0
        ),
        "top_seller": (
            seller_performance.iloc[0]["seller"]
            if not seller_performance.empty
            else None
        ),
        "top_seller_revenue": (
            float(seller_performance.iloc[0]["revenue"])
            if not seller_performance.empty
            else 0.0
        ),
        "top_market": (
            market_performance.iloc[0]["market"]
            if not market_performance.empty
            else None
        ),
        "top_market_revenue": (
            float(market_performance.iloc[0]["revenue"])
            if not market_performance.empty
            else 0.0
        ),
        "top_region": (
            geographic_performance.iloc[0]["region"]
            if not geographic_performance.empty
            else None
        ),
        "top_region_revenue": (
            float(geographic_performance.iloc[0]["revenue"])
            if not geographic_performance.empty
            else 0.0
        ),
    }


# ---------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------

def _build_capabilities(
    working: pd.DataFrame,
) -> Dict[str, bool]:
    """Expose exactly which market/seller analyses are available."""

    return {
        "seller_analysis": (
            "seller_name" in working.columns
            or "seller_id" in working.columns
        ),
        "market_analysis": "market" in working.columns,
        "regional_analysis": "region" in working.columns,
        "seller_profitability": (
            (
                "seller_name" in working.columns
                or "seller_id" in working.columns
            )
            and "profit" in working.columns
        ),
        "product_region_analysis": (
            "region" in working.columns
            and "product_name" in working.columns
        ),
        "seller_region_analysis": (
            "region" in working.columns
            and (
                "seller_name" in working.columns
                or "seller_id" in working.columns
            )
        ),
    }


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_market_seller_analytics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run Market & Seller Intelligence.

    Parameters
    ----------
    df:
        Analysis-ready source dataframe.

    mapping:
        Schema mapping returned by src.schema_mapper.map_schema().

    Returns
    -------
    dict
        Stable analytics contract consumed by app.py and pages/market.py.
    """

    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "available": False,
            "summary": {},
            "seller_performance": pd.DataFrame(),
            "market_performance": pd.DataFrame(),
            "geographic_performance": pd.DataFrame(),
            "product_region_performance": pd.DataFrame(),
            "seller_region_performance": pd.DataFrame(),
            "regional_opportunities": pd.DataFrame(),
            "concentration": {},
            "capabilities": {},
            "warnings": [
                "A valid pandas DataFrame is required."
            ],
        }

    if df.empty:
        return {
            "available": False,
            "summary": {},
            "seller_performance": pd.DataFrame(),
            "market_performance": pd.DataFrame(),
            "geographic_performance": pd.DataFrame(),
            "product_region_performance": pd.DataFrame(),
            "seller_region_performance": pd.DataFrame(),
            "regional_opportunities": pd.DataFrame(),
            "concentration": {},
            "capabilities": {},
            "warnings": [
                "The dataset is empty."
            ],
        }

    working = _prepare_dataframe(
        df,
        mapping,
    )

    seller_available = (
        "seller_name" in working.columns
        or "seller_id" in working.columns
    )

    market_available = "market" in working.columns
    region_available = "region" in working.columns

    if not seller_available and not market_available and not region_available:
        return {
            "available": False,
            "summary": {},
            "seller_performance": pd.DataFrame(),
            "market_performance": pd.DataFrame(),
            "geographic_performance": pd.DataFrame(),
            "product_region_performance": pd.DataFrame(),
            "seller_region_performance": pd.DataFrame(),
            "regional_opportunities": pd.DataFrame(),
            "concentration": {},
            "capabilities": _build_capabilities(working),
            "warnings": [
                "Market & Seller Intelligence requires at least one "
                "seller, market or region field."
            ],
        }

    seller_performance = _build_seller_performance(
        working
    )

    market_performance = _build_market_performance(
        working
    )

    geographic_performance = _build_geographic_performance(
        working
    )

    product_region_performance = (
        _build_product_region_performance(working)
    )

    seller_region_performance = (
        _build_seller_region_performance(working)
    )

    regional_opportunities = _build_regional_opportunities(
        geographic_performance
    )

    concentration = {
        "seller": _build_concentration(
            seller_performance,
            "seller",
        ),
        "market": _build_concentration(
            market_performance,
            "market",
        ),
        "region": _build_concentration(
            geographic_performance,
            "region",
        ),
    }

    summary = _build_summary(
        working,
        seller_performance,
        market_performance,
        geographic_performance,
    )

    capabilities = _build_capabilities(
        working
    )

    warnings = []

    if not seller_available:
        warnings.append(
            "Seller fields are unavailable; seller-level analysis is limited."
        )

    if not market_available:
        warnings.append(
            "Market field is unavailable; market comparison is limited."
        )

    if not region_available:
        warnings.append(
            "Region field is unavailable; geographic opportunity analysis "
            "is limited."
        )

    return {
        "available": True,
        "summary": summary,
        "seller_performance": seller_performance,
        "market_performance": market_performance,
        "geographic_performance": geographic_performance,
        "product_region_performance": product_region_performance,
        "seller_region_performance": seller_region_performance,
        "regional_opportunities": regional_opportunities,
        "concentration": concentration,
        "capabilities": capabilities,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------

run_market_analytics = run_market_seller_analytics