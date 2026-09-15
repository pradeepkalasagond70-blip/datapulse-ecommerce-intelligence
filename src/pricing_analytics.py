"""
DataPulse - Pricing & Discount Intelligence

Purpose
-------
Analyzes pricing, discounting and their relationship with revenue,
orders and profitability.

Covers:
- Average selling price
- Unit price analysis
- Discount analysis
- Revenue by discount band
- Profitability by discount band
- Product pricing performance
- Category pricing performance
- Discount vs quantity
- Discount vs profit margin
- Pricing opportunities
- Discount risk signals

Stable interface
----------------
run_pricing_analytics(df, mapping) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
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


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_text(
    series: pd.Series,
    default: str = "Unknown",
) -> pd.Series:
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
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    ).fillna(0)

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    return (
        numerator
        .div(denominator.replace(0, np.nan))
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def _prepare_dataframe(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build an internal standardized dataframe.

    Original dataframe is never modified.
    """

    fields = [
        "order_id",
        "order_date",
        "sales_amount",
        "quantity",
        "unit_price",
        "cost",
        "profit",
        "discount",
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "brand",
        "customer_id",
        "campaign_id",
        "campaign_name",
        "coupon_code",
        "channel",
    ]

    working = pd.DataFrame(index=df.index)

    for field in fields:
        source_column = _get_source_column(
            mapping,
            field,
        )

        if (
            source_column
            and source_column in df.columns
        ):
            working[field] = df[source_column]

    # Numeric fields
    for field in [
        "sales_amount",
        "quantity",
        "unit_price",
        "cost",
        "profit",
        "discount",
    ]:
        if field in working.columns:
            working[field] = _safe_numeric(
                working[field]
            )

    # Text fields
    for field in [
        "order_id",
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "brand",
        "customer_id",
        "campaign_id",
        "campaign_name",
        "coupon_code",
        "channel",
    ]:
        if field in working.columns:
            working[field] = _safe_text(
                working[field]
            )

    # Derive profit when cost exists but profit does not
    if (
        "profit" not in working.columns
        and "cost" in working.columns
        and "sales_amount" in working.columns
    ):
        working["profit"] = (
            working["sales_amount"]
            - working["cost"]
        )

    # Derive unit price when possible
    if (
        "unit_price" not in working.columns
        and "sales_amount" in working.columns
        and "quantity" in working.columns
    ):
        working["unit_price"] = _safe_divide(
            working["sales_amount"],
            working["quantity"],
        )

    return working


# ---------------------------------------------------------------------
# Discount normalization
# ---------------------------------------------------------------------

def _normalize_discount(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize discount values into percentage form.

    Handles common cases:
    - 10 -> 10%
    - 0.10 -> 10%
    """

    result = working.copy()

    if "discount" not in result.columns:
        return result

    discount = result["discount"].copy()

    valid = discount.dropna()

    if valid.empty:
        result["discount_pct"] = 0.0
        return result

    # If almost all values are <= 1, interpret them as fractions.
    if valid.abs().quantile(0.95) <= 1:
        discount = discount * 100

    result["discount_pct"] = (
        discount
        .clip(lower=0)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return result


# ---------------------------------------------------------------------
# Discount bands
# ---------------------------------------------------------------------

def _assign_discount_band(
    discount: float,
) -> str:
    """Assign a business-friendly discount band."""

    if discount <= 0:
        return "No Discount"

    if discount <= 5:
        return "0-5%"

    if discount <= 10:
        return "5-10%"

    if discount <= 20:
        return "10-20%"

    if discount <= 30:
        return "20-30%"

    return "30%+"


def _build_discount_bands(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze performance across discount bands."""

    if "discount_pct" not in working.columns:
        return _empty(
            [
                "discount_band",
                "orders",
                "revenue",
                "quantity",
                "profit",
                "average_discount_pct",
                "average_order_value",
                "profit_margin",
                "revenue_contribution_pct",
            ]
        )

    data = working.copy()

    data["discount_band"] = data[
        "discount_pct"
    ].apply(
        _assign_discount_band
    )

    rows = []

    for band, group in data.groupby(
        "discount_band",
        dropna=False,
    ):
        orders = (
            group["order_id"].nunique()
            if "order_id" in group.columns
            else len(group)
        )

        revenue = float(
            group["sales_amount"].sum()
        )

        quantity = float(
            group["quantity"].sum()
        ) if "quantity" in group.columns else 0.0

        profit = float(
            group["profit"].sum()
        ) if "profit" in group.columns else 0.0

        average_discount = float(
            group["discount_pct"].mean()
        )

        aov = (
            revenue / orders
            if orders > 0
            else 0.0
        )

        margin = (
            profit / revenue * 100
            if revenue != 0
            else 0.0
        )

        rows.append(
            {
                "discount_band": band,
                "orders": int(orders),
                "revenue": revenue,
                "quantity": quantity,
                "profit": profit,
                "average_discount_pct": round(
                    average_discount,
                    2,
                ),
                "average_order_value": round(
                    aov,
                    2,
                ),
                "profit_margin": round(
                    margin,
                    2,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    total_revenue = result["revenue"].sum()

    result["revenue_contribution_pct"] = (
        result["revenue"]
        / total_revenue
        * 100
        if total_revenue
        else 0
    )

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# Pricing summary
# ---------------------------------------------------------------------

def _build_summary(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """Build high-level pricing KPIs."""

    revenue = float(
        working["sales_amount"].sum()
    ) if "sales_amount" in working.columns else 0.0

    quantity = float(
        working["quantity"].sum()
    ) if "quantity" in working.columns else 0.0

    orders = (
        working["order_id"].nunique()
        if "order_id" in working.columns
        else len(working)
    )

    profit = float(
        working["profit"].sum()
    ) if "profit" in working.columns else 0.0

    if "unit_price" in working.columns:
        prices = working["unit_price"].dropna()

        average_unit_price = (
            float(prices.mean())
            if not prices.empty
            else 0.0
        )

        median_unit_price = (
            float(prices.median())
            if not prices.empty
            else 0.0
        )

    else:
        average_unit_price = (
            revenue / quantity
            if quantity > 0
            else 0.0
        )

        median_unit_price = 0.0

    if "discount_pct" in working.columns:
        discount = working["discount_pct"]

        average_discount = float(
            discount.mean()
        )

        median_discount = float(
            discount.median()
        )

        discounted_rows = int(
            (discount > 0).sum()
        )

        discount_rate = (
            discounted_rows
            / len(discount)
            * 100
            if len(discount)
            else 0.0
        )

    else:
        average_discount = 0.0
        median_discount = 0.0
        discounted_rows = 0
        discount_rate = 0.0

    discounted_rows_mask = (
        working["discount_pct"] > 0
        if "discount_pct" in working.columns
        else pd.Series(False, index=working.index)
    )

    discounted_revenue = float(
        working.loc[
            discounted_rows_mask,
            "sales_amount",
        ].sum()
    )

    discounted_profit = float(
        working.loc[
            discounted_rows_mask,
            "profit",
        ].sum()
    ) if "profit" in working.columns else 0.0

    non_discounted_profit = float(
        working.loc[
            ~discounted_rows_mask,
            "profit",
        ].sum()
    ) if "profit" in working.columns else 0.0

    return {
        "revenue": revenue,
        "quantity": quantity,
        "orders": int(orders),
        "profit": profit,
        "profit_margin": (
            profit / revenue * 100
            if revenue
            else 0.0
        ),
        "average_unit_price": round(
            average_unit_price,
            2,
        ),
        "median_unit_price": round(
            median_unit_price,
            2,
        ),
        "average_discount_pct": round(
            average_discount,
            2,
        ),

        "average_discount": round(
            average_discount,
            2,
        ),
        "median_discount_pct": round(
            median_discount,
            2,
        ),
        "discounted_order_rate": round(
            discount_rate,
            2,
        ),
        "discounted_rows": discounted_rows,
        "discounted_revenue": round(
            discounted_revenue,
            2,
        ),
        "profit_impact": round(
            discounted_profit - non_discounted_profit,
            2,
        ),
    }


# ---------------------------------------------------------------------
# Product pricing
# ---------------------------------------------------------------------

def _build_product_pricing(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze pricing and discounts by product."""

    product_field = None

    if "product_name" in working.columns:
        product_field = "product_name"
    elif "product_id" in working.columns:
        product_field = "product_id"

    if product_field is None:
        return _empty(
            [
                "product",
                "revenue",
                "quantity",
                "orders",
                "average_unit_price",
                "average_discount_pct",
                "profit",
                "profit_margin",
                "pricing_signal",
            ]
        )

    grouped = (
        working.groupby(
            product_field,
            dropna=False,
        )
        .agg(
            revenue=("sales_amount", "sum"),
            quantity=("quantity", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            average_unit_price=("unit_price", "mean")
            if "unit_price" in working.columns
            else ("sales_amount", "mean"),
            average_discount_pct=("discount_pct", "mean")
            if "discount_pct" in working.columns
            else ("sales_amount", lambda x: 0.0),
            profit=("profit", "sum")
            if "profit" in working.columns
            else ("sales_amount", lambda x: 0.0),
        )
        .reset_index()
        .rename(
            columns={
                product_field: "product",
            }
        )
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    median_revenue = grouped["revenue"].median()
    median_margin = grouped["profit_margin"].median()

    def pricing_signal(row: pd.Series) -> str:
        if (
            row["revenue"] >= median_revenue
            and row["profit_margin"] >= median_margin
        ):
            return "Strong Price Economics"

        if (
            row["revenue"] >= median_revenue
            and row["profit_margin"] < median_margin
        ):
            return "High Volume / Margin Pressure"

        if (
            row["revenue"] < median_revenue
            and row["profit_margin"] >= median_margin
        ):
            return "Pricing Growth Opportunity"

        return "Review Pricing"

    grouped["pricing_signal"] = grouped.apply(
        pricing_signal,
        axis=1,
    )

    return grouped.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Category pricing
# ---------------------------------------------------------------------

def _build_category_pricing(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze pricing and discounts by category."""

    if "category" not in working.columns:
        return _empty(
            [
                "category",
                "revenue",
                "quantity",
                "orders",
                "average_unit_price",
                "average_discount_pct",
                "profit",
                "profit_margin",
            ]
        )

    grouped = (
        working.groupby(
            "category",
            dropna=False,
        )
        .agg(
            revenue=("sales_amount", "sum"),
            quantity=("quantity", "sum"),
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            average_unit_price=("unit_price", "mean")
            if "unit_price" in working.columns
            else ("sales_amount", "mean"),
            average_discount_pct=("discount_pct", "mean")
            if "discount_pct" in working.columns
            else ("sales_amount", lambda x: 0.0),
            profit=("profit", "sum")
            if "profit" in working.columns
            else ("sales_amount", lambda x: 0.0),
        )
        .reset_index()
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    return grouped.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Discount impact
# ---------------------------------------------------------------------

def _build_discount_impact(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare discounted and non-discounted transactions.

    This is descriptive analysis and should not be interpreted as
    causal evidence that discounts caused the observed changes.
    """

    if "discount_pct" not in working.columns:
        return {
            "available": False,
            "discounted": {},
            "non_discounted": {},
            "comparison": {},
        }

    data = working.copy()

    discounted = data[
        data["discount_pct"] > 0
    ]

    non_discounted = data[
        data["discount_pct"] <= 0
    ]

    def metrics(
        group: pd.DataFrame,
    ) -> Dict[str, float]:
        revenue = float(
            group["sales_amount"].sum()
        )

        profit = (
            float(group["profit"].sum())
            if "profit" in group.columns
            else 0.0
        )

        quantity = (
            float(group["quantity"].sum())
            if "quantity" in group.columns
            else 0.0
        )

        orders = (
            group["order_id"].nunique()
            if "order_id" in group.columns
            else len(group)
        )

        return {
            "orders": int(orders),
            "revenue": revenue,
            "quantity": quantity,
            "profit": profit,
            "average_order_value": (
                revenue / orders
                if orders
                else 0.0
            ),
            "profit_margin": (
                profit / revenue * 100
                if revenue
                else 0.0
            ),
        }

    discounted_metrics = metrics(
        discounted
    )

    non_discounted_metrics = metrics(
        non_discounted
    )

    return {
        "available": True,
        "discounted": discounted_metrics,
        "non_discounted": non_discounted_metrics,
        "comparison": {
            "aov_difference": (
                discounted_metrics[
                    "average_order_value"
                ]
                - non_discounted_metrics[
                    "average_order_value"
                ]
            ),
            "margin_difference": (
                discounted_metrics[
                    "profit_margin"
                ]
                - non_discounted_metrics[
                    "profit_margin"
                ]
            ),
            "profit_difference": (
                discounted_metrics["profit"]
                - non_discounted_metrics["profit"]
            ),
        },
        "interpretation_note": (
            "Differences are descriptive associations. "
            "They do not establish causal impact of discounting."
        ),
    }


# ---------------------------------------------------------------------
# High-discount risk
# ---------------------------------------------------------------------

def _build_discount_risk(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """Identify potentially risky discounting patterns."""

    if "discount_pct" not in working.columns:
        return {
            "available": False,
            "high_discount_threshold": 0.0,
            "high_discount_order_count": 0,
            "high_discount_revenue": 0.0,
            "high_discount_profit": 0.0,
            "high_discount_margin": 0.0,
            "risk_level": "Unavailable",
        }

    threshold = 20.0

    high_discount = working[
        working["discount_pct"] >= threshold
    ]

    orders = (
        high_discount["order_id"].nunique()
        if "order_id" in high_discount.columns
        else len(high_discount)
    )

    revenue = float(
        high_discount["sales_amount"].sum()
    )

    profit = (
        float(high_discount["profit"].sum())
        if "profit" in high_discount.columns
        else 0.0
    )

    margin = (
        profit / revenue * 100
        if revenue
        else 0.0
    )

    if margin < 0:
        risk = "Critical"

    elif margin < 5:
        risk = "High"

    elif margin < 10:
        risk = "Medium"

    else:
        risk = "Low"

    return {
        "available": True,
        "high_discount_threshold": threshold,
        "high_discount_order_count": int(orders),
        "high_discount_revenue": revenue,
        "high_discount_profit": profit,
        "high_discount_margin": round(
            margin,
            2,
        ),
        "risk_level": risk,
    }


# ---------------------------------------------------------------------
# Pricing opportunities
# ---------------------------------------------------------------------

def _build_pricing_opportunities(
    summary: Dict[str, Any],
    discount_bands: pd.DataFrame,
    product_pricing: pd.DataFrame,
) -> list[Dict[str, Any]]:
    """Generate pricing and discount recommendations."""

    opportunities = []

    average_discount = summary.get(
        "average_discount_pct",
        0.0,
    )

    if average_discount >= 20:
        opportunities.append(
            {
                "priority": "High",
                "type": "Heavy Discounting",
                "message": (
                    f"Average discount is {average_discount:.1f}%. "
                    "Review whether discount depth is eroding margins."
                ),
            }
        )

    if not discount_bands.empty:
        high_discount = discount_bands[
            discount_bands["average_discount_pct"] >= 20
        ]

        if not high_discount.empty:
            weak_margin = high_discount[
                high_discount["profit_margin"] < 10
            ]

            if not weak_margin.empty:
                opportunities.append(
                    {
                        "priority": "High",
                        "type": "Discount Margin Risk",
                        "message": (
                            "High-discount bands are generating "
                            "relatively weak margins. Review pricing "
                            "floors and promotion rules."
                        ),
                    }
                )

    if not product_pricing.empty:
        growth_products = product_pricing[
            product_pricing["pricing_signal"]
            == "Pricing Growth Opportunity"
        ]

        if not growth_products.empty:
            opportunities.append(
                {
                    "priority": "Medium",
                    "type": "Pricing Growth",
                    "message": (
                        f"{len(growth_products)} product(s) show "
                        "stronger-than-median margins but lower revenue. "
                        "Test targeted demand-generation strategies."
                    ),
                }
            )

        margin_pressure = product_pricing[
            product_pricing["pricing_signal"]
            == "High Volume / Margin Pressure"
        ]

        if not margin_pressure.empty:
            opportunities.append(
                {
                    "priority": "High",
                    "type": "Margin Pressure",
                    "message": (
                        f"{len(margin_pressure)} high-revenue "
                        "product(s) show below-median margins. "
                        "Review price, discount and cost structure."
                    ),
                }
            )

    return opportunities


# ---------------------------------------------------------------------
# Channel pricing
# ---------------------------------------------------------------------

def _build_channel_pricing(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze pricing and discount behavior by sales channel."""

    if "channel" not in working.columns:
        return _empty(
            [
                "channel",
                "orders",
                "revenue",
                "average_discount_pct",
                "profit",
                "profit_margin",
            ]
        )

    grouped = (
        working.groupby(
            "channel",
            dropna=False,
        )
        .agg(
            orders=("order_id", "nunique")
            if "order_id" in working.columns
            else ("sales_amount", "size"),
            revenue=("sales_amount", "sum"),
            average_discount_pct=("discount_pct", "mean")
            if "discount_pct" in working.columns
            else ("sales_amount", lambda x: 0.0),
            profit=("profit", "sum")
            if "profit" in working.columns
            else ("sales_amount", lambda x: 0.0),
        )
        .reset_index()
    )

    grouped["profit_margin"] = _safe_divide(
        grouped["profit"] * 100,
        grouped["revenue"],
    )

    return grouped.sort_values(
        "revenue",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------

def _build_capabilities(
    working: pd.DataFrame,
) -> Dict[str, bool]:
    """Expose available pricing analytics."""

    return {
        "pricing_analysis": (
            "unit_price" in working.columns
            or (
                "sales_amount" in working.columns
                and "quantity" in working.columns
            )
        ),
        "discount_analysis": (
            "discount" in working.columns
        ),
        "product_pricing": (
            "product_name" in working.columns
            or "product_id" in working.columns
        ),
        "category_pricing": (
            "category" in working.columns
        ),
        "channel_pricing": (
            "channel" in working.columns
        ),
        "profitability_analysis": (
            "profit" in working.columns
        ),
    }


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_pricing_analytics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run Pricing & Discount Intelligence.

    Parameters
    ----------
    df:
        Analysis-ready dataframe.

    mapping:
        Schema mapping generated by schema_mapper.py.

    Returns
    -------
    dict
        Stable Pricing Intelligence result contract.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return {
            "available": False,
            "summary": {},
            "discount_bands": pd.DataFrame(),
            "product_pricing": pd.DataFrame(),
            "category_pricing": pd.DataFrame(),
            "channel_pricing": pd.DataFrame(),
            "discount_impact": {},
            "discount_risk": {},
            "pricing_opportunities": [],
            "capabilities": {},
            "warnings": [
                "A non-empty pandas DataFrame is required."
            ],
        }

    working = _prepare_dataframe(
        df,
        mapping,
    )

    capabilities = _build_capabilities(
        working
    )

    if not capabilities["pricing_analysis"]:
        return {
            "available": False,
            "summary": {},
            "discount_bands": pd.DataFrame(),
            "product_pricing": pd.DataFrame(),
            "category_pricing": pd.DataFrame(),
            "channel_pricing": pd.DataFrame(),
            "discount_impact": {},
            "discount_risk": {},
            "pricing_opportunities": [],
            "capabilities": capabilities,
            "warnings": [
                "Pricing information is unavailable. "
                "Provide unit price or sales amount plus quantity."
            ],
        }

    working = _normalize_discount(
        working
    )

    summary = _build_summary(
        working
    )

    discount_bands = _build_discount_bands(
        working
    )

    product_pricing = _build_product_pricing(
        working
    )

    category_pricing = _build_category_pricing(
        working
    )

    channel_pricing = _build_channel_pricing(
        working
    )

    discount_impact = _build_discount_impact(
        working
    )

    discount_risk = _build_discount_risk(
        working
    )

    pricing_opportunities = (
        _build_pricing_opportunities(
            summary,
            discount_bands,
            product_pricing,
        )
    )

    warnings = []

    if not capabilities["discount_analysis"]:
        warnings.append(
            "Discount field is unavailable; "
            "discount-specific analytics are limited."
        )

    if not capabilities["profitability_analysis"]:
        warnings.append(
            "Profit field is unavailable; "
            "profitability-based pricing analysis is limited."
        )

    if not capabilities["product_pricing"]:
        warnings.append(
            "Product fields are unavailable; "
            "product-level pricing analysis is limited."
        )

    return {
        "available": True,
        "summary": summary,
        "discount_bands": discount_bands,
        "product_pricing": product_pricing,
        "category_pricing": category_pricing,
        "channel_pricing": channel_pricing,
        "discount_impact": discount_impact,
        "discount_risk": discount_risk,
        "pricing_opportunities": pricing_opportunities,
        "capabilities": capabilities,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------

run_pricing_intelligence = run_pricing_analytics