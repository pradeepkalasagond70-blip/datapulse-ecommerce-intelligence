"""
DataPulse - Insight Engine
--------------------------
Converts analytics and machine-learning outputs into concise,
business-oriented insights and recommendations.

Public interface:
    generate_insights(analytics_results)

The engine is intentionally rule-based for the MVP. This makes the
insights deterministic, explainable, fast, and safe for deployment.

Expected analytics_results structure:
    {
        "core": {...},
        "customer": {...},
        "product": {...},
        "campaign_impact": {...},
        "market": {...},
        "delivery": {...},
        "pricing": {...},
        "reviews": {...},
        "churn": {...},
        "segmentation": {...},
        "forecast": {...},
        "anomaly": {...},
    }

The engine is defensive: unavailable modules or missing metrics do not
cause the complete DataPulse pipeline to fail.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_INSIGHTS = 25
MAX_RECOMMENDATIONS = 15


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely convert a value to float."""

    if value is None:
        return default

    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_dict(
    results: dict[str, Any],
    *keys: str,
) -> dict[str, Any]:
    """Safely retrieve a nested dictionary."""

    current: Any = results

    for key in keys:

        if not isinstance(current, dict):
            return {}

        current = current.get(key)

    return current if isinstance(current, dict) else {}


def _get_dataframe(
    results: dict[str, Any],
    *keys: str,
) -> pd.DataFrame:
    """Safely retrieve a DataFrame."""

    current: Any = results

    for key in keys:

        if not isinstance(current, dict):
            return pd.DataFrame()

        current = current.get(key)

    if isinstance(current, pd.DataFrame):
        return current

    return pd.DataFrame()


def _is_available(
    results: dict[str, Any],
    key: str,
) -> bool:
    """Check whether an analytics module is available."""

    module = results.get(
        key,
        {},
    )

    return (
        isinstance(module, dict)
        and bool(
            module.get(
                "available",
                False,
            )
        )
    )


def _round(
    value: Any,
    digits: int = 2,
) -> Optional[float]:
    """Safely round numeric values."""

    number = _safe_float(value)

    if number is None:
        return None

    return round(
        number,
        digits,
    )


# ---------------------------------------------------------------------
# Insight creation
# ---------------------------------------------------------------------

def _make_insight(
    category: str,
    title: str,
    message: str,
    priority: str = "Medium",
    metric: Any = None,
    metric_label: Optional[str] = None,
) -> dict[str, Any]:
    """Create a standardized insight."""

    return {
        "category": category,
        "title": title,
        "message": message,
        "priority": priority,
        "metric": _round(metric),
        "metric_label": metric_label,
    }


def _make_recommendation(
    category: str,
    title: str,
    action: str,
    priority: str = "Medium",
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Create a standardized recommendation."""

    return {
        "category": category,
        "title": title,
        "action": action,
        "priority": priority,
        "reason": reason or "",
    }


# ---------------------------------------------------------------------
# Core business insights
# ---------------------------------------------------------------------

def _core_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "core",
    ):
        return insights, recommendations

    core = results.get(
        "core",
        {},
    )

    summary = core.get(
        "summary",
        {},
    )

    revenue = _safe_float(
        summary.get(
            "total_revenue"
        )
    )

    orders = _safe_int(
        summary.get(
            "total_orders"
        )
    )

    profit = _safe_float(
        summary.get(
            "total_profit"
        )
    )

    margin = _safe_float(
        summary.get(
            "profit_margin_pct"
        )
    )

    if revenue is not None:
        insights.append(
            _make_insight(
                "Executive",
                "Revenue base identified",
                (
                    f"DataPulse analyzed approximately "
                    f"{revenue:,.2f} in total revenue across "
                    f"{orders:,} orders."
                ),
                "Medium",
                revenue,
                "Total Revenue",
            )
        )

    if margin is not None:

        if margin < 5:
            priority = "Critical"

            message = (
                f"Overall profit margin is only {margin:.2f}%. "
                "Margin protection should be a top business priority."
            )

            recommendations.append(
                _make_recommendation(
                    "Profitability",
                    "Protect margins",
                    (
                        "Review discounting, product mix, pricing "
                        "and low-margin sellers or regions."
                    ),
                    "Critical",
                    "Overall profit margin is below 5%.",
                )
            )

        elif margin < 15:
            priority = "High"

            message = (
                f"Overall profit margin is {margin:.2f}%. "
                "There is meaningful room to improve profitability."
            )

            recommendations.append(
                _make_recommendation(
                    "Profitability",
                    "Improve margin",
                    (
                        "Identify low-margin products and high-discount "
                        "transactions and optimize pricing."
                    ),
                    "High",
                    "Overall margin is below 15%.",
                )
            )

        else:
            priority = "Low"

            message = (
                f"Overall profit margin is {margin:.2f}%, "
                "indicating a comparatively healthy aggregate margin."
            )

        insights.append(
            _make_insight(
                "Profitability",
                "Profitability signal",
                message,
                priority,
                margin,
                "Profit Margin %",
            )
        )

    if profit is not None and profit < 0:

        insights.append(
            _make_insight(
                "Profitability",
                "Business is operating at an aggregate loss",
                (
                    f"Total profit is {profit:,.2f}. "
                    "Revenue growth alone may not improve business performance "
                    "until cost and margin drivers are addressed."
                ),
                "Critical",
                profit,
                "Total Profit",
            )
        )

        recommendations.append(
            _make_recommendation(
                "Profitability",
                "Prioritize profitability recovery",
                (
                    "Investigate negative-margin products, excessive discounts, "
                    "seller economics and operational costs."
                ),
                "Critical",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Customer insights
# ---------------------------------------------------------------------

def _customer_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "customer",
    ):
        return insights, recommendations

    customer = results.get(
        "customer",
        {},
    )

    summary = customer.get(
        "summary",
        {},
    )

    total_customers = _safe_int(
        summary.get(
            "total_customers"
        )
    )

    repeat_rate = _safe_float(
        summary.get(
            "repeat_customer_rate_pct"
        )
    )

    avg_order_value = _safe_float(
        summary.get(
            "average_order_value"
        )
    )

    if total_customers:

        insights.append(
            _make_insight(
                "Customer",
                "Customer base analyzed",
                (
                    f"DataPulse identified {total_customers:,} "
                    "customers in the analysis dataset."
                ),
                "Low",
                total_customers,
                "Customers",
            )
        )

    if repeat_rate is not None:

        if repeat_rate < 20:

            insights.append(
                _make_insight(
                    "Customer",
                    "Repeat purchase rate is low",
                    (
                        f"Only {repeat_rate:.1f}% of customers "
                        "are classified as repeat customers."
                    ),
                    "High",
                    repeat_rate,
                    "Repeat Customer Rate %",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Customer",
                    "Strengthen retention",
                    (
                        "Build targeted retention journeys using "
                        "purchase history, customer value and recency."
                    ),
                    "High",
                    "Low repeat customer rate.",
                )
            )

        elif repeat_rate >= 50:

            insights.append(
                _make_insight(
                    "Customer",
                    "Strong repeat-purchase behavior",
                    (
                        f"{repeat_rate:.1f}% of customers are "
                        "classified as repeat customers."
                    ),
                    "Low",
                    repeat_rate,
                    "Repeat Customer Rate %",
                )
            )

    if avg_order_value is not None:

        insights.append(
            _make_insight(
                "Customer",
                "Average order value benchmark",
                (
                    f"Average order value is "
                    f"{avg_order_value:,.2f}."
                ),
                "Low",
                avg_order_value,
                "Average Order Value",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Product insights
# ---------------------------------------------------------------------

def _product_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "product",
    ):
        return insights, recommendations

    product = results.get(
        "product",
        {},
    )

    top_products = product.get(
        "top_products",
        pd.DataFrame(),
    )

    if (
        isinstance(
            top_products,
            pd.DataFrame,
        )
        and not top_products.empty
    ):

        first = top_products.iloc[0]

        product_name = first.get(
            "product_name",
            first.get(
                "product_id",
                "Top Product",
            ),
        )

        revenue = _safe_float(
            first.get(
                "revenue"
            )
        )

        insights.append(
            _make_insight(
                "Product",
                "Top revenue-driving product",
                (
                    f"{product_name} is currently the leading "
                    "product by revenue in the available product analysis."
                ),
                "Medium",
                revenue,
                "Top Product Revenue",
            )
        )

    product_risk = product.get(
        "product_risk",
        pd.DataFrame(),
    )

    if (
        isinstance(
            product_risk,
            pd.DataFrame,
        )
        and not product_risk.empty
    ):

        recommendations.append(
            _make_recommendation(
                "Product",
                "Review product-level risks",
                (
                    "Prioritize products flagged by DataPulse for "
                    "weak revenue, profitability or demand performance."
                ),
                "Medium",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Campaign Impact insights
# ---------------------------------------------------------------------

def _campaign_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "campaign_impact",
    ):
        return insights, recommendations

    campaign = results.get(
        "campaign_impact",
        {},
    )

    summary = campaign.get(
        "summary",
        {},
    )

    roi = _safe_float(
        summary.get(
            "average_roi"
        )
    )

    campaign_count = _safe_int(
        summary.get(
            "campaign_count"
        )
    )

    if campaign_count:
        insights.append(
            _make_insight(
                "Campaign Impact",
                "Campaign performance analyzed",
                (
                    f"DataPulse evaluated {campaign_count:,} "
                    "campaign records."
                ),
                "Low",
                campaign_count,
                "Campaign Records",
            )
        )

    if roi is not None:

        if roi < 0:

            insights.append(
                _make_insight(
                    "Campaign Impact",
                    "Campaign ROI requires attention",
                    (
                        f"Average campaign ROI is {roi:.2f}. "
                        "Campaign spend should be reviewed against "
                        "incremental business impact."
                    ),
                    "High",
                    roi,
                    "Average ROI",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Campaign Impact",
                    "Optimize campaign allocation",
                    (
                        "Reduce investment in consistently weak campaigns "
                        "and redirect budget toward campaigns with stronger "
                        "revenue and profitability outcomes."
                    ),
                    "High",
                )
            )

        else:

            insights.append(
                _make_insight(
                    "Campaign Impact",
                    "Campaigns are generating positive return",
                    (
                        f"Average campaign ROI is {roi:.2f}, "
                        "indicating positive return across the "
                        "available campaign analysis."
                    ),
                    "Low",
                    roi,
                    "Average ROI",
                )
            )

    return insights, recommendations


# ---------------------------------------------------------------------
# Market & Seller insights
# ---------------------------------------------------------------------

def _market_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "market",
    ):
        return insights, recommendations

    market = results.get(
        "market",
        {},
    )

    opportunities = market.get(
        "regional_opportunities",
        pd.DataFrame(),
    )

    if (
        isinstance(
            opportunities,
            pd.DataFrame,
        )
        and not opportunities.empty
    ):

        insights.append(
            _make_insight(
                "Market & Seller",
                "Regional opportunities identified",
                (
                    f"DataPulse identified {len(opportunities):,} "
                    "regional opportunities based on available "
                    "market-performance signals."
                ),
                "Medium",
                len(opportunities),
                "Regional Opportunities",
            )
        )

        recommendations.append(
            _make_recommendation(
                "Market & Seller",
                "Prioritize regional opportunities",
                (
                    "Compare high-potential regions against current "
                    "revenue, orders, seller coverage and product demand "
                    "before expanding."
                ),
                "Medium",
            )
        )

    concentration = market.get(
        "concentration",
        {},
    )

    top_seller_share = _safe_float(
        concentration.get(
            "top_seller_revenue_share_pct"
        )
    )

    if (
        top_seller_share is not None
        and top_seller_share > 50
    ):

        insights.append(
            _make_insight(
                "Market & Seller",
                "Seller concentration risk",
                (
                    f"The largest seller contributes approximately "
                    f"{top_seller_share:.1f}% of analyzed revenue."
                ),
                "High",
                top_seller_share,
                "Top Seller Revenue Share %",
            )
        )

        recommendations.append(
            _make_recommendation(
                "Market & Seller",
                "Reduce seller concentration risk",
                (
                    "Diversify seller contribution and identify "
                    "additional high-performing sellers or markets."
                ),
                "High",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Delivery insights
# ---------------------------------------------------------------------

def _delivery_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "delivery",
    ):
        return insights, recommendations

    delivery = results.get(
        "delivery",
        {},
    )

    summary = delivery.get(
        "summary",
        {},
    )

    return_rate = _safe_float(
        summary.get(
            "return_rate_pct"
        )
    )

    delayed_rate = _safe_float(
        summary.get(
            "delayed_rate_pct"
        )
    )

    if return_rate is not None:

        if return_rate >= 10:

            insights.append(
                _make_insight(
                    "Delivery",
                    "Return rate is elevated",
                    (
                        f"The analyzed return rate is "
                        f"{return_rate:.1f}%."
                    ),
                    "High",
                    return_rate,
                    "Return Rate %",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Delivery",
                    "Investigate return drivers",
                    (
                        "Break down returns by product, seller, region "
                        "and delivery method to identify recurring causes."
                    ),
                    "High",
                )
            )

    if delayed_rate is not None:

        if delayed_rate >= 10:

            insights.append(
                _make_insight(
                    "Delivery",
                    "Delivery delays need attention",
                    (
                        f"Approximately {delayed_rate:.1f}% "
                        "of analyzed deliveries are delayed."
                    ),
                    "High",
                    delayed_rate,
                    "Delayed Delivery Rate %",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Delivery",
                    "Reduce delivery delays",
                    (
                        "Identify high-delay regions, sellers and "
                        "shipping methods and prioritize corrective action."
                    ),
                    "High",
                )
            )

    return insights, recommendations


# ---------------------------------------------------------------------
# Pricing insights
# ---------------------------------------------------------------------

def _pricing_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "pricing",
    ):
        return insights, recommendations

    pricing = results.get(
        "pricing",
        {},
    )

    discount_risk = pricing.get(
        "discount_risk",
        pd.DataFrame(),
    )

    if (
        isinstance(
            discount_risk,
            pd.DataFrame,
        )
        and not discount_risk.empty
    ):

        insights.append(
            _make_insight(
                "Pricing",
                "Discount-related risks detected",
                (
                    f"{len(discount_risk):,} pricing records or "
                    "entities were flagged for potential discount risk."
                ),
                "High",
                len(discount_risk),
                "Pricing Risk Records",
            )
        )

        recommendations.append(
            _make_recommendation(
                "Pricing",
                "Review discount effectiveness",
                (
                    "Identify discount levels that increase sales "
                    "without creating disproportionate margin erosion."
                ),
                "High",
            )
        )

    discount_impact = pricing.get(
        "discount_impact",
        pd.DataFrame(),
    )

    if (
        isinstance(
            discount_impact,
            pd.DataFrame,
        )
        and not discount_impact.empty
    ):

        recommendations.append(
            _make_recommendation(
                "Pricing",
                "Use discount elasticity signals",
                (
                    "Compare revenue, order volume and profitability "
                    "across discount bands before changing promotional pricing."
                ),
                "Medium",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Review & NLP insights
# ---------------------------------------------------------------------

def _review_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "reviews",
    ):
        return insights, recommendations

    reviews = results.get(
        "reviews",
        {},
    )

    summary = reviews.get(
        "summary",
        {},
    )

    negative_rate = _safe_float(
        summary.get(
            "negative_sentiment_pct"
        )
    )

    average_rating = _safe_float(
        summary.get(
            "average_rating"
        )
    )

    if negative_rate is not None:

        if negative_rate >= 30:

            insights.append(
                _make_insight(
                    "Reviews & NLP",
                    "Customer sentiment requires attention",
                    (
                        f"{negative_rate:.1f}% of analyzed reviews "
                        "are classified as negative."
                    ),
                    "High",
                    negative_rate,
                    "Negative Sentiment %",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Reviews & NLP",
                    "Investigate negative customer themes",
                    (
                        "Use product-level and keyword-level review "
                        "analysis to identify the most common customer pain points."
                    ),
                    "High",
                )
            )

    if average_rating is not None:

        insights.append(
            _make_insight(
                "Reviews & NLP",
                "Customer rating benchmark",
                (
                    f"Average review rating is "
                    f"{average_rating:.2f}."
                ),
                "Low",
                average_rating,
                "Average Rating",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Churn insights
# ---------------------------------------------------------------------

def _churn_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "churn",
    ):
        return insights, recommendations

    churn = results.get(
        "churn",
        {},
    )

    risk_summary = churn.get(
        "risk_summary",
        {},
    )

    high_risk_count = _safe_int(
        risk_summary.get(
            "high_risk_customers"
        )
    )

    high_value_risk = churn.get(
        "high_value_risk",
        pd.DataFrame(),
    )

    if high_risk_count > 0:

        insights.append(
            _make_insight(
                "ML - Churn",
                "Customers at elevated churn risk",
                (
                    f"{high_risk_count:,} customers have been "
                    "identified as high churn-risk candidates."
                ),
                "High",
                high_risk_count,
                "High-Risk Customers",
            )
        )

        recommendations.append(
            _make_recommendation(
                "ML - Churn",
                "Launch targeted retention actions",
                (
                    "Prioritize high-risk customers using customer value, "
                    "recency and purchase behavior to design retention offers."
                ),
                "High",
            )
        )

    if (
        isinstance(
            high_value_risk,
            pd.DataFrame,
        )
        and not high_value_risk.empty
    ):

        insights.append(
            _make_insight(
                "ML - Churn",
                "High-value customers require retention focus",
                (
                    f"{len(high_value_risk):,} high-value customers "
                    "are included in the elevated-risk group."
                ),
                "Critical",
                len(high_value_risk),
                "High-Value At-Risk Customers",
            )
        )

        recommendations.append(
            _make_recommendation(
                "ML - Churn",
                "Protect high-value customers",
                (
                    "Create a priority retention list for high-value "
                    "customers showing declining engagement."
                ),
                "Critical",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Segmentation insights
# ---------------------------------------------------------------------

def _segmentation_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "segmentation",
    ):
        return insights, recommendations

    segmentation = results.get(
        "segmentation",
        {},
    )

    metrics = segmentation.get(
        "metrics",
        {},
    )

    cluster_count = _safe_int(
        metrics.get(
            "cluster_count"
        )
    )

    silhouette = _safe_float(
        metrics.get(
            "silhouette_score"
        )
    )

    profiles = segmentation.get(
        "segment_profiles",
        pd.DataFrame(),
    )

    if cluster_count:

        insights.append(
            _make_insight(
                "ML - Segmentation",
                "Customer segments discovered",
                (
                    f"DataPulse identified {cluster_count} "
                    "behavioral customer segments."
                ),
                "Medium",
                cluster_count,
                "Customer Segments",
            )
        )

    if silhouette is not None:

        insights.append(
            _make_insight(
                "ML - Segmentation",
                "Segmentation quality signal",
                (
                    f"The selected segmentation has a silhouette "
                    f"score of {silhouette:.2f}."
                ),
                "Low",
                silhouette,
                "Silhouette Score",
            )
        )

    if (
        isinstance(
            profiles,
            pd.DataFrame,
        )
        and not profiles.empty
    ):

        recommendations.append(
            _make_recommendation(
                "ML - Segmentation",
                "Activate segment-specific strategies",
                (
                    "Use the discovered customer segments to tailor "
                    "campaigns, offers, retention and product recommendations."
                ),
                "Medium",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Forecasting insights
# ---------------------------------------------------------------------

def _forecast_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "forecast",
    ):
        return insights, recommendations

    forecast = results.get(
        "forecast",
        {},
    )

    trend = forecast.get(
        "trend",
        {},
    )

    metrics = forecast.get(
        "metrics",
        {},
    )

    direction = trend.get(
        "forecast_direction",
        "Stable",
    )

    growth = _safe_float(
        trend.get(
            "forecast_growth_pct"
        )
    )

    selected_model = metrics.get(
        "selected_model"
    )

    if direction == "Growing":

        insights.append(
            _make_insight(
                "ML - Forecast",
                "Sales forecast indicates growth",
                (
                    "The forecast model indicates an upward sales "
                    "direction across the projected periods."
                ),
                "High",
                growth,
                "Forecast Growth %",
            )
        )

        recommendations.append(
            _make_recommendation(
                "ML - Forecast",
                "Prepare for projected demand",
                (
                    "Align inventory, fulfillment capacity and "
                    "commercial planning with the expected demand increase."
                ),
                "High",
            )
        )

    elif direction == "Declining":

        insights.append(
            _make_insight(
                "ML - Forecast",
                "Sales forecast indicates decline",
                (
                    "The forecast model indicates a downward sales "
                    "direction across the projected periods."
                ),
                "High",
                growth,
                "Forecast Growth %",
            )
        )

        recommendations.append(
            _make_recommendation(
                "ML - Forecast",
                "Investigate projected sales decline",
                (
                    "Review product, customer, market, pricing and "
                    "campaign signals before committing to inventory expansion."
                ),
                "High",
            )
        )

    else:

        insights.append(
            _make_insight(
                "ML - Forecast",
                "Sales forecast is relatively stable",
                (
                    "The forecast does not indicate a strong upward "
                    "or downward direction."
                ),
                "Medium",
                growth,
                "Forecast Growth %",
            )
        )

    if selected_model:

        insights.append(
            _make_insight(
                "ML - Forecast",
                "Forecast model selected automatically",
                (
                    f"DataPulse selected {selected_model} based on "
                    "time-ordered historical validation."
                ),
                "Low",
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Anomaly insights
# ---------------------------------------------------------------------

def _anomaly_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    if not _is_available(
        results,
        "anomaly",
    ):
        return insights, recommendations

    anomaly = results.get(
        "anomaly",
        {},
    )

    summary = anomaly.get(
        "anomaly_summary",
        {},
    )

    anomaly_count = _safe_int(
        summary.get(
            "anomaly_count"
        )
    )

    anomaly_rate = _safe_float(
        summary.get(
            "anomaly_rate_pct"
        )
    )

    critical_count = _safe_int(
        summary.get(
            "critical_count"
        )
    )

    if anomaly_count > 0:

        priority = (
            "Critical"
            if critical_count > 0
            else "High"
        )

        insights.append(
            _make_insight(
                "ML - Anomaly",
                "Unusual transaction patterns detected",
                (
                    f"DataPulse detected {anomaly_count:,} "
                    f"anomalous observations "
                    f"({anomaly_rate or 0:.1f}% of analyzed records)."
                ),
                priority,
                anomaly_rate,
                "Anomaly Rate %",
            )
        )

        recommendations.append(
            _make_recommendation(
                "ML - Anomaly",
                "Review anomalous transactions",
                (
                    "Investigate unusual sales values, quantities, "
                    "prices, discounts and profit patterns for data "
                    "quality or business exceptions."
                ),
                priority,
            )
        )

    return insights, recommendations


# ---------------------------------------------------------------------
# Cross-module intelligence
# ---------------------------------------------------------------------

def _cross_module_insights(
    results: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    insights = []
    recommendations = []

    # ---------------------------------------------------------------
    # Churn + customer value
    # ---------------------------------------------------------------

    churn = results.get(
        "churn",
        {},
    )

    customer = results.get(
        "customer",
        {},
    )

    if (
        _is_available(results, "churn")
        and _is_available(results, "customer")
    ):

        high_value_risk = churn.get(
            "high_value_risk",
            pd.DataFrame(),
        )

        customer_summary = customer.get(
            "summary",
            {},
        )

        total_customers = _safe_int(
            customer_summary.get(
                "total_customers"
            )
        )

        if (
            isinstance(
                high_value_risk,
                pd.DataFrame,
            )
            and not high_value_risk.empty
            and total_customers > 0
        ):

            risk_share = (
                len(high_value_risk)
                / total_customers
                * 100
            )

            insights.append(
                _make_insight(
                    "Cross-Module",
                    "High-value retention opportunity",
                    (
                        f"{risk_share:.1f}% of the analyzed customer "
                        "base is represented by high-value customers "
                        "with elevated churn risk."
                    ),
                    "Critical",
                    risk_share,
                    "High-Value Risk Share %",
                )
            )

    # ---------------------------------------------------------------
    # Pricing + reviews
    # ---------------------------------------------------------------

    pricing = results.get(
        "pricing",
        {}
    )

    reviews = results.get(
        "reviews",
        {}
    )

    if (
        _is_available(results, "pricing")
        and _is_available(results, "reviews")
    ):

        negative_rate = _safe_float(
            _get_dict(
                results,
                "reviews",
                "summary",
            ).get(
                "negative_sentiment_pct"
            )
        )

        if (
            negative_rate is not None
            and negative_rate >= 30
        ):

            insights.append(
                _make_insight(
                    "Cross-Module",
                    "Pricing and customer experience should be reviewed together",
                    (
                        "Elevated negative sentiment combined with "
                        "pricing signals warrants investigation of "
                        "value perception and product experience."
                    ),
                    "High",
                    negative_rate,
                    "Negative Sentiment %",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Cross-Module",
                    "Connect pricing with customer feedback",
                    (
                        "Compare negative review themes with products, "
                        "discount levels and price positioning before "
                        "making pricing changes."
                    ),
                    "High",
                )
            )

    # ---------------------------------------------------------------
    # Forecast + inventory / delivery
    # ---------------------------------------------------------------

    forecast = results.get(
        "forecast",
        {}
    )

    delivery = results.get(
        "delivery",
        {}
    )

    if (
        _is_available(results, "forecast")
        and _is_available(results, "delivery")
    ):

        direction = _get_dict(
            results,
            "forecast",
            "trend",
        ).get(
            "forecast_direction"
        )

        delayed_rate = _safe_float(
            _get_dict(
                results,
                "delivery",
                "summary",
            ).get(
                "delayed_rate_pct"
            )
        )

        if (
            direction == "Growing"
            and delayed_rate is not None
            and delayed_rate >= 10
        ):

            insights.append(
                _make_insight(
                    "Cross-Module",
                    "Growth and delivery capacity may conflict",
                    (
                        "Sales are forecast to grow while delivery "
                        "delays are already elevated."
                    ),
                    "Critical",
                    delayed_rate,
                    "Delayed Delivery Rate %",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Cross-Module",
                    "Prepare operational capacity before growth",
                    (
                        "Increase fulfillment readiness and review "
                        "delivery bottlenecks before scaling demand."
                    ),
                    "Critical",
                )
            )

    # ---------------------------------------------------------------
    # Campaign + forecast
    # ---------------------------------------------------------------

    campaign = results.get(
        "campaign_impact",
        {}
    )

    if (
        _is_available(results, "campaign_impact")
        and _is_available(results, "forecast")
    ):

        campaign_summary = _get_dict(
            results,
            "campaign_impact",
            "summary",
        )

        roi = _safe_float(
            campaign_summary.get(
                "average_roi"
            )
        )

        direction = _get_dict(
            results,
            "forecast",
            "trend",
        ).get(
            "forecast_direction"
        )

        if (
            roi is not None
            and roi > 0
            and direction == "Growing"
        ):

            insights.append(
                _make_insight(
                    "Cross-Module",
                    "Commercial momentum is positive",
                    (
                        "Campaign performance shows positive return "
                        "while the sales forecast also indicates growth."
                    ),
                    "Medium",
                    roi,
                    "Average Campaign ROI",
                )
            )

            recommendations.append(
                _make_recommendation(
                    "Cross-Module",
                    "Scale proven commercial activity carefully",
                    (
                        "Consider increasing investment in proven campaigns "
                        "while monitoring incremental revenue and margin."
                    ),
                    "Medium",
                )
            )

    return insights, recommendations


# ---------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------

def _priority_score(
    priority: str,
) -> int:
    """Convert priority to sortable score."""

    scores = {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    return scores.get(
        priority,
        0,
    )


def _sort_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort insights/recommendations by priority."""

    return sorted(
        items,
        key=lambda item: _priority_score(
            str(
                item.get(
                    "priority",
                    "Low",
                )
            )
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------

def _build_executive_summary(
    insights: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact executive-level summary."""

    critical = sum(
        1
        for item in insights
        if item.get("priority") == "Critical"
    )

    high = sum(
        1
        for item in insights
        if item.get("priority") == "High"
    )

    if critical:
        status = "Immediate Attention"

    elif high >= 2:
        status = "Needs Attention"

    else:
        status = "Stable"

    top_priorities = [
        item
        for item in recommendations
        if item.get("priority")
        in {
            "Critical",
            "High",
        }
    ][:5]

    return {
        "business_status": status,
        "critical_insights": critical,
        "high_priority_insights": high,
        "total_insights": len(insights),
        "total_recommendations": len(
            recommendations
        ),
        "top_priorities": top_priorities,
    }


# ---------------------------------------------------------------------
# Insight tables
# ---------------------------------------------------------------------

def _insights_dataframe(
    insights: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert insights to a display-ready DataFrame."""

    if not insights:
        return pd.DataFrame(
            columns=[
                "category",
                "title",
                "message",
                "priority",
                "metric",
                "metric_label",
            ]
        )

    return pd.DataFrame(
        insights
    )


def _recommendations_dataframe(
    recommendations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert recommendations to a display-ready DataFrame."""

    if not recommendations:
        return pd.DataFrame(
            columns=[
                "category",
                "title",
                "action",
                "priority",
                "reason",
            ]
        )

    return pd.DataFrame(
        recommendations
    )


# ---------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------

def generate_insights(
    analytics_results: Optional[
        dict[str, Any]
    ] = None,
) -> dict[str, Any]:
    """
    Generate DataPulse business intelligence.

    Parameters
    ----------
    analytics_results:
        Dictionary containing outputs from analytics and ML modules.

    Returns
    -------
    dict
        Standardized insights and recommendations.
    """

    if not isinstance(
        analytics_results,
        dict,
    ):
        analytics_results = {}

    all_insights: list[
        dict[str, Any]
    ] = []

    all_recommendations: list[
        dict[str, Any]
    ] = []

    engines = [
        _core_insights,
        _customer_insights,
        _product_insights,
        _campaign_insights,
        _market_insights,
        _delivery_insights,
        _pricing_insights,
        _review_insights,
        _churn_insights,
        _segmentation_insights,
        _forecast_insights,
        _anomaly_insights,
        _cross_module_insights,
    ]

    for engine in engines:

        try:
            insights, recommendations = engine(
                analytics_results
            )

            all_insights.extend(
                insights
            )

            all_recommendations.extend(
                recommendations
            )

        except Exception:
            # One insight engine must never break the complete
            # DataPulse analytics pipeline.
            continue

    # ---------------------------------------------------------------
    # Sort and de-duplicate
    # ---------------------------------------------------------------

    all_insights = _sort_items(
        all_insights
    )

    all_recommendations = _sort_items(
        all_recommendations
    )

    seen_insights = set()
    unique_insights = []

    for insight in all_insights:

        key = (
            insight.get("category"),
            insight.get("title"),
        )

        if key in seen_insights:
            continue

        seen_insights.add(key)

        unique_insights.append(
            insight
        )

    seen_recommendations = set()
    unique_recommendations = []

    for recommendation in all_recommendations:

        key = (
            recommendation.get(
                "category"
            ),
            recommendation.get(
                "title"
            ),
        )

        if key in seen_recommendations:
            continue

        seen_recommendations.add(key)

        unique_recommendations.append(
            recommendation
        )

    insights = unique_insights[
        :MAX_INSIGHTS
    ]

    recommendations = unique_recommendations[
        :MAX_RECOMMENDATIONS
    ]

    # ---------------------------------------------------------------
    # Executive summary
    # ---------------------------------------------------------------

    executive_summary = (
        _build_executive_summary(
            insights,
            recommendations,
        )
    )

    # ---------------------------------------------------------------
    # Module availability
    # ---------------------------------------------------------------

    module_names = [
        "core",
        "customer",
        "product",
        "campaign_impact",
        "market",
        "delivery",
        "pricing",
        "reviews",
        "churn",
        "segmentation",
        "forecast",
        "anomaly",
    ]

    module_status = {
        module: _is_available(
            analytics_results,
            module,
        )
        for module in module_names
    }

    available_modules = sum(
        module_status.values()
    )

    # ---------------------------------------------------------------
    # Return stable result
    # ---------------------------------------------------------------

    return {
        "available": bool(
            insights
            or recommendations
        ),
        "insights": insights,
        "recommendations": recommendations,
        "insights_df": _insights_dataframe(
            insights
        ),
        "recommendations_df": (
            _recommendations_dataframe(
                recommendations
            )
        ),
        "executive_summary": executive_summary,
        "module_status": module_status,
        "available_module_count": (
            available_modules
        ),
        "total_module_count": len(
            module_names
        ),
        "warnings": [],
    }


# ---------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------

def get_top_insights(
    result: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the highest-priority insights."""

    insights = result.get(
        "insights",
        [],
    )

    if not isinstance(
        insights,
        list,
    ):
        return []

    return insights[
        :max(
            0,
            limit,
        )
    ]


def get_top_recommendations(
    result: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the highest-priority recommendations."""

    recommendations = result.get(
        "recommendations",
        [],
    )

    if not isinstance(
        recommendations,
        list,
    ):
        return []

    return recommendations[
        :max(
            0,
            limit,
        )
    ]


def get_executive_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return the executive summary."""

    return result.get(
        "executive_summary",
        {},
    )


# Backward-compatible aliases.
run_insight_engine = generate_insights
build_insights = generate_insights