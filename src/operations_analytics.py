"""
DataPulse - Delivery Intelligence

Purpose
-------
Analyzes order fulfillment and delivery performance.

Covers:
- Delivery status
- On-time vs late delivery
- Delivery duration
- Shipping duration
- Promised vs actual delivery
- Shipping method performance
- Return analysis
- Delivery performance by region
- Delivery performance by seller
- Operational recommendations

Stable interface
----------------
run_delivery_analytics(df, mapping) -> dict
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
    """Return source column mapped to a canonical field."""

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


def _safe_percentage(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator * 100)


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
    Create an internal standardized dataframe.

    Original dataframe is never modified.
    """

    fields = [
        "order_id",
        "order_date",
        "order_status",
        "sales_amount",
        "quantity",
        "seller_id",
        "seller_name",
        "region",
        "market",
        "shipping_date",
        "delivery_date",
        "promised_delivery_date",
        "delivery_status",
        "shipping_method",
        "return_flag",
        "return_date",
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

    # Numeric
    for field in [
        "sales_amount",
        "quantity",
    ]:
        if field in working.columns:
            working[field] = _safe_numeric(
                working[field]
            )

    # Text
    for field in [
        "order_id",
        "order_status",
        "seller_id",
        "seller_name",
        "region",
        "market",
        "delivery_status",
        "shipping_method",
    ]:
        if field in working.columns:
            working[field] = _safe_text(
                working[field]
            )

    # Dates
    for field in [
        "order_date",
        "shipping_date",
        "delivery_date",
        "promised_delivery_date",
        "return_date",
    ]:
        if field in working.columns:
            working[field] = pd.to_datetime(
                working[field],
                errors="coerce",
            )

    # Return flag normalization
    if "return_flag" in working.columns:
        raw = working["return_flag"]

        if pd.api.types.is_bool_dtype(raw):
            working["return_flag"] = raw.fillna(False)

        else:
            normalized = (
                raw.astype("string")
                .str.strip()
                .str.lower()
            )

            working["return_flag"] = normalized.isin(
                {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "returned",
                    "return",
                }
            )

    return working


# ---------------------------------------------------------------------
# Delivery duration
# ---------------------------------------------------------------------

def _calculate_delivery_metrics(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate shipping and delivery durations."""

    result = working.copy()

    if (
        "shipping_date" in result.columns
        and "delivery_date" in result.columns
    ):
        result["shipping_days"] = (
            result["shipping_date"]
            - result["order_date"]
        ).dt.total_seconds() / 86400

        result["delivery_days"] = (
            result["delivery_date"]
            - result["shipping_date"]
        ).dt.total_seconds() / 86400

        result["total_delivery_days"] = (
            result["delivery_date"]
            - result["order_date"]
        ).dt.total_seconds() / 86400

    elif (
        "order_date" in result.columns
        and "delivery_date" in result.columns
    ):
        result["total_delivery_days"] = (
            result["delivery_date"]
            - result["order_date"]
        ).dt.total_seconds() / 86400

    if (
        "delivery_date" in result.columns
        and "promised_delivery_date" in result.columns
    ):
        result["delivery_delay_days"] = (
            result["delivery_date"]
            - result["promised_delivery_date"]
        ).dt.total_seconds() / 86400

        result["is_late"] = (
            result["delivery_delay_days"] > 0
        )

        result["is_on_time"] = (
            result["delivery_delay_days"] <= 0
        )

    return result


# ---------------------------------------------------------------------
# Overall delivery performance
# ---------------------------------------------------------------------

def _build_delivery_summary(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """Build overall delivery KPIs."""

    total_rows = len(working)

    if "order_id" in working.columns:
        total_orders = working["order_id"].nunique()
    else:
        total_orders = total_rows

    delivered_orders = total_orders

    if "delivery_date" in working.columns:
        delivered_orders = (
            working["delivery_date"]
            .notna()
            .sum()
        )

    if "delivery_status" in working.columns:
        status = (
            working["delivery_status"]
            .astype("string")
            .str.lower()
        )

        delivered_mask = status.str.contains(
            "deliver",
            na=False,
        )

        if delivered_mask.any():
            delivered_orders = (
                working.loc[delivered_mask, "order_id"]
                .nunique()
                if "order_id" in working.columns
                else int(delivered_mask.sum())
            )

    on_time_orders = 0
    late_orders = 0

    if "is_on_time" in working.columns:
        valid = working["is_on_time"].notna()

        on_time_orders = int(
            working.loc[valid, "is_on_time"].sum()
        )

        late_orders = int(
            valid.sum() - on_time_orders
        )

    delivery_days = (
        working["total_delivery_days"]
        if "total_delivery_days" in working.columns
        else pd.Series(dtype=float)
    )

    valid_delivery_days = delivery_days.dropna()

    average_delivery_days = (
        float(valid_delivery_days.mean())
        if not valid_delivery_days.empty
        else None
    )

    median_delivery_days = (
        float(valid_delivery_days.median())
        if not valid_delivery_days.empty
        else 0.0
    )

    average_delay_days = 0.0

    if "delivery_delay_days" in working.columns:
        delays = working["delivery_delay_days"].dropna()

        if not delays.empty:
            average_delay_days = float(
                delays.mean()
            )

    on_time_rate = _safe_percentage(
        on_time_orders,
        on_time_orders + late_orders,
    )

    late_rate = _safe_percentage(
        late_orders,
        on_time_orders + late_orders,
    )

    returned_orders = 0

    if "return_flag" in working.columns:
        returned_orders = int(
            working["return_flag"].sum()
        )

    return {
        "total_rows": int(total_rows),
        "total_orders": int(total_orders),
        "delivered_orders": int(delivered_orders),
        "on_time_orders": int(on_time_orders),
        "late_orders": int(late_orders),
        "on_time_rate": round(on_time_rate, 2),
        "late_rate": round(late_rate, 2),
        "delivery_rate_pct": round(
            _safe_percentage(
                delivered_orders,
                total_orders,
            ),
            2,
        ),
        "average_delivery_days": (
            round(average_delivery_days, 2)
            if average_delivery_days is not None
            else None
        ),
        "median_delivery_days": (
            round(median_delivery_days, 2)
            if not valid_delivery_days.empty
            else None
        ),
        "average_delay_days": round(
            average_delay_days,
            2,
        ),
        "returned_orders": int(returned_orders),
        "return_rate": round(
            _safe_percentage(
                returned_orders,
                total_orders,
            ),
            2,
        ),
    }


# ---------------------------------------------------------------------
# Delivery status
# ---------------------------------------------------------------------

def _build_delivery_status(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze delivery statuses."""

    if "delivery_status" not in working.columns:
        return _empty(
            [
                "delivery_status",
                "orders",
                "revenue",
                "share_pct",
            ]
        )

    group_field = "order_id" if "order_id" in working.columns else None

    if group_field:
        result = (
            working.groupby("delivery_status")
            .agg(
                orders=("order_id", "nunique"),
                revenue=("sales_amount", "sum"),
            )
            .reset_index()
        )
    else:
        result = (
            working.groupby("delivery_status")
            .agg(
                orders=("sales_amount", "size"),
                revenue=("sales_amount", "sum"),
            )
            .reset_index()
        )

    total_orders = result["orders"].sum()

    result["share_pct"] = (
        result["orders"]
        / total_orders
        * 100
        if total_orders
        else 0
    )

    return result.sort_values(
        "orders",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Shipping method
# ---------------------------------------------------------------------

def _build_shipping_method_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Compare shipping methods."""

    if "shipping_method" not in working.columns:
        return _empty(
            [
                "shipping_method",
                "orders",
                "revenue",
                "average_delivery_days",
                "on_time_rate",
                "late_rate",
            ]
        )

    rows = []

    for method, group in working.groupby(
        "shipping_method",
        dropna=False,
    ):
        orders = (
            group["order_id"].nunique()
            if "order_id" in group.columns
            else len(group)
        )

        revenue = (
            group["sales_amount"].sum()
            if "sales_amount" in group.columns
            else 0
        )

        if "total_delivery_days" in group.columns:
            delivery_days = (
                group["total_delivery_days"]
                .dropna()
            )

            average_delivery_days = (
                float(delivery_days.mean())
                if not delivery_days.empty
                else 0.0
            )
        else:
            average_delivery_days = 0.0

        if "is_on_time" in group.columns:
            valid = group["is_on_time"].dropna()

            on_time_rate = (
                float(valid.mean() * 100)
                if not valid.empty
                else 0.0
            )
        else:
            on_time_rate = 0.0

        rows.append(
            {
                "shipping_method": method,
                "orders": int(orders),
                "revenue": float(revenue),
                "average_delivery_days": round(
                    average_delivery_days,
                    2,
                ),
                "on_time_rate": round(
                    on_time_rate,
                    2,
                ),
                "late_rate": round(
                    100 - on_time_rate,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "orders",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Region performance
# ---------------------------------------------------------------------

def _build_region_delivery_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze delivery performance by region."""

    if "region" not in working.columns:
        return _empty(
            [
                "region",
                "orders",
                "revenue",
                "average_delivery_days",
                "on_time_rate",
                "late_rate",
                "return_rate",
            ]
        )

    rows = []

    for region, group in working.groupby(
        "region",
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

        if "total_delivery_days" in group.columns:
            days = (
                group["total_delivery_days"]
                .dropna()
            )

            average_days = (
                float(days.mean())
                if not days.empty
                else 0.0
            )
        else:
            average_days = 0.0

        if "is_on_time" in group.columns:
            valid = group["is_on_time"].dropna()

            on_time_rate = (
                float(valid.mean() * 100)
                if not valid.empty
                else 0.0
            )
        else:
            on_time_rate = 0.0

        if "return_flag" in group.columns:
            return_rate = (
                float(
                    group["return_flag"].mean()
                    * 100
                )
            )
        else:
            return_rate = 0.0

        rows.append(
            {
                "region": region,
                "orders": int(orders),
                "revenue": revenue,
                "average_delivery_days": round(
                    average_days,
                    2,
                ),
                "on_time_rate": round(
                    on_time_rate,
                    2,
                ),
                "late_rate": round(
                    100 - on_time_rate,
                    2,
                ),
                "return_rate": round(
                    return_rate,
                    2,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    median_on_time = result["on_time_rate"].median()
    median_return = result["return_rate"].median()

    def classify(row: pd.Series) -> str:
        if (
            row["on_time_rate"] >= median_on_time
            and row["return_rate"] <= median_return
        ):
            return "Strong Operations"

        if row["on_time_rate"] < median_on_time:
            return "Delivery Risk"

        if row["return_rate"] > median_return:
            return "Return Risk"

        return "Monitor"

    result["decision_class"] = result.apply(
        classify,
        axis=1,
    )

    return result.sort_values(
        "orders",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Seller delivery performance
# ---------------------------------------------------------------------

def _build_seller_delivery_performance(
    working: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze delivery performance by seller."""

    seller_field = None

    if "seller_name" in working.columns:
        seller_field = "seller_name"
    elif "seller_id" in working.columns:
        seller_field = "seller_id"

    if seller_field is None:
        return _empty(
            [
                "seller",
                "orders",
                "revenue",
                "average_delivery_days",
                "on_time_rate",
                "late_rate",
                "return_rate",
            ]
        )

    rows = []

    for seller, group in working.groupby(
        seller_field,
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

        if "total_delivery_days" in group.columns:
            days = (
                group["total_delivery_days"]
                .dropna()
            )

            average_days = (
                float(days.mean())
                if not days.empty
                else 0.0
            )
        else:
            average_days = 0.0

        if "is_on_time" in group.columns:
            valid = group["is_on_time"].dropna()

            on_time_rate = (
                float(valid.mean() * 100)
                if not valid.empty
                else 0.0
            )
        else:
            on_time_rate = 0.0

        if "return_flag" in group.columns:
            return_rate = float(
                group["return_flag"].mean()
                * 100
            )
        else:
            return_rate = 0.0

        rows.append(
            {
                "seller": seller,
                "orders": int(orders),
                "revenue": revenue,
                "average_delivery_days": round(
                    average_days,
                    2,
                ),
                "on_time_rate": round(
                    on_time_rate,
                    2,
                ),
                "late_rate": round(
                    100 - on_time_rate,
                    2,
                ),
                "return_rate": round(
                    return_rate,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "orders",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Return analysis
# ---------------------------------------------------------------------

def _build_return_analysis(
    working: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze returns."""

    if "return_flag" not in working.columns:
        return {
            "available": False,
            "returned_orders": 0,
            "return_rate": 0.0,
            "returned_revenue": 0.0,
            "non_returned_revenue": 0.0,
        }

    returned = working["return_flag"]

    total_orders = (
        working["order_id"].nunique()
        if "order_id" in working.columns
        else len(working)
    )

    returned_orders = int(
        returned.sum()
    )

    returned_revenue = float(
        working.loc[
            returned,
            "sales_amount",
        ].sum()
    )

    non_returned_revenue = float(
        working.loc[
            ~returned,
            "sales_amount",
        ].sum()
    )

    return {
        "available": True,
        "returned_orders": returned_orders,
        "return_rate": round(
            _safe_percentage(
                returned_orders,
                total_orders,
            ),
            2,
        ),
        "returned_revenue": returned_revenue,
        "non_returned_revenue": non_returned_revenue,
    }


# ---------------------------------------------------------------------
# Operational alerts
# ---------------------------------------------------------------------

def _build_operational_alerts(
    summary: Dict[str, Any],
    region_performance: pd.DataFrame,
    seller_performance: pd.DataFrame,
) -> list[Dict[str, Any]]:
    """Generate actionable operational alerts."""

    alerts = []

    late_rate = summary.get(
        "late_rate",
        0,
    )

    if late_rate >= 20:
        alerts.append(
            {
                "priority": "High",
                "type": "Late Delivery",
                "message": (
                    f"Late delivery rate is "
                    f"{late_rate:.1f}%. Review fulfillment "
                    "and logistics performance."
                ),
            }
        )
    elif late_rate >= 10:
        alerts.append(
            {
                "priority": "Medium",
                "type": "Late Delivery",
                "message": (
                    f"Late delivery rate is "
                    f"{late_rate:.1f}%. Monitor delivery "
                    "performance closely."
                ),
            }
        )

    return_rate = summary.get(
        "return_rate",
        0,
    )

    if return_rate >= 10:
        alerts.append(
            {
                "priority": "High",
                "type": "Returns",
                "message": (
                    f"Return rate is "
                    f"{return_rate:.1f}%. Investigate "
                    "product, seller and fulfillment drivers."
                ),
            }
        )

    if not region_performance.empty:
        risky_regions = region_performance[
            region_performance["decision_class"]
            == "Delivery Risk"
        ]

        if not risky_regions.empty:
            alerts.append(
                {
                    "priority": "High",
                    "type": "Regional Delivery Risk",
                    "message": (
                        f"{len(risky_regions)} region(s) show "
                        "below-median on-time delivery performance."
                    ),
                }
            )

    if not seller_performance.empty:
        risky_sellers = seller_performance[
            seller_performance["on_time_rate"] < 80
        ]

        if not risky_sellers.empty:
            alerts.append(
                {
                    "priority": "Medium",
                    "type": "Seller Fulfillment Risk",
                    "message": (
                        f"{len(risky_sellers)} seller(s) have "
                        "on-time delivery below 80%."
                    ),
                }
            )

    return alerts


# ---------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------

def _build_capabilities(
    working: pd.DataFrame,
) -> Dict[str, bool]:
    """Return available delivery analytics."""

    return {
        "delivery_analysis": (
            "delivery_date" in working.columns
            or "delivery_status" in working.columns
        ),
        "on_time_analysis": (
            "delivery_date" in working.columns
            and "promised_delivery_date" in working.columns
        ),
        "shipping_method_analysis": (
            "shipping_method" in working.columns
        ),
        "regional_delivery_analysis": (
            "region" in working.columns
        ),
        "seller_delivery_analysis": (
            "seller_name" in working.columns
            or "seller_id" in working.columns
        ),
        "return_analysis": (
            "return_flag" in working.columns
        ),
        "delivery_duration_analysis": (
            "order_date" in working.columns
            and "delivery_date" in working.columns
        ),
    }


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_delivery_analytics(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run Delivery Intelligence.

    Parameters
    ----------
    df:
        Analysis-ready dataframe.

    mapping:
        Schema mapping generated by schema_mapper.py.

    Returns
    -------
    dict
        Stable Delivery Intelligence result contract.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return {
            "available": False,
            "summary": {},
            "delivery_status": pd.DataFrame(),
            "shipping_method_performance": pd.DataFrame(),
            "region_performance": pd.DataFrame(),
            "seller_performance": pd.DataFrame(),
            "return_analysis": {},
            "operational_alerts": [],
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

    if not capabilities["delivery_analysis"]:
        return {
            "available": False,
            "summary": {},
            "delivery_status": pd.DataFrame(),
            "shipping_method_performance": pd.DataFrame(),
            "region_performance": pd.DataFrame(),
            "seller_performance": pd.DataFrame(),
            "return_analysis": {},
            "operational_alerts": [],
            "capabilities": capabilities,
            "warnings": [
                "Delivery fields are unavailable in the dataset."
            ],
        }

    working = _calculate_delivery_metrics(
        working
    )

    summary = _build_delivery_summary(
        working
    )

    delivery_status = _build_delivery_status(
        working
    )

    shipping_method_performance = (
        _build_shipping_method_performance(
            working
        )
    )

    region_performance = (
        _build_region_delivery_performance(
            working
        )
    )

    seller_performance = (
        _build_seller_delivery_performance(
            working
        )
    )

    return_analysis = _build_return_analysis(
        working
    )

    operational_alerts = _build_operational_alerts(
        summary,
        region_performance,
        seller_performance,
    )

    warnings = []

    if not capabilities["on_time_analysis"]:
        warnings.append(
            "Promised delivery date is unavailable; "
            "on-time/late classification is limited."
        )

    if not capabilities["shipping_method_analysis"]:
        warnings.append(
            "Shipping method is unavailable; "
            "shipping-method comparison is limited."
        )

    if not capabilities["return_analysis"]:
        warnings.append(
            "Return flag is unavailable; "
            "return analysis cannot be performed."
        )

    return {
        "available": True,
        "summary": summary,
        "delivery_status": delivery_status,
        "shipping_method_performance": shipping_method_performance,
        "region_performance": region_performance,
        "seller_performance": seller_performance,
        "return_analysis": return_analysis,
        "operational_alerts": operational_alerts,
        "capabilities": capabilities,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------

run_operations_analytics = run_delivery_analytics