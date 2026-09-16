"""
DataPulse - ML Intelligence
---------------------------
Streamlit page for DataPulse machine-learning intelligence.

Modules:
    - Churn Prediction
    - Customer Segmentation
    - Sales Forecasting
    - Anomaly Detection
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="DataPulse | ML Intelligence",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.html(
    """
    <style>

    .ml-hero {
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        background:
            linear-gradient(
                135deg,
                #0F172A 0%,
                #172554 55%,
                #1E3A8A 100%
            );
        color: white;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .ml-hero h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .ml-hero p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 15px;
        color: #CBD5E1;
    }

    .ml-card {
        box-sizing: border-box;
        width: 100%;
        height: 145px;
        min-height: 145px;
        padding: 20px;
        border-radius: 16px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 14px rgba(15,23,42,0.04);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .ml-card-title {
        font-size: 14px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 10px;
    }

    .ml-card-value {
        font-size: 28px;
        font-weight: 750;
        color: #0F172A;
        line-height: 1.15;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: 0;
        margin-top: auto;
    }

    .ml-card-subtitle {
        font-size: 12px;
        color: #64748B;
        margin-top: 6px;
        min-height: 16px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #0F172A;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .insight-box {
        padding: 15px 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        background: #F8FAFC;
        margin-bottom: 10px;
    }

    .insight-title {
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 4px;
    }

    .insight-text {
        color: #475569;
        font-size: 13px;
        line-height: 1.5;
    }

    .priority {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 999px;
        margin-bottom: 7px;
        background: #E2E8F0;
        color: #334155;
    }

    .model-status {
        box-sizing: border-box;
        width: 100%;
        height: 114px;
        min-height: 114px;
        max-height: 114px;
        padding: 16px 18px;
        border-radius: 12px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 8px;
        overflow: hidden;
    }

    .model-status strong {
        display: block;
        color: #0F172A;
        font-size: 17px;
        font-weight: 750;
        line-height: 1.35;
        margin: 0;
        min-width: 0;
        overflow: hidden;
    }

    .model-status span {
        display: block;
        color: #64748B;
        font-size: 16px;
        line-height: 1.25;
        margin: 0;
    }

    @media (max-width: 768px) {
        .main .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
        [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0 !important; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
        .ml-hero { padding: 20px 18px; }
        .ml-hero h1 { font-size: 25px; line-height: 1.12; overflow-wrap: anywhere; }
        .ml-hero p, .section-title, .insight-title, .insight-text, .model-status strong, .model-status span { overflow-wrap: anywhere; }
        .ml-card { height: auto; min-height: 118px; padding: 16px; }
        .ml-card-value, .ml-card-subtitle { white-space: normal; overflow-wrap: anywhere; }
        .ml-card-value { font-size: 23px; }
        .model-status { height: auto; min-height: 100px; max-height: none; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { max-width: 100% !important; overflow-x: auto !important; }
        button { min-height: 44px; }
    }

    </style>
    """,

)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_result(
    name: str,
) -> dict[str, Any]:
    """Safely retrieve an ML result from session state."""

    results = st.session_state.get(
        "analytics_results",
        {},
    )

    if not isinstance(
        results,
        dict,
    ):
        return {}

    result = results.get(
        name,
        {},
    )

    return result if isinstance(
        result,
        dict,
    ) else {}


def _format_number(
    value: Any,
) -> str:
    """Format a numeric value for display."""

    try:
        number = float(value)

        if abs(number) >= 1_000_000:
            return f"{number / 1_000_000:.2f}M"

        if abs(number) >= 1_000:
            return f"{number / 1_000:.1f}K"

        return f"{number:,.0f}"

    except (
        TypeError,
        ValueError,
    ):
        return "—"


def _format_currency(
    value: Any,
) -> str:
    """Format currency-like numeric values."""

    try:
        number = float(value)

        if abs(number) >= 1_000_000:
            return f"₹{number / 1_000_000:.2f}M"

        if abs(number) >= 1_000:
            return f"₹{number / 1_000:.1f}K"

        return f"₹{number:,.0f}"

    except (
        TypeError,
        ValueError,
    ):
        return "—"


def _show_empty_state(
    message: str,
) -> None:
    """Display an ML unavailable message."""

    st.info(
        message
    )


def _priority_badge(
    priority: str,
) -> str:
    """Return a simple priority label."""

    return (
        f'<span class="priority">{priority}</span>'
    )


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------

st.html(
    """
    <div class="ml-hero">
        <h1>Machine Learning Intelligence</h1>
        <p>
            Predict customer behavior, discover customer segments,
            forecast sales and detect unusual business activity.
        </p>
    </div>
    """,

)


# ---------------------------------------------------------------------
# Dataset status
# ---------------------------------------------------------------------

metadata = st.session_state.get(
    "dataset_metadata"
)

if metadata is None:
    st.warning(
        "Upload and process an analysis-ready dataset from the DataPulse "
        "workspace before using ML Intelligence."
    )

    st.stop()


st.caption(
    f"Dataset: {metadata.file_name} · "
    f"{metadata.rows:,} rows · "
    f"{metadata.columns:,} columns"
)


# ---------------------------------------------------------------------
# Retrieve models
# ---------------------------------------------------------------------

churn = _get_result(
    "churn"
)

segmentation = _get_result(
    "segmentation"
)

forecast = _get_result(
    "forecast"
)

anomaly = _get_result(
    "anomaly"
)


# ---------------------------------------------------------------------
# ML overview cards
# ---------------------------------------------------------------------

st.html(
    '<div class="section-title">ML Overview</div>',

)

overview_cols = st.columns(
    4
)


# Churn
with overview_cols[0]:

    risk_summary = churn.get(
        "risk_summary",
        {},
    )

    high_risk = risk_summary.get(
        "high_risk_customers",
        0,
    )

    st.html(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">
                Churn Risk
            </div>
            <div class="ml-card-value">
                {_format_number(high_risk)}
            </div>
            <div class="ml-card-subtitle">
                High-risk customers
            </div>
        </div>
        """,

    )


# Segmentation
with overview_cols[1]:

    segmentation_metrics = segmentation.get(
        "metrics",
        {},
    )

    cluster_count = segmentation_metrics.get(
        "cluster_count",
        0,
    )

    st.html(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">
                Segmentation
            </div>
            <div class="ml-card-value">
                {_format_number(cluster_count)}
            </div>
            <div class="ml-card-subtitle">
                Customer segments
            </div>
        </div>
        """,

    )


# Forecast
with overview_cols[2]:

    forecast_trend = forecast.get(
        "trend",
        {},
    )

    direction = forecast_trend.get(
        "forecast_direction",
        "—",
    )

    st.html(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">
                Sales Forecast
            </div>
            <div class="ml-card-value">
                {direction}
            </div>
            <div class="ml-card-subtitle">
                Projected direction
            </div>
        </div>
        """,

    )


# Anomaly
with overview_cols[3]:

    anomaly_summary = anomaly.get(
        "anomaly_summary",
        {},
    )

    anomaly_count = anomaly_summary.get(
        "anomaly_count",
        0,
    )

    st.html(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">
                Anomalies
            </div>
            <div class="ml-card-value">
                {_format_number(anomaly_count)}
            </div>
            <div class="ml-card-subtitle">
                Unusual observations
            </div>
        </div>
        """,

    )


# ---------------------------------------------------------------------
# Model tabs
# ---------------------------------------------------------------------

st.html(
    '<div class="section-title">ML Models</div>',

)

tabs = st.tabs(
    [
        "Churn Prediction",
        "Customer Segmentation",
        "Sales Forecasting",
        "Anomaly Detection",
    ]
)


# =====================================================================
# CHURN
# =====================================================================

with tabs[0]:

    st.subheader(
        "Customer Churn Prediction"
    )

    if not churn.get(
        "available",
        False,
    ):
        _show_empty_state(
            "Churn prediction is unavailable for this dataset. "
            "DataPulse needs sufficient customer-level purchase history."
        )

    else:

        churn_metrics = churn.get(
            "metrics",
            {},
        )

        risk_summary = churn.get(
            "risk_summary",
            {},
        )

        cols = st.columns(
            4
        )

        with cols[0]:
            st.metric(
                "Customers",
                _format_number(
                    churn_metrics.get(
                        "customer_count"
                    )
                ),
            )

        with cols[1]:
            st.metric(
                "High Risk",
                _format_number(
                    risk_summary.get(
                        "high_risk_customers"
                    )
                ),
            )

        with cols[2]:
            st.metric(
                "Medium Risk",
                _format_number(
                    risk_summary.get(
                        "medium_risk_customers"
                    )
                ),
            )

        with cols[3]:
            auc = churn_metrics.get(
                "roc_auc"
            )

            st.metric(
                "ROC-AUC",
                (
                    f"{float(auc):.2f}"
                    if auc is not None
                    else "—"
                ),
            )

        st.html(
            '<div class="section-title">High-Value Customers at Risk</div>',

        )

        high_value_risk = churn.get(
            "high_value_risk",
            pd.DataFrame(),
        )

        if (
            isinstance(
                high_value_risk,
                pd.DataFrame,
            )
            and not high_value_risk.empty
        ):

            display_columns = [
                column
                for column in [
                    "customer_id",
                    "churn_probability",
                    "churn_risk",
                    "revenue",
                    "orders",
                    "recency_days",
                    "profit",
                    "average_order_value",
                    "business_action",
                ]
                if column in high_value_risk.columns
            ]

            st.dataframe(
                high_value_risk[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No high-value customers were flagged in the current model output."
            )

        st.html(
            '<div class="section-title">Feature Importance</div>',

        )

        importance = churn.get(
            "feature_importance",
            pd.DataFrame(),
        )

        if (
            isinstance(
                importance,
                pd.DataFrame,
            )
            and not importance.empty
        ):

            st.bar_chart(
                importance.set_index(
                    "feature"
                )[
                    "importance"
                ]
            )

        recommendations = churn.get(
            "recommendations",
            [],
        )

        if recommendations:

            st.html(
                '<div class="section-title">Model Recommendations</div>',

            )

            for recommendation in recommendations:

                if isinstance(
                    recommendation,
                    dict,
                ):

                    priority = recommendation.get(
                        "priority",
                        "Medium",
                    )

                    title = recommendation.get(
                        "title",
                        "Recommendation",
                    )

                    action = recommendation.get(
                        "recommendation",
                        recommendation.get(
                            "action",
                            "",
                        ),
                    )

                    st.html(
                        f"""
                        <div class="insight-box">
                            {_priority_badge(priority)}
                            <div class="insight-title">
                                {title}
                            </div>
                            <div class="insight-text">
                                {action}
                            </div>
                        </div>
                        """,

                    )


# =====================================================================
# SEGMENTATION
# =====================================================================

with tabs[1]:

    st.subheader(
        "Customer Segmentation"
    )

    if not segmentation.get(
        "available",
        False,
    ):
        _show_empty_state(
            "Customer segmentation is unavailable. "
            "DataPulse needs enough customer-level behavioral data."
        )

    else:

        metrics = segmentation.get(
            "metrics",
            {},
        )

        profiles = segmentation.get(
            "segment_profiles",
            pd.DataFrame(),
        )

        assignments = segmentation.get(
            "customer_assignments",
            pd.DataFrame(),
        )

        cols = st.columns(
            3
        )

        with cols[0]:
            st.metric(
                "Segments",
                _format_number(
                    metrics.get(
                        "cluster_count"
                    )
                ),
            )

        with cols[1]:
            silhouette = metrics.get(
                "silhouette_score"
            )

            st.metric(
                "Silhouette Score",
                (
                    f"{float(silhouette):.2f}"
                    if silhouette is not None
                    else "—"
                ),
            )

        with cols[2]:
            st.metric(
                "Customers",
                _format_number(
                    metrics.get(
                        "customer_count"
                    )
                ),
            )

        st.html(
            '<div class="section-title">Segment Profiles</div>',

        )

        if (
            isinstance(
                profiles,
                pd.DataFrame,
            )
            and not profiles.empty
        ):

            profile_columns = [
                column
                for column in [
                    "segment_id",
                    "segment_name",
                    "customer_count",
                    "customer_share_pct",
                    "revenue",
                    "orders",
                    "profit",
                    "average_order_value",
                    "recency_days",
                    "profit_margin",
                ]
                if column in profiles.columns
            ]

            st.dataframe(
                profiles[
                    profile_columns
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.html(
            '<div class="section-title">Customer Distribution</div>',

        )

        distribution = segmentation.get(
            "segment_distribution",
            pd.DataFrame(),
        )

        if (
            isinstance(
                distribution,
                pd.DataFrame,
            )
            and not distribution.empty
        ):

            chart_data = distribution[
                [
                    "segment_name",
                    "customer_count",
                ]
            ].copy()

            chart_data = chart_data.set_index(
                "segment_name"
            )

            st.bar_chart(
                chart_data
            )

        st.html(
            '<div class="section-title">Customer Assignments</div>',

        )

        if (
            isinstance(
                assignments,
                pd.DataFrame,
            )
            and not assignments.empty
        ):

            st.dataframe(
                assignments,
                use_container_width=True,
                hide_index=True,
            )

        recommendations = segmentation.get(
            "recommendations",
            [],
        )

        if recommendations:

            st.html(
                '<div class="section-title">Segment Actions</div>',

            )

            for recommendation in recommendations:

                if not isinstance(
                    recommendation,
                    dict,
                ):
                    continue

                priority = recommendation.get(
                    "priority",
                    "Medium",
                )

                segment_name = recommendation.get(
                    "segment_name",
                    "Customer Segment",
                )

                action = recommendation.get(
                    "recommendation",
                    "",
                )

                st.html(
                    f"""
                    <div class="insight-box">
                        {_priority_badge(priority)}
                        <div class="insight-title">
                            {segment_name}
                        </div>
                        <div class="insight-text">
                            {action}
                        </div>
                    </div>
                    """,

                )


# =====================================================================
# FORECASTING
# =====================================================================

with tabs[2]:

    st.subheader(
        "Sales Forecasting"
    )

    if not forecast.get(
        "available",
        False,
    ):
        _show_empty_state(
            "Sales forecasting is unavailable. "
            "DataPulse requires sufficient historical sales periods."
        )

    else:

        metrics = forecast.get(
            "metrics",
            {},
        )

        trend = forecast.get(
            "trend",
            {},
        )

        forecast_data = forecast.get(
            "forecast",
            pd.DataFrame(),
        )

        cols = st.columns(
            4
        )

        with cols[0]:
            st.metric(
                "Model",
                metrics.get(
                    "selected_model",
                    "—",
                ),
            )

        with cols[1]:
            st.metric(
                "Frequency",
                metrics.get(
                    "frequency",
                    "—",
                ).title(),
            )

        with cols[2]:
            mape = metrics.get(
                "mape"
            )

            st.metric(
                "MAPE",
                (
                    f"{float(mape):.1f}%"
                    if mape is not None
                    else "—"
                ),
            )

        with cols[3]:
            st.metric(
                "Direction",
                trend.get(
                    "forecast_direction",
                    "—",
                ),
            )

        st.html(
            '<div class="section-title">Historical vs Forecast Revenue</div>',

        )

        historical = forecast.get(
            "historical",
            pd.DataFrame(),
        )

        if (
            isinstance(
                historical,
                pd.DataFrame,
            )
            and not historical.empty
            and isinstance(
                forecast_data,
                pd.DataFrame,
            )
            and not forecast_data.empty
        ):

            historical_chart = historical[
                [
                    "period",
                    "revenue",
                ]
            ].copy()

            historical_chart["period"] = (
                pd.to_datetime(
                    historical_chart["period"]
                )
            )

            historical_chart = (
                historical_chart
                .set_index("period")
                .rename(
                    columns={
                        "revenue": "Historical Revenue"
                    }
                )
            )

            future_chart = forecast_data[
                [
                    "period",
                    "forecast_revenue",
                ]
            ].copy()

            future_chart["period"] = (
                pd.to_datetime(
                    future_chart["period"]
                )
            )

            future_chart = (
                future_chart
                .set_index("period")
                .rename(
                    columns={
                        "forecast_revenue": "Forecast Revenue"
                    }
                )
            )

            chart = historical_chart.join(
                future_chart,
                how="outer",
            )

            st.line_chart(
                chart
            )

        st.html(
            '<div class="section-title">Projected Revenue</div>',

        )

        if (
            isinstance(
                forecast_data,
                pd.DataFrame,
            )
            and not forecast_data.empty
        ):

            display_forecast = forecast_data.copy()

            if "forecast_revenue" in display_forecast.columns:
                display_forecast[
                    "forecast_revenue"
                ] = display_forecast[
                    "forecast_revenue"
                ].map(
                    _format_currency
                )

            st.dataframe(
                display_forecast,
                use_container_width=True,
                hide_index=True,
            )

        comparison = forecast.get(
            "model_comparison",
            pd.DataFrame(),
        )

        if (
            isinstance(
                comparison,
                pd.DataFrame,
            )
            and not comparison.empty
        ):

            st.html(
                '<div class="section-title">Model Comparison</div>',

            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

        recommendations = forecast.get(
            "recommendations",
            [],
        )

        if recommendations:

            st.html(
                '<div class="section-title">Forecast Actions</div>',

            )

            for recommendation in recommendations:

                if not isinstance(
                    recommendation,
                    dict,
                ):
                    continue

                st.html(
                    f"""
                    <div class="insight-box">
                        {_priority_badge(
                            recommendation.get(
                                "priority",
                                "Medium",
                            )
                        )}
                        <div class="insight-title">
                            {recommendation.get(
                                "area",
                                "Forecast",
                            )}
                        </div>
                        <div class="insight-text">
                            {recommendation.get(
                                "recommendation",
                                "",
                            )}
                        </div>
                    </div>
                    """,

                )


# =====================================================================
# ANOMALY DETECTION
# =====================================================================

with tabs[3]:

    st.subheader(
        "Anomaly Detection"
    )

    if not anomaly.get(
        "available",
        False,
    ):
        _show_empty_state(
            "Anomaly detection is unavailable. "
            "DataPulse needs sufficient transaction-level observations."
        )

    else:

        summary = anomaly.get(
            "anomaly_summary",
            {},
        )

        metrics = anomaly.get(
            "metrics",
            {},
        )

        cols = st.columns(
            4
        )

        with cols[0]:
            st.metric(
                "Observations",
                _format_number(
                    summary.get(
                        "total_observations"
                    )
                ),
            )

        with cols[1]:
            st.metric(
                "Anomalies",
                _format_number(
                    summary.get(
                        "anomaly_count"
                    )
                ),
            )

        with cols[2]:
            st.metric(
                "Anomaly Rate",
                (
                    f"{float(summary.get('anomaly_rate_pct', 0)):.1f}%"
                ),
            )

        with cols[3]:
            st.metric(
                "Critical",
                _format_number(
                    summary.get(
                        "critical_count"
                    )
                ),
            )

        st.html(
            '<div class="section-title">Severity Distribution</div>',

        )

        severity = anomaly.get(
            "severity_summary",
            pd.DataFrame(),
        )

        if (
            isinstance(
                severity,
                pd.DataFrame,
            )
            and not severity.empty
        ):

            severity_chart = severity[
                [
                    "anomaly_severity",
                    "observation_count",
                ]
            ].copy()

            severity_chart = (
                severity_chart
                .set_index(
                    "anomaly_severity"
                )
            )

            st.bar_chart(
                severity_chart
            )

        st.html(
            '<div class="section-title">Detected Anomalies</div>',

        )

        anomalies = anomaly.get(
            "anomalies",
            pd.DataFrame(),
        )

        if (
            isinstance(
                anomalies,
                pd.DataFrame,
            )
            and not anomalies.empty
        ):

            st.dataframe(
                anomalies,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No anomalous observations were detected."
            )

        recommendations = anomaly.get(
            "recommendations",
            [],
        )

        if recommendations:

            st.html(
                '<div class="section-title">Anomaly Actions</div>',

            )

            for recommendation in recommendations:

                if not isinstance(
                    recommendation,
                    dict,
                ):
                    continue

                st.html(
                    f"""
                    <div class="insight-box">
                        {_priority_badge(
                            recommendation.get(
                                "priority",
                                "Medium",
                            )
                        )}
                        <div class="insight-title">
                            {recommendation.get(
                                "area",
                                "Anomaly Detection",
                            )}
                        </div>
                        <div class="insight-text">
                            {recommendation.get(
                                "recommendation",
                                "",
                            )}
                        </div>
                    </div>
                    """,

                )


# ---------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------

st.html(
    '<div class="section-title">Model Status</div>',

)

status_columns = st.columns(
    4
)

model_status = [
    (
        "Churn Prediction",
        churn,
    ),
    (
        "Customer Segmentation",
        segmentation,
    ),
    (
        "Sales Forecasting",
        forecast,
    ),
    (
        "Anomaly Detection",
        anomaly,
    ),
]

for column, (
    name,
    result,
) in zip(
    status_columns,
    model_status,
):

    with column:

        available = result.get(
            "available",
            False,
        )

        status = (
            "Ready"
            if available
            else "Unavailable"
        )

        st.html(
            f"""
            <div class="model-status">
                <strong>{name}</strong>
                <span>{status}</span>
            </div>
            """,

        )