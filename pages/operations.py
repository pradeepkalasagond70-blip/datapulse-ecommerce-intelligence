# pages/operations.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Delivery Intelligence | DataPulse",
    page_icon="🚚",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .operations-header {
        padding: 18px 22px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #0F172A 0%,
            #172554 55%,
            #1E40AF 100%
        );
        color: white;
        margin-bottom: 20px;
    }

    .operations-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 750;
    }

    .operations-header p {
        margin: 6px 0 0 0;
        color: #CBD5E1;
        font-size: 14px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .metric-card {
        box-sizing: border-box;
        width: 100%;
        height: 122px;
        min-height: 122px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        position: relative;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
    }

    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: #CBD5E1;
    }

    .metric-revenue {
        background: #F0FDF4;
        border-color: #BBF7D0;
    }

    .metric-revenue::before {
        background: #16A34A;
    }

    .metric-orders {
        background: #EFF6FF;
        border-color: #BFDBFE;
    }

    .metric-orders::before {
        background: #2563EB;
    }

    .metric-customers {
        background: #F5F3FF;
        border-color: #DDD6FE;
    }

    .metric-customers::before {
        background: #7C3AED;
    }

    .metric-profit {
        background: #FFFBEB;
        border-color: #FDE68A;
    }

    .metric-profit::before {
        background: #D97706;
    }

    .metric-aov {
        background: #ECFEFF;
        border-color: #A5F3FC;
    }

    .metric-aov::before {
        background: #0891B2;
    }

    .metric-growth {
        background: #F0FDFA;
        border-color: #99F6E4;
    }

    .metric-growth::before {
        background: #0D9488;
    }

    .metric-risk {
        background: #FEF2F2;
        border-color: #FECACA;
    }

    .metric-risk::before {
        background: #DC2626;
    }

    .metric-label {
        color: #64748B;
        font-size: 10px;
        font-weight: 650;
        line-height: 1.2;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .metric-value {
        color: #0F172A;
        font-size: 25px;
        font-weight: 850;
        line-height: 1.1;
        margin-top: 10px;
        min-width: 0;
        white-space: nowrap;
        overflow: visible;
        text-overflow: clip;
    }

    .metric-sub {
        color: #94A3B8;
        font-size: 10px;
        line-height: 1.25;
        margin-top: auto;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }

    .insight-card {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 13px 15px;
        border-radius: 10px;
        margin-bottom: 9px;
    }

    .insight-title {
        font-weight: 700;
        color: #0F172A;
        font-size: 14px;
    }

    .insight-text {
        color: #475569;
        font-size: 13px;
        margin-top: 4px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def first_existing(data, keys, default=None):
    """Return the first available value from a dictionary."""

    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def fmt_number(value, decimals=0):
    """Format numeric values safely."""

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except Exception:
        return "—"

    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return str(value)


def fmt_currency(value):
    """Format currency values."""

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except Exception:
        return "—"

    try:
        value = float(value)

        if abs(value) >= 10_000_000:
            return f"₹{value / 10_000_000:.2f} Cr"

        if abs(value) >= 100_000:
            return f"₹{value / 100_000:.2f} L"

        if abs(value) >= 1_000:
            return f"₹{value / 1_000:.1f}K"

        return f"₹{value:,.0f}"

    except Exception:
        return str(value)


def fmt_percent(value):
    """Format percentage values."""

    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except Exception:
        return "—"

    try:
        return f"{float(value):.1f}%"
    except Exception:
        return str(value)


def _metric_tone(label):
    """Choose a KPI color tone from the metric meaning."""

    label_text = str(label).lower()

    tone_keywords = [
        (
            "risk",
            [
                "return",
                "returned",
                "cancel",
                "risk",
                "negative",
                "anomal",
                "critical",
            ],
        ),
        (
            "aov",
            [
                "aov",
                "average order value",
                "avg customer value",
                "avg product revenue",
                "average value",
            ],
        ),
        ("profit", ["profit", "margin"]),
        ("revenue", ["revenue", "sales"]),
        ("orders", ["order", "transaction", "units sold", "quantity"]),
        ("customers", ["customer", "segment"]),
        (
            "growth",
            [
                "growth",
                "conversion",
                "uplift",
                "roi",
                "roas",
                "rate",
                "delivered",
                "delivery",
                "positive",
            ],
        ),
    ]

    for tone, keywords in tone_keywords:
        if any(keyword in label_text for keyword in keywords):
            return tone

    return "default"


def metric_card(label, value, subtitle=""):
    """Render KPI card."""

    tone = _metric_tone(label)

    st.html(
        f"""
        <div class="metric-card metric-{tone}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """
    )


def clean_df(df):
    """Prepare dataframe for UI display."""

    if not isinstance(df, pd.DataFrame):
        return None

    if df.empty:
        return None

    result = df.copy()

    result.columns = [
        str(column).replace("_", " ").title()
        for column in result.columns
    ]

    return result


def find_column(df, keywords):
    """Find a dataframe column using keyword matching."""

    if df is None:
        return None

    for column in df.columns:

        normalized = str(column).lower().replace(
            " ",
            "_",
        )

        for keyword in keywords:

            if keyword in normalized:
                return column

    return None


def get_results():
    """Retrieve Delivery Intelligence results."""

    analytics_results = st.session_state.get(
        "analytics_results",
        {},
    )

    if not isinstance(analytics_results, dict):
        return {}

    return analytics_results.get(
        "delivery",
        {},
    ) or {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="operations-header">
        <h1>🚚 Delivery Intelligence</h1>
        <p>
            Monitor delivery performance, shipping methods, operational
            bottlenecks, returns and seller or regional fulfillment patterns.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================

results = get_results()

if not results:

    st.info(
        "Upload and process an analysis-ready dataset from the "
        "DataPulse home page to activate Delivery Intelligence."
    )

    st.stop()


if results.get("available") is False:

    st.warning(
        "Delivery Intelligence is not available for the current dataset."
    )

    for warning in results.get("warnings", []):
        st.caption(f"• {warning}")

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

summary = results.get(
    "summary",
    {},
) or {}


total_orders = first_existing(
    summary,
    [
        "total_orders",
        "orders",
        "order_count",
    ],
    0,
)

delivered_orders = first_existing(
    summary,
    [
        "delivered_orders",
        "completed_orders",
    ],
    0,
)

cancelled_orders = first_existing(
    summary,
    [
        "cancelled_orders",
        "canceled_orders",
        "cancelled_count",
    ],
    0,
)

returned_orders = first_existing(
    summary,
    [
        "returned_orders",
        "returns",
        "returned_count",
    ],
    0,
)

delivery_rate = first_existing(
    summary,
    [
        "delivery_rate_pct",
        "delivered_rate_pct",
        "delivery_success_rate_pct",
    ],
    None,
)

return_rate = first_existing(
    summary,
    [
        "return_rate_pct",
        "returns_rate_pct",
        "return_rate",
    ],
    None,
)

cancellation_rate = first_existing(
    summary,
    [
        "cancellation_rate_pct",
        "cancel_rate_pct",
    ],
    None,
)

avg_delivery_days = first_existing(
    summary,
    [
        "average_delivery_days",
        "avg_delivery_days",
        "delivery_time_days",
    ],
    None,
)


# ============================================================
# KPI ROW
# ============================================================

cols = st.columns(6)

with cols[0]:

    metric_card(
        "Orders",
        fmt_number(total_orders),
        "Orders analysed",
    )

with cols[1]:

    metric_card(
        "Delivered",
        fmt_number(delivered_orders),
        "Successfully delivered",
    )

with cols[2]:

    metric_card(
        "Delivery Rate",
        fmt_percent(delivery_rate),
        "Successful delivery share",
    )

with cols[3]:

    metric_card(
        "Returns",
        fmt_number(returned_orders),
        "Returned orders",
    )

with cols[4]:

    metric_card(
        "Return Rate",
        fmt_percent(return_rate),
        "Share of orders returned",
    )

with cols[5]:

    metric_card(
        "Avg Delivery",
        (
            f"{float(avg_delivery_days):.1f} days"
            if avg_delivery_days is not None
            else "—"
        ),
        "Average delivery time",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_status, tab_shipping, tab_regions, tab_returns = st.tabs(
    [
        "Overview",
        "Delivery Status",
        "Shipping Methods",
        "Regional Performance",
        "Returns & Risks",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Operational Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.15, 1])

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    with col1:

        overview_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Orders",
                    "Delivered Orders",
                    "Cancelled Orders",
                    "Returned Orders",
                    "Delivery Rate",
                    "Cancellation Rate",
                    "Return Rate",
                    "Average Delivery Time",
                ],
                "Value": [
                    fmt_number(total_orders),
                    fmt_number(delivered_orders),
                    fmt_number(cancelled_orders),
                    fmt_number(returned_orders),
                    fmt_percent(delivery_rate),
                    fmt_percent(cancellation_rate),
                    fmt_percent(return_rate),
                    (
                        f"{float(avg_delivery_days):.1f} days"
                        if avg_delivery_days is not None
                        else "—"
                    ),
                ],
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # ORDER OUTCOME
    # --------------------------------------------------------

    with col2:

        outcome_data = pd.DataFrame(
            {
                "Status": [
                    "Delivered",
                    "Cancelled",
                    "Returned",
                ],
                "Orders": [
                    delivered_orders,
                    cancelled_orders,
                    returned_orders,
                ],
            }
        )

        outcome_data["Orders"] = pd.to_numeric(
            outcome_data["Orders"],
            errors="coerce",
        ).fillna(0)

        if outcome_data["Orders"].sum() > 0:

            fig = px.pie(
                outcome_data,
                names="Status",
                values="Orders",
                title="Order Outcome Distribution",
                hole=0.45,
            )

            fig.update_layout(
                height=390,
                margin=dict(
                    l=10,
                    r=10,
                    t=50,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "Order outcome distribution is unavailable."
            )


# ============================================================
# DELIVERY STATUS
# ============================================================

with tab_status:

    st.markdown(
        '<div class="section-title">Delivery Status Analysis</div>',
        unsafe_allow_html=True,
    )

    status_df = first_existing(
        results,
        [
            "delivery_status",
            "status_performance",
            "delivery_status_analysis",
        ],
        None,
    )

    status_df = clean_df(status_df)

    if status_df is None:

        st.info(
            "Detailed delivery-status analysis is not available."
        )

    else:

        status_column = find_column(
            status_df,
            [
                "status",
                "delivery_status",
                "order_status",
            ],
        )

        count_column = find_column(
            status_df,
            [
                "orders",
                "count",
                "order_count",
            ],
        )

        percentage_column = find_column(
            status_df,
            [
                "percentage",
                "percent",
                "rate",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if status_column and count_column:

                chart_df = status_df.copy()

                chart_df[count_column] = pd.to_numeric(
                    chart_df[count_column],
                    errors="coerce",
                ).fillna(0)

                fig = px.bar(
                    chart_df,
                    x=status_column,
                    y=count_column,
                    title="Orders by Delivery Status",
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    xaxis_tickangle=-25,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Delivery status counts are unavailable."
                )

        with col2:

            if status_column and percentage_column:

                chart_df = status_df.copy()

                chart_df[percentage_column] = pd.to_numeric(
                    chart_df[percentage_column],
                    errors="coerce",
                )

                fig = px.bar(
                    chart_df,
                    x=status_column,
                    y=percentage_column,
                    title="Delivery Status Share",
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    xaxis_tickangle=-25,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Delivery status percentage data is unavailable."
                )

        st.markdown("### Delivery Status Table")

        st.dataframe(
            status_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# SHIPPING METHODS
# ============================================================

with tab_shipping:

    st.markdown(
        '<div class="section-title">Shipping Method Performance</div>',
        unsafe_allow_html=True,
    )

    shipping_df = first_existing(
        results,
        [
            "shipping_method_performance",
            "shipping_methods",
            "shipping_analysis",
        ],
        None,
    )

    shipping_df = clean_df(
        shipping_df
    )

    if shipping_df is None:

        st.info(
            "Shipping-method analysis is not available."
        )

    else:

        method_column = find_column(
            shipping_df,
            [
                "shipping_method",
                "ship_method",
                "delivery_method",
                "shipping",
            ],
        )

        orders_column = find_column(
            shipping_df,
            [
                "orders",
                "order_count",
            ],
        )

        delivery_column = find_column(
            shipping_df,
            [
                "delivery_rate",
                "delivered_rate",
                "success_rate",
            ],
        )

        days_column = find_column(
            shipping_df,
            [
                "delivery_days",
                "avg_delivery",
                "delivery_time",
            ],
        )

        return_column = find_column(
            shipping_df,
            [
                "return_rate",
                "returns_rate",
            ],
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # ORDER VOLUME
        # ----------------------------------------------------

        with col1:

            if method_column and orders_column:

                chart_df = shipping_df.sort_values(
                    orders_column,
                    ascending=False,
                )

                fig = px.bar(
                    chart_df,
                    x=method_column,
                    y=orders_column,
                    title="Orders by Shipping Method",
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    xaxis_tickangle=-30,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Shipping order volume is unavailable."
                )

        # ----------------------------------------------------
        # DELIVERY TIME
        # ----------------------------------------------------

        with col2:

            if method_column and days_column:

                chart_df = shipping_df.sort_values(
                    days_column,
                    ascending=True,
                )

                fig = px.bar(
                    chart_df,
                    x=method_column,
                    y=days_column,
                    title="Average Delivery Time",
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    xaxis_tickangle=-30,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Shipping delivery-time data is unavailable."
                )

        # ----------------------------------------------------
        # DELIVERY RATE
        # ----------------------------------------------------

        if method_column and delivery_column:

            chart_df = shipping_df.sort_values(
                delivery_column,
                ascending=False,
            )

            fig = px.bar(
                chart_df,
                x=method_column,
                y=delivery_column,
                title="Delivery Success Rate by Shipping Method",
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=10,
                    r=10,
                    t=50,
                    b=10,
                ),
                xaxis_tickangle=-30,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### Shipping Method Table")

        st.dataframe(
            shipping_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# REGIONAL PERFORMANCE
# ============================================================

with tab_regions:

    st.markdown(
        '<div class="section-title">Regional Delivery Performance</div>',
        unsafe_allow_html=True,
    )

    region_df = first_existing(
        results,
        [
            "region_performance",
            "regional_performance",
            "geographic_performance",
        ],
        None,
    )

    region_df = clean_df(
        region_df
    )

    if region_df is None:

        st.info(
            "Regional delivery performance is not available."
        )

    else:

        region_column = find_column(
            region_df,
            [
                "region",
                "state",
                "city",
                "location",
                "market",
            ],
        )

        orders_column = find_column(
            region_df,
            [
                "orders",
                "order_count",
            ],
        )

        delivery_column = find_column(
            region_df,
            [
                "delivery_rate",
                "delivered_rate",
                "success_rate",
            ],
        )

        return_column = find_column(
            region_df,
            [
                "return_rate",
                "returns_rate",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if region_column and delivery_column:

                chart_df = (
                    region_df
                    .sort_values(
                        delivery_column,
                        ascending=False,
                    )
                    .head(20)
                )

                fig = px.bar(
                    chart_df,
                    x=delivery_column,
                    y=region_column,
                    orientation="h",
                    title="Delivery Rate by Region",
                )

                fig.update_layout(
                    height=520,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    yaxis=dict(
                        categoryorder="total ascending"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Regional delivery-rate data is unavailable."
                )

        with col2:

            if region_column and return_column:

                chart_df = (
                    region_df
                    .sort_values(
                        return_column,
                        ascending=False,
                    )
                    .head(20)
                )

                fig = px.bar(
                    chart_df,
                    x=return_column,
                    y=region_column,
                    orientation="h",
                    title="Return Rate by Region",
                )

                fig.update_layout(
                    height=520,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    yaxis=dict(
                        categoryorder="total ascending"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Regional return-rate data is unavailable."
                )

        if region_column and orders_column:

            chart_df = (
                region_df
                .sort_values(
                    orders_column,
                    ascending=False,
                )
                .head(20)
            )

            fig = px.bar(
                chart_df,
                x=orders_column,
                y=region_column,
                orientation="h",
                title="Order Volume by Region",
            )

            fig.update_layout(
                height=500,
                margin=dict(
                    l=10,
                    r=10,
                    t=50,
                    b=10,
                ),
                yaxis=dict(
                    categoryorder="total ascending"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### Regional Delivery Table")

        st.dataframe(
            region_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RETURNS & RISKS
# ============================================================

with tab_returns:

    st.markdown(
        '<div class="section-title">Returns & Operational Risks</div>',
        unsafe_allow_html=True,
    )

    return_df = first_existing(
        results,
        [
            "return_analysis",
            "returns",
            "return_performance",
            "returns_analysis",
        ],
        None,
    )

    return_df = clean_df(
        return_df
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        metric_card(
            "Returned Orders",
            fmt_number(returned_orders),
            "Total returned orders",
        )

    with col2:

        metric_card(
            "Return Rate",
            fmt_percent(return_rate),
            "Share of orders returned",
        )

    with col3:

        metric_card(
            "Cancelled",
            fmt_number(cancelled_orders),
            "Cancelled orders",
        )

    if return_df is not None:

        category_column = find_column(
            return_df,
            [
                "reason",
                "return_reason",
                "category",
                "product",
                "region",
                "status",
            ],
        )

        count_column = find_column(
            return_df,
            [
                "returns",
                "returned_orders",
                "count",
                "orders",
            ],
        )

        if category_column and count_column:

            chart_df = (
                return_df
                .sort_values(
                    count_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=count_column,
                y=category_column,
                orientation="h",
                title="Return Drivers",
            )

            fig.update_layout(
                height=500,
                margin=dict(
                    l=10,
                    r=10,
                    t=50,
                    b=10,
                ),
                yaxis=dict(
                    categoryorder="total ascending"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### Return Analysis Table")

        st.dataframe(
            return_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Detailed return analysis is not available."
        )

    # --------------------------------------------------------
    # OPERATIONAL ALERTS
    # --------------------------------------------------------

    alerts = results.get(
        "operational_alerts",
        [],
    )

    st.markdown(
        '<div class="section-title">Operational Alerts</div>',
        unsafe_allow_html=True,
    )

    if isinstance(alerts, pd.DataFrame):

        alerts = alerts.to_dict(
            "records"
        )

    if isinstance(alerts, list) and alerts:

        for alert in alerts[:10]:

            if isinstance(alert, dict):

                title = first_existing(
                    alert,
                    [
                        "title",
                        "alert",
                        "name",
                    ],
                    "Operational Alert",
                )

                text = first_existing(
                    alert,
                    [
                        "description",
                        "message",
                        "text",
                        "detail",
                    ],
                    "",
                )

            else:

                title = "Operational Alert"
                text = str(alert)

            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        generated_alerts = []

        try:

            if (
                return_rate is not None
                and float(return_rate) >= 20
            ):

                generated_alerts.append(
                    (
                        "High Return Rate",
                        f"Return rate is approximately "
                        f"{float(return_rate):.1f}%. "
                        "Investigate product quality, fulfilment "
                        "and customer-experience drivers.",
                    )
                )

        except Exception:
            pass

        try:

            if (
                cancellation_rate is not None
                and float(cancellation_rate) >= 10
            ):

                generated_alerts.append(
                    (
                        "Cancellation Risk",
                        f"Cancellation rate is approximately "
                        f"{float(cancellation_rate):.1f}%. "
                        "Review inventory availability, processing "
                        "times and order fulfilment issues.",
                    )
                )

        except Exception:
            pass

        try:

            if (
                delivery_rate is not None
                and float(delivery_rate) < 80
            ):

                generated_alerts.append(
                    (
                        "Delivery Performance Risk",
                        f"Delivery success rate is approximately "
                        f"{float(delivery_rate):.1f}%. "
                        "Operational performance requires attention.",
                    )
                )

        except Exception:
            pass

        if not generated_alerts:

            generated_alerts.append(
                (
                    "Operations Stable",
                    "No major operational warning was detected "
                    "from the available delivery metrics.",
                )
            )

        for title, text in generated_alerts:

            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="insight-title">{title}</div>
                    <div class="insight-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DATASET STATUS
# ============================================================

metadata = st.session_state.get(
    "dataset_metadata"
)

if metadata:

    st.divider()

    st.caption(
        f"Dataset: {getattr(metadata, 'file_name', 'Uploaded dataset')} "
        f"• {getattr(metadata, 'rows', '—'):,} rows "
        f"• {getattr(metadata, 'columns', '—')} columns"
    )
