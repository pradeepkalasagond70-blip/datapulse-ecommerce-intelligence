# pages/sales.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sales Intelligence | DataPulse",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .sales-header {
        padding: 18px 22px;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #0F172A 0%,
            #172554 55%,
            #1E3A8A 100%
        );
        color: white;
        margin-bottom: 20px;
    }

    .sales-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 750;
    }

    .sales-header p {
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
        font-size: 22px;
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
    """Format currency values in INR-style notation."""

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
    """Render a consistent KPI card."""

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


def clean_dataframe(df):
    """Safely clean dataframe for display."""

    if df is None:
        return None

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


def first_existing(data, keys, default=None):
    """Return first existing non-null dictionary value."""

    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def get_sales_results():
    """Retrieve Sales Intelligence results from session state."""

    analytics_results = st.session_state.get("analytics_results", {})

    if not isinstance(analytics_results, dict):
        return {}

    return analytics_results.get("core", {}) or {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="sales-header">
        <h1>📈 Sales Intelligence</h1>
        <p>
            Understand revenue performance, order trends, growth momentum,
            profitability and the commercial drivers behind your business.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================

results = get_sales_results()


if not results:
    st.info(
        "Upload and process an analysis-ready dataset from the DataPulse "
        "home page to activate Sales Intelligence."
    )
    st.stop()


available = results.get("available", True)

if available is False:
    st.warning(
        "Sales Intelligence is not available for the current dataset."
    )

    warnings = results.get("warnings", [])

    if warnings:
        for warning in warnings:
            st.caption(f"• {warning}")

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

summary = results.get("summary", {}) or {}
growth_results = results.get("growth", {}) or {}


# Support multiple reasonable summary key names so this page
# remains compatible with the analytics engine.

revenue = first_existing(
    summary,
    [
        "total_revenue",
        "revenue",
        "sales",
        "gross_revenue",
    ],
    0,
)

orders = first_existing(
    summary,
    [
        "total_orders",
        "orders",
        "order_count",
    ],
    0,
)

quantity = first_existing(
    summary,
    [
        "total_quantity",
        "quantity",
        "units_sold",
    ],
    0,
)

profit = first_existing(
    summary,
    [
        "total_profit",
        "profit",
        "gross_profit",
    ],
    0,
)

aov = first_existing(
    summary,
    [
        "average_order_value",
        "aov",
        "avg_order_value",
    ],
    None,
)

profit_margin = first_existing(
    summary,
    [
        "profit_margin_pct",
        "profit_margin",
        "margin_pct",
    ],
    None,
)

growth = first_existing(
    summary,
    [
        "revenue_growth_pct",
        "growth_pct",
        "growth",
        "sales_growth_pct",
    ],
    first_existing(
        growth_results,
        [
            "revenue_growth",
            "revenue_growth_pct",
            "growth_pct",
            "growth",
        ],
        None,
    ),
)


# ============================================================
# KPI ROW
# ============================================================

cols = st.columns(6)

with cols[0]:
    metric_card(
        "Revenue",
        fmt_currency(revenue),
        "Total sales generated",
    )

with cols[1]:
    metric_card(
        "Orders",
        fmt_number(orders),
        "Total transactions",
    )

with cols[2]:
    metric_card(
        "Units Sold",
        fmt_number(quantity),
        "Total quantity",
    )

with cols[3]:
    metric_card(
        "Profit",
        fmt_currency(profit),
        "Total business profit",
    )

with cols[4]:
    metric_card(
        "AOV",
        fmt_currency(aov),
        "Average order value",
    )

with cols[5]:
    metric_card(
        "Growth",
        fmt_percent(growth),
        "Revenue growth",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_trends, tab_products, tab_regions, tab_profitability = st.tabs(
    [
        "Overview",
        "Sales Trends",
        "Product Performance",
        "Regional Performance",
        "Profitability",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Sales Performance Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.5, 1])

    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------

    with col1:

        performance_data = {
            "Metric": [
                "Revenue",
                "Orders",
                "Units Sold",
                "Profit",
                "Average Order Value",
                "Profit Margin",
            ],
            "Value": [
                fmt_currency(revenue),
                fmt_number(orders),
                fmt_number(quantity),
                fmt_currency(profit),
                fmt_currency(aov),
                fmt_percent(profit_margin),
            ],
        }

        st.dataframe(
            pd.DataFrame(performance_data),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # SALES HEALTH
    # --------------------------------------------------------

    with col2:

        st.markdown("### Sales Health")

        health_score = first_existing(
            summary,
            [
                "sales_health_score",
                "performance_score",
                "health_score",
            ],
            None,
        )

        if health_score is not None:

            try:
                score = max(0, min(100, float(health_score)))

                st.progress(
                    score / 100,
                    text=f"Sales Health Score: {score:.0f}/100",
                )

            except Exception:
                st.metric("Sales Health Score", health_score)

        else:

            # Derive a simple health indicator when the backend
            # does not explicitly provide one.

            score = 50

            if growth is not None:
                try:
                    growth_value = float(growth)

                    if growth_value > 20:
                        score += 25
                    elif growth_value > 5:
                        score += 15
                    elif growth_value < -20:
                        score -= 25
                    elif growth_value < -5:
                        score -= 15

                except Exception:
                    pass

            if profit_margin is not None:
                try:
                    margin_value = float(profit_margin)

                    if margin_value > 20:
                        score += 20
                    elif margin_value > 10:
                        score += 10
                    elif margin_value < 0:
                        score -= 20

                except Exception:
                    pass

            score = max(0, min(100, score))

            st.progress(
                score / 100,
                text=f"Sales Health Score: {score:.0f}/100",
            )

        st.caption(
            "Composite commercial indicator based on available sales "
            "performance metrics."
        )


# ============================================================
# SALES TRENDS
# ============================================================

with tab_trends:

    st.markdown(
        '<div class="section-title">Revenue & Order Trends</div>',
        unsafe_allow_html=True,
    )

    trend_df = first_existing(
        results,
        [
            "sales_trend",
            "sales_trends",
            "time_series",
            "revenue_trend",
            "monthly_sales",
        ],
        None,
    )

    trend_df = clean_dataframe(trend_df)

    if trend_df is None:

        st.info(
            "A time-based sales trend is not available for the current dataset."
        )

    else:

        # Identify date/period column.

        date_column = None

        for column in trend_df.columns:

            normalized = column.lower().replace(" ", "_")

            if normalized in {
                "date",
                "period",
                "month",
                "year_month",
                "order_date",
                "sales_date",
            }:
                date_column = column
                break

        if date_column is None:

            date_column = trend_df.columns[0]

        try:
            trend_df[date_column] = pd.to_datetime(
                trend_df[date_column],
                errors="coerce",
            )
        except Exception:
            pass

        numeric_columns = trend_df.select_dtypes(
            include="number"
        ).columns.tolist()

        revenue_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(" ", "_")

            if "revenue" in normalized or "sales" in normalized:
                revenue_column = column
                break

        orders_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(" ", "_")

            if "order" in normalized:
                orders_column = column
                break

        col1, col2 = st.columns(2)

        with col1:

            if revenue_column:

                fig = px.line(
                    trend_df,
                    x=date_column,
                    y=revenue_column,
                    markers=True,
                    title="Revenue Trend",
                )

                fig.update_layout(
                    height=390,
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info("Revenue trend column unavailable.")

        with col2:

            if orders_column:

                fig = px.line(
                    trend_df,
                    x=date_column,
                    y=orders_column,
                    markers=True,
                    title="Order Trend",
                )

                fig.update_layout(
                    height=390,
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info("Order trend column unavailable.")

        st.markdown("### Trend Data")

        st.dataframe(
            trend_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

with tab_products:

    st.markdown(
        '<div class="section-title">Product Sales Performance</div>',
        unsafe_allow_html=True,
    )

    product_df = first_existing(
        results,
        [
            "product_performance",
            "product_sales",
            "top_products",
            "products",
        ],
        None,
    )

    product_df = clean_dataframe(product_df)

    if product_df is None:

        st.info(
            "Product-level sales analysis is not available for the current dataset."
        )

    else:

        product_column = None

        for column in product_df.columns:

            normalized = column.lower().replace(" ", "_")

            if normalized in {
                "product",
                "product_id",
                "product_name",
                "sku",
            } or "product" in normalized:

                product_column = column
                break

        numeric_columns = product_df.select_dtypes(
            include="number"
        ).columns.tolist()

        sales_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(" ", "_")

            if "revenue" in normalized or "sales" in normalized:

                sales_column = column
                break

        col1, col2 = st.columns([1.4, 1])

        with col1:

            if product_column and sales_column:

                chart_df = (
                    product_df
                    .sort_values(
                        sales_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=sales_column,
                    y=product_column,
                    orientation="h",
                    title="Top Products by Revenue",
                )

                fig.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis=dict(categoryorder="total ascending"),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Product revenue fields are not available."
                )

        with col2:

            profit_column = None

            for column in numeric_columns:

                normalized = column.lower().replace(" ", "_")

                if "profit" in normalized:

                    profit_column = column
                    break

            if product_column and profit_column:

                chart_df = (
                    product_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(10)
                )

                fig = px.bar(
                    chart_df,
                    x=profit_column,
                    y=product_column,
                    orientation="h",
                    title="Top Products by Profit",
                )

                fig.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10),
                    yaxis=dict(categoryorder="total ascending"),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Product profit fields are not available."
                )

        st.markdown("### Product Performance Table")

        st.dataframe(
            product_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# REGIONAL PERFORMANCE
# ============================================================

with tab_regions:

    st.markdown(
        '<div class="section-title">Regional Sales Performance</div>',
        unsafe_allow_html=True,
    )

    regional_df = first_existing(
        results,
        [
            "regional_performance",
            "region_performance",
            "geographic_performance",
            "state_performance",
            "market_performance",
        ],
        None,
    )

    regional_df = clean_dataframe(regional_df)

    if regional_df is None:

        st.info(
            "Regional sales analysis is not available for the current dataset."
        )

    else:

        region_column = None

        for column in regional_df.columns:

            normalized = column.lower().replace(" ", "_")

            if any(
                keyword in normalized
                for keyword in [
                    "region",
                    "state",
                    "city",
                    "country",
                    "market",
                ]
            ):

                region_column = column
                break

        numeric_columns = regional_df.select_dtypes(
            include="number"
        ).columns.tolist()

        sales_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(" ", "_")

            if "revenue" in normalized or "sales" in normalized:

                sales_column = column
                break

        if region_column and sales_column:

            chart_df = (
                regional_df
                .sort_values(
                    sales_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=sales_column,
                y=region_column,
                orientation="h",
                title="Revenue by Region",
            )

            fig.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(categoryorder="total ascending"),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### Regional Performance Table")

        st.dataframe(
            regional_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PROFITABILITY
# ============================================================

with tab_profitability:

    st.markdown(
        '<div class="section-title">Sales Profitability</div>',
        unsafe_allow_html=True,
    )

    profitability_df = first_existing(
        results,
        [
            "profitability",
            "profitability_analysis",
            "profit_analysis",
            "margin_analysis",
        ],
        None,
    )

    profitability_df = clean_dataframe(
        profitability_df
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Revenue",
            fmt_currency(revenue),
            "Total sales",
        )

    with col2:
        metric_card(
            "Profit",
            fmt_currency(profit),
            "Total profit",
        )

    with col3:
        metric_card(
            "Margin",
            fmt_percent(profit_margin),
            "Profitability",
        )

    if profitability_df is not None:

        numeric_columns = profitability_df.select_dtypes(
            include="number"
        ).columns.tolist()

        margin_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(" ", "_")

            if "margin" in normalized:

                margin_column = column
                break

        category_column = None

        for column in profitability_df.columns:

            normalized = column.lower().replace(" ", "_")

            if any(
                keyword in normalized
                for keyword in [
                    "category",
                    "product",
                    "region",
                    "segment",
                    "channel",
                ]
            ):

                category_column = column
                break

        if category_column and margin_column:

            chart_df = profitability_df.copy()

            fig = px.bar(
                chart_df.sort_values(
                    margin_column,
                    ascending=False,
                ).head(15),
                x=margin_column,
                y=category_column,
                orientation="h",
                title="Profit Margin Performance",
            )

            fig.update_layout(
                height=480,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(categoryorder="total ascending"),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.dataframe(
            profitability_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Detailed profitability breakdown is not available. "
            "The headline profitability metrics above remain available."
        )


# ============================================================
# SALES INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">Sales Intelligence Signals</div>',
    unsafe_allow_html=True,
)

insights = first_existing(
    results,
    [
        "insights",
        "sales_insights",
        "signals",
    ],
    [],
)

if isinstance(insights, pd.DataFrame):

    insights = insights.to_dict("records")

if isinstance(insights, list) and insights:

    for insight in insights[:8]:

        if isinstance(insight, dict):

            title = first_existing(
                insight,
                [
                    "title",
                    "insight",
                    "name",
                ],
                "Sales Signal",
            )

            text = first_existing(
                insight,
                [
                    "description",
                    "message",
                    "text",
                    "detail",
                ],
                "",
            )

        else:

            title = "Sales Signal"
            text = str(insight)

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

    # Generate lightweight UI-level observations only when
    # the analytics engine does not provide explicit insights.

    generated_signals = []

    try:

        if growth is not None:

            growth_value = float(growth)

            if growth_value > 10:

                generated_signals.append(
                    (
                        "Positive Growth",
                        f"Revenue is growing at approximately "
                        f"{growth_value:.1f}%, indicating positive sales momentum.",
                    )
                )

            elif growth_value < -10:

                generated_signals.append(
                    (
                        "Growth Warning",
                        f"Revenue is declining by approximately "
                        f"{abs(growth_value):.1f}%. Investigate demand, "
                        f"pricing, customer and product drivers.",
                    )
                )

    except Exception:
        pass

    try:

        if profit_margin is not None:

            margin_value = float(profit_margin)

            if margin_value < 0:

                generated_signals.append(
                    (
                        "Profitability Risk",
                        "Overall sales are generating a negative profit margin. "
                        "Pricing, discounting and cost structure should be reviewed.",
                    )
                )

            elif margin_value > 20:

                generated_signals.append(
                    (
                        "Strong Margin",
                        f"The business is generating a healthy "
                        f"{margin_value:.1f}% profit margin.",
                    )
                )

    except Exception:
        pass

    if not generated_signals:

        generated_signals.append(
            (
                "Sales Analysis Ready",
                "Sales performance metrics are available for deeper "
                "customer, product, campaign, pricing and ML analysis.",
            )
        )

    for title, text in generated_signals:

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
