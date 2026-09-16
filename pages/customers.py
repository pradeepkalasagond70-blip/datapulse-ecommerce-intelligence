# pages/customers.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence | DataPulse",
    page_icon="👥",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .customer-header {
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

    .customer-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 750;
    }

    .customer-header p {
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

    @media (max-width: 768px) {
        .main .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
        [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0 !important; }
        [data-testid="stColumn"], .stColumn { width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
        [data-testid="stColumn"] > div, .stColumn > div { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
        .customer-header { padding: 15px 16px; }
        .customer-header h1 { font-size: 24px; line-height: 1.15; overflow-wrap: anywhere; }
        .customer-header p, .section-title, .insight-title, .insight-text { overflow-wrap: anywhere; }
        .metric-card { height: auto; min-height: 108px; padding: 15px; }
        .metric-label, .metric-value { white-space: normal; overflow-wrap: anywhere; }
        .metric-value { font-size: 20px; }
        [data-testid="stPlotlyChart"] { max-width: 100% !important; overflow: hidden !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stDataEditor"] { max-width: 100% !important; min-width: 0 !important; overflow-x: auto !important; }
        [data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] iframe { width: 100% !important; max-width: 100% !important; }
        input, textarea, select, [data-baseweb="select"] { max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
        button { min-height: 44px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def first_existing(data, keys, default=None):
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def fmt_number(value, decimals=0):
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


def get_results():
    analytics_results = st.session_state.get(
        "analytics_results",
        {},
    )

    if not isinstance(analytics_results, dict):
        return {}

    return analytics_results.get(
        "customer",
        {},
    ) or {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="customer-header">
        <h1>👥 Customer Intelligence</h1>
        <p>
            Understand customer value, purchasing behaviour, retention,
            loyalty and the segments driving your business.
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
        "DataPulse home page to activate Customer Intelligence."
    )
    st.stop()


if results.get("available") is False:

    st.warning(
        "Customer Intelligence is not available for the current dataset."
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

repeat_customer = results.get(
    "repeat_customer",
    {},
) or {}


total_customers = first_existing(
    summary,
    [
        "total_customers",
        "customers",
        "customer_count",
        "unique_customers",
    ],
    0,
)

repeat_customers = first_existing(
    summary,
    [
        "repeat_customers",
        "repeat_customer_count",
    ],
    first_existing(
        repeat_customer,
        ["repeat_customers"],
        0,
    ),
)

repeat_rate = first_existing(
    summary,
    [
        "repeat_customer_rate_pct",
        "repeat_rate_pct",
        "repeat_rate",
        "repeat_customer_rate",
    ],
    None,
)

customer_revenue = first_existing(
    summary,
    [
        "total_revenue",
        "total_customer_revenue",
        "customer_revenue",
        "revenue",
    ],
    0,
)

customer_profit = first_existing(
    summary,
    [
        "total_profit",
        "total_customer_profit",
        "customer_profit",
        "profit",
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

avg_customer_value = first_existing(
    summary,
    [
        "average_customer_value",
        "average_customer_revenue",
        "avg_customer_value",
        "revenue_per_customer",
    ],
    None,
)


# ============================================================
# KPI ROW
# ============================================================

cols = st.columns(6)

with cols[0]:
    metric_card(
        "Customers",
        fmt_number(total_customers),
        "Unique customers",
    )

with cols[1]:
    metric_card(
        "Repeat Customers",
        fmt_number(repeat_customers),
        "Customers with repeat purchases",
    )

with cols[2]:
    metric_card(
        "Repeat Rate",
        fmt_percent(repeat_rate),
        "Customer retention signal",
    )

with cols[3]:
    metric_card(
        "Customer Revenue",
        fmt_currency(customer_revenue),
        "Revenue generated",
    )

with cols[4]:
    metric_card(
        "Customer Profit",
        fmt_currency(customer_profit),
        "Profit contribution",
    )

with cols[5]:
    metric_card(
        "Avg Customer Value",
        fmt_currency(avg_customer_value),
        "Average value per customer",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_value, tab_segments, tab_retention, tab_table = st.tabs(
    [
        "Overview",
        "Customer Value",
        "Customer Segments",
        "Retention",
        "Customer Explorer",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Customer Base Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # CUSTOMER METRICS
    # --------------------------------------------------------

    with col1:

        overview_data = pd.DataFrame(
            {
                "Metric": [
                    "Total Customers",
                    "Repeat Customers",
                    "Repeat Rate",
                    "Revenue",
                    "Profit",
                    "Average Order Value",
                ],
                "Value": [
                    fmt_number(total_customers),
                    fmt_number(repeat_customers),
                    fmt_percent(repeat_rate),
                    fmt_currency(customer_revenue),
                    fmt_currency(customer_profit),
                    fmt_currency(aov),
                ],
            }
        )

        st.dataframe(
            overview_data,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # CUSTOMER DISTRIBUTION
    # --------------------------------------------------------

    with col2:

        distribution = first_existing(
            results,
            [
                "customer_distribution",
                "customer_type_distribution",
                "segment_distribution",
            ],
            None,
        )

        distribution = clean_df(distribution)

        if distribution is not None:

            category_column = distribution.columns[0]

            numeric_columns = distribution.select_dtypes(
                include="number"
            ).columns.tolist()

            value_column = (
                numeric_columns[0]
                if numeric_columns
                else None
            )

            if value_column:

                fig = px.pie(
                    distribution,
                    names=category_column,
                    values=value_column,
                    title="Customer Distribution",
                    hole=0.45,
                )

                fig.update_layout(
                    height=360,
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

                st.dataframe(
                    distribution,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.info(
                "Customer distribution is not available."
            )


# ============================================================
# CUSTOMER VALUE
# ============================================================

with tab_value:

    st.markdown(
        '<div class="section-title">Customer Value Analysis</div>',
        unsafe_allow_html=True,
    )

    value_df = first_existing(
        results,
        [
            "customer_value",
            "customer_value_analysis",
            "value_distribution",
            "top_customers",
            "customer_performance",
        ],
        None,
    )

    value_df = clean_df(value_df)

    if value_df is None:

        st.info(
            "Customer value analysis is not available for this dataset."
        )

    else:

        numeric_columns = value_df.select_dtypes(
            include="number"
        ).columns.tolist()

        customer_column = None

        for column in value_df.columns:

            normalized = column.lower().replace(
                " ",
                "_",
            )

            if (
                "customer" in normalized
                or normalized in {"id", "customer_id"}
            ):
                customer_column = column
                break

        revenue_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(
                " ",
                "_",
            )

            if "revenue" in normalized:

                revenue_column = column
                break

        profit_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(
                " ",
                "_",
            )

            if "profit" in normalized:

                profit_column = column
                break

        col1, col2 = st.columns(2)

        with col1:

            if customer_column and revenue_column:

                chart_df = (
                    value_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=revenue_column,
                    y=customer_column,
                    orientation="h",
                    title="Top Customers by Revenue",
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

            else:

                st.info(
                    "Customer revenue fields are unavailable."
                )

        with col2:

            if customer_column and profit_column:

                chart_df = (
                    value_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=profit_column,
                    y=customer_column,
                    orientation="h",
                    title="Top Customers by Profit",
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

            else:

                st.info(
                    "Customer profit fields are unavailable."
                )

        st.markdown("### Customer Value Table")

        st.dataframe(
            value_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

with tab_segments:

    st.markdown(
        '<div class="section-title">Customer Segmentation</div>',
        unsafe_allow_html=True,
    )

    segment_df = first_existing(
        results,
        [
            "segments",
            "segment_summary",
            "customer_segments",
            "segment_profiles",
        ],
        None,
    )

    segment_df = clean_df(segment_df)

    if segment_df is None:

        st.info(
            "Customer segmentation data is not available."
        )

    else:

        numeric_columns = segment_df.select_dtypes(
            include="number"
        ).columns.tolist()

        segment_column = None

        for column in segment_df.columns:

            normalized = column.lower().replace(
                " ",
                "_",
            )

            if "segment" in normalized or "cluster" in normalized:

                segment_column = column
                break

        count_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(
                " ",
                "_",
            )

            if (
                "customer" in normalized
                or "count" in normalized
                or "size" in normalized
            ):
                count_column = column
                break

        if segment_column and count_column:

            fig = px.bar(
                segment_df,
                x=segment_column,
                y=count_column,
                title="Customer Segment Distribution",
            )

            fig.update_layout(
                height=420,
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

        st.markdown("### Segment Profiles")

        st.dataframe(
            segment_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RETENTION
# ============================================================

with tab_retention:

    st.markdown(
        '<div class="section-title">Customer Retention Analysis</div>',
        unsafe_allow_html=True,
    )

    retention_df = first_existing(
        results,
        [
            "retention",
            "retention_analysis",
            "repeat_purchase_analysis",
            "customer_retention",
        ],
        None,
    )

    retention_df = clean_df(retention_df)

    col1, col2 = st.columns(2)

    with col1:

        metric_card(
            "Repeat Customers",
            fmt_number(repeat_customers),
            "Customers returning to purchase",
        )

    with col2:

        metric_card(
            "Repeat Rate",
            fmt_percent(repeat_rate),
            "Share of customers making repeat purchases",
        )

    if retention_df is not None:

        numeric_columns = retention_df.select_dtypes(
            include="number"
        ).columns.tolist()

        category_column = retention_df.columns[0]

        rate_column = None

        for column in numeric_columns:

            normalized = column.lower().replace(
                " ",
                "_",
            )

            if "rate" in normalized or "percent" in normalized:

                rate_column = column
                break

        if rate_column:

            fig = px.bar(
                retention_df,
                x=category_column,
                y=rate_column,
                title="Retention / Repeat Purchase Rate",
            )

            fig.update_layout(
                height=400,
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

        st.dataframe(
            retention_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Detailed retention breakdown is not available."
        )


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

with tab_table:

    st.markdown(
        '<div class="section-title">Customer Explorer</div>',
        unsafe_allow_html=True,
    )

    explorer_df = first_existing(
        results,
        [
            "customers",
            "customer_table",
            "customer_performance",
            "customer_value",
            "rfm",
        ],
        None,
    )

    explorer_df = clean_df(explorer_df)

    if explorer_df is None:

        st.info(
            "A customer-level explorer is not available for the current dataset."
        )

    else:

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        search = st.text_input(
            "Search customers",
            placeholder="Search by customer ID or customer name...",
        )

        filtered_df = explorer_df.copy()

        if search.strip():

            mask = filtered_df.astype(
                str
            ).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ).any(axis=1)

            filtered_df = filtered_df.loc[mask]

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        numeric_columns = filtered_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            sort_column = st.selectbox(
                "Sort customers by",
                numeric_columns,
            )

            filtered_df = filtered_df.sort_values(
                sort_column,
                ascending=False,
            )

        st.caption(
            f"Showing {len(filtered_df):,} customer records."
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CUSTOMER INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">Customer Intelligence Signals</div>',
    unsafe_allow_html=True,
)

insights = first_existing(
    results,
    [
        "insights",
        "customer_insights",
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
                "Customer Signal",
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

            title = "Customer Signal"
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

    generated_signals = []

    try:

        if repeat_rate is not None:

            repeat_rate_value = float(
                repeat_rate
            )

            if repeat_rate_value >= 50:

                generated_signals.append(
                    (
                        "Strong Customer Loyalty",
                        f"Approximately {repeat_rate_value:.1f}% "
                        "of customers are repeat purchasers, indicating "
                        "strong customer retention.",
                    )
                )

            elif repeat_rate_value < 20:

                generated_signals.append(
                    (
                        "Retention Opportunity",
                        f"Repeat purchase rate is approximately "
                        f"{repeat_rate_value:.1f}%. Customer retention "
                        "and re-engagement should be investigated.",
                    )
                )

    except Exception:
        pass

    if total_customers:

        try:

            customer_count = float(
                total_customers
            )

            if customer_count > 0:

                generated_signals.append(
                    (
                        "Customer Base",
                        f"DataPulse identified approximately "
                        f"{customer_count:,.0f} unique customers "
                        "in the analysis dataset.",
                    )
                )

        except Exception:
            pass

    if not generated_signals:

        generated_signals.append(
            (
                "Customer Analysis Ready",
                "Customer behaviour can now be connected with "
                "segmentation, churn prediction, product performance "
                "and campaign response.",
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
