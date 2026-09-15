# pages/campaign_impact.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Campaign Impact | DataPulse",
    page_icon="🎯",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .campaign-header {
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

    .campaign-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 750;
    }

    .campaign-header p {
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
    """Prepare dataframe for display."""

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
    """Find a column using keyword matching."""

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
    """Get Campaign Impact results from session state."""

    analytics_results = st.session_state.get(
        "analytics_results",
        {},
    )

    if not isinstance(analytics_results, dict):
        return {}

    return analytics_results.get(
        "campaign_impact",
        {},
    ) or {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="campaign-header">
        <h1>🎯 Campaign Impact</h1>
        <p>
            Measure campaign performance, customer response, revenue impact,
            profitability and the campaigns creating real business value.
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
        "DataPulse home page to activate Campaign Impact."
    )

    st.stop()


if results.get("available") is False:

    st.warning(
        "Campaign Impact is not available for the current dataset."
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


campaign_count = first_existing(
    summary,
    [
        "total_campaigns",
        "campaigns",
        "campaign_count",
        "unique_campaigns",
    ],
    0,
)

campaign_revenue = first_existing(
    summary,
    [
        "campaign_revenue",
        "total_campaign_revenue",
        "revenue",
    ],
    0,
)

campaign_orders = first_existing(
    summary,
    [
        "campaign_orders",
        "total_campaign_orders",
        "orders",
    ],
    0,
)

campaign_customers = first_existing(
    summary,
    [
        "campaign_customers",
        "customers_acquired",
        "campaign_customers_acquired",
        "unique_customers",
    ],
    0,
)

campaign_profit = first_existing(
    summary,
    [
        "campaign_profit",
        "total_campaign_profit",
        "profit",
    ],
    0,
)

roi = first_existing(
    summary,
    [
        "roi_pct",
        "roi",
        "campaign_roi",
    ],
    None,
)

roas = first_existing(
    summary,
    [
        "roas",
        "roas_ratio",
        "campaign_roas",
    ],
    None,
)

conversion_rate = first_existing(
    summary,
    [
        "conversion_rate_pct",
        "conversion_pct",
        "campaign_conversion_rate",
    ],
    None,
)

uplift = first_existing(
    summary,
    [
        "revenue_uplift_pct",
        "uplift_pct",
        "campaign_uplift_pct",
    ],
    None,
)


# ============================================================
# KPI ROW
# ============================================================

cols = st.columns(6)

with cols[0]:

    metric_card(
        "Campaigns",
        fmt_number(campaign_count),
        "Campaigns detected",
    )

with cols[1]:

    metric_card(
        "Campaign Revenue",
        fmt_currency(campaign_revenue),
        "Revenue associated with campaigns",
    )

with cols[2]:

    metric_card(
        "Orders",
        fmt_number(campaign_orders),
        "Campaign-associated orders",
    )

with cols[3]:

    metric_card(
        "Campaign Profit",
        fmt_currency(campaign_profit),
        "Profit contribution",
    )

with cols[4]:

    metric_card(
        "ROI",
        fmt_percent(roi),
        "Return on campaign investment",
    )

with cols[5]:

    metric_card(
        "Revenue Uplift",
        fmt_percent(uplift),
        "Estimated campaign uplift",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_campaigns, tab_impact, tab_customers, tab_products = st.tabs(
    [
        "Overview",
        "Campaign Performance",
        "Business Impact",
        "Customer Response",
        "Product Impact",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Campaign Performance Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    # --------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------

    with col1:

        overview_df = pd.DataFrame(
            {
                "Metric": [
                    "Campaigns",
                    "Campaign Revenue",
                    "Campaign Orders",
                    "Campaign Customers",
                    "Campaign Profit",
                    "ROI",
                    "ROAS",
                    "Conversion Rate",
                    "Revenue Uplift",
                ],
                "Value": [
                    fmt_number(campaign_count),
                    fmt_currency(campaign_revenue),
                    fmt_number(campaign_orders),
                    fmt_number(campaign_customers),
                    fmt_currency(campaign_profit),
                    fmt_percent(roi),
                    fmt_number(roas, 2),
                    fmt_percent(conversion_rate),
                    fmt_percent(uplift),
                ],
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # CAMPAIGN MIX
    # --------------------------------------------------------

    with col2:

        campaign_df = first_existing(
            results,
            [
                "campaign_performance",
                "campaigns",
                "campaign_analysis",
                "campaign_summary",
            ],
            None,
        )

        campaign_df = clean_df(campaign_df)

        if campaign_df is not None:

            campaign_column = find_column(
                campaign_df,
                [
                    "campaign_name",
                    "campaign_id",
                    "campaign",
                ],
            )

            revenue_column = find_column(
                campaign_df,
                [
                    "revenue",
                    "sales",
                ],
            )

            if campaign_column and revenue_column:

                chart_df = (
                    campaign_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(10)
                )

                fig = px.pie(
                    chart_df,
                    names=campaign_column,
                    values=revenue_column,
                    title="Revenue Contribution",
                    hole=0.45,
                )

                fig.update_layout(
                    height=380,
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
                    "Campaign revenue breakdown is unavailable."
                )

        else:

            st.info(
                "Campaign-level performance data is unavailable."
            )


# ============================================================
# CAMPAIGN PERFORMANCE
# ============================================================

with tab_campaigns:

    st.markdown(
        '<div class="section-title">Campaign Performance</div>',
        unsafe_allow_html=True,
    )

    campaign_df = first_existing(
        results,
        [
            "campaign_performance",
            "campaigns",
            "campaign_analysis",
            "campaign_summary",
        ],
        None,
    )

    campaign_df = clean_df(campaign_df)

    if campaign_df is None:

        st.info(
            "Campaign-level performance data is not available."
        )

    else:

        campaign_column = find_column(
            campaign_df,
            [
                "campaign_name",
                "campaign_id",
                "campaign",
            ],
        )

        revenue_column = find_column(
            campaign_df,
            [
                "revenue",
                "sales",
            ],
        )

        profit_column = find_column(
            campaign_df,
            [
                "profit",
            ],
        )

        orders_column = find_column(
            campaign_df,
            [
                "orders",
                "order_count",
            ],
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # REVENUE
        # ----------------------------------------------------

        with col1:

            if campaign_column and revenue_column:

                chart_df = (
                    campaign_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=revenue_column,
                    y=campaign_column,
                    orientation="h",
                    title="Top Campaigns by Revenue",
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
                    "Campaign revenue data is unavailable."
                )

        # ----------------------------------------------------
        # PROFIT
        # ----------------------------------------------------

        with col2:

            if campaign_column and profit_column:

                chart_df = (
                    campaign_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=profit_column,
                    y=campaign_column,
                    orientation="h",
                    title="Top Campaigns by Profit",
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
                    "Campaign profit data is unavailable."
                )

        # ----------------------------------------------------
        # ORDERS
        # ----------------------------------------------------

        if campaign_column and orders_column:

            chart_df = (
                campaign_df
                .sort_values(
                    orders_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=orders_column,
                y=campaign_column,
                orientation="h",
                title="Campaigns by Orders",
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

        st.markdown("### Campaign Performance Table")

        st.dataframe(
            campaign_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# BUSINESS IMPACT
# ============================================================

with tab_impact:

    st.markdown(
        '<div class="section-title">Business Impact Analysis</div>',
        unsafe_allow_html=True,
    )

    impact_df = first_existing(
        results,
        [
            "impact_analysis",
            "campaign_impact",
            "before_after",
            "uplift_analysis",
            "impact",
        ],
        None,
    )

    impact_df = clean_df(impact_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "Revenue",
            fmt_currency(campaign_revenue),
            "Campaign revenue",
        )

    with col2:

        metric_card(
            "Profit",
            fmt_currency(campaign_profit),
            "Campaign profit",
        )

    with col3:

        metric_card(
            "ROI",
            fmt_percent(roi),
            "Investment return",
        )

    with col4:

        metric_card(
            "ROAS",
            fmt_number(roas, 2),
            "Revenue / ad spend",
        )

    if impact_df is not None:

        numeric_columns = impact_df.select_dtypes(
            include="number"
        ).columns.tolist()

        category_column = None

        for column in impact_df.columns:

            normalized = str(column).lower().replace(
                " ",
                "_",
            )

            if any(
                keyword in normalized
                for keyword in [
                    "campaign",
                    "period",
                    "stage",
                    "group",
                    "segment",
                ]
            ):

                category_column = column
                break

        value_column = None

        for column in numeric_columns:

            normalized = str(column).lower().replace(
                " ",
                "_",
            )

            if (
                "uplift" in normalized
                or "revenue" in normalized
                or "impact" in normalized
            ):

                value_column = column
                break

        if category_column and value_column:

            fig = px.bar(
                impact_df,
                x=category_column,
                y=value_column,
                title="Campaign Business Impact",
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

        st.markdown("### Impact Analysis Table")

        st.dataframe(
            impact_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Detailed before-vs-after or uplift analysis is not "
            "available for the current dataset."
        )


# ============================================================
# CUSTOMER RESPONSE
# ============================================================

with tab_customers:

    st.markdown(
        '<div class="section-title">Customer Response to Campaigns</div>',
        unsafe_allow_html=True,
    )

    customer_df = first_existing(
        results,
        [
            "customer_response",
            "campaign_customer_response",
            "customer_segment_response",
            "segment_response",
            "campaign_segments",
        ],
        None,
    )

    customer_df = clean_df(
        customer_df
    )

    if customer_df is None:

        st.info(
            "Campaign customer-response data is not available."
        )

    else:

        segment_column = find_column(
            customer_df,
            [
                "segment",
                "customer_segment",
                "customer_type",
            ],
        )

        revenue_column = find_column(
            customer_df,
            [
                "revenue",
                "sales",
            ],
        )

        orders_column = find_column(
            customer_df,
            [
                "orders",
                "order_count",
            ],
        )

        conversion_column = find_column(
            customer_df,
            [
                "conversion",
                "response_rate",
                "conversion_rate",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if segment_column and revenue_column:

                fig = px.bar(
                    customer_df.sort_values(
                        revenue_column,
                        ascending=False,
                    ),
                    x=segment_column,
                    y=revenue_column,
                    title="Campaign Revenue by Customer Segment",
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
                    "Customer segment revenue data is unavailable."
                )

        with col2:

            if segment_column and conversion_column:

                fig = px.bar(
                    customer_df.sort_values(
                        conversion_column,
                        ascending=False,
                    ),
                    x=segment_column,
                    y=conversion_column,
                    title="Campaign Response by Segment",
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
                    "Customer response-rate data is unavailable."
                )

        if segment_column and orders_column:

            fig = px.bar(
                customer_df.sort_values(
                    orders_column,
                    ascending=False,
                ),
                x=segment_column,
                y=orders_column,
                title="Campaign Orders by Customer Segment",
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

        st.markdown("### Customer Response Table")

        st.dataframe(
            customer_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PRODUCT IMPACT
# ============================================================

with tab_products:

    st.markdown(
        '<div class="section-title">Product Impact from Campaigns</div>',
        unsafe_allow_html=True,
    )

    product_df = first_existing(
        results,
        [
            "product_impact",
            "campaign_product_impact",
            "product_campaign_performance",
            "campaign_products",
        ],
        None,
    )

    product_df = clean_df(
        product_df
    )

    if product_df is None:

        st.info(
            "Campaign product-impact data is not available."
        )

    else:

        product_column = find_column(
            product_df,
            [
                "product_name",
                "product_id",
                "product",
                "sku",
            ],
        )

        revenue_column = find_column(
            product_df,
            [
                "revenue",
                "sales",
            ],
        )

        profit_column = find_column(
            product_df,
            [
                "profit",
            ],
        )

        quantity_column = find_column(
            product_df,
            [
                "quantity",
                "units",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if product_column and revenue_column:

                chart_df = (
                    product_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=revenue_column,
                    y=product_column,
                    orientation="h",
                    title="Products Driving Campaign Revenue",
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
                    "Campaign product revenue data is unavailable."
                )

        with col2:

            if product_column and profit_column:

                chart_df = (
                    product_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=profit_column,
                    y=product_column,
                    orientation="h",
                    title="Products Driving Campaign Profit",
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
                    "Campaign product profit data is unavailable."
                )

        if product_column and quantity_column:

            chart_df = (
                product_df
                .sort_values(
                    quantity_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=quantity_column,
                y=product_column,
                orientation="h",
                title="Products Driving Campaign Volume",
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

        st.markdown("### Product Campaign Impact Table")

        st.dataframe(
            product_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# CAMPAIGN INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">Campaign Intelligence Signals</div>',
    unsafe_allow_html=True,
)

insights = first_existing(
    results,
    [
        "insights",
        "campaign_insights",
        "signals",
        "recommendations",
    ],
    [],
)

if isinstance(insights, pd.DataFrame):

    insights = insights.to_dict(
        "records"
    )


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
                "Campaign Signal",
            )

            text = first_existing(
                insight,
                [
                    "description",
                    "message",
                    "text",
                    "detail",
                    "recommendation",
                ],
                "",
            )

        else:

            title = "Campaign Signal"
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

    # --------------------------------------------------------
    # ROI SIGNAL
    # --------------------------------------------------------

    try:

        if roi is not None:

            roi_value = float(roi)

            if roi_value > 100:

                generated_signals.append(
                    (
                        "Strong Campaign Return",
                        f"Campaign ROI is approximately "
                        f"{roi_value:.1f}%, indicating strong "
                        "return relative to campaign investment.",
                    )
                )

            elif roi_value < 0:

                generated_signals.append(
                    (
                        "Campaign ROI Risk",
                        "Campaign ROI is negative. "
                        "Campaign spend, targeting and conversion "
                        "performance should be reviewed.",
                    )
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # UPLIFT SIGNAL
    # --------------------------------------------------------

    try:

        if uplift is not None:

            uplift_value = float(uplift)

            if uplift_value > 10:

                generated_signals.append(
                    (
                        "Positive Revenue Uplift",
                        f"Campaign-associated revenue uplift is "
                        f"approximately {uplift_value:.1f}%.",
                    )
                )

            elif uplift_value < 0:

                generated_signals.append(
                    (
                        "Negative Revenue Uplift",
                        f"Estimated revenue uplift is "
                        f"{uplift_value:.1f}%. Campaign effectiveness "
                        "should be investigated.",
                    )
                )

    except Exception:
        pass

    if not generated_signals:

        generated_signals.append(
            (
                "Campaign Analysis Ready",
                "Campaign performance can be connected with "
                "customer segments, products, pricing and "
                "profitability to identify the campaigns "
                "creating the strongest business outcomes.",
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
