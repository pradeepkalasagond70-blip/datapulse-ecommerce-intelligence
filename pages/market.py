# pages/market.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Market & Seller Intelligence | DataPulse",
    page_icon="🌍",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .market-header {
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

    .market-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 750;
    }

    .market-header p {
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
    """Format percentages."""

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
    """Render a KPI card."""

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
    """Prepare a dataframe for display."""

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
    """Retrieve Market & Seller Intelligence results."""

    analytics_results = st.session_state.get(
        "analytics_results",
        {},
    )

    if not isinstance(analytics_results, dict):
        return {}

    return analytics_results.get(
        "market",
        {},
    ) or {}


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="market-header">
        <h1>🌍 Market & Seller Intelligence</h1>
        <p>
            Understand where your business performs, which sellers drive
            results, how markets compare and where the next opportunities lie.
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
        "DataPulse home page to activate Market & Seller Intelligence."
    )

    st.stop()


if results.get("available") is False:

    st.warning(
        "Market & Seller Intelligence is not available for the current dataset."
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


total_sellers = first_existing(
    summary,
    [
        "total_sellers",
        "sellers",
        "seller_count",
        "unique_sellers",
    ],
    0,
)

total_markets = first_existing(
    summary,
    [
        "total_markets",
        "markets",
        "market_count",
        "unique_markets",
    ],
    0,
)

total_regions = first_existing(
    summary,
    [
        "total_regions",
        "regions",
        "region_count",
        "unique_regions",
    ],
    0,
)

market_revenue = first_existing(
    summary,
    [
        "total_revenue",
        "market_revenue",
        "revenue",
    ],
    0,
)

market_profit = first_existing(
    summary,
    [
        "total_profit",
        "market_profit",
        "profit",
    ],
    0,
)

top_seller_share = first_existing(
    summary,
    [
        "top_seller_revenue_share_pct",
        "top_seller_share_pct",
        "seller_concentration_pct",
    ],
    None,
)


# ============================================================
# KPI ROW
# ============================================================

cols = st.columns(6)

with cols[0]:

    metric_card(
        "Sellers",
        fmt_number(total_sellers),
        "Unique sellers",
    )

with cols[1]:

    metric_card(
        "Markets",
        fmt_number(total_markets),
        "Markets detected",
    )

with cols[2]:

    metric_card(
        "Regions",
        fmt_number(total_regions),
        "Geographic coverage",
    )

with cols[3]:

    metric_card(
        "Revenue",
        fmt_currency(market_revenue),
        "Market-generated revenue",
    )

with cols[4]:

    metric_card(
        "Profit",
        fmt_currency(market_profit),
        "Market-generated profit",
    )

with cols[5]:

    metric_card(
        "Top Seller Share",
        fmt_percent(top_seller_share),
        "Revenue concentration",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_sellers, tab_markets, tab_geography, tab_opportunities = st.tabs(
    [
        "Overview",
        "Seller Intelligence",
        "Market Comparison",
        "Geographic Intelligence",
        "Opportunities",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Market & Seller Overview</div>',
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
                    "Sellers",
                    "Markets",
                    "Regions",
                    "Revenue",
                    "Profit",
                    "Top Seller Revenue Share",
                ],
                "Value": [
                    fmt_number(total_sellers),
                    fmt_number(total_markets),
                    fmt_number(total_regions),
                    fmt_currency(market_revenue),
                    fmt_currency(market_profit),
                    fmt_percent(top_seller_share),
                ],
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # CONCENTRATION
    # --------------------------------------------------------

    with col2:

        concentration = results.get(
            "concentration",
            {},
        )

        if isinstance(concentration, dict):

            concentration_rows = []

            for key, value in concentration.items():

                if isinstance(value, (int, float)):

                    label = (
                        str(key)
                        .replace("_", " ")
                        .title()
                    )

                    concentration_rows.append(
                        {
                            "Metric": label,
                            "Value": value,
                        }
                    )

            if concentration_rows:

                concentration_df = pd.DataFrame(
                    concentration_rows
                )

                st.markdown("### Market Concentration")

                st.dataframe(
                    concentration_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "Seller concentration metrics are unavailable."
                )

        else:

            st.info(
                "Seller concentration analysis is unavailable."
            )


# ============================================================
# SELLER INTELLIGENCE
# ============================================================

with tab_sellers:

    st.markdown(
        '<div class="section-title">Seller Performance</div>',
        unsafe_allow_html=True,
    )

    seller_df = first_existing(
        results,
        [
            "seller_performance",
            "sellers",
            "seller_analysis",
        ],
        None,
    )

    seller_df = clean_df(
        seller_df
    )

    if seller_df is None:

        st.info(
            "Seller-level performance data is not available."
        )

    else:

        seller_column = find_column(
            seller_df,
            [
                "seller_name",
                "seller_id",
                "seller",
                "vendor",
            ],
        )

        revenue_column = find_column(
            seller_df,
            [
                "revenue",
                "sales",
            ],
        )

        profit_column = find_column(
            seller_df,
            [
                "profit",
            ],
        )

        orders_column = find_column(
            seller_df,
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

            if seller_column and revenue_column:

                chart_df = (
                    seller_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=revenue_column,
                    y=seller_column,
                    orientation="h",
                    title="Top Sellers by Revenue",
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
                    "Seller revenue data is unavailable."
                )

        # ----------------------------------------------------
        # PROFIT
        # ----------------------------------------------------

        with col2:

            if seller_column and profit_column:

                chart_df = (
                    seller_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=profit_column,
                    y=seller_column,
                    orientation="h",
                    title="Top Sellers by Profit",
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
                    "Seller profit data is unavailable."
                )

        # ----------------------------------------------------
        # ORDERS
        # ----------------------------------------------------

        if seller_column and orders_column:

            chart_df = (
                seller_df
                .sort_values(
                    orders_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=orders_column,
                y=seller_column,
                orientation="h",
                title="Top Sellers by Orders",
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

        st.markdown("### Seller Performance Table")

        st.dataframe(
            seller_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MARKET COMPARISON
# ============================================================

with tab_markets:

    st.markdown(
        '<div class="section-title">Market Comparison</div>',
        unsafe_allow_html=True,
    )

    market_df = first_existing(
        results,
        [
            "market_performance",
            "markets",
            "market_analysis",
        ],
        None,
    )

    market_df = clean_df(
        market_df
    )

    if market_df is None:

        st.info(
            "Market-level performance data is not available."
        )

    else:

        market_column = find_column(
            market_df,
            [
                "market",
                "market_name",
                "market_id",
            ],
        )

        revenue_column = find_column(
            market_df,
            [
                "revenue",
                "sales",
            ],
        )

        profit_column = find_column(
            market_df,
            [
                "profit",
            ],
        )

        orders_column = find_column(
            market_df,
            [
                "orders",
                "order_count",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if market_column and revenue_column:

                chart_df = (
                    market_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=market_column,
                    y=revenue_column,
                    title="Revenue by Market",
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
                    "Market revenue data is unavailable."
                )

        with col2:

            if market_column and profit_column:

                chart_df = (
                    market_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    chart_df,
                    x=market_column,
                    y=profit_column,
                    title="Profit by Market",
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
                    "Market profit data is unavailable."
                )

        if market_column and orders_column:

            fig = px.bar(
                market_df
                .sort_values(
                    orders_column,
                    ascending=False,
                )
                .head(15),
                x=market_column,
                y=orders_column,
                title="Orders by Market",
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

        st.markdown("### Market Performance Table")

        st.dataframe(
            market_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# GEOGRAPHIC INTELLIGENCE
# ============================================================

with tab_geography:

    st.markdown(
        '<div class="section-title">Geographic Performance</div>',
        unsafe_allow_html=True,
    )

    geographic_df = first_existing(
        results,
        [
            "geographic_performance",
            "regional_performance",
            "region_performance",
            "geography",
        ],
        None,
    )

    geographic_df = clean_df(
        geographic_df
    )

    if geographic_df is None:

        st.info(
            "Geographic performance data is not available."
        )

    else:

        geographic_column = find_column(
            geographic_df,
            [
                "region",
                "state",
                "city",
                "country",
                "location",
                "geography",
            ],
        )

        revenue_column = find_column(
            geographic_df,
            [
                "revenue",
                "sales",
            ],
        )

        profit_column = find_column(
            geographic_df,
            [
                "profit",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if geographic_column and revenue_column:

                chart_df = (
                    geographic_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(20)
                )

                fig = px.bar(
                    chart_df,
                    x=revenue_column,
                    y=geographic_column,
                    orientation="h",
                    title="Revenue by Geography",
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
                    "Geographic revenue data is unavailable."
                )

        with col2:

            if geographic_column and profit_column:

                chart_df = (
                    geographic_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(20)
                )

                fig = px.bar(
                    chart_df,
                    x=profit_column,
                    y=geographic_column,
                    orientation="h",
                    title="Profit by Geography",
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
                    "Geographic profit data is unavailable."
                )

        st.markdown("### Geographic Performance Table")

        st.dataframe(
            geographic_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# OPPORTUNITIES
# ============================================================

with tab_opportunities:

    st.markdown(
        '<div class="section-title">Regional & Market Opportunities</div>',
        unsafe_allow_html=True,
    )

    opportunity_df = first_existing(
        results,
        [
            "regional_opportunities",
            "market_opportunities",
            "opportunities",
            "growth_opportunities",
        ],
        None,
    )

    opportunity_df = clean_df(
        opportunity_df
    )

    if opportunity_df is not None:

        numeric_columns = opportunity_df.select_dtypes(
            include="number"
        ).columns.tolist()

        opportunity_column = None

        for column in opportunity_df.columns:

            normalized = str(column).lower().replace(
                " ",
                "_",
            )

            if any(
                keyword in normalized
                for keyword in [
                    "opportunity",
                    "score",
                    "potential",
                    "growth",
                ]
            ):

                if column in numeric_columns:

                    opportunity_column = column
                    break

        category_column = None

        for column in opportunity_df.columns:

            if column == opportunity_column:
                continue

            normalized = str(column).lower().replace(
                " ",
                "_",
            )

            if any(
                keyword in normalized
                for keyword in [
                    "region",
                    "market",
                    "state",
                    "city",
                    "seller",
                    "product",
                ]
            ):

                category_column = column
                break

        if category_column and opportunity_column:

            chart_df = (
                opportunity_df
                .sort_values(
                    opportunity_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=opportunity_column,
                y=category_column,
                orientation="h",
                title="Highest-Potential Opportunities",
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

        st.markdown("### Opportunity Table")

        st.dataframe(
            opportunity_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "The current dataset does not contain enough information "
            "to generate a detailed opportunity table."
        )


# ============================================================
# PRODUCT-REGION PERFORMANCE
# ============================================================

product_region_df = first_existing(
    results,
    [
        "product_region_performance",
    ],
    None,
)

product_region_df = clean_df(
    product_region_df
)

if product_region_df is not None:

    st.markdown(
        '<div class="section-title">Product × Region Performance</div>',
        unsafe_allow_html=True,
    )

    product_column = find_column(
        product_region_df,
        [
            "product",
            "sku",
        ],
    )

    region_column = find_column(
        product_region_df,
        [
            "region",
            "state",
            "city",
            "market",
        ],
    )

    revenue_column = find_column(
        product_region_df,
        [
            "revenue",
            "sales",
        ],
    )

    if (
        product_column
        and region_column
        and revenue_column
    ):

        pivot_df = product_region_df.pivot_table(
            index=product_column,
            columns=region_column,
            values=revenue_column,
            aggfunc="sum",
            fill_value=0,
        )

        st.dataframe(
            pivot_df,
            use_container_width=True,
        )

    else:

        st.dataframe(
            product_region_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MARKET INTELLIGENCE SIGNALS
# ============================================================

st.markdown(
    '<div class="section-title">Market Intelligence Signals</div>',
    unsafe_allow_html=True,
)

insights = first_existing(
    results,
    [
        "insights",
        "market_insights",
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
                "Market Signal",
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

            title = "Market Signal"
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
    # SELLER CONCENTRATION
    # --------------------------------------------------------

    try:

        if top_seller_share is not None:

            share = float(
                top_seller_share
            )

            if share >= 50:

                generated_signals.append(
                    (
                        "High Seller Concentration",
                        f"The leading seller contributes approximately "
                        f"{share:.1f}% of revenue. Business dependency "
                        "on a small number of sellers should be monitored.",
                    )
                )

            elif share < 20:

                generated_signals.append(
                    (
                        "Diversified Seller Base",
                        f"The leading seller contributes approximately "
                        f"{share:.1f}% of revenue, suggesting a relatively "
                        "diversified seller base.",
                    )
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # MARKET COVERAGE
    # --------------------------------------------------------

    try:

        if float(total_markets) > 0:

            generated_signals.append(
                (
                    "Market Coverage",
                    f"DataPulse identified approximately "
                    f"{float(total_markets):,.0f} distinct markets "
                    "in the analysis dataset.",
                )
            )

    except Exception:
        pass

    if not generated_signals:

        generated_signals.append(
            (
                "Market Analysis Ready",
                "Seller, geographic and market-level performance "
                "can be combined to identify concentration risks "
                "and expansion opportunities.",
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
