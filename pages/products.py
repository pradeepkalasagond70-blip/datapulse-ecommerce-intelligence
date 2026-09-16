# pages/products.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Product Intelligence | DataPulse",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .product-header {
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

    .product-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 750;
    }

    .product-header p {
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
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
        .product-header { padding: 15px 16px; }
        .product-header h1 { font-size: 24px; line-height: 1.15; overflow-wrap: anywhere; }
        .product-header p, .section-title, .insight-title, .insight-text { overflow-wrap: anywhere; }
        .metric-card { height: auto; min-height: 108px; padding: 15px; }
        .metric-label, .metric-value { white-space: normal; overflow-wrap: anywhere; }
        .metric-value { font-size: 20px; }
        [data-testid="stPlotlyChart"] { max-width: 100% !important; overflow: hidden !important; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { max-width: 100% !important; overflow-x: auto !important; }
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


def get_results():
    """Get Product Intelligence results from session state."""

    analytics_results = st.session_state.get(
        "analytics_results",
        {},
    )

    if not isinstance(analytics_results, dict):
        return {}

    return analytics_results.get(
        "product",
        {},
    ) or {}


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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="product-header">
        <h1>📦 Product Intelligence</h1>
        <p>
            Identify winning products, weak performers, category trends,
            profitability drivers and product-level growth opportunities.
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
        "DataPulse home page to activate Product Intelligence."
    )

    st.stop()


if results.get("available") is False:

    st.warning(
        "Product Intelligence is not available for the current dataset."
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


total_products = first_existing(
    summary,
    [
        "total_products",
        "products",
        "unique_products",
        "product_count",
    ],
    0,
)

total_categories = first_existing(
    summary,
    [
        "total_categories",
        "categories",
        "unique_categories",
        "category_count",
    ],
    0,
)

product_revenue = first_existing(
    summary,
    [
        "total_revenue",
        "total_product_revenue",
        "product_revenue",
        "revenue",
    ],
    0,
)

product_profit = first_existing(
    summary,
    [
        "total_profit",
        "total_product_profit",
        "product_profit",
        "profit",
    ],
    0,
)

units_sold = first_existing(
    summary,
    [
        "total_quantity",
        "total_product_quantity",
        "quantity",
        "units_sold",
    ],
    0,
)

average_product_revenue = first_existing(
    summary,
    [
        "average_product_revenue",
        "avg_product_revenue",
    ],
    None,
)

average_product_profit = first_existing(
    summary,
    [
        "average_product_profit",
        "avg_product_profit",
    ],
    None,
)


# ============================================================
# KPI ROW
# ============================================================

cols = st.columns(6)

with cols[0]:

    metric_card(
        "Products",
        fmt_number(total_products),
        "Unique products",
    )

with cols[1]:

    metric_card(
        "Categories",
        fmt_number(total_categories),
        "Product categories",
    )

with cols[2]:

    metric_card(
        "Revenue",
        fmt_currency(product_revenue),
        "Product-generated revenue",
    )

with cols[3]:

    metric_card(
        "Profit",
        fmt_currency(product_profit),
        "Product-generated profit",
    )

with cols[4]:

    metric_card(
        "Units Sold",
        fmt_number(units_sold),
        "Total quantity sold",
    )

with cols[5]:

    metric_card(
        "Avg Product Revenue",
        fmt_currency(average_product_revenue),
        "Average revenue per product",
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_winners, tab_categories, tab_profit, tab_explorer = st.tabs(
    [
        "Overview",
        "Winning Products",
        "Category Intelligence",
        "Profitability",
        "Product Explorer",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-title">Product Portfolio Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    with col1:

        overview_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Products",
                    "Total Categories",
                    "Revenue",
                    "Profit",
                    "Units Sold",
                    "Average Product Revenue",
                    "Average Product Profit",
                ],
                "Value": [
                    fmt_number(total_products),
                    fmt_number(total_categories),
                    fmt_currency(product_revenue),
                    fmt_currency(product_profit),
                    fmt_number(units_sold),
                    fmt_currency(average_product_revenue),
                    fmt_currency(average_product_profit),
                ],
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # PRODUCT DISTRIBUTION
    # --------------------------------------------------------

    with col2:

        distribution = first_existing(
            results,
            [
                "product_distribution",
                "category_distribution",
                "product_mix",
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
                    title="Product Portfolio Distribution",
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

                st.dataframe(
                    distribution,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.info(
                "Product distribution is not available."
            )


# ============================================================
# WINNING PRODUCTS
# ============================================================

with tab_winners:

    st.markdown(
        '<div class="section-title">Winning & Underperforming Products</div>',
        unsafe_allow_html=True,
    )

    product_df = first_existing(
        results,
        [
            "product_performance",
            "products",
            "top_products",
            "product_analysis",
        ],
        None,
    )

    product_df = clean_df(product_df)

    if product_df is None:

        st.info(
            "Product-level performance data is not available."
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

        # ----------------------------------------------------
        # TOP REVENUE PRODUCTS
        # ----------------------------------------------------

        with col1:

            if product_column and revenue_column:

                top_revenue = (
                    product_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    top_revenue,
                    x=revenue_column,
                    y=product_column,
                    orientation="h",
                    title="Top Products by Revenue",
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
                    "Product revenue information is unavailable."
                )

        # ----------------------------------------------------
        # TOP PROFIT PRODUCTS
        # ----------------------------------------------------

        with col2:

            if product_column and profit_column:

                top_profit = (
                    product_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                    .head(15)
                )

                fig = px.bar(
                    top_profit,
                    x=profit_column,
                    y=product_column,
                    orientation="h",
                    title="Top Products by Profit",
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
                    "Product profit information is unavailable."
                )

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        if product_column and quantity_column:

            volume_df = (
                product_df
                .sort_values(
                    quantity_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                volume_df,
                x=quantity_column,
                y=product_column,
                orientation="h",
                title="Top Products by Units Sold",
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


# ============================================================
# CATEGORY INTELLIGENCE
# ============================================================

with tab_categories:

    st.markdown(
        '<div class="section-title">Category Intelligence</div>',
        unsafe_allow_html=True,
    )

    category_df = first_existing(
        results,
        [
            "category_performance",
            "categories",
            "category_analysis",
            "category_sales",
        ],
        None,
    )

    category_df = clean_df(category_df)

    if category_df is None:

        st.info(
            "Category-level analysis is not available."
        )

    else:

        category_column = find_column(
            category_df,
            [
                "category",
                "product_category",
            ],
        )

        revenue_column = find_column(
            category_df,
            [
                "revenue",
                "sales",
            ],
        )

        profit_column = find_column(
            category_df,
            [
                "profit",
            ],
        )

        quantity_column = find_column(
            category_df,
            [
                "quantity",
                "units",
            ],
        )

        col1, col2 = st.columns(2)

        with col1:

            if category_column and revenue_column:

                chart_df = (
                    category_df
                    .sort_values(
                        revenue_column,
                        ascending=False,
                    )
                )

                fig = px.bar(
                    chart_df,
                    x=category_column,
                    y=revenue_column,
                    title="Revenue by Category",
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    xaxis_tickangle=-35,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Category revenue data is unavailable."
                )

        with col2:

            if category_column and profit_column:

                chart_df = (
                    category_df
                    .sort_values(
                        profit_column,
                        ascending=False,
                    )
                )

                fig = px.bar(
                    chart_df,
                    x=category_column,
                    y=profit_column,
                    title="Profit by Category",
                )

                fig.update_layout(
                    height=430,
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                    xaxis_tickangle=-35,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Category profit data is unavailable."
                )

        if category_column and quantity_column:

            fig = px.bar(
                category_df.sort_values(
                    quantity_column,
                    ascending=False,
                ),
                x=category_column,
                y=quantity_column,
                title="Units Sold by Category",
            )

            fig.update_layout(
                height=420,
                margin=dict(
                    l=10,
                    r=10,
                    t=50,
                    b=10,
                ),
                xaxis_tickangle=-35,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### Category Performance Table")

        st.dataframe(
            category_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PROFITABILITY
# ============================================================

with tab_profit:

    st.markdown(
        '<div class="section-title">Product Profitability</div>',
        unsafe_allow_html=True,
    )

    profitability_df = first_existing(
        results,
        [
            "profitability",
            "product_profitability",
            "margin_analysis",
            "product_performance",
        ],
        None,
    )

    profitability_df = clean_df(
        profitability_df
    )

    col1, col2 = st.columns(2)

    with col1:

        metric_card(
            "Product Revenue",
            fmt_currency(product_revenue),
            "Revenue generated by products",
        )

    with col2:

        metric_card(
            "Product Profit",
            fmt_currency(product_profit),
            "Profit generated by products",
        )

    if profitability_df is not None:

        product_column = find_column(
            profitability_df,
            [
                "product",
                "sku",
            ],
        )

        margin_column = find_column(
            profitability_df,
            [
                "margin",
                "profit_margin",
            ],
        )

        profit_column = find_column(
            profitability_df,
            [
                "profit",
            ],
        )

        if product_column and margin_column:

            chart_df = (
                profitability_df
                .sort_values(
                    margin_column,
                    ascending=False,
                )
                .head(15)
            )

            fig = px.bar(
                chart_df,
                x=margin_column,
                y=product_column,
                orientation="h",
                title="Highest-Margin Products",
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

        elif product_column and profit_column:

            chart_df = (
                profitability_df
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
                title="Most Profitable Products",
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

        st.markdown("### Product Profitability Table")

        st.dataframe(
            profitability_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Detailed product profitability data is not available."
        )


# ============================================================
# PRODUCT EXPLORER
# ============================================================

with tab_explorer:

    st.markdown(
        '<div class="section-title">Product Explorer</div>',
        unsafe_allow_html=True,
    )

    explorer_df = first_existing(
        results,
        [
            "product_performance",
            "products",
            "product_table",
            "product_analysis",
        ],
        None,
    )

    explorer_df = clean_df(explorer_df)

    if explorer_df is None:

        st.info(
            "Product explorer data is not available."
        )

    else:

        search = st.text_input(
            "Search products",
            placeholder="Search product, SKU or category...",
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

        numeric_columns = filtered_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            sort_column = st.selectbox(
                "Sort products by",
                numeric_columns,
            )

            filtered_df = filtered_df.sort_values(
                sort_column,
                ascending=False,
            )

        st.caption(
            f"Showing {len(filtered_df):,} product records."
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PRODUCT INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">Product Intelligence Signals</div>',
    unsafe_allow_html=True,
)

insights = first_existing(
    results,
    [
        "insights",
        "product_insights",
        "signals",
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
                "Product Signal",
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

            title = "Product Signal"
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
    # PRODUCT CONCENTRATION SIGNAL
    # --------------------------------------------------------

    try:

        if (
            product_df is not None
            and revenue_column
            and len(product_df) > 0
        ):

            total_revenue_value = product_df[
                revenue_column
            ].sum()

            top_5_revenue = (
                product_df
                .sort_values(
                    revenue_column,
                    ascending=False,
                )
                .head(5)[revenue_column]
                .sum()
            )

            if total_revenue_value > 0:

                concentration = (
                    top_5_revenue
                    / total_revenue_value
                    * 100
                )

                if concentration >= 60:

                    generated_signals.append(
                        (
                            "High Product Concentration",
                            f"The top five products contribute approximately "
                            f"{concentration:.1f}% of product revenue. "
                            "Revenue concentration should be monitored.",
                        )
                    )

    except Exception:
        pass

    # --------------------------------------------------------
    # PORTFOLIO SCALE
    # --------------------------------------------------------

    try:

        if float(total_products) > 0:

            generated_signals.append(
                (
                    "Product Portfolio",
                    f"DataPulse identified approximately "
                    f"{float(total_products):,.0f} products "
                    "across the analysis dataset.",
                )
            )

    except Exception:
        pass

    if not generated_signals:

        generated_signals.append(
            (
                "Product Analysis Ready",
                "Product performance can be connected with "
                "customer behaviour, pricing, campaigns, "
                "regional performance and forecasting.",
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
