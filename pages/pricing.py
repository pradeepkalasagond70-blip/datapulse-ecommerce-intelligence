from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="DataPulse | Pricing & Discount Intelligence",
    page_icon="◇",
    layout="wide",
)


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str = "—") -> str:
    if value is None:
        return default

    value = str(value).strip()

    return value if value else default


def _esc(value: Any, default: str = "—") -> str:
    return html.escape(_text(value, default))


def _df(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, list):
        try:
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()

    if isinstance(value, dict):
        try:
            if value and all(
                not isinstance(
                    v,
                    (list, tuple, pd.Series, pd.Index),
                )
                for v in value.values()
            ):
                return pd.DataFrame([value])

            return pd.DataFrame(value)

        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def _money(value: Any) -> str:
    value = _safe_float(value)

    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f}K"

    return f"₹{value:,.0f}"


def _pct(value: Any) -> str:
    return f"{_safe_float(value):.1f}%"


def _render_html(content: str) -> None:
    """
    Render one complete HTML block.

    Important:
    Never split an opening <div> and closing </div>
    between separate Streamlit calls.
    """
    st.html(content)


# ============================================================
# PAGE CSS
# ============================================================

_render_html(
    """
<style>

.dp-eyebrow {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .12em;
    color: #64748B;
    margin-bottom: 7px;
}

.dp-title {
    font-size: 30px;
    line-height: 1.15;
    font-weight: 850;
    color: #0F172A;
    margin-bottom: 7px;
}

.dp-subtitle {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 22px;
}

.dp-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
}

.dp-card-title {
    color: #0F172A;
    font-size: 16px;
    font-weight: 800;
    margin-bottom: 5px;
}

.dp-card-subtitle {
    color: #64748B;
    font-size: 13px;
    line-height: 1.5;
}

.dp-kpi-grid {
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 8px 0 20px 0;
}

.dp-kpi {
    box-sizing: border-box;
    width: 100%;
    min-width: 0;
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
    box-shadow: 0 5px 18px rgba(15,23,42,0.035);
}

.dp-kpi::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: #CBD5E1;
}

.dp-kpi-revenue {
    background: #F0FDF4;
    border-color: #BBF7D0;
}

.dp-kpi-revenue::before {
    background: #16A34A;
}

.dp-kpi-orders {
    background: #EFF6FF;
    border-color: #BFDBFE;
}

.dp-kpi-orders::before {
    background: #2563EB;
}

.dp-kpi-customers {
    background: #F5F3FF;
    border-color: #DDD6FE;
}

.dp-kpi-customers::before {
    background: #7C3AED;
}

.dp-kpi-profit {
    background: #FFFBEB;
    border-color: #FDE68A;
}

.dp-kpi-profit::before {
    background: #D97706;
}

.dp-kpi-aov {
    background: #ECFEFF;
    border-color: #A5F3FC;
}

.dp-kpi-aov::before {
    background: #0891B2;
}

.dp-kpi-growth {
    background: #F0FDFA;
    border-color: #99F6E4;
}

.dp-kpi-growth::before {
    background: #0D9488;
}

.dp-kpi-risk {
    background: #FEF2F2;
    border-color: #FECACA;
}

.dp-kpi-risk::before {
    background: #DC2626;
}

.dp-kpi-label {
    color: #64748B;
    font-size: 10px;
    font-weight: 650;
    line-height: 1.2;
    letter-spacing: .06em;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.dp-kpi-value {
    color: #0F172A;
    font-size: 25px;
    font-weight: 850;
    line-height: 1.1;
    margin-top: 10px;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.dp-kpi-note {
    color: #94A3B8;
    font-size: 10px;
    line-height: 1.25;
    margin-top: auto;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.dp-action {
    border-left: 4px solid #2563EB;
    background: #F8FAFC;

    border-radius: 10px;

    padding: 13px 14px;
    margin: 9px 0;
}

.dp-action-title {
    font-size: 13px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 4px;
}

.dp-action-text {
    font-size: 12px;
    color: #475569;
    line-height: 1.5;
}

.dp-muted {
    color: #94A3B8;
    font-size: 12px;
}

.dp-warning {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 12px 14px;
    color: #92400E;
    font-size: 12px;
    margin-bottom: 10px;
}

@media (max-width: 900px) {

    .dp-kpi-grid {
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }

}

</style>
"""
)


# ============================================================
# SESSION DATA
# ============================================================

analytics_results = st.session_state.get(
    "analytics_results",
    {},
)

pricing = analytics_results.get(
    "pricing",
    {},
)

if not isinstance(pricing, dict):
    pricing = {}


dataset_name = st.session_state.get(
    "dataset_name"
)

available = bool(
    pricing.get("available", False)
)


# ============================================================
# HEADER
# ============================================================

_render_html(
    """
<div class="dp-eyebrow">
    DATAPULSE INTELLIGENCE
</div>

<div class="dp-title">
    Pricing &amp; Discount Intelligence
</div>

<div class="dp-subtitle">
    Pricing performance, discount effectiveness,
    profitability risk and optimization opportunities.
</div>
"""
)


# ============================================================
# DATASET CHECK
# ============================================================

if not dataset_name:

    _render_html(
        """
<div class="dp-card">

    <div class="dp-card-title">
        Dataset Required
    </div>

    <div class="dp-card-subtitle">
        Connect an analysis-ready dataset from the
        Overview page before using Pricing &amp;
        Discount Intelligence.
    </div>

</div>
"""
    )

    st.stop()


# ============================================================
# AVAILABILITY CHECK
# ============================================================

if not available:

    warnings = pricing.get(
        "warnings",
        [],
    )

    _render_html(
        f"""
<div class="dp-card">

    <div class="dp-card-title">
        Pricing intelligence is unavailable
    </div>

    <div class="dp-card-subtitle">
        The connected dataset does not currently contain
        enough compatible pricing, discount or revenue
        fields for this module.
    </div>

    <div
        class="dp-muted"
        style="margin-top:12px;"
    >
        Dataset: {_esc(dataset_name)}
    </div>

</div>
"""
    )

    for warning in warnings:
        st.warning(str(warning))

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

summary = pricing.get(
    "summary",
    {},
)

if not isinstance(summary, dict):
    summary = {}


total_revenue = summary.get(
    "total_revenue",
    summary.get(
        "revenue",
        0,
    ),
)

avg_discount = summary.get(
    "average_discount",
    summary.get(
        "avg_discount",
        summary.get(
            "discount_rate",
            0,
        ),
    ),
)

discounted_revenue = summary.get(
    "discounted_revenue",
    0,
)

profit_impact = summary.get(
    "profit_impact",
    summary.get(
        "discount_profit_impact",
        0,
    ),
)


# ============================================================
# KPI CARDS
# ============================================================

_render_html(
    f"""
<div class="dp-kpi-grid">

    <div class="dp-kpi dp-kpi-revenue">

        <div class="dp-kpi-label">
            TOTAL REVENUE
        </div>

        <div class="dp-kpi-value">
            {_money(total_revenue)}
        </div>

        <div class="dp-kpi-note">
            Revenue analyzed
        </div>

    </div>


    <div class="dp-kpi">

        <div class="dp-kpi-label">
            AVERAGE DISCOUNT
        </div>

        <div class="dp-kpi-value">
            {_pct(avg_discount)}
        </div>

        <div class="dp-kpi-note">
            Average discount level
        </div>

    </div>


    <div class="dp-kpi dp-kpi-revenue">

        <div class="dp-kpi-label">
            DISCOUNTED REVENUE
        </div>

        <div class="dp-kpi-value">
            {_money(discounted_revenue)}
        </div>

        <div class="dp-kpi-note">
            Revenue from discounted sales
        </div>

    </div>


    <div class="dp-kpi dp-kpi-profit">

        <div class="dp-kpi-label">
            PROFIT IMPACT
        </div>

        <div class="dp-kpi-value">
            {_money(profit_impact)}
        </div>

        <div class="dp-kpi-note">
            Estimated discount impact
        </div>

    </div>

</div>
"""
)


# ============================================================
# DISCOUNT BANDS
# ============================================================

discount_bands = _df(
    pricing.get(
        "discount_bands"
    )
)

product_pricing = _df(
    pricing.get(
        "product_pricing"
    )
)

category_pricing = _df(
    pricing.get(
        "category_pricing"
    )
)

channel_pricing = _df(
    pricing.get(
        "channel_pricing"
    )
)


left, right = st.columns(
    2,
    gap="large",
)


# ============================================================
# DISCOUNT BAND ANALYSIS
# ============================================================

with left:

    _render_html(
        """
<div class="dp-card">

    <div class="dp-card-title">
        Discount Band Performance
    </div>

    <div class="dp-card-subtitle">
        Revenue and business performance across
        discount levels.
    </div>

</div>
"""
    )

    if not discount_bands.empty:

        numeric_candidates = [
            col
            for col in discount_bands.columns
            if pd.api.types.is_numeric_dtype(
                discount_bands[col]
            )
        ]

        if len(discount_bands.columns) >= 2:

            x_col = discount_bands.columns[0]

            y_col = (
                numeric_candidates[-1]
                if numeric_candidates
                else discount_bands.columns[-1]
            )

            fig = px.bar(
                discount_bands,
                x=x_col,
                y=y_col,
                template="plotly_white",
            )

            fig.update_layout(
                height=340,
                margin=dict(
                    l=10,
                    r=10,
                    t=15,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                },
            )

            st.dataframe(
                discount_bands,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.dataframe(
                discount_bands,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "Discount band analysis is not available."
        )


# ============================================================
# CATEGORY PRICING
# ============================================================

with right:

    _render_html(
        """
<div class="dp-card">

    <div class="dp-card-title">
        Category Pricing Intelligence
    </div>

    <div class="dp-card-subtitle">
        Pricing and discount performance by category.
    </div>

</div>
"""
    )

    if not category_pricing.empty:

        st.dataframe(
            category_pricing,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Category pricing analysis is not available."
        )


# ============================================================
# PRODUCT PRICING
# ============================================================

_render_html(
    """
<div class="dp-card">

    <div class="dp-card-title">
        Product Pricing Intelligence
    </div>

    <div class="dp-card-subtitle">
        Identify products with strong pricing performance,
        discount pressure or profitability concerns.
    </div>

</div>
"""
)

if not product_pricing.empty:

    st.dataframe(
        product_pricing,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Product-level pricing intelligence is not available."
    )


# ============================================================
# CHANNEL PRICING
# ============================================================

_render_html(
    """
<div class="dp-card">

    <div class="dp-card-title">
        Channel Pricing Performance
    </div>

    <div class="dp-card-subtitle">
        Compare pricing and discount behavior across
        sales channels.
    </div>

</div>
"""
)

if not channel_pricing.empty:

    st.dataframe(
        channel_pricing,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Channel pricing analysis is not available."
    )


# ============================================================
# DISCOUNT IMPACT
# ============================================================

discount_impact = _df(
    pricing.get(
        "discount_impact"
    )
)

discount_risk = _df(
    pricing.get(
        "discount_risk"
    )
)


left, right = st.columns(
    2,
    gap="large",
)


with left:

    _render_html(
        """
<div class="dp-card">

    <div class="dp-card-title">
        Discount Impact
    </div>

    <div class="dp-card-subtitle">
        Business impact associated with discounting.
    </div>

</div>
"""
    )

    if not discount_impact.empty:

        st.dataframe(
            discount_impact,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Discount impact analysis is not available."
        )


with right:

    _render_html(
        """
<div class="dp-card">

    <div class="dp-card-title">
        Discount Risk
    </div>

    <div class="dp-card-subtitle">
        Products, categories or pricing patterns
        showing potential discount risk.
    </div>

</div>
"""
    )

    if not discount_risk.empty:

        st.dataframe(
            discount_risk,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Discount risk analysis is not available."
        )


# ============================================================
# PRICING OPPORTUNITIES
# ============================================================

opportunities = pricing.get(
    "pricing_opportunities",
    [],
)

if isinstance(
    opportunities,
    pd.DataFrame,
):

    opportunity_df = opportunities

else:

    opportunity_df = _df(
        opportunities
    )


_render_html(
    """
<div class="dp-card">

    <div class="dp-card-title">
        Pricing Opportunities
    </div>

    <div class="dp-card-subtitle">
        Potential opportunities to improve pricing,
        margin and discount efficiency.
    </div>

</div>
"""
)


if not opportunity_df.empty:

    st.dataframe(
        opportunity_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No structured pricing opportunities are available."
    )


# ============================================================
# ACTION RECOMMENDATIONS
# ============================================================

recommendations = pricing.get(
    "recommendations",
    [],
)


_render_html(
    """
<div class="dp-card">

    <div class="dp-card-title">
        Pricing Actions
    </div>

    <div class="dp-card-subtitle">
        Recommended actions generated from pricing
        and discount intelligence.
    </div>

</div>
"""
)


if isinstance(
    recommendations,
    dict,
):

    recommendations = list(
        recommendations.values()
    )


if recommendations:

    for index, item in enumerate(
        recommendations[:15],
        start=1,
    ):

        if isinstance(
            item,
            dict,
        ):

            title = item.get(
                "title",
                item.get(
                    "action",
                    item.get(
                        "recommendation",
                        f"Pricing Action {index}",
                    ),
                ),
            )

            body = item.get(
                "description",
                item.get(
                    "reason",
                    item.get(
                        "rationale",
                        item.get(
                            "details",
                            "",
                        ),
                    ),
                ),
            )

        else:

            title = (
                f"Pricing Action {index}"
            )

            body = item


        _render_html(
            f"""
<div class="dp-action">

    <div class="dp-action-title">
        {_esc(title)}
    </div>

    <div class="dp-action-text">
        {_esc(body)}
    </div>

</div>
"""
        )

else:

    st.info(
        "No pricing actions were generated."
    )


# ============================================================
# CAPABILITIES
# ============================================================

capabilities = pricing.get(
    "capabilities",
    [],
)

if capabilities:

    _render_html(
        f"""
<div class="dp-card">

    <div class="dp-card-title">
        Pricing Intelligence Coverage
    </div>

    <div class="dp-card-subtitle">
        {_esc(", ".join(map(str, capabilities)))}
    </div>

</div>
"""
    )


# ============================================================
# WARNINGS
# ============================================================

warnings = pricing.get(
    "warnings",
    [],
)

if warnings:

    for warning in warnings:

        _render_html(
            f"""
<div class="dp-warning">
    {_esc(warning)}
</div>
"""
        )


# ============================================================
# FOOTER
# ============================================================

_render_html(
    """
<div
    style="
        text-align:center;
        color:#94A3B8;
        font-size:11px;
        padding:18px 0 8px 0;
    "
>
    DataPulse · Pricing &amp; Discount Intelligence
</div>
"""
)
