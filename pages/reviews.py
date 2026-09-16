
from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="DataPulse | Review & NLP Analytics",
    page_icon="▤",
    layout="wide",
)


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_number(value: Any) -> str:
    return f"{_safe_float(value):,.0f}"


def _fmt_pct(value: Any) -> str:
    return f"{_safe_float(value):.1f}%"


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
            # Dict of scalar values is a record, not columns.
            if value and all(not isinstance(v, (list, tuple, pd.Series, pd.Index)) for v in value.values()):
                return pd.DataFrame([value])
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _render_html(content: str) -> None:
    # IMPORTANT: every HTML block is complete and is rendered with st.html.
    # Never split an opening/closing div across separate Streamlit calls.
    st.html(content)


# ============================================================
# PAGE CSS
# ============================================================

_render_html(
    """
<style>
.dp-page {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
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
    grid-template-columns: repeat(4, minmax(0, 1fr));
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
.dp-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 800;
    background: #EFF6FF;
    color: #2563EB;
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
@media (max-width: 900px) {
    .dp-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 768px) {
    .main .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0 !important; }
    [data-testid="stColumn"], .stColumn { width: 100% !important; max-width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
    [data-testid="stColumn"] > div, .stColumn > div { width: 100% !important; max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
    .dp-title { font-size: 25px; line-height: 1.12; overflow-wrap: anywhere; }
    .dp-subtitle, .dp-card-subtitle, .dp-action-title, .dp-action-text, .dp-muted { overflow-wrap: anywhere; }
    .dp-card { padding: 16px; }
    .dp-kpi-grid { grid-template-columns: 1fr; gap: 10px; }
    .dp-kpi { height: auto; min-height: 108px; padding: 15px; }
    .dp-kpi-label, .dp-kpi-value { white-space: normal; overflow-wrap: anywhere; }
    .dp-kpi-value { font-size: 20px; }
    [data-testid="stPlotlyChart"] { max-width: 100% !important; overflow: hidden !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stDataEditor"] { max-width: 100% !important; min-width: 0 !important; overflow-x: auto !important; }
    [data-testid="stPlotlyChart"] > div, [data-testid="stPlotlyChart"] iframe { width: 100% !important; max-width: 100% !important; }
    input, textarea, select, [data-baseweb="select"] { max-width: 100% !important; min-width: 0 !important; box-sizing: border-box !important; }
    button { min-height: 44px; }
}
</style>
"""
)


# ============================================================
# LOAD SESSION DATA
# ============================================================

analytics_results = st.session_state.get("analytics_results", {})
reviews = analytics_results.get("reviews", {})

if not isinstance(reviews, dict):
    reviews = {}

dataset_name = st.session_state.get("dataset_name")
available = bool(reviews.get("available", False))


# ============================================================
# HEADER
# ============================================================

_render_html(
    f"""
<div class="dp-page">
    <div class="dp-eyebrow">DATAPULSE INTELLIGENCE</div>
    <div class="dp-title">Review &amp; NLP Analytics</div>
    <div class="dp-subtitle">
        Customer review intelligence, sentiment, themes and experience actions.
    </div>
</div>
"""
)


if not dataset_name:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Dataset Required</div>
    <div class="dp-card-subtitle">
        Connect an analysis-ready dataset from the Overview page before using
        Review &amp; NLP Analytics.
    </div>
</div>
"""
    )
    st.stop()


if not available:
    warnings = reviews.get("warnings", [])
    _render_html(
        f"""
<div class="dp-card">
    <div class="dp-card-title">Review intelligence is unavailable</div>
    <div class="dp-card-subtitle">
        The uploaded dataset does not currently contain enough compatible
        review/rating fields for this module.
    </div>
    <div style="margin-top:12px" class="dp-muted">
        Dataset: {_esc(dataset_name)}
    </div>
</div>
"""
    )
    if warnings:
        for warning in warnings:
            st.warning(str(warning))
    st.stop()


# ============================================================
# SUMMARY
# ============================================================

summary = reviews.get("summary", {})
if not isinstance(summary, dict):
    summary = {}

total_reviews = _safe_int(
    summary.get("total_reviews", summary.get("review_count", 0))
)
avg_rating = _safe_float(
    summary.get("average_rating", summary.get("avg_rating", 0))
)
positive = _safe_float(
    summary.get(
        "positive_sentiment_pct",
        summary.get(
            "positive_pct",
            summary.get("positive_rate", 0),
        ),
    )
)
negative = _safe_float(
    summary.get(
        "negative_sentiment_pct",
        summary.get(
            "negative_pct",
            summary.get("negative_rate", 0),
        ),
    )
)

_render_html(
    f"""
<div class="dp-kpi-grid">
    <div class="dp-kpi">
        <div class="dp-kpi-label">TOTAL REVIEWS</div>
        <div class="dp-kpi-value">{_fmt_number(total_reviews)}</div>
        <div class="dp-kpi-note">Analyzed customer reviews</div>
    </div>
    <div class="dp-kpi dp-kpi-customers">
        <div class="dp-kpi-label">AVERAGE RATING</div>
        <div class="dp-kpi-value">{avg_rating:.1f} / 5</div>
        <div class="dp-kpi-note">Overall customer rating</div>
    </div>
    <div class="dp-kpi dp-kpi-growth">
        <div class="dp-kpi-label">POSITIVE SENTIMENT</div>
        <div class="dp-kpi-value">{_fmt_pct(positive)}</div>
        <div class="dp-kpi-note">Share of positive feedback</div>
    </div>
    <div class="dp-kpi dp-kpi-risk">
        <div class="dp-kpi-label">NEGATIVE SENTIMENT</div>
        <div class="dp-kpi-value">{_fmt_pct(negative)}</div>
        <div class="dp-kpi-note">Share requiring attention</div>
    </div>
</div>
"""
)


# ============================================================
# DISTRIBUTIONS
# ============================================================

rating_df = _df(reviews.get("rating_distribution"))
sentiment_df = _df(reviews.get("sentiment_distribution"))

left, right = st.columns(2, gap="large")

with left:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Rating Distribution</div>
    <div class="dp-card-subtitle">How customers rate the overall experience.</div>
</div>
"""
    )
    if not rating_df.empty:
        st.plotly_chart(
            px.bar(
                rating_df,
                x=rating_df.columns[0],
                y=rating_df.columns[-1],
                template="plotly_white",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Rating distribution is not available.")

with right:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Sentiment Distribution</div>
    <div class="dp-card-subtitle">Customer feedback grouped by sentiment.</div>
</div>
"""
    )
    if not sentiment_df.empty:
        st.plotly_chart(
            px.bar(
                sentiment_df,
                x=sentiment_df.columns[0],
                y=sentiment_df.columns[-1],
                template="plotly_white",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.info("Sentiment distribution is not available.")


# ============================================================
# TREND + MATRIX
# ============================================================

trend_df = _df(reviews.get("review_trend"))
matrix_df = _df(reviews.get("rating_sentiment_matrix"))

left, right = st.columns(2, gap="large")

with left:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Review Trend</div>
    <div class="dp-card-subtitle">Review volume and customer feedback over time.</div>
</div>
"""
    )
    if not trend_df.empty:
        x_col = trend_df.columns[0]
        y_col = trend_df.columns[-1]
        fig = px.line(
            trend_df,
            x=x_col,
            y=y_col,
            markers=True,
            template="plotly_white",
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Review trend is not available.")

with right:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Rating × Sentiment</div>
    <div class="dp-card-subtitle">Relationship between ratings and detected sentiment.</div>
</div>
"""
    )
    if not matrix_df.empty:
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    else:
        st.info("Rating × sentiment analysis is not available.")


# ============================================================
# KEYWORDS + NEGATIVE THEMES
# ============================================================

keywords = _df(reviews.get("keywords"))
positive_keywords = _df(reviews.get("positive_keywords"))
negative_keywords = _df(reviews.get("negative_keywords"))
negative_themes = _df(reviews.get("negative_themes"))

_render_html(
    """
<div class="dp-card">
    <div class="dp-card-title">Customer Voice</div>
    <div class="dp-card-subtitle">
        The words and themes appearing most often in customer feedback.
    </div>
</div>
"""
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Top Keywords", "Positive Keywords", "Negative Keywords", "Negative Themes"]
)

with tab1:
    if not keywords.empty:
        st.dataframe(keywords, use_container_width=True, hide_index=True)
    else:
        st.info("Top keywords are not available.")

with tab2:
    if not positive_keywords.empty:
        st.dataframe(positive_keywords, use_container_width=True, hide_index=True)
    else:
        st.info("Positive keywords are not available.")

with tab3:
    if not negative_keywords.empty:
        st.dataframe(negative_keywords, use_container_width=True, hide_index=True)
    else:
        st.info("Negative keywords are not available.")

with tab4:
    if not negative_themes.empty:
        st.dataframe(negative_themes, use_container_width=True, hide_index=True)
    else:
        st.info("Negative themes are not available.")


# ============================================================
# PRODUCT / CATEGORY REVIEW INTELLIGENCE
# ============================================================

product_reviews = _df(reviews.get("product_reviews"))
category_reviews = _df(reviews.get("category_reviews"))

left, right = st.columns(2, gap="large")

with left:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Product Review Intelligence</div>
    <div class="dp-card-subtitle">
        Products receiving the strongest positive or negative customer signals.
    </div>
</div>
"""
    )
    if not product_reviews.empty:
        st.dataframe(product_reviews, use_container_width=True, hide_index=True)
    else:
        st.info("Product-level review intelligence is not available.")

with right:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Category Review Intelligence</div>
    <div class="dp-card-subtitle">
        Category-level customer experience performance.
    </div>
</div>
"""
    )
    if not category_reviews.empty:
        st.dataframe(category_reviews, use_container_width=True, hide_index=True)
    else:
        st.info("Category-level review intelligence is not available.")


# ============================================================
# REVIEW QUALITY + ACTIONS
# ============================================================

quality = reviews.get("review_quality", {})
recommendations = reviews.get("recommendations", [])

_render_html(
    """
<div class="dp-card">
    <div class="dp-card-title">Review Quality</div>
    <div class="dp-card-subtitle">
        Coverage and quality checks for the review intelligence layer.
    </div>
</div>
"""
)

if isinstance(quality, dict) and quality:
    quality_rows = []
    for key, value in quality.items():
        quality_rows.append(
            {"Metric": str(key).replace("_", " ").title(), "Value": value}
        )
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
else:
    st.info("Review quality metrics are not available.")

_render_html(
    """
<div class="dp-card">
    <div class="dp-card-title">Customer Experience Actions</div>
    <div class="dp-card-subtitle">
        Recommended actions derived from review and sentiment signals.
    </div>
</div>
"""
)

if isinstance(recommendations, dict):
    recommendations = list(recommendations.values())

if recommendations:
    for index, item in enumerate(recommendations[:12], start=1):
        if isinstance(item, dict):
            title = item.get("title", item.get("action", f"Action {index}"))
            body = item.get(
                "description",
                item.get("recommendation", item.get("reason", "")),
            )
        else:
            title = f"Action {index}"
            body = item

        _render_html(
            f"""
<div class="dp-action">
    <div class="dp-action-title">{_esc(title)}</div>
    <div class="dp-action-text">{_esc(body)}</div>
</div>
"""
        )
else:
    st.info("No review-driven actions were generated.")


# ============================================================
# CAPABILITIES / WARNINGS
# ============================================================

capabilities = reviews.get("capabilities", [])
warnings = reviews.get("warnings", [])

if capabilities:
    _render_html(
        f"""
<div class="dp-card">
    <div class="dp-card-title">NLP Coverage</div>
    <div class="dp-card-subtitle">
        {_esc(", ".join(map(str, capabilities)))}
    </div>
</div>
"""
    )

if warnings:
    for warning in warnings:
        st.warning(str(warning))

_render_html(
    """
<div style="text-align:center; color:#94A3B8; font-size:11px; padding:18px 0 8px 0;">
    DataPulse · Review &amp; NLP Intelligence
</div>
"""
)
