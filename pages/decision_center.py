
from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="DataPulse | Business Decision Center",
    page_icon="◉",
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


def _is_html_fragment(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(re.fullmatch(r"</?[A-Za-z][^>]*>", text))


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
            if value and all(not isinstance(v, (list, tuple, pd.Series, pd.Index)) for v in value.values()):
                return pd.DataFrame([value])
            return pd.DataFrame(value)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _render_html(content: str) -> None:
    # Every HTML block is complete. Do not open a div in one Streamlit
    # call and close it in another.
    st.html(content)


# ============================================================
# CSS
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
.dp-decision-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}
.dp-decision-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 15px 16px;
}
.dp-decision-panel-title {
    color: #0F172A;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 7px;
}
.dp-decision-panel-body {
    color: #475569;
    font-size: 12px;
    line-height: 1.55;
}
.dp-decision-status {
    color: #0F172A;
    font-size: 20px;
    font-weight: 850;
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
.dp-recommendation {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    border-radius: 12px;
    padding: 15px 16px;
    margin-bottom: 11px;
}
.dp-rec-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 7px;
}
.dp-rec-rank {
    min-width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #EFF6FF;
    color: #2563EB;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 850;
}
.dp-rec-title {
    color: #0F172A;
    font-size: 14px;
    font-weight: 800;
}
.dp-rec-body {
    color: #475569;
    font-size: 12px;
    line-height: 1.55;
}
.dp-rec-meta {
    margin-top: 8px;
    color: #64748B;
    font-size: 11px;
}
.dp-status {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 9px;
    background: #F1F5F9;
    color: #475569;
    font-size: 10px;
    font-weight: 800;
}
@media (max-width: 900px) {
    .dp-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .dp-decision-grid { grid-template-columns: 1fr; }
}

@media (max-width: 768px) {
    .main .block-container { padding-left: .75rem !important; padding-right: .75rem !important; }
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; gap: 0 !important; }
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
    .dp-title { font-size: 25px; line-height: 1.12; overflow-wrap: anywhere; }
    .dp-subtitle, .dp-card-subtitle, .dp-decision-panel-body, .dp-rec-title, .dp-rec-body, .dp-rec-meta { overflow-wrap: anywhere; }
    .dp-card { padding: 16px; }
    .dp-kpi-grid { grid-template-columns: 1fr; gap: 10px; }
    .dp-kpi { height: auto; min-height: 108px; padding: 15px; }
    .dp-kpi-label, .dp-kpi-value { white-space: normal; overflow-wrap: anywhere; }
    .dp-kpi-value { font-size: 20px; }
    .dp-rec-head { align-items: flex-start; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { max-width: 100% !important; overflow-x: auto !important; }
    button { min-height: 44px; }
}
</style>
"""
)


# ============================================================
# SESSION DATA
# ============================================================

analytics_results = st.session_state.get("analytics_results", {})
insight_result = analytics_results.get("insights", {})

if not isinstance(insight_result, dict):
    insight_result = {}

dataset_name = st.session_state.get("dataset_name")
insights = insight_result.get("insights", [])
recommendations = insight_result.get("recommendations", [])
executive_summary = insight_result.get("executive_summary", "")
module_status = insight_result.get("module_status", {})
available_module_count = _safe_int(
    insight_result.get("available_module_count", 0)
)
total_module_count = _safe_int(
    insight_result.get("total_module_count", 0)
)


# ============================================================
# HEADER
# ============================================================

_render_html(
    f"""
<div class="dp-eyebrow">DATAPULSE DECISION INTELLIGENCE</div>
<div class="dp-title">Business Decision Center</div>
<div class="dp-subtitle">
    Convert analytics and machine-learning signals into prioritized business actions.
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
        the Business Decision Center.
    </div>
</div>
"""
    )
    st.stop()


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

summary_data = (
    executive_summary
    if isinstance(executive_summary, dict)
    else {}
)

business_status = _text(
    summary_data.get(
        "business_status"
    ),
    "Stable",
)

critical_count = _safe_int(
    summary_data.get(
        "critical_insights",
        0,
    )
)

high_priority_count = _safe_int(
    summary_data.get(
        "high_priority_insights",
        0,
    )
)

total_insight_count = _safe_int(
    summary_data.get(
        "total_insights",
        len(insights)
        if isinstance(insights, list)
        else 0,
    )
)

total_recommendation_count = _safe_int(
    summary_data.get(
        "total_recommendations",
        len(recommendations)
        if isinstance(recommendations, list)
        else 0,
    )
)

top_priorities = summary_data.get(
    "top_priorities",
    [],
)

if not isinstance(top_priorities, list):
    top_priorities = []

_render_html(
    f"""
<div class="dp-card">
    <div class="dp-card-title">Executive Decision Summary</div>
    <div class="dp-card-subtitle"></div>
</div>

<div class="dp-decision-grid">
    <div class="dp-decision-panel">
        <div class="dp-decision-panel-title">Business Status</div>
        <div class="dp-decision-status">{_esc(business_status)}</div>
        <div class="dp-decision-panel-body">
            Current priority level based on the strongest available signals.
        </div>
    </div>
    <div class="dp-decision-panel">
        <div class="dp-decision-panel-title">Why It Matters</div>
        <div class="dp-decision-panel-body">
            {critical_count + high_priority_count:,} high-priority signal(s) need attention across {total_insight_count:,} findings and {total_recommendation_count:,} recommended actions.
        </div>
    </div>
</div>

<div class="dp-decision-grid">
    <div class="dp-decision-panel">
        <div class="dp-decision-panel-title">Key Findings</div>
        <div class="dp-decision-panel-body">
            DataPulse identified {total_insight_count:,} business finding(s), including {critical_count:,} critical issue(s) and {high_priority_count:,} high-priority issue(s).
        </div>
    </div>
    <div class="dp-decision-panel">
        <div class="dp-decision-panel-title">Recommended Actions</div>
        <div class="dp-decision-panel-body">
            {total_recommendation_count:,} prioritized action(s) are available to guide the next business decisions.
        </div>
    </div>
</div>
"""
)

if critical_count:
    _render_html(
        '<div class="dp-card"><div class="dp-card-title">Critical Issues</div>'
        '<div class="dp-card-subtitle">Immediate signals requiring executive attention.</div></div>'
    )
else:
    st.info("No critical issues were identified by the available intelligence.")

if high_priority_count:
    _render_html(
        '<div class="dp-card"><div class="dp-card-title">High-Priority Issues</div>'
        '<div class="dp-card-subtitle">Important risks or opportunities to address soon.</div></div>'
    )
else:
    st.info("No high-priority issues were identified by the available intelligence.")

_render_html(
    f"""
<div class="dp-kpi-grid">
    <div class="dp-kpi">
        <div class="dp-kpi-label">AVAILABLE MODULES</div>
        <div class="dp-kpi-value">{available_module_count:,}</div>
        <div class="dp-kpi-note">Intelligence modules ready</div>
    </div>
    <div class="dp-kpi">
        <div class="dp-kpi-label">TOTAL MODULES</div>
        <div class="dp-kpi-value">{total_module_count:,}</div>
        <div class="dp-kpi-note">Configured DataPulse modules</div>
    </div>
    <div class="dp-kpi">
        <div class="dp-kpi-label">INSIGHTS</div>
        <div class="dp-kpi-value">{len(insights) if isinstance(insights, list) else 0:,}</div>
        <div class="dp-kpi-note">Generated intelligence signals</div>
    </div>
    <div class="dp-kpi">
        <div class="dp-kpi-label">RECOMMENDATIONS</div>
        <div class="dp-kpi-value">{len(recommendations) if isinstance(recommendations, list) else 0:,}</div>
        <div class="dp-kpi-note">Prioritized actions</div>
    </div>
</div>
"""
)


# ============================================================
# MODULE STATUS
# ============================================================

_render_html(
    """
<div class="dp-card">
    <div class="dp-card-title">Intelligence Coverage</div>
    <div class="dp-card-subtitle">
        Status of the analytics and ML modules used to build the decision layer.
    </div>
</div>
"""
)

if isinstance(module_status, dict) and module_status:
    status_rows = []
    for module, status in module_status.items():
        if isinstance(status, dict):
            state = status.get("available", status.get("status", ""))
            detail = status.get("reason", status.get("message", ""))
        else:
            state = status
            detail = ""
        status_rows.append(
            {
                "Module": str(module).replace("_", " ").title(),
                "Status": "Available" if bool(state) else "Unavailable",
                "Detail": detail,
            }
        )
    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Module status is not available.")


# ============================================================
# PRIORITIZED RECOMMENDATIONS
# ============================================================

_render_html(
    """
<div class="dp-card">
    <div class="dp-card-title">Prioritized Recommendations</div>
    <div class="dp-card-subtitle">
        The highest-value actions generated from DataPulse intelligence.
    </div>
</div>
"""
)

if isinstance(recommendations, dict):
    recommendations = list(recommendations.values())

if recommendations:
    for rank, item in enumerate(recommendations[:15], start=1):
        if isinstance(item, dict):
            title = item.get(
                "title",
                item.get(
                    "action",
                    item.get("recommendation", f"Business Action {rank}"),
                ),
            )
            body = item.get(
                "description",
                item.get(
                    "reason",
                    item.get(
                        "rationale",
                        item.get("details", ""),
                    ),
                ),
            )
            priority = item.get("priority", item.get("severity", ""))
            impact = item.get(
                "impact",
                item.get(
                    "expected_impact",
                    item.get("business_impact", ""),
                ),
            )
        else:
            title = f"Business Action {rank}"
            body = item
            priority = ""
            impact = ""

        if _is_html_fragment(title) or _is_html_fragment(body):
            continue

        meta_parts = []
        if priority:
            meta_parts.append(f"Priority: {_text(priority)}")
        if impact:
            meta_parts.append(f"Impact: {_text(impact)}")

        meta_html = (
            f'<div class="dp-rec-meta">{_esc(" · ".join(meta_parts))}</div>'
            if meta_parts
            else ""
        )

        # IMPORTANT: the complete recommendation card is one HTML string.
        # This eliminates the raw </div> leak seen in the previous version.
        _render_html(
            f"""
<div class="dp-recommendation">
    <div class="dp-rec-head">
        <span class="dp-rec-rank">{rank}</span>
        <span class="dp-rec-title">{_esc(title)}</span>
    </div>
    <div class="dp-rec-body">{_esc(body)}</div>
    {meta_html}
</div>
"""
        )
else:
    st.info("No prioritized recommendations were generated yet.")


# ============================================================
# INSIGHTS
# ============================================================

_render_html(
    """
<div class="dp-card">
    <div class="dp-card-title">Key Intelligence Signals</div>
    <div class="dp-card-subtitle">
        Supporting insights behind the recommended actions.
    </div>
</div>
"""
)

if isinstance(insights, dict):
    insights = list(insights.values())

if insights:
    for rank, item in enumerate(insights[:20], start=1):
        if isinstance(item, dict):
            title = item.get(
                "title",
                item.get(
                    "name",
                    item.get("insight", f"Insight {rank}"),
                ),
            )
            body = item.get(
                "description",
                item.get(
                    "message",
                    item.get(
                        "detail",
                        item.get("reason", ""),
                    ),
                ),
            )
            category = item.get("category", item.get("module", ""))
        else:
            title = f"Insight {rank}"
            body = item
            category = ""

        if _is_html_fragment(title) or _is_html_fragment(body):
            continue

        badge = (
            f'<span class="dp-status">{_esc(category)}</span>'
            if category
            else ""
        )

        _render_html(
            f"""
<div class="dp-recommendation">
    <div class="dp-rec-head">
        <span class="dp-rec-rank">{rank}</span>
        <span class="dp-rec-title">{_esc(title)}</span>
        {badge}
    </div>
    <div class="dp-rec-body">{_esc(body)}</div>
</div>
"""
        )
else:
    st.info("No additional insight records are available.")


# ============================================================
# TOP PRIORITIES
# ============================================================

if top_priorities:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Top Priorities</div>
    <div class="dp-card-subtitle">
        The highest-priority actions selected by the decision layer.
    </div>
</div>
"""
    )

    for rank, item in enumerate(top_priorities[:5], start=1):
        if not isinstance(item, dict):
            continue

        title = item.get(
            "title",
            item.get("action", "Business Action"),
        )
        action = item.get(
            "action",
            item.get("description", ""),
        )
        reason = item.get(
            "reason",
            item.get("rationale", ""),
        )

        _render_html(
            f"""
<div class="dp-recommendation">
    <div class="dp-rec-head">
        <span class="dp-rec-rank">{rank}</span>
        <span class="dp-rec-title">{_esc(title)}</span>
    </div>
    <div class="dp-rec-body">{_esc(action)}</div>
    {f'<div class="dp-rec-meta">Why it matters: {_esc(reason)}</div>' if reason else ''}
</div>
"""
        )
else:
    st.info("No top priorities are available yet.")


# ============================================================
# WARNINGS
# ============================================================

warnings = insight_result.get("warnings", [])

if warnings:
    _render_html(
        """
<div class="dp-card">
    <div class="dp-card-title">Decision Layer Notes</div>
    <div class="dp-card-subtitle">
        Some recommendations may be limited by dataset coverage or module availability.
    </div>
</div>
"""
    )
    for warning in warnings:
        st.warning(str(warning))


_render_html(
    """
<div style="text-align:center; color:#94A3B8; font-size:11px; padding:18px 0 8px 0;">
    DataPulse · Business Decision Intelligence
</div>
"""
)
