from pathlib import Path
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.data_loader import DataLoadError, load_dataset
from src.schema_mapper import map_schema
from src.validator import validate_dataset
from src.analytics import run_core_analytics
from src.customer_analytics import run_customer_analytics
from src.product_analytics import run_product_analytics
from src.marketing_analytics import run_campaign_impact
from src.market_analytics import run_market_seller_analytics
from src.operations_analytics import run_delivery_analytics
from src.pricing_analytics import run_pricing_analytics
from src.nlp_engine import run_review_nlp
from src.feature_engineering import build_ml_features
from src.ml.churn import run_churn_prediction
from src.ml.segmentation import run_customer_segmentation
from src.ml.forecasting import run_sales_forecast
from src.ml.anomaly_detection import run_anomaly_detection
from src.insight_engine import generate_insights


# ============================================================
# DATAPULSE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataPulse | E-Commerce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

for folder in [
    DATA_DIR,
    UPLOAD_DIR,
    PROCESSED_DIR,
    SAMPLE_DIR,
    MODEL_DIR,
    ASSETS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "DataPulse"
APP_SUBTITLE = "E-Commerce Intelligence Platform"

LINKEDIN_URL = (
    "https://www.linkedin.com/in/"
    "pradeep-kalasagond-95579a230/"
)

CREATOR_NAME = "Pradeep Kalasagond"
CREATOR_ROLE = "Data Analytics • Machine Learning"


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "active_page": "Overview",
    "dataset_name": None,
    "dataset_path": None,
    "dataset_source": None,
    "dataset": None,
    "schema_mapping": {},
    "validation_result": None,
    "analysis_ready": False,
    "models_trained": {},
    "model_results": {},
    "insights": [],
    "dataset_metadata": None,
    "dataset_upload_signature": None,
    "analytics_results": {},
    "ml_features": None,
    "processing_error": None,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HTML RENDERER
# ============================================================
#
# IMPORTANT:
#
# Do NOT use st.markdown() for our dashboard HTML.
#
# st.html() directly renders HTML and prevents the dashboard
# markup from appearing as visible text.
#
# ============================================================

def render_html(html_content: str):
    st.html(html_content)


# ============================================================
# GLOBAL CSS
# ============================================================

render_html(
    """
<style>

#MainMenu {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

header {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* Hide Streamlit automatic multipage navigation */
[data-testid="stSidebarNav"] {
    display: none !important;
}


/* ==========================================================
   APPLICATION
   ========================================================== */

.stApp {
    background: #F5F7FB !important;
    color: #0F172A !important;
}

.main .block-container {
    max-width: 1500px !important;

    padding-top: 1.1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 3rem !important;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #080F1D !important;

    border-right:
        1px solid
        #1E293B !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 0.8rem !important;
    padding-left: 0.65rem !important;
    padding-right: 0.65rem !important;

    scrollbar-width: auto !important;

    scrollbar-color:
        #64748B
        #080F1D !important;
}


/* ==========================================================
   SIDEBAR SCROLLBAR
   ========================================================== */

section[data-testid="stSidebar"] > div::-webkit-scrollbar {
    width: 10px !important;
}

section[data-testid="stSidebar"] > div::-webkit-scrollbar-track {
    background: #050A13 !important;
}

section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
    background: #64748B !important;

    border-radius: 10px !important;

    border:
        2px solid
        #080F1D !important;
}

section[data-testid="stSidebar"] > div::-webkit-scrollbar-thumb:hover {
    background: #94A3B8 !important;
}


/* ==========================================================
   SIDEBAR BRAND
   ========================================================== */

.dp-brand {
    padding:
        8px
        8px
        18px
        8px;

    margin-bottom: 12px;

    border-bottom:
        1px solid
        rgba(148,163,184,0.14);
}

.dp-brand-row {
    display: flex;

    align-items: center;

    justify-content: flex-start;

    gap: 10px;
}

.dp-logo {
    width: 36px;
    height: 36px;

    flex-shrink: 0;

    border-radius: 10px;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #2563EB,
            #6366F1
        );

    color: #FFFFFF;

    font-size: 22px;

    font-weight: 800;
}

.dp-brand-name {
    color: #FFFFFF;

    font-size: 21px;

    font-weight: 800;

    letter-spacing: -0.5px;
}

.dp-brand-subtitle {
    color: #94A3B8;

    font-size: 8px;

    margin-left: 46px;

    margin-top: 5px;
}


/* ==========================================================
   SIDEBAR SECTION
   ========================================================== */

.dp-section {
    color: #718096;

    font-size: 10px;

    font-weight: 850;

    letter-spacing: 1.4px;

    text-transform: uppercase;

    margin:
        18px
        8px
        7px
        8px;
}


/* ==========================================================
   SIDEBAR BUTTONS
   ========================================================== */

section[data-testid="stSidebar"] div.stButton {
    width: 100% !important;

    margin-bottom: 2px !important;
}

section[data-testid="stSidebar"]
div.stButton > button {

    width: 100% !important;

    min-height: 38px !important;

    padding:
        0
        11px !important;

    background:
        transparent !important;

    border:
        1px solid
        transparent !important;

    border-radius: 8px !important;

    color: #CBD5E1 !important;

    font-size: 10px !important;

    font-weight: 550 !important;

    text-align: left !important;

    justify-content: flex-start !important;

    box-shadow: none !important;
}

section[data-testid="stSidebar"]
div.stButton > button > div {

    width: 100% !important;

    justify-content: flex-start !important;

    text-align: left !important;
}

section[data-testid="stSidebar"]
div.stButton > button p {

    width: 100% !important;

    text-align: left !important;

    margin: 0 !important;
}

section[data-testid="stSidebar"]
div.stButton > button:hover {

    background:
        rgba(37,99,235,0.16) !important;

    color:
        #FFFFFF !important;

    border-color:
        rgba(96,165,250,0.20) !important;
}


/* ==========================================================
   ACTIVE NAV
   ========================================================== */

.dp-active {

    width: 100%;

    min-height: 38px;

    display: flex;

    align-items: center;

    justify-content: flex-start;

    padding:
        0
        11px;

    margin-bottom: 3px;

    border-radius: 8px;

    background:
        linear-gradient(
            90deg,
            #2563EB,
            #1D4ED8
        );

    color: #FFFFFF;

    font-size: 10px;

    font-weight: 750;

    box-shadow:
        0
        7px
        18px
        rgba(37,99,235,0.25);
}

.dp-active-icon {

    width: 23px;

    flex-shrink: 0;

    color: #BFDBFE;

    text-align: left;
}


/* ==========================================================
   SIDEBAR CREATOR
   ========================================================== */

.dp-sidebar-credit {
    margin-top: 16px;
    margin-bottom: 8px;
    padding: 12px 10px;
    border-top: 1px solid rgba(148,163,184,0.14);
}

.dp-sidebar-credit-label {
    color: #64748B;
    font-size: 7px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.dp-sidebar-credit-name {
    color: #FFFFFF;
    font-size: 10px;
    font-weight: 750;
    margin-top: 4px;
}

.dp-sidebar-credit-role {
    color: #94A3B8;
    font-size: 8px;
    margin-top: 3px;
}

.dp-sidebar-credit-link {
    display: inline-block;
    margin-top: 7px;
    color: #60A5FA !important;
    font-size: 8px;
    font-weight: 700;
    text-decoration: none !important;
}

.dp-sidebar-credit-link:hover {
    color: #BFDBFE !important;
    text-decoration: underline !important;
}

.dp-creator {

    margin-top: 18px;

    padding: 13px;

    border-radius: 11px;

    background:
        linear-gradient(
            145deg,
            rgba(37,99,235,0.18),
            rgba(15,23,42,0.85)
        );

    border:
        1px solid
        rgba(96,165,250,0.16);
}

.dp-created-label {

    color: #64748B;

    font-size: 7px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 1px;
}

.dp-created-name {

    color: #FFFFFF;

    font-size: 11px;

    font-weight: 750;

    margin-top: 4px;
}

.dp-created-role {

    color: #94A3B8;

    font-size: 8px;

    margin-top: 3px;
}

.dp-link {

    display: inline-block;

    margin-top: 8px;

    color: #60A5FA !important;

    font-size: 8px;

    font-weight: 700;

    text-decoration: none !important;
}

.dp-link:hover {

    color: #BFDBFE !important;

    text-decoration: underline !important;
}


/* ==========================================================
   TOP BAR
   ========================================================== */

.dp-topbar {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 24px;

    margin-bottom: 22px;
}

.dp-search {

    flex: 1;

    max-width: 760px;

    height: 42px;

    background: #FFFFFF;

    border:
        1px solid
        #E2E8F0;

    border-radius: 10px;

    display: flex;

    align-items: center;

    padding:
        0
        13px;

    color: #64748B;

    font-size: 11px;

    box-shadow:
        0
        3px
        12px
        rgba(15,23,42,0.03);
}

.dp-search-icon {

    font-size: 16px;

    margin-right: 8px;
}

.dp-shortcut {

    margin-left: auto;

    background: #F1F5F9;

    padding:
        3px
        6px;

    border-radius: 4px;

    font-size: 8px;

    font-weight: 700;
}

.dp-top-right {

    display: flex;

    align-items: center;

    gap: 10px;
}

.dp-date {

    background: #FFFFFF;

    border:
        1px solid
        #E2E8F0;

    border-radius: 9px;

    padding:
        10px
        13px;

    font-size: 10px;

    font-weight: 600;

    color: #334155;
}

.dp-avatar {

    width: 38px;

    height: 38px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #172554,
            #2563EB
        );

    color: #FFFFFF;

    font-size: 12px;

    font-weight: 800;
}



/* ==========================================================
   HEADER / CONTENT ALIGNMENT
   ========================================================== */

.dp-topbar > div {
    min-width: 0;
}

.dp-top-right {
    flex-shrink: 0;
}

.dp-page-title,
.dp-page-description,
.dp-card-title,
.dp-intelligence-title,
.dp-pipeline-title {
    letter-spacing: -0.15px;
}

/* ==========================================================
   PAGE HEADER
   ========================================================== */

.dp-eyebrow {

    color: #64748B;

    font-size: 10px;

    font-weight: 850;

    text-transform: uppercase;

    letter-spacing: 1.3px;
}

.dp-page-title {

    color: #0F172A;

    font-size: 30px;

    font-weight: 850;

    letter-spacing: -0.9px;

    margin-top: 3px;
}

.dp-page-description {

    color: #64748B;

    font-size: 12px;

    margin-top: 5px;

    margin-bottom: 20px;
}


/* ==========================================================
   HERO
   ========================================================== */

.dp-hero {
    position: relative;
    overflow: hidden;
    min-height: 205px;
    padding: 0;
    border-radius: 20px;
    background:
        radial-gradient(circle at 83% 50%, rgba(0,122,255,0.26), transparent 28%),
        radial-gradient(circle at 100% 0%, rgba(0,153,255,0.20), transparent 30%),
        linear-gradient(108deg, #03112D 0%, #041B42 46%, #063A82 100%);
    border: 1px solid rgba(70,150,255,0.24);
    box-shadow: 0 20px 50px rgba(15,23,42,0.18), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 14px;
}

.dp-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 13% 63%, rgba(20,150,255,0.14) 0 1px, transparent 1.5px),
        radial-gradient(circle at 61% 18%, rgba(20,150,255,0.12) 0 1px, transparent 1.5px);
    background-size: 15px 15px, 13px 13px;
    opacity: .65;
    pointer-events: none;
}

.dp-hero::after {
    content: "";
    position: absolute;
    width: 850px;
    height: 850px;
    right: -360px;
    top: -370px;
    border-radius: 50%;
    border: 1px solid rgba(58,150,255,0.22);
    box-shadow:
        0 0 0 85px rgba(58,150,255,0.025),
        0 0 0 170px rgba(58,150,255,0.018);
    pointer-events: none;
}

.dp-hero-inner {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: stretch;
    width: 100%;
    min-height: 205px;
}

.dp-hero-content {
    position: relative;
    z-index: 10;
    width: 52%;
    min-width: 0;
    padding: 18px 0 16px 34px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.dp-hero-kicker {
    color: #35BDF4;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 9px;
}

.dp-hero-kicker::after {
    content: "";
    display: block;
    width: 84px;
    height: 2px;
    margin-top: 7px;
    background: linear-gradient(90deg, #1E9BFF, rgba(30,155,255,0));
    box-shadow: 0 0 10px rgba(30,155,255,.45);
}

.dp-hero-title {
    color: #FFFFFF;
    font-size: clamp(34px, 3vw, 44px);
    font-weight: 850;
    line-height: .98;
    letter-spacing: -2.1px;
    max-width: 780px;
}

.dp-hero-accent {
    display: block;
    color: #20D4F2;
    background: linear-gradient(90deg, #16D7F4 0%, #18B9F5 62%, #2A8FFF 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dp-hero-description {
    color: #D3DFEF;
    font-size: 12px;
    line-height: 1.55;
    max-width: 700px;
    margin-top: 9px;
}

.dp-hero-note {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    margin-top: 10px;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(2,17,45,.58);
    border: 1px solid rgba(18,161,255,.75);
    color: #F0F8FF;
    font-size: 9px;
    font-weight: 700;
    box-shadow: 0 0 18px rgba(0,157,255,.12), inset 0 0 18px rgba(0,157,255,.04);
}

/* Right-side illustration recreated in CSS/SVG to match the supplied reference. */
.dp-hero-visual {
    position: absolute;
    inset: 0 0 0 47%;
    overflow: hidden;
    pointer-events: none;
}

.dp-hero-visual::before {
    content: "";
    position: absolute;
    left: -10%;
    top: 28%;
    width: 70%;
    height: 45%;
    border: 1px solid rgba(27,159,255,.30);
    border-left-color: transparent;
    border-bottom-color: transparent;
    border-radius: 50%;
    transform: rotate(-20deg);
}

.dp-hero-visual::after {
    content: "";
    position: absolute;
    left: 4%;
    top: 40%;
    width: 58%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #20BFFF, transparent);
    transform: rotate(-17deg);
    box-shadow: 0 0 10px rgba(32,191,255,.65);
}

.dp-orbit {
    position: absolute;
    width: 460px;
    height: 245px;
    right: -45px;
    top: 12px;
    border: 1px solid rgba(48,151,255,.20);
    border-radius: 50%;
    transform: rotate(-14deg);
}

.dp-orbit::before {
    content: "";
    position: absolute;
    inset: -18px 70px 15px 80px;
    border: 1px solid rgba(37,157,255,.16);
    border-radius: 50%;
}

.dp-grid-floor {
    position: absolute;
    right: -10px;
    bottom: -65px;
    width: 72%;
    height: 150px;
    transform: perspective(280px) rotateX(58deg);
    background-image:
        linear-gradient(rgba(31,166,255,.11) 1px, transparent 1px),
        linear-gradient(90deg, rgba(31,166,255,.11) 1px, transparent 1px);
    background-size: 30px 22px;
    mask-image: linear-gradient(to top, black, transparent);
    -webkit-mask-image: linear-gradient(to top, black, transparent);
}

.dp-data-card {
    position: absolute;
    z-index: 5;
    background: rgba(8,49,101,.34);
    border: 1px solid rgba(58,170,255,.55);
    border-radius: 10px;
    box-shadow: 0 0 25px rgba(0,128,255,.08), inset 0 0 20px rgba(0,128,255,.04);
    backdrop-filter: blur(4px);
}

.dp-data-card small { display: none; }

.dp-data-card.card-a {
    right: 3%;
    top: 6%;
    width: 155px;
    height: 96px;
    padding: 0;
}

.dp-data-card.card-b {
    left: 12%;
    top: 38%;
    width: 100px;
    height: 88px;
    padding: 0;
}

.dp-data-card.card-c {
    right: 7%;
    top: 31%;
    width: 90px;
    height: 72px;
    padding: 0;
}

.dp-bars {
    position: absolute;
    inset: 20px 18px 18px;
    display: flex;
    align-items: end;
    gap: 7px;
    border-bottom: 1px solid rgba(96,200,255,.30);
    border-left: 1px solid rgba(96,200,255,.20);
    padding-left: 8px;
}

.dp-bars span {
    display: block;
    width: 13px;
    border-radius: 2px 2px 0 0;
    background: linear-gradient(to top, #1689F8, #55E5FF);
    box-shadow: 0 0 10px rgba(34,211,238,.30);
}

.dp-bars span:nth-child(1) { height: 27px; }
.dp-bars span:nth-child(2) { height: 39px; }
.dp-bars span:nth-child(3) { height: 53px; }
.dp-bars span:nth-child(4) { height: 70px; }
.dp-bars span:nth-child(5) { height: 87px; }

.dp-mini-chart {
    position: absolute;
    inset: 28px 12px 17px;
}

.dp-mini-chart::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: linear-gradient(rgba(82,185,255,.10) 1px, transparent 1px), linear-gradient(90deg, rgba(82,185,255,.08) 1px, transparent 1px);
    background-size: 22px 22px;
}

.dp-mini-chart svg { width:100%; height:100%; position:relative; z-index:2; }

.dp-cart {
    position: absolute;
    z-index: 8;
    left: 27%;
    top: 35%;
    width: 200px;
    height: 143px;
    filter: drop-shadow(0 0 13px rgba(34,211,238,.55));
}

.dp-cart svg { width:100%; height:100%; }

.dp-glow-dot {
    position: absolute;
    z-index: 9;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #6BEAFF;
    box-shadow: 0 0 15px 5px rgba(32,205,255,.42);
}

.dp-glow-dot.one { right: 27%; top: 28%; }
.dp-glow-dot.two { right: 5%; top: 57%; }
.dp-glow-dot.three { left: 26%; top: 55%; }

/* Extra visual elements present in the reference: donut, payment tile, shirt tile and globe. */
.dp-hero-visual .card-a::before {
    content: "";
    position: absolute;
    width: 47px;
    height: 47px;
    left: 20px;
    top: 30px;
    border-radius: 50%;
    border: 10px solid #43D9FF;
    border-right-color: #1476E8;
    transform: rotate(30deg);
    box-shadow: 0 0 14px rgba(40,205,255,.22);
}

.dp-hero-visual .card-a::after {
    content: "";
    position: absolute;
    left: 112px;
    top: 24px;
    width: 55px;
    height: 68px;
    border-radius: 6px;
    background: repeating-linear-gradient(to top, #29B8FF 0 6px, transparent 6px 12px);
    opacity: .9;
}

.dp-hero-visual .card-c::before {
    content: "";
    position: absolute;
    left: 16px;
    top: 23px;
    width: 72px;
    height: 39px;
    border-radius: 5px;
    background: linear-gradient(180deg, #57DBFF 0 9px, rgba(87,219,255,.18) 9px 100%);
    box-shadow: 0 0 13px rgba(50,198,255,.22);
}

.dp-hero-visual .card-c::after {
    content: "";
    position: absolute;
    right: 13px;
    top: 16px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #D7F8FF;
    box-shadow: 0 0 8px #43D9FF;
}

.dp-hero-visual .card-b::after {
    content: "";
    position: absolute;
    left: 38px;
    top: 22px;
    width: 40px;
    height: 55px;
    background: #35CFFF;
    clip-path: polygon(15% 0, 35% 10%, 65% 10%, 85% 0, 100% 30%, 76% 40%, 72% 100%, 28% 100%, 24% 40%, 0 30%);
    filter: drop-shadow(0 0 8px rgba(35,207,255,.45));
}

.dp-hero-visual .dp-globe {
    position: absolute;
    z-index: 3;
    right: -28px;
    bottom: -88px;
    width: 245px;
    height: 245px;
    border-radius: 50%;
    border: 1px solid rgba(47,181,255,.55);
    background:
        radial-gradient(circle at 42% 34%, rgba(21,168,255,.24), transparent 43%),
        repeating-radial-gradient(ellipse at center, transparent 0 28px, rgba(46,180,255,.12) 29px 30px),
        repeating-linear-gradient(90deg, transparent 0 27px, rgba(46,180,255,.12) 28px 29px);
    box-shadow: inset 0 0 55px rgba(0,123,255,.24), 0 0 25px rgba(0,140,255,.14);
    transform: rotate(-15deg);
}

.dp-hero-visual .dp-globe::before {
    content: "";
    position: absolute;
    left: 53px;
    top: 48px;
    width: 175px;
    height: 125px;
    border-radius: 48% 52% 43% 57%;
    background: rgba(27,172,255,.23);
    clip-path: polygon(4% 33%, 20% 18%, 31% 23%, 40% 7%, 54% 14%, 62% 30%, 75% 35%, 84% 54%, 68% 65%, 61% 82%, 44% 77%, 36% 93%, 24% 70%, 8% 68%);
    filter: blur(.2px);
}

.dp-hero-visual .dp-data-stream {
    position: absolute;
    z-index: 2;
    left: 8%;
    bottom: 16%;
    width: 58%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #19C7FF, transparent);
    transform: rotate(-8deg);
    box-shadow: 0 0 12px rgba(25,199,255,.65);
}

@media (max-width: 1100px) {
    .dp-hero-content { width: 56%; padding-left: 30px; }
    .dp-hero-title { font-size: 38px; }
    .dp-cart { left: 25%; width: 175px; }
}

/* ==========================================================
   BUILT BY
   ========================================================== */

.dp-built-by {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    min-height: 34px;
    margin: 7px 3px 10px 3px;
    padding: 0 2px;
}

.dp-built-left {
    display: flex;
    align-items: baseline;
    gap: 7px;
    min-width: 0;
}

.dp-built-label {
    color: #94A3B8;
    font-size: 8px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .9px;
}

.dp-built-name {
    color: #334155;
    font-size: 10px;
    font-weight: 800;
}

.dp-built-role {
    color: #94A3B8;
    font-size: 9px;
}

.dp-built-link {
    flex-shrink: 0;
    color: #2563EB !important;
    font-size: 9px;
    font-weight: 750;
    text-decoration: none !important;
}

.dp-built-link:hover {
    text-decoration: underline !important;
}

.dp-dashboard-credit {
    display: none !important;
}

/* ==========================================================
   KPI
   ========================================================== */

.dp-kpi {

    box-sizing: border-box;
    width: 100%;
    height: 122px;
    min-height: 122px;
    display: flex;
    flex-direction: column;

    background: #FFFFFF;

    border:
        1px solid
        #E2E8F0;

    border-radius: 14px;

    padding: 19px;

    box-shadow:
        0
        5px
        18px
        rgba(15,23,42,0.035);

    overflow: visible;
    position: relative;
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

.dp-kpi-aov {
    background: #ECFEFF;
    border-color: #A5F3FC;
}

.dp-kpi-aov::before {
    background: #0891B2;
}

.dp-kpi-profit {
    background: #FFFBEB;
    border-color: #FDE68A;
}

.dp-kpi-profit::before {
    background: #D97706;
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

    text-transform: uppercase;

    letter-spacing: 0.06em;

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}

.dp-kpi-value {

    color: #0F172A;

    font-size: 22px;

    font-weight: 850;

    margin-top: 10px;
    min-width: 0;
    line-height: 1.1;
    white-space: nowrap;
    overflow: visible;
    text-overflow: clip;
}

.dp-kpi-note {

    color: #94A3B8;

    font-size: 10px;

    margin-top: auto;

    line-height: 1.25;

    overflow: hidden;

    display: -webkit-box;

    -webkit-line-clamp: 2;

    -webkit-box-orient: vertical;
}


/* ==========================================================
   CARDS
   ========================================================== */

.dp-card {

    background: #FFFFFF;

    border:
        1px solid
        #E2E8F0;

    border-radius: 16px;

    padding: 20px;

    box-shadow:
        0
        5px
        18px
        rgba(15,23,42,0.035);
}

.dp-card-title {

    color: #0F172A;

    font-size: 16px;

    font-weight: 800;
}

.dp-card-description {

    color: #64748B;

    font-size: 11px;

    margin-top: 5px;

    margin-bottom: 16px;

    line-height: 1.5;
}


/* ==========================================================
   INTELLIGENCE ITEMS
   ========================================================== */

.dp-intelligence {

    background: #F8FAFC;

    border:
        1px solid
        #E2E8F0;

    border-radius: 11px;

    padding: 16px;

    min-height: 88px;

    margin-bottom: 11px;
}

.dp-intelligence-title {

    color: #0F172A;

    font-size: 13px;

    font-weight: 800;
}

.dp-intelligence-text {

    color: #64748B;

    font-size: 10px;

    line-height: 1.55;

    margin-top: 6px;
}


/* ==========================================================
   PIPELINE
   ========================================================== */

.dp-pipeline {

    padding:
        14px
        0;

    border-bottom:
        1px solid
        #F1F5F9;
}

.dp-pipeline-number {

    display: inline-flex;

    width: 27px;

    height: 27px;

    align-items: center;

    justify-content: center;

    border-radius: 50%;

    background: #EFF6FF;

    color: #2563EB;

    font-size: 10px;

    font-weight: 850;

    margin-right: 8px;
}

.dp-pipeline-title {

    color: #0F172A;

    font-size: 12px;

    font-weight: 800;
}

.dp-pipeline-text {

    color: #64748B;

    font-size: 10px;

    margin-left: 36px;

    margin-top: -3px;
}


/* ==========================================================
   INFO
   ========================================================== */

.dp-info {

    background: #EFF6FF;

    border:
        1px solid
        #DBEAFE;

    border-radius: 10px;

    padding: 12px;

    color: #1E3A8A;

    font-size: 10px;

    line-height: 1.55;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.dp-footer {

    margin-top: 28px;

    padding-top: 18px;

    border-top:
        1px solid
        #E2E8F0;

    display: flex;

    justify-content: space-between;

    color: #94A3B8;

    font-size: 10px;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .dp-topbar {
        flex-direction: column;
        align-items: stretch;
    }

    .dp-search {
        max-width: 100%;
    }

    .dp-top-right {
        justify-content: flex-end;
    }

    .dp-hero {
        min-height: 300px;
    }

    .dp-hero-content {
        width: 100%;
        padding: 28px 26px 24px;
    }

    .dp-hero-inner {
        min-height: 300px;
    }

    .dp-hero-visual {
        inset: 150px 0 0 20%;
        opacity: .62;
    }

    .dp-hero-title {
        font-size: 39px;
    }
}

@media (max-width: 600px) {
    .dp-hero {
        min-height: 330px;
    }

    .dp-hero-title {
        font-size: 31px;
        letter-spacing: -1.2px;
    }

    .dp-hero-description {
        font-size: 12px;
    }

    .dp-hero-visual {
        min-height: 190px;
        transform: scale(0.88);
        transform-origin: top right;
        margin-bottom: -25px;
    }

}

</style>
"""
)



# ============================================================
# DATAPULSE DATA + INTELLIGENCE PIPELINE
# ============================================================

def _safe_filename(filename: str) -> str:
    """Keep uploaded filenames inside the upload directory."""
    return Path(filename).name


def _upload_signature(uploaded_file) -> str:
    """Create a stable signature so the same upload is not reprocessed."""
    raw = uploaded_file.getvalue()
    return f"{uploaded_file.name}:{len(raw)}:{hash(raw)}"


def _summary_value(summary, *keys, default=None):
    """Read a metric from a module summary using fallback key names."""
    if not isinstance(summary, dict):
        return default

    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value

    return default


def _format_metric(value, currency=False):
    """Format dashboard KPI values."""
    if value is None:
        return "—"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if currency:
        if abs(value) >= 10_000_000:
            return f"₹{value / 10_000_000:.2f} Cr"
        if abs(value) >= 100_000:
            return f"₹{value / 100_000:.2f} L"
        if abs(value) >= 1_000:
            return f"₹{value / 1_000:.1f}K"
        return f"₹{value:,.0f}"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def reset_dataset_state():
    """Clear all derived state when a new dataset is uploaded."""
    st.session_state.dataset = None
    st.session_state.dataset_name = None
    st.session_state.dataset_path = None
    st.session_state.dataset_source = None
    st.session_state.schema_mapping = {}
    st.session_state.validation_result = None
    st.session_state.analysis_ready = False
    st.session_state.models_trained = {}
    st.session_state.model_results = {}
    st.session_state.insights = []
    st.session_state.dataset_metadata = None
    st.session_state.analytics_results = {}
    st.session_state.ml_features = None
    st.session_state.processing_error = None


def process_uploaded_dataset(uploaded_file):
    """
    Run the complete DataPulse pipeline once for a new upload.

    Contract:
        Upload → Load → Schema Map → Validate → Analytics
        → Feature Engineering → ML → Insight Engine.
    """
    raw_bytes = uploaded_file.getvalue()
    original_name = _safe_filename(uploaded_file.name)

    # Reset derived state before processing a genuinely new dataset.
    reset_dataset_state()

    try:
        # --------------------------------------------------------
        # LOAD
        # --------------------------------------------------------
        dataframe, metadata = load_dataset(
            BytesIO(raw_bytes),
            file_name=original_name,
            max_size_mb=200,
        )

        # --------------------------------------------------------
        # SCHEMA MAPPING
        # --------------------------------------------------------
        mapping = map_schema(dataframe)

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------
        validation_result = validate_dataset(
            dataframe,
            mapping,
        )

        # The validator may expose readiness under different names.
        validation_ready = bool(
            validation_result.get("valid", False)
            or validation_result.get("is_valid", False)
            or validation_result.get("analysis_ready", False)
        )

        # Some valid datasets may report availability instead of a
        # single valid flag. Treat a mapping/validation result with
        # no explicit failure as usable.
        if not validation_ready:
            errors = validation_result.get("errors", [])

            if errors:
                raise ValueError(
                    "Dataset validation failed: "
                    + "; ".join(str(error) for error in errors[:8])
                )

        # --------------------------------------------------------
        # BUSINESS ANALYTICS
        # --------------------------------------------------------
        analytics_results = {}

        analytics_results["core"] = run_core_analytics(
            dataframe,
            mapping,
        )

        analytics_results["customer"] = run_customer_analytics(
            dataframe,
            mapping,
        )

        analytics_results["product"] = run_product_analytics(
            dataframe,
            mapping,
        )

        analytics_results["campaign_impact"] = run_campaign_impact(
            dataframe,
            mapping,
        )

        analytics_results["market"] = run_market_seller_analytics(
            dataframe,
            mapping,
        )

        analytics_results["delivery"] = run_delivery_analytics(
            dataframe,
            mapping,
        )

        analytics_results["pricing"] = run_pricing_analytics(
            dataframe,
            mapping,
        )

        analytics_results["reviews"] = run_review_nlp(
            dataframe,
            mapping,
        )

        # --------------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------------
        ml_features = build_ml_features(
            dataframe,
            mapping,
        )

        # --------------------------------------------------------
        # MACHINE LEARNING
        # --------------------------------------------------------
        analytics_results["churn"] = run_churn_prediction(
            dataframe,
            mapping,
            feature_result=ml_features,
        )

        analytics_results["segmentation"] = run_customer_segmentation(
            dataframe,
            mapping,
            feature_result=ml_features,
        )

        analytics_results["forecast"] = run_sales_forecast(
            dataframe,
            mapping,
            feature_result=ml_features,
        )

        analytics_results["anomaly"] = run_anomaly_detection(
            dataframe,
            mapping,
            feature_result=ml_features,
        )

        # --------------------------------------------------------
        # INSIGHT ENGINE
        # IMPORTANT: generate_insights receives ONE dictionary.
        # --------------------------------------------------------
        insight_results = generate_insights(
            analytics_results
        )

        analytics_results["insights"] = insight_results

        # --------------------------------------------------------
        # COMMIT RESULTS TO SESSION STATE ONLY AFTER SUCCESS
        # --------------------------------------------------------
        upload_path = UPLOAD_DIR / original_name
        upload_path.write_bytes(raw_bytes)

        st.session_state.dataset = dataframe
        st.session_state.dataset_name = original_name
        st.session_state.dataset_path = str(upload_path)
        st.session_state.dataset_source = "User Upload"
        st.session_state.schema_mapping = mapping
        st.session_state.validation_result = validation_result
        st.session_state.analysis_ready = True
        st.session_state.dataset_metadata = metadata
        st.session_state.analytics_results = analytics_results
        st.session_state.ml_features = ml_features
        st.session_state.model_results = {
            key: analytics_results[key]
            for key in (
                "churn",
                "segmentation",
                "forecast",
                "anomaly",
            )
            if key in analytics_results
        }
        st.session_state.models_trained = {
            key: bool(
                isinstance(analytics_results.get(key), dict)
                and analytics_results[key].get(
                    "model_ready",
                    False,
                )
            )
            for key in (
                "churn",
                "segmentation",
                "forecast",
                "anomaly",
            )
        }
        st.session_state.insights = insight_results.get(
            "insights",
            [],
        ) if isinstance(insight_results, dict) else []

        st.session_state.processing_error = None

        return True, None

    except (
        DataLoadError,
        ValueError,
        KeyError,
        TypeError,
        ImportError,
        OSError,
    ) as exc:

        st.session_state.processing_error = str(exc)
        return False, str(exc)

    except Exception as exc:

        st.session_state.processing_error = str(exc)
        return False, str(exc)


def render_processing_status():
    """Render compact status information after dataset processing."""
    if not st.session_state.get("analysis_ready"):
        return

    metadata = st.session_state.get("dataset_metadata")

    if metadata is None:
        return

    rows = getattr(metadata, "rows", None)
    columns = getattr(metadata, "columns", None)

    dataset_name = st.session_state.get(
        "dataset_name",
        "Dataset",
    )

    if rows is not None and columns is not None:
        st.success(
            f"DataPulse processed **{dataset_name}** successfully — "
            f"{rows:,} rows × {columns:,} columns. "
            "Analytics, ML and decision intelligence are ready."
        )
    else:
        st.success(
            f"DataPulse processed **{dataset_name}** successfully."
        )



# ============================================================
# NAVIGATION FUNCTION
# ============================================================

# Custom DataPulse sidebar -> real Streamlit page mapping.
# ML sub-pages intentionally point to ml_lab.py because the four
# ML models are presented together in the ML Intelligence workspace.
PAGE_ROUTES = {
    "Overview": "app.py",
    "Sales Intelligence": "pages/sales.py",
    "Customer Intelligence": "pages/customers.py",
    "Product Intelligence": "pages/products.py",
    "Campaign Impact": "pages/campaign_impact.py",
    "ML Intelligence": "pages/ml_lab.py",
    "Churn Prediction": "pages/ml_lab.py",
    "Customer Segmentation": "pages/ml_lab.py",
    "Sales Forecasting": "pages/ml_lab.py",
    "Anomaly Detection": "pages/ml_lab.py",
    "Delivery Intelligence": "pages/operations.py",
    "Review & NLP Analytics": "pages/reviews.py",
    "Pricing & Discount Intelligence": "pages/pricing.py",
    "Market & Seller Intelligence": "pages/market.py",
    "Business Decision Center": "pages/decision_center.py",
}


def navigate(label: str, icon: str, page: str):

    if st.session_state.active_page == page:

        render_html(
            f"""
<div class="dp-active">
    <span class="dp-active-icon">{icon}</span>
    <span>{label}</span>
</div>
"""
        )

    else:

        if st.button(
            f"{icon}   {label}",
            key=f"nav_{page}",
            use_container_width=True,
        ):

            st.session_state.active_page = page

            target = PAGE_ROUTES.get(page)

            if target:
                st.switch_page(target)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    render_html(
        """
<div class="dp-brand">

    <div class="dp-brand-row">

        <div class="dp-logo">
            ∿
        </div>

        <div class="dp-brand-name">
            DataPulse
        </div>

    </div>

    <div class="dp-brand-subtitle">
        E-Commerce Intelligence Platform
    </div>

</div>
"""
    )


    # --------------------------------------------------------
    # WORKSPACE
    # --------------------------------------------------------

    render_html(
        """
<div class="dp-section">
    Workspace
</div>
"""
    )

    navigate(
        "Overview",
        "⌂",
        "Overview",
    )

    navigate(
        "Sales Intelligence",
        "▥",
        "Sales Intelligence",
    )

    navigate(
        "Customer Intelligence",
        "♙",
        "Customer Intelligence",
    )

    navigate(
        "Product Intelligence",
        "▣",
        "Product Intelligence",
    )

    navigate(
        "Campaign Impact",
        "◈",
        "Campaign Impact",
    )


    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    render_html(
        """
<div class="dp-section">
    ML Intelligence
</div>
"""
    )

    navigate(
        "ML Intelligence",
        "◎",
        "ML Intelligence",
    )

    navigate(
        "Churn Prediction",
        "◌",
        "Churn Prediction",
    )

    navigate(
        "Customer Segmentation",
        "◉",
        "Customer Segmentation",
    )

    navigate(
        "Sales Forecasting",
        "⌁",
        "Sales Forecasting",
    )

    navigate(
        "Anomaly Detection",
        "△",
        "Anomaly Detection",
    )


    # --------------------------------------------------------
    # BUSINESS INTELLIGENCE
    # --------------------------------------------------------

    render_html(
        """
<div class="dp-section">
    Business Intelligence
</div>
"""
    )

    navigate(
        "Delivery Intelligence",
        "▰",
        "Delivery Intelligence",
    )

    navigate(
        "Review & NLP Analytics",
        "▤",
        "Review & NLP Analytics",
    )

    navigate(
        "Pricing & Discount Intelligence",
        "◇",
        "Pricing & Discount Intelligence",
    )

    navigate(
        "Market & Seller Intelligence",
        "◎",
        "Market & Seller Intelligence",
    )


    # --------------------------------------------------------
    # DECISIONS
    # --------------------------------------------------------

    render_html(
        """
<div class="dp-section">
    Decisions
</div>
"""
    )

    navigate(
        "Business Decision Center",
        "◉",
        "Business Decision Center",
    )


    # --------------------------------------------------------
    # SIDEBAR CREATOR CREDIT
    # --------------------------------------------------------

    render_html(
        f"""
<div class="dp-sidebar-credit">
    <div class="dp-sidebar-credit-label">Built by</div>
    <div class="dp-sidebar-credit-name">{CREATOR_NAME}</div>
    <div class="dp-sidebar-credit-role">{CREATOR_ROLE}</div>
    <a
        class="dp-sidebar-credit-link"
        href="{LINKEDIN_URL}"
        target="_blank"
        rel="noopener noreferrer"
    >
        LinkedIn ↗
    </a>
</div>
"""
    )


# ============================================================
# TOP BAR
# ============================================================

render_html(
    f"""
<div class="dp-topbar">

    <div class="dp-search">

        <span class="dp-search-icon">
            ⌕
        </span>

        <span>
            Search features, insights, or help...
        </span>

        <span class="dp-shortcut">
            Ctrl K
        </span>

    </div>

    <div class="dp-top-right">

        <div class="dp-date">
            ◷ &nbsp; {datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%b %d, %Y")}
        </div>

        <div class="dp-avatar">
            PK
        </div>

    </div>

</div>
"""
)


# ============================================================
# ACTIVE PAGE
# ============================================================

active_page = st.session_state.active_page


# ============================================================
# OVERVIEW
# ============================================================

if active_page == "Overview":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    render_html(
        f"""
<div class="dp-hero">

    <div class="dp-hero-inner">

        <div class="dp-hero-content">

            <div class="dp-hero-kicker">
                E-Commerce Intelligence
            </div>

            <div class="dp-hero-title">
                Turn Data Into
                <span class="dp-hero-accent">Better Decisions.</span>
            </div>

            <div class="dp-hero-description">
                Connect an analysis-ready dataset and let DataPulse transform your
                business data into analytics, machine-learning predictions and
                actionable business intelligence.
            </div>

            <div class="dp-hero-note">
                Analyze • Predict • Optimize • Grow
            </div>

        </div>

        <div class="dp-hero-visual" aria-hidden="true">

            <div class="dp-orbit"></div>
            <div class="dp-grid-floor"></div>

            <div class="dp-data-card card-a">
                <div class="dp-bars">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
            </div>

            <div class="dp-data-card card-b">
                <div class="dp-mini-chart">
                    <svg viewBox="0 0 100 30" preserveAspectRatio="none">
                        <polyline
                            points="0,25 14,20 27,22 41,12 55,16 69,7 83,10 100,2"
                            fill="none"
                            stroke="#55E5FF"
                            stroke-width="2.2"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </div>
            </div>

            <div class="dp-data-card card-c"></div>

            <div class="dp-cart">
                <svg viewBox="0 0 180 130" fill="none">
                    <path d="M18 17H37L50 82C52 91 60 97 69 97H137C146 97 153 91 156 83L166 50H47"
                          stroke="#38BDF8" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M65 55H150M69 67H146"
                          stroke="#0EA5E9" stroke-width="3" stroke-linecap="round" opacity=".9"/>
                    <circle cx="70" cy="111" r="9" stroke="#67E8F9" stroke-width="4"/>
                    <circle cx="139" cy="111" r="9" stroke="#67E8F9" stroke-width="4"/>
                    <path d="M54 30H88L98 18H130L141 30"
                          stroke="#67E8F9" stroke-width="3.5" stroke-linecap="round" opacity=".9"/>
                </svg>
            </div>

            <div class="dp-glow-dot one"></div>
            <div class="dp-glow-dot two"></div>
            <div class="dp-glow-dot three"></div>
            <div class="dp-globe"></div>
            <div class="dp-data-stream"></div>

        </div>

    </div>

</div>
"""
        )


    # --------------------------------------------------------
    # CREATOR CREDIT
    # --------------------------------------------------------

    render_html(
        f"""
<div class="dp-built-by">
    <div class="dp-built-left">
        <span class="dp-built-label">Built by</span>
        <span class="dp-built-name">{CREATOR_NAME}</span>
        <span class="dp-built-role">· {CREATOR_ROLE}</span>
    </div>
    <a
        class="dp-built-link"
        href="{LINKEDIN_URL}"
        target="_blank"
        rel="noopener noreferrer"
    >
        LinkedIn ↗
    </a>
</div>
"""
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    core_results = st.session_state.get(
        "analytics_results",
        {},
    ).get(
        "core",
        {},
    )

    core_summary = (
        core_results.get("summary", {})
        if isinstance(core_results, dict)
        else {}
    )

    revenue_value = _summary_value(
        core_summary,
        "total_revenue",
        "revenue",
        "sales",
        "total_sales",
        default=None,
    )

    orders_value = _summary_value(
        core_summary,
        "total_orders",
        "orders",
        "order_count",
        default=None,
    )

    customers_value = _summary_value(
        core_summary,
        "unique_customers",
        "customers",
        "customer_count",
        "total_customers",
        default=None,
    )

    aov_value = _summary_value(
        core_summary,
        "average_order_value",
        "avg_order_value",
        "aov",
        default=None,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi_data = [
        (
            kpi1,
            "Revenue",
            _format_metric(revenue_value, currency=True),
            "Total sales generated",
        ),
        (
            kpi2,
            "Orders",
            _format_metric(orders_value),
            "Total orders analyzed",
        ),
        (
            kpi3,
            "Customers",
            _format_metric(customers_value),
            "Unique customers",
        ),
        (
            kpi4,
            "Average Order Value",
            _format_metric(aov_value, currency=True),
            "Average revenue per order",
        ),
    ]

    kpi_color_classes = {
        "Revenue": "revenue",
        "Orders": "orders",
        "Customers": "customers",
        "Average Order Value": "aov",
    }

    for column, label, value, note in kpi_data:

        label_class = kpi_color_classes.get(label, "default")

        with column:

            render_html(
                f"""
<div class="dp-kpi dp-kpi-{label_class}">

    <div class="dp-kpi-label">
        {label}
    </div>

    <div class="dp-kpi-value">
        {value}
    </div>

    <div class="dp-kpi-note">
        {note}
    </div>

</div>
"""
            )


    # --------------------------------------------------------
    # CONNECT DATA
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    render_html(
        """
<div class="dp-card">

    <div class="dp-card-title">
        Connect Your Data
    </div>

    <div class="dp-card-description">
        Upload an EDA-completed, analysis-ready CSV
        or Excel dataset to activate the DataPulse
        intelligence engine.
    </div>

</div>
"""
    )


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload analysis-ready dataset",

        type=[
            "csv",
            "xlsx",
            "xls",
        ],

        label_visibility="collapsed",

        help=(
            "Upload an EDA-completed and "
            "analysis-ready dataset."
        ),
    )


    if uploaded_file is not None:

        current_signature = _upload_signature(uploaded_file)

        # Process only when this is a new file/signature.
        if (
            st.session_state.get("dataset_upload_signature")
            != current_signature
        ):

            st.session_state.dataset_upload_signature = (
                current_signature
            )

            with st.spinner(
                "DataPulse is mapping, validating, analyzing and "
                "preparing machine-learning intelligence..."
            ):
                success, error = process_uploaded_dataset(
                    uploaded_file
                )

            if success:
                st.success(
                    f"'{uploaded_file.name}' "
                    "was uploaded and processed successfully."
                )
            else:
                st.error(
                    "DataPulse could not process the dataset: "
                    f"{error}"
                )

        elif st.session_state.get("analysis_ready"):

            render_processing_status()

        elif st.session_state.get("processing_error"):

            st.error(
                "DataPulse could not process the dataset: "
                f"{st.session_state.processing_error}"
            )

        elif st.session_state.get("analysis_ready"):

            render_processing_status()


    # --------------------------------------------------------
    # LOWER DASHBOARD
    # --------------------------------------------------------

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.55, 1],
        gap="large",
    )


    # ========================================================
    # INTELLIGENCE WORKSPACE
    # ========================================================

    with left:

        render_html(
            """
<div class="dp-card">

    <div class="dp-card-title">
        Intelligence Workspace
    </div>

    <div class="dp-card-description">
        One platform for e-commerce analytics,
        machine learning and decision intelligence.
    </div>
"""
        )


        intelligence = [

            (
                "▥",
                "Sales Intelligence",
                "Revenue, growth, orders and profitability.",
            ),

            (
                "♙",
                "Customer Intelligence",
                "RFM, cohorts, CLV and customer behavior.",
            ),

            (
                "▣",
                "Product Intelligence",
                "Product, category and demand performance.",
            ),

            (
                "◈",
                "Campaign Impact",
                "Campaign performance, revenue impact and ROI.",
            ),

            (
                "◎",
                "ML Intelligence",
                "Predictive models and machine-learning insights.",
            ),

            (
                "▰",
                "Delivery Intelligence",
                "Delivery, returns and operational performance.",
            ),

            (
                "▤",
                "Review & NLP Analytics",
                "Sentiment and customer review themes.",
            ),

            (
                "◇",
                "Pricing & Discount Intelligence",
                "Pricing, discount and profitability analysis.",
            ),

            (
                "◎",
                "Market & Seller Intelligence",
                "Seller, regional and market performance.",
            ),

            (
                "◉",
                "Business Decision Center",
                "Recommendations and business actions.",
            ),
        ]


        for icon, title, description in intelligence:

            render_html(
                f"""
<div class="dp-intelligence">

    <div class="dp-intelligence-title">
        {icon}
        &nbsp;&nbsp;
        {title}
    </div>

    <div class="dp-intelligence-text">
        {description}
    </div>

</div>
"""
            )


        render_html(
            """
</div>
"""
        )


    # ========================================================
    # DATAPULSE PIPELINE
    # ========================================================

    with right:

        render_html(
            """
<div class="dp-card">

    <div class="dp-card-title">
        DataPulse Intelligence
    </div>

    <div class="dp-card-description">
        End-to-end analytics pipeline
    </div>
"""
        )


        pipeline = [

            (
                "01",
                "Connect",
                "Upload your analysis-ready dataset.",
            ),

            (
                "02",
                "Configure",
                "Map your dataset columns.",
            ),

            (
                "03",
                "Validate",
                "Check compatibility and data quality.",
            ),

            (
                "04",
                "Analyze",
                "Generate business metrics and insights.",
            ),

            (
                "05",
                "Predict",
                "Train and evaluate ML models.",
            ),

            (
                "06",
                "Decide",
                "Turn intelligence into business actions.",
            ),
        ]


        for number, title, description in pipeline:

            render_html(
                f"""
<div class="dp-pipeline">

    <span class="dp-pipeline-number">
        {number}
    </span>

    <span class="dp-pipeline-title">
        {title}
    </span>

    <div class="dp-pipeline-text">
        {description}
    </div>

</div>
"""
            )


        render_html(
            """
<br>

<div class="dp-info">

    <strong>
        Data requirement
    </strong>

    <br><br>

    EDA should be completed before upload.
    DataPulse handles schema mapping, validation,
    analytics, machine learning and business
    intelligence after the dataset is connected.

</div>

</div>
"""
        )


# ============================================================
# OTHER MODULES
# ============================================================

# Non-Overview modules are opened by st.switch_page() from the
# custom sidebar. This fallback should normally never be reached.
else:

    render_html(
        f"""
<div class="dp-eyebrow">
    DATAPULSE INTELLIGENCE
</div>

<div class="dp-page-title">
    {active_page}
</div>

<div class="dp-page-description">
    Opening the selected DataPulse intelligence module.
</div>

<div class="dp-card">

    <div class="dp-card-title">
        Intelligence Module
    </div>

    <div class="dp-card-description">
        Use the DataPulse sidebar to open this module.
    </div>

</div>
"""
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
<div class="dp-footer">

    <span>
        DataPulse · E-Commerce Intelligence Platform
    </span>

    <span>
        Built by Pradeep Kalasagond · Data Analytics • ML
    </span>

</div>
"""
)
