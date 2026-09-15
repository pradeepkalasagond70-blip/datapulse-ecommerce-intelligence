from __future__ import annotations

from pathlib import Path


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "DataPulse"

APP_SUBTITLE = (
    "E-Commerce Intelligence Platform"
)

APP_TITLE = (
    "DataPulse | E-Commerce Intelligence"
)

APP_DESCRIPTION = (
    "Turn Data Into Better Decisions."
)

APP_TAGLINE = (
    "Analyze • Predict • Optimize • Grow"
)


# ============================================================
# CREATOR
# ============================================================

CREATOR_NAME = "Pradeep Kalasagond"

CREATOR_ROLE = (
    "Data Analytics • Machine Learning"
)

LINKEDIN_URL = (
    "https://www.linkedin.com/in/"
    "pradeep-kalasagond-95579a230/"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"

PROCESSED_DIR = DATA_DIR / "processed"

SAMPLE_DIR = DATA_DIR / "sample"

MODEL_DIR = BASE_DIR / "models"

ASSETS_DIR = BASE_DIR / "assets"

LOGO_DIR = ASSETS_DIR / "logo"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}

SUPPORTED_EXCEL_EXTENSIONS = {
    ".xlsx",
    ".xls",
}

MAX_UPLOAD_SIZE_MB = 200

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB
    * 1024
    * 1024
)


# ============================================================
# ANALYTICS CONFIGURATION
# ============================================================

DEFAULT_TOP_N = 10

DEFAULT_TOP_PRODUCTS = 10

DEFAULT_TOP_CUSTOMERS = 10

DEFAULT_TOP_SELLERS = 10

DEFAULT_TOP_REGIONS = 10

DEFAULT_TOP_CAMPAIGNS = 10


# ============================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================

# Churn
CHURN_THRESHOLD_DAYS = 90

MIN_CHURN_CUSTOMERS = 20


# Segmentation
MIN_SEGMENT_CUSTOMERS = 20

MIN_SEGMENT_CLUSTERS = 2

MAX_SEGMENT_CLUSTERS = 6

DEFAULT_SEGMENT_CLUSTERS = 4


# Forecasting
MIN_FORECAST_PERIODS = 12

DEFAULT_FORECAST_PERIODS = 6

MAX_FORECAST_PERIODS = 24

FORECAST_TEST_SIZE = 0.20


# Anomaly detection
MIN_ANOMALY_ROWS = 20

ANOMALY_CONTAMINATION = 0.05

ANOMALY_RANDOM_STATE = 42


# General ML
ML_RANDOM_STATE = 42


# ============================================================
# MODEL FEATURE CONFIGURATION
# ============================================================

CUSTOMER_FEATURES = [
    "revenue",
    "orders",
    "quantity",
    "profit",
    "average_order_value",
    "recency_days",
    "lifetime_days",
    "purchase_frequency",
    "avg_days_between_orders",
    "profit_margin",
    "discount_rate",
    "recency_score",
    "frequency_score",
    "monetary_score",
    "rfm_score",
    "ltv_proxy",
    "annualized_revenue_proxy",
    "average_discount_pct",
]


PRODUCT_FEATURES = [
    "revenue",
    "quantity",
    "orders",
    "profit",
    "average_unit_price",
    "average_discount_pct",
    "profit_margin",
]


FORECAST_FEATURES = [
    "time_index",
    "month_number",
    "quarter",
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
    "revenue_rolling_3",
    "revenue_rolling_6",
    "revenue_rolling_std_3",
]


ANOMALY_FEATURES = [
    "sales_amount",
    "quantity",
    "unit_price",
    "profit",
    "discount_pct",
    "day_of_week",
    "month",
    "log_sales_amount",
    "log_quantity",
    "profit_margin",
]


# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

# The dataset is expected to be EDA-completed before upload.
EDA_REQUIRED_BEFORE_UPLOAD = True

ALLOW_EMPTY_DATASET = False

ALLOW_DUPLICATE_COLUMNS = False

ALLOW_BLANK_COLUMNS = False


# ============================================================
# UI CONFIGURATION
# ============================================================

PAGE_LAYOUT = "wide"

INITIAL_SIDEBAR_STATE = "expanded"

PAGE_ICON = "📊"


# ============================================================
# INTELLIGENCE MODULES
# ============================================================

MODULES = [
    "Overview",
    "Sales Intelligence",
    "Customer Intelligence",
    "Product Intelligence",
    "Campaign Impact",
    "ML Intelligence",
    "Churn Prediction",
    "Customer Segmentation",
    "Sales Forecasting",
    "Anomaly Detection",
    "Delivery Intelligence",
    "Review & NLP Analytics",
    "Pricing & Discount Intelligence",
    "Market & Seller Intelligence",
    "Business Decision Center",
]


MODULE_DESCRIPTIONS = {
    "Overview": (
        "Executive business intelligence "
        "and dataset readiness."
    ),

    "Sales Intelligence": (
        "Revenue, orders, growth and "
        "profitability analytics."
    ),

    "Customer Intelligence": (
        "RFM, cohorts, CLV, retention "
        "and customer behavior."
    ),

    "Product Intelligence": (
        "Product, category, demand and "
        "profitability analytics."
    ),

    "Campaign Impact": (
        "Campaign performance, revenue "
        "impact, ROI and uplift."
    ),

    "ML Intelligence": (
        "Machine-learning models and "
        "predictive intelligence."
    ),

    "Churn Prediction": (
        "Predict customers who are "
        "at risk of churn."
    ),

    "Customer Segmentation": (
        "Discover meaningful customer "
        "groups from behavior."
    ),

    "Sales Forecasting": (
        "Forecast future sales, "
        "revenue and demand."
    ),

    "Anomaly Detection": (
        "Identify unusual transactions "
        "and business patterns."
    ),

    "Delivery Intelligence": (
        "Delivery, returns and "
        "fulfillment performance."
    ),

    "Review & NLP Analytics": (
        "Customer review sentiment "
        "and NLP analytics."
    ),

    "Pricing & Discount Intelligence": (
        "Pricing, discounts and "
        "profitability analysis."
    ),

    "Market & Seller Intelligence": (
        "Seller, region and market "
        "performance."
    ),

    "Business Decision Center": (
        "Convert intelligence into "
        "business decisions."
    ),
}


# ============================================================
# PAGE ROUTES
# ============================================================

PAGE_ROUTES = {
    "Overview": "app.py",

    "Sales Intelligence":
        "pages/sales.py",

    "Customer Intelligence":
        "pages/customers.py",

    "Product Intelligence":
        "pages/products.py",

    "Campaign Impact":
        "pages/campaign_impact.py",

    "ML Intelligence":
        "pages/ml_lab.py",

    "Churn Prediction":
        "pages/ml_lab.py",

    "Customer Segmentation":
        "pages/ml_lab.py",

    "Sales Forecasting":
        "pages/ml_lab.py",

    "Anomaly Detection":
        "pages/ml_lab.py",

    "Delivery Intelligence":
        "pages/operations.py",

    "Review & NLP Analytics":
        "pages/reviews.py",

    "Pricing & Discount Intelligence":
        "pages/pricing.py",

    "Market & Seller Intelligence":
        "pages/market.py",

    "Business Decision Center":
        "pages/decision_center.py",
}


# ============================================================
# ANALYTICS RESULT KEYS
# ============================================================

ANALYTICS_RESULT_KEYS = [
    "core",
    "customer",
    "product",
    "campaign_impact",
    "market",
    "delivery",
    "pricing",
    "reviews",
    "churn",
    "segmentation",
    "forecast",
    "anomaly",
]


# ============================================================
# MODEL RESULT KEYS
# ============================================================

MODEL_RESULT_KEYS = [
    "churn",
    "segmentation",
    "forecast",
    "anomaly",
]


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

PROJECT_DIRECTORIES = [
    DATA_DIR,
    UPLOAD_DIR,
    PROCESSED_DIR,
    SAMPLE_DIR,
    MODEL_DIR,
    ASSETS_DIR,
    LOGO_DIR,
]


def ensure_project_directories() -> None:
    """Create DataPulse project directories if required."""

    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration() -> list[str]:
    """
    Validate important static DataPulse configuration.

    Returns:
        List of configuration errors.
        Empty list means configuration is valid.
    """

    errors: list[str] = []

    if MAX_UPLOAD_SIZE_MB <= 0:
        errors.append(
            "MAX_UPLOAD_SIZE_MB must be greater than zero."
        )

    if not SUPPORTED_EXTENSIONS:
        errors.append(
            "At least one dataset extension must be supported."
        )

    if MIN_SEGMENT_CLUSTERS < 2:
        errors.append(
            "MIN_SEGMENT_CLUSTERS must be at least 2."
        )

    if MAX_SEGMENT_CLUSTERS < MIN_SEGMENT_CLUSTERS:
        errors.append(
            "MAX_SEGMENT_CLUSTERS must be >= MIN_SEGMENT_CLUSTERS."
        )

    if DEFAULT_FORECAST_PERIODS <= 0:
        errors.append(
            "DEFAULT_FORECAST_PERIODS must be positive."
        )

    if MIN_FORECAST_PERIODS <= 0:
        errors.append(
            "MIN_FORECAST_PERIODS must be positive."
        )

    if not 0 < FORECAST_TEST_SIZE < 1:
        errors.append(
            "FORECAST_TEST_SIZE must be between 0 and 1."
        )

    if not 0 < ANOMALY_CONTAMINATION < 0.5:
        errors.append(
            "ANOMALY_CONTAMINATION must be between 0 and 0.5."
        )

    if not MODULES:
        errors.append(
            "DataPulse must define at least one module."
        )

    for module in MODULES:

        if module not in MODULE_DESCRIPTIONS:
            errors.append(
                f"Missing description for module: {module}"
            )

        if module not in PAGE_ROUTES:
            errors.append(
                f"Missing page route for module: {module}"
            )

    return errors


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Application
    "APP_NAME",
    "APP_SUBTITLE",
    "APP_TITLE",
    "APP_DESCRIPTION",
    "APP_TAGLINE",

    # Creator
    "CREATOR_NAME",
    "CREATOR_ROLE",
    "LINKEDIN_URL",

    # Paths
    "BASE_DIR",
    "DATA_DIR",
    "UPLOAD_DIR",
    "PROCESSED_DIR",
    "SAMPLE_DIR",
    "MODEL_DIR",
    "ASSETS_DIR",
    "LOGO_DIR",

    # Dataset
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_EXCEL_EXTENSIONS",
    "MAX_UPLOAD_SIZE_MB",
    "MAX_UPLOAD_SIZE_BYTES",

    # Analytics
    "DEFAULT_TOP_N",
    "DEFAULT_TOP_PRODUCTS",
    "DEFAULT_TOP_CUSTOMERS",
    "DEFAULT_TOP_SELLERS",
    "DEFAULT_TOP_REGIONS",
    "DEFAULT_TOP_CAMPAIGNS",

    # ML
    "CHURN_THRESHOLD_DAYS",
    "MIN_CHURN_CUSTOMERS",
    "MIN_SEGMENT_CUSTOMERS",
    "MIN_SEGMENT_CLUSTERS",
    "MAX_SEGMENT_CLUSTERS",
    "DEFAULT_SEGMENT_CLUSTERS",
    "MIN_FORECAST_PERIODS",
    "DEFAULT_FORECAST_PERIODS",
    "MAX_FORECAST_PERIODS",
    "FORECAST_TEST_SIZE",
    "MIN_ANOMALY_ROWS",
    "ANOMALY_CONTAMINATION",
    "ANOMALY_RANDOM_STATE",
    "ML_RANDOM_STATE",

    # Features
    "CUSTOMER_FEATURES",
    "PRODUCT_FEATURES",
    "FORECAST_FEATURES",
    "ANOMALY_FEATURES",

    # Validation
    "EDA_REQUIRED_BEFORE_UPLOAD",
    "ALLOW_EMPTY_DATASET",
    "ALLOW_DUPLICATE_COLUMNS",
    "ALLOW_BLANK_COLUMNS",

    # UI
    "PAGE_LAYOUT",
    "INITIAL_SIDEBAR_STATE",
    "PAGE_ICON",

    # Modules
    "MODULES",
    "MODULE_DESCRIPTIONS",
    "PAGE_ROUTES",

    # Results
    "ANALYTICS_RESULT_KEYS",
    "MODEL_RESULT_KEYS",

    # Functions
    "ensure_project_directories",
    "validate_configuration",
]