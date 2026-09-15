"""
DataPulse Machine Learning Package

Contains the machine-learning engines used by DataPulse:

- Churn Prediction
- Customer Segmentation
- Sales Forecasting
- Anomaly Detection

Each ML module is designed to:
1. Accept analysis-ready business data.
2. Use the shared schema mapping and feature-engineering layer.
3. Train models at runtime when the uploaded dataset is processed.
4. Return structured results for Streamlit pages and the Insight Engine.
5. Gracefully report when the dataset is insufficient for a model.

The package intentionally keeps model implementations separated from
application orchestration so that app.py remains a stable shell.
"""

__version__ = "1.0.0"
__package_name__ = "DataPulse ML"
__author__ = "Pradeep Kalasagond"

# Public ML module names
ML_MODULES = (
    "churn",
    "segmentation",
    "forecasting",
    "anomaly_detection",
)

ML_MODULE_DESCRIPTIONS = {
    "churn": "Predict customer churn risk using behavioral and transactional features.",
    "segmentation": "Group customers into actionable behavioral and value segments.",
    "forecasting": "Forecast future sales using historical time-series patterns.",
    "anomaly_detection": "Detect unusual sales and transaction behavior.",
}

__all__ = [
    "ML_MODULES",
    "ML_MODULE_DESCRIPTIONS",
]