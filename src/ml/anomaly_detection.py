"""
DataPulse - Anomaly Detection
-----------------------------
Detect unusual transactions / sales observations using Isolation Forest.

Public interface:
    run_anomaly_detection(df, mapping, feature_result=None)

Expected feature_result:
    Output from src.feature_engineering.build_ml_features()

Dependencies:
    pandas
    numpy
    scikit-learn
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_OBSERVATIONS = 20

CONTAMINATION = 0.05

RANDOM_STATE = 42

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


# ---------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------

def _empty_result(
    warnings: Optional[list[str]] = None,
    capabilities: Optional[dict[str, bool]] = None,
) -> dict[str, Any]:
    """Return a consistent unavailable anomaly result."""

    return {
        "available": False,
        "model_ready": False,
        "model": None,
        "anomalies": pd.DataFrame(),
        "anomaly_summary": {},
        "feature_statistics": pd.DataFrame(),
        "severity_summary": pd.DataFrame(),
        "metrics": {},
        "recommendations": [],
        "capabilities": capabilities or {},
        "training": {},
        "warnings": warnings or [],
    }


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def _prepare_features(
    feature_result: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract anomaly features from feature engineering output.
    """

    anomaly_features = feature_result.get(
        "anomaly_features"
    )

    if (
        anomaly_features is None
        or not isinstance(
            anomaly_features,
            pd.DataFrame,
        )
        or anomaly_features.empty
    ):
        return pd.DataFrame(), []

    data = anomaly_features.copy()

    usable_features = [
        column
        for column in ANOMALY_FEATURES
        if column in data.columns
    ]

    if not usable_features:
        return pd.DataFrame(), []

    for column in usable_features:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Remove columns with no usable information.
    usable_features = [
        column
        for column in usable_features
        if data[column].notna().sum() > 0
        and data[column].nunique(
            dropna=True
        ) > 1
    ]

    if not usable_features:
        return pd.DataFrame(), []

    return data, usable_features


def _build_pipeline() -> Pipeline:
    """Build robust anomaly detection pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=CONTAMINATION,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------

def _severity_from_score(
    score: float,
) -> str:
    """
    Convert anomaly score into a business-friendly severity.

    Isolation Forest decision_function:
        positive -> more normal
        negative -> more anomalous
    """

    if score <= -0.20:
        return "Critical"

    if score <= -0.10:
        return "High"

    if score < 0:
        return "Medium"

    return "Normal"


def _build_business_reason(
    row: pd.Series,
    feature_medians: pd.Series,
) -> str:
    """Generate a human-readable reason for an anomaly."""

    reasons: list[str] = []

    checks = [
        (
            "sales_amount",
            "sales value",
        ),
        (
            "quantity",
            "quantity",
        ),
        (
            "unit_price",
            "unit price",
        ),
        (
            "profit",
            "profit",
        ),
        (
            "discount_pct",
            "discount",
        ),
    ]

    for column, label in checks:

        if (
            column not in row.index
            or column not in feature_medians.index
        ):
            continue

        value = row.get(
            column,
            np.nan,
        )

        median = feature_medians.get(
            column,
            np.nan,
        )

        if (
            pd.isna(value)
            or pd.isna(median)
            or abs(float(median)) < 1e-9
        ):
            continue

        ratio = abs(
            float(value)
        ) / abs(
            float(median)
        )

        if ratio >= 3:
            if value > median:
                reasons.append(
                    f"unusually high {label}"
                )
            else:
                reasons.append(
                    f"unusually low {label}"
                )

    if not reasons:
        return (
            "The transaction has an unusual combination "
            "of business characteristics compared with normal observations."
        )

    if len(reasons) == 1:
        return reasons[0].capitalize() + "."

    return (
        "Multiple unusual signals: "
        + ", ".join(reasons[:3])
        + "."
    )


# ---------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------

def _build_feature_statistics(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Build descriptive statistics for anomaly features."""

    records = []

    for feature in features:

        values = pd.to_numeric(
            data[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        records.append(
            {
                "feature": feature,
                "mean": float(
                    values.mean()
                ),
                "median": float(
                    values.median()
                ),
                "std": float(
                    values.std()
                )
                if len(values) > 1
                else 0.0,
                "minimum": float(
                    values.min()
                ),
                "maximum": float(
                    values.max()
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def _build_summary(
    result: pd.DataFrame,
) -> dict[str, Any]:
    """Create anomaly summary metrics."""

    if result.empty:
        return {
            "total_observations": 0,
            "anomaly_count": 0,
            "normal_count": 0,
            "anomaly_rate_pct": 0.0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
        }

    total = len(result)

    anomaly_count = int(
        result["is_anomaly"]
        .sum()
    )

    normal_count = (
        total - anomaly_count
    )

    anomaly_rate = (
        anomaly_count
        / max(total, 1)
        * 100
    )

    severity_counts = (
        result[
            result["is_anomaly"]
        ]["anomaly_severity"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_observations": int(
            total
        ),
        "anomaly_count": anomaly_count,
        "normal_count": int(
            normal_count
        ),
        "anomaly_rate_pct": round(
            float(anomaly_rate),
            2,
        ),
        "critical_count": int(
            severity_counts.get(
                "Critical",
                0,
            )
        ),
        "high_count": int(
            severity_counts.get(
                "High",
                0,
            )
        ),
        "medium_count": int(
            severity_counts.get(
                "Medium",
                0,
            )
        ),
    }


# ---------------------------------------------------------------------
# Severity summary
# ---------------------------------------------------------------------

def _build_severity_summary(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """Create severity distribution."""

    if result.empty:
        return pd.DataFrame(
            columns=[
                "anomaly_severity",
                "observation_count",
                "share_pct",
            ]
        )

    counts = (
        result[
            result["is_anomaly"]
        ]["anomaly_severity"]
        .value_counts()
        .rename_axis(
            "anomaly_severity"
        )
        .reset_index(
            name="observation_count"
        )
    )

    total_anomalies = max(
        int(
            result["is_anomaly"].sum()
        ),
        1,
    )

    counts["share_pct"] = (
        counts["observation_count"]
        / total_anomalies
        * 100
    ).round(2)

    return counts


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

def _build_recommendations(
    summary: dict[str, Any],
    anomalies: pd.DataFrame,
) -> list[dict[str, str]]:
    """Generate business recommendations."""

    recommendations: list[
        dict[str, str]
    ] = []

    anomaly_rate = summary.get(
        "anomaly_rate_pct",
        0,
    )

    critical_count = summary.get(
        "critical_count",
        0,
    )

    high_count = summary.get(
        "high_count",
        0,
    )

    if critical_count > 0:
        recommendations.append(
            {
                "priority": "Critical",
                "area": "Transaction Review",
                "recommendation": (
                    "Immediately review critical anomalies for "
                    "fraud, data-quality issues, pricing errors, "
                    "or unusually large transactions."
                ),
            }
        )

    if high_count > 0:
        recommendations.append(
            {
                "priority": "High",
                "area": "Operational Monitoring",
                "recommendation": (
                    "Investigate high-severity anomalies and determine "
                    "whether they represent genuine business events "
                    "or abnormal transactions."
                ),
            }
        )

    if anomaly_rate > 10:
        recommendations.append(
            {
                "priority": "High",
                "area": "Data Quality",
                "recommendation": (
                    "The anomaly rate is relatively high. Review "
                    "data collection, pricing, discount and transaction "
                    "processes for systematic issues."
                ),
            }
        )

    elif anomaly_rate <= 2:
        recommendations.append(
            {
                "priority": "Low",
                "area": "Monitoring",
                "recommendation": (
                    "Anomaly levels are relatively low. Continue "
                    "monitoring unusual transactions for emerging patterns."
                ),
            }
        )

    if not anomalies.empty:

        high_discount = pd.DataFrame()

        if "discount_pct" in anomalies.columns:
            high_discount = anomalies[
                anomalies[
                    "discount_pct"
                ]
                >= 50
            ]

        if not high_discount.empty:
            recommendations.append(
                {
                    "priority": "Medium",
                    "area": "Discount Control",
                    "recommendation": (
                        "Review unusually high-discount transactions "
                        "to confirm promotional rules and protect margins."
                    ),
                }
            )

    return recommendations


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_anomaly_detection(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    feature_result: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Run Isolation Forest anomaly detection.

    Parameters
    ----------
    df:
        Original analysis-ready dataset.

    mapping:
        Schema mapping from schema_mapper.py.

    feature_result:
        Output from feature_engineering.build_ml_features().

    Returns
    -------
    dict
        Stable DataPulse anomaly detection result.
    """

    del df
    del mapping

    capabilities = {
        "anomaly_features": False,
        "isolation_forest": False,
        "severity_scoring": False,
        "feature_statistics": False,
        "business_reasons": False,
        "recommendations": False,
    }

    if not feature_result:
        return _empty_result(
            warnings=[
                "ML feature engineering results were not provided."
            ],
            capabilities=capabilities,
        )

    data, features = _prepare_features(
        feature_result
    )

    if data.empty:
        return _empty_result(
            warnings=[
                "Transaction-level anomaly features are unavailable."
            ],
            capabilities=capabilities,
        )

    if len(data) < MIN_OBSERVATIONS:
        return _empty_result(
            warnings=[
                f"Anomaly detection requires at least "
                f"{MIN_OBSERVATIONS} observations; "
                f"only {len(data)} were available."
            ],
            capabilities=capabilities,
        )

    capabilities[
        "anomaly_features"
    ] = True

    # -----------------------------------------------------------------
    # Model input
    # -----------------------------------------------------------------

    X = data[
        features
    ].copy()

    # -----------------------------------------------------------------
    # Train Isolation Forest
    # -----------------------------------------------------------------

    pipeline = _build_pipeline()

    try:
        pipeline.fit(
            X
        )

    except Exception as exc:
        return _empty_result(
            warnings=[
                f"Anomaly detection training failed: {exc}"
            ],
            capabilities=capabilities,
        )

    capabilities[
        "isolation_forest"
    ] = True

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------

    try:
        predictions = pipeline.predict(
            X
        )

        decision_scores = (
            pipeline.decision_function(
                X
            )
        )

    except Exception as exc:
        return _empty_result(
            warnings=[
                f"Anomaly scoring failed: {exc}"
            ],
            capabilities=capabilities,
        )

    # Isolation Forest:
    #   -1 = anomaly
    #    1 = normal

    result = data.copy()

    result["is_anomaly"] = (
        predictions == -1
    )

    result["anomaly_score"] = (
        decision_scores
    )

    result["anomaly_severity"] = [
        _severity_from_score(
            float(score)
        )
        if prediction == -1
        else "Normal"
        for prediction, score in zip(
            predictions,
            decision_scores,
        )
    ]

    # -----------------------------------------------------------------
    # Business reasons
    # -----------------------------------------------------------------

    feature_medians = (
        X.median(
            numeric_only=True
        )
    )

    result["anomaly_reason"] = [
        _build_business_reason(
            row,
            feature_medians,
        )
        if is_anomaly
        else "Within normal transaction pattern."
        for (_, row), is_anomaly in zip(
            result.iterrows(),
            result["is_anomaly"],
        )
    ]

    capabilities[
        "severity_scoring"
    ] = True

    capabilities[
        "business_reasons"
    ] = True

    # -----------------------------------------------------------------
    # Observation identifier
    # -----------------------------------------------------------------

    if "transaction_id" in result.columns:
        result["observation_id"] = (
            result["transaction_id"]
            .astype(str)
        )

    elif "order_id" in result.columns:
        result["observation_id"] = (
            result["order_id"]
            .astype(str)
        )

    else:
        result["observation_id"] = (
            result.index.astype(str)
        )

    # -----------------------------------------------------------------
    # Sort anomalies first
    # -----------------------------------------------------------------

    result = result.sort_values(
        [
            "is_anomaly",
            "anomaly_score",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    summary = _build_summary(
        result
    )

    # -----------------------------------------------------------------
    # Feature statistics
    # -----------------------------------------------------------------

    feature_statistics = (
        _build_feature_statistics(
            data,
            features,
        )
    )

    capabilities[
        "feature_statistics"
    ] = True

    # -----------------------------------------------------------------
    # Severity distribution
    # -----------------------------------------------------------------

    severity_summary = (
        _build_severity_summary(
            result
        )
    )

    # -----------------------------------------------------------------
    # Anomaly-only dataset
    # -----------------------------------------------------------------

    anomalies = result[
        result["is_anomaly"]
    ].copy()

    # Keep anomaly records easy to display.
    preferred_columns = [
        "observation_id",
        "is_anomaly",
        "anomaly_severity",
        "anomaly_score",
        "anomaly_reason",
        "transaction_id",
        "order_id",
        "customer_id",
        "product_id",
        "sales_amount",
        "quantity",
        "unit_price",
        "profit",
        "discount_pct",
    ]

    anomaly_columns = [
        column
        for column in preferred_columns
        if column in anomalies.columns
    ]

    if anomaly_columns:
        anomalies = anomalies[
            anomaly_columns
        ]

    # -----------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------

    recommendations = (
        _build_recommendations(
            summary,
            anomalies,
        )
    )

    capabilities[
        "recommendations"
    ] = True

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    anomaly_rate = summary[
        "anomaly_rate_pct"
    ]

    metrics = {
        "observations": int(
            len(result)
        ),
        "anomalies": int(
            summary["anomaly_count"]
        ),
        "anomaly_rate_pct": round(
            float(anomaly_rate),
            2,
        ),
        "contamination": CONTAMINATION,
        "model": "Isolation Forest",
    }

    # -----------------------------------------------------------------
    # Training metadata
    # -----------------------------------------------------------------

    training = {
        "algorithm": "Isolation Forest",
        "n_estimators": 300,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "features": features,
        "feature_count": len(features),
        "observations": int(
            len(data)
        ),
        "preprocessing": [
            "Median imputation",
            "Standard scaling",
        ],
    }

    return {
        "available": True,
        "model_ready": True,
        "model": pipeline,
        "anomalies": anomalies,
        "anomaly_summary": summary,
        "feature_statistics": feature_statistics,
        "severity_summary": severity_summary,
        "metrics": metrics,
        "recommendations": recommendations,
        "capabilities": capabilities,
        "training": training,
        "warnings": [],
    }


# ---------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------

def get_anomalies(
    result: dict[str, Any],
) -> pd.DataFrame:
    """Return detected anomalies."""

    return result.get(
        "anomalies",
        pd.DataFrame(),
    )


def get_anomaly_summary(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return anomaly summary."""

    return result.get(
        "anomaly_summary",
        {},
    )


def get_anomaly_recommendations(
    result: dict[str, Any],
) -> list[dict[str, str]]:
    """Return anomaly recommendations."""

    return result.get(
        "recommendations",
        [],
    )


# Backward-compatible alias.
run_anomaly_detection_engine = run_anomaly_detection