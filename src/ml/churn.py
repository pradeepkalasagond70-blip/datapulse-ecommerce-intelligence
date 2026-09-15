"""
DataPulse - Churn Prediction

Purpose
-------
Predict which customers are most likely to become inactive/churn.

Approach
--------
- Uses customer-level behavioral features.
- Creates a practical churn label from historical inactivity.
- Trains a classification model when enough usable data exists.
- Uses Random Forest for robust tabular classification.
- Produces churn probability, risk level and business actions.
- Includes feature importance for explainability.

Stable interface
----------------
run_churn_prediction(
    df,
    mapping,
    feature_result=None,
) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

RANDOM_STATE = 42

MIN_CUSTOMERS = 20

CHURN_THRESHOLD_DAYS = 90

FEATURE_COLUMNS = [
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
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _empty_predictions() -> pd.DataFrame:
    """Return a stable empty prediction dataframe."""

    return pd.DataFrame(
        columns=[
            "customer_id",
            "churn_probability",
            "churn_risk",
            "predicted_churn",
            "revenue",
            "orders",
            "recency_days",
            "profit",
            "average_order_value",
            "business_action",
        ]
    )


def _risk_level(
    probability: float,
) -> str:
    """Convert churn probability into a business risk level."""

    if probability >= 0.75:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if probability >= 0.25:
        return "Medium"

    return "Low"


def _business_action(
    risk: str,
) -> str:
    """Generate a practical retention action."""

    actions = {
        "Critical": (
            "Immediate retention intervention"
        ),
        "High": (
            "Launch targeted retention campaign"
        ),
        "Medium": (
            "Monitor and test re-engagement"
        ),
        "Low": (
            "Maintain engagement"
        ),
    }

    return actions.get(
        risk,
        "Monitor customer",
    )


# ---------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------

def _get_customer_features(
    feature_result: Optional[Dict[str, Any]],
    df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Retrieve customer features.

    Prefer the centralized feature-engineering output.
    """

    if feature_result:
        features = feature_result.get(
            "customer_features"
        )

        if (
            isinstance(features, pd.DataFrame)
            and not features.empty
        ):
            return features.copy()

    # Fallback is intentionally conservative.
    # The centralized feature-engineering pipeline is preferred.
    if (
        isinstance(df, pd.DataFrame)
        and not df.empty
    ):
        possible_columns = [
            column
            for column in [
                "customer_id",
                *FEATURE_COLUMNS,
            ]
            if column in df.columns
        ]

        if (
            "customer_id"
            in possible_columns
            and len(possible_columns) > 1
        ):
            return df[
                possible_columns
            ].copy()

    return pd.DataFrame()


# ---------------------------------------------------------------------
# Churn label construction
# ---------------------------------------------------------------------

def _create_churn_label(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a historical churn label.

    Customers with recency >= 90 days are treated as churned/inactive.

    Important:
    This is a behavioral proxy label, not a ground-truth cancellation
    label. It is suitable for portfolio/business analytics when an
    explicit churn field is unavailable.
    """

    result = features.copy()

    if "recency_days" not in result.columns:
        return pd.DataFrame()

    result["churn_label"] = (
        result["recency_days"]
        >= CHURN_THRESHOLD_DAYS
    ).astype(int)

    return result


# ---------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------

def _prepare_training_data(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare X, y and metadata for model training.
    """

    required = [
        "customer_id",
        "churn_label",
    ]

    if not all(
        column in features.columns
        for column in required
    ):
        return (
            pd.DataFrame(),
            pd.Series(dtype=int),
            pd.DataFrame(),
        )

    available_features = [
        column
        for column in FEATURE_COLUMNS
        if column in features.columns
    ]

    if not available_features:
        return (
            pd.DataFrame(),
            pd.Series(dtype=int),
            pd.DataFrame(),
        )

    metadata = features[
        [
            "customer_id"
        ]
        + [
            column
            for column in [
                "revenue",
                "orders",
                "recency_days",
                "profit",
                "average_order_value",
            ]
            if column in features.columns
        ]
    ].copy()

    X = features[
        available_features
    ].copy()

    y = features[
        "churn_label"
    ].astype(int)

    # Remove infinite values.
    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # Ensure numeric.
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    return X, y, metadata


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def _build_model() -> Pipeline:
    """
    Build the churn classification pipeline.

    Pipeline guarantees consistent preprocessing during training
    and prediction.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ---------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------

def _evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Calculate classification metrics."""

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    result = {
        "accuracy": round(
            float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_score": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_test,
                predictions,
            ).tolist()
        ),
        "classification_report": (
            classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            )
        ),
    }

    # ROC-AUC requires both classes in the test set.
    if y_test.nunique() == 2:
        result["roc_auc"] = round(
            float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),
            4,
        )
    else:
        result["roc_auc"] = None

    return result


# ---------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------

def _build_feature_importance(
    model: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract Random Forest feature importance."""

    try:
        estimator = model.named_steps[
            "model"
        ]

        importance = estimator.feature_importances_

    except Exception:
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
                "importance_pct",
            ]
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    total = result["importance"].sum()

    if total > 0:
        result["importance_pct"] = (
            result["importance"]
            / total
            * 100
        )
    else:
        result["importance_pct"] = 0.0

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------

def _build_predictions(
    model: Pipeline,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate customer churn predictions."""

    if features.empty:
        return _empty_predictions()

    X = features[
        feature_columns
    ].copy()

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predicted = (
        probabilities >= 0.50
    ).astype(int)

    result = pd.DataFrame(
        {
            "customer_id": features[
                "customer_id"
            ].values,
            "churn_probability": (
                probabilities
            ),
            "predicted_churn": predicted,
        }
    )

    result["churn_probability"] = (
        result["churn_probability"]
        .clip(0, 1)
        * 100
    )

    result["churn_probability"] = (
        result["churn_probability"]
        .round(2)
    )

    result["churn_risk"] = (
        result["churn_probability"]
        .apply(
            lambda value: _risk_level(
                value / 100
            )
        )
    )

    result["business_action"] = (
        result["churn_risk"]
        .apply(
            _business_action
        )
    )

    metadata_columns = [
        column
        for column in [
            "customer_id",
            "revenue",
            "orders",
            "recency_days",
            "profit",
            "average_order_value",
        ]
        if column in features.columns
    ]

    metadata = features[
        metadata_columns
    ].copy()

    result = result.drop(
        columns=[
            column
            for column in [
                "revenue",
                "orders",
                "recency_days",
                "profit",
                "average_order_value",
            ]
            if column in result.columns
        ],
        errors="ignore",
    )

    result = result.merge(
        metadata,
        on="customer_id",
        how="left",
    )

    preferred_order = [
        "customer_id",
        "churn_probability",
        "churn_risk",
        "predicted_churn",
        "revenue",
        "orders",
        "recency_days",
        "profit",
        "average_order_value",
        "business_action",
    ]

    existing = [
        column
        for column in preferred_order
        if column in result.columns
    ]

    result = result[
        existing
    ]

    return result.sort_values(
        "churn_probability",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------

def _build_risk_summary(
    predictions: pd.DataFrame,
) -> Dict[str, Any]:
    """Summarize customer churn risk."""

    if predictions.empty:
        return {
            "customers_scored": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "at_risk_customers": 0,
            "at_risk_rate": 0.0,
            "at_risk_revenue": 0.0,
        }

    risk_counts = (
        predictions[
            "churn_risk"
        ]
        .value_counts()
        .to_dict()
    )

    at_risk = predictions[
        predictions["churn_risk"].isin(
            [
                "Critical",
                "High",
            ]
        )
    ]

    total_customers = len(
        predictions
    )

    return {
        "customers_scored": int(
            total_customers
        ),
        "critical": int(
            risk_counts.get(
                "Critical",
                0,
            )
        ),
        "high": int(
            risk_counts.get(
                "High",
                0,
            )
        ),
        "medium": int(
            risk_counts.get(
                "Medium",
                0,
            )
        ),
        "low": int(
            risk_counts.get(
                "Low",
                0,
            )
        ),
        "at_risk_customers": int(
            len(at_risk)
        ),
        "at_risk_rate": round(
            len(at_risk)
            / total_customers
            * 100
            if total_customers
            else 0.0,
            2,
        ),
        "at_risk_revenue": float(
            at_risk["revenue"].sum()
        )
        if "revenue" in at_risk.columns
        else 0.0,
    }


# ---------------------------------------------------------------------
# High-value churn risk
# ---------------------------------------------------------------------

def _build_high_value_risk(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Identify high-value customers with elevated churn risk."""

    if (
        predictions.empty
        or "revenue" not in predictions.columns
    ):
        return pd.DataFrame(
            columns=[
                "customer_id",
                "revenue",
                "churn_probability",
                "churn_risk",
                "business_action",
            ]
        )

    revenue_threshold = (
        predictions["revenue"]
        .median()
    )

    result = predictions[
        (
            predictions["revenue"]
            >= revenue_threshold
        )
        & (
            predictions["churn_risk"]
            .isin(
                [
                    "Critical",
                    "High",
                ]
            )
        )
    ].copy()

    columns = [
        "customer_id",
        "revenue",
        "churn_probability",
        "churn_risk",
        "business_action",
    ]

    return result[
        [
            column
            for column in columns
            if column in result.columns
        ]
    ].sort_values(
        "churn_probability",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

def _build_recommendations(
    risk_summary: Dict[str, Any],
    high_value_risk: pd.DataFrame,
) -> list[Dict[str, Any]]:
    """Generate business recommendations."""

    recommendations = []

    at_risk = risk_summary.get(
        "at_risk_customers",
        0,
    )

    if at_risk > 0:
        recommendations.append(
            {
                "priority": "High",
                "type": "Retention",
                "message": (
                    f"{at_risk} customer(s) have high or critical "
                    "churn risk. Prioritize targeted re-engagement."
                ),
            }
        )

    if not high_value_risk.empty:
        revenue = float(
            high_value_risk[
                "revenue"
            ].sum()
        )

        recommendations.append(
            {
                "priority": "Critical",
                "type": "High-Value Retention",
                "message": (
                    f"{len(high_value_risk)} high-value customer(s) "
                    f"are at elevated churn risk, representing "
                    f"approximately {revenue:,.2f} in historical revenue."
                ),
            }
        )

    if risk_summary.get(
        "at_risk_rate",
        0,
    ) >= 30:
        recommendations.append(
            {
                "priority": "High",
                "type": "Portfolio Risk",
                "message": (
                    "A large share of customers is classified as "
                    "at risk. Review retention strategy and customer "
                    "engagement frequency."
                ),
            }
        )

    return recommendations


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_churn_prediction(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    feature_result: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Run customer churn prediction.

    Parameters
    ----------
    df:
        Original analysis-ready dataframe.

    mapping:
        Schema mapping.

    feature_result:
        Output from build_ml_features().

    Returns
    -------
    dict
        Stable churn prediction contract.
    """

    empty_response = {
        "available": False,
        "model_ready": False,
        "model": None,
        "predictions": _empty_predictions(),
        "risk_summary": {},
        "high_value_risk": pd.DataFrame(),
        "feature_importance": pd.DataFrame(),
        "metrics": {},
        "recommendations": [],
        "capabilities": {},
        "warnings": [],
    }

    # -------------------------------------------------------------
    # Retrieve features
    # -------------------------------------------------------------

    features = _get_customer_features(
        feature_result,
        df,
    )

    if features.empty:
        empty_response["warnings"] = [
            "Customer feature data is unavailable."
        ]

        return empty_response

    # -------------------------------------------------------------
    # Validate minimum customer count
    # -------------------------------------------------------------

    if len(features) < MIN_CUSTOMERS:
        empty_response["warnings"] = [
            (
                f"Churn prediction requires at least "
                f"{MIN_CUSTOMERS} customers. "
                f"Only {len(features)} customer(s) are available."
            )
        ]

        return empty_response

    # -------------------------------------------------------------
    # Create churn label
    # -------------------------------------------------------------

    labeled = _create_churn_label(
        features
    )

    if labeled.empty:
        empty_response["warnings"] = [
            "Recency information is unavailable for churn labeling."
        ]

        return empty_response

    # -------------------------------------------------------------
    # Check class balance
    # -------------------------------------------------------------

    class_counts = (
        labeled["churn_label"]
        .value_counts()
    )

    if len(class_counts) < 2:
        empty_response["warnings"] = [
            (
                "The dataset does not contain both active and "
                "inactive customer groups using the current "
                f"{CHURN_THRESHOLD_DAYS}-day churn threshold."
            )
        ]

        return empty_response

    if class_counts.min() < 2:
        empty_response["warnings"] = [
            "There are too few examples in one churn class "
            "to train a reliable classification model."
        ]

        return empty_response

    # -------------------------------------------------------------
    # Prepare X / y
    # -------------------------------------------------------------

    X, y, metadata = _prepare_training_data(
        labeled
    )

    if X.empty or y.empty:
        empty_response["warnings"] = [
            "No usable churn features are available."
        ]

        return empty_response

    # -------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------

    try:
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        )

    except ValueError:
        empty_response["warnings"] = [
            "The dataset is too small or imbalanced for a stratified "
            "churn train/test split."
        ]

        return empty_response

    # -------------------------------------------------------------
    # Build and train model
    # -------------------------------------------------------------

    model = _build_model()

    try:
        model.fit(
            X_train,
            y_train,
        )

    except Exception as exc:
        empty_response["warnings"] = [
            f"Churn model training failed: {exc}"
        ]

        return empty_response

    # -------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------

    metrics = _evaluate_model(
        model,
        X_test,
        y_test,
    )

    # -------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------

    feature_importance = (
        _build_feature_importance(
            model,
            list(X.columns),
        )
    )

    # -------------------------------------------------------------
    # Score all customers
    # -------------------------------------------------------------

    predictions = _build_predictions(
        model,
        labeled,
        list(X.columns),
    )

    risk_summary = _build_risk_summary(
        predictions
    )

    high_value_risk = (
        _build_high_value_risk(
            predictions
        )
    )

    recommendations = (
        _build_recommendations(
            risk_summary,
            high_value_risk,
        )
    )

    # -------------------------------------------------------------
    # Capabilities
    # -------------------------------------------------------------

    capabilities = {
        "churn_prediction": True,
        "risk_scoring": not predictions.empty,
        "high_value_churn_detection": (
            not high_value_risk.empty
        ),
        "feature_importance": (
            not feature_importance.empty
        ),
        "model_evaluation": bool(metrics),
    }

    return {
        "available": True,
        "model_ready": True,
        "model": model,
        "predictions": predictions,
        "risk_summary": risk_summary,
        "high_value_risk": high_value_risk,
        "feature_importance": feature_importance,
        "metrics": metrics,
        "recommendations": recommendations,
        "capabilities": capabilities,
        "training": {
            "customers": int(len(labeled)),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "churn_threshold_days": CHURN_THRESHOLD_DAYS,
            "churn_class_distribution": {
                str(key): int(value)
                for key, value
                in class_counts.items()
            },
            "features_used": list(
                X.columns
            ),
        },
        "warnings": [
            (
                "Churn is modeled using a "
                f"{CHURN_THRESHOLD_DAYS}-day inactivity proxy. "
                "It should be interpreted as predicted inactivity "
                "risk rather than confirmed customer cancellation."
            )
        ],
    }


# ---------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------

def get_high_risk_customers(
    churn_result: Dict[str, Any],
    minimum_probability: float = 50.0,
) -> pd.DataFrame:
    """Return customers above a specified churn probability."""

    predictions = churn_result.get(
        "predictions",
        pd.DataFrame(),
    )

    if predictions.empty:
        return _empty_predictions()

    if (
        "churn_probability"
        not in predictions.columns
    ):
        return _empty_predictions()

    return predictions[
        predictions[
            "churn_probability"
        ]
        >= minimum_probability
    ].copy()


def get_churn_feature_importance(
    churn_result: Dict[str, Any],
) -> pd.DataFrame:
    """Return churn feature importance."""

    return churn_result.get(
        "feature_importance",
        pd.DataFrame(),
    )