"""
DataPulse - Customer Segmentation
---------------------------------
Customer segmentation using RFM/business features and K-Means clustering.

Public interface:
    run_customer_segmentation(df, mapping, feature_result=None)

Expected feature_result:
    Output from src.feature_engineering.build_ml_features()

Dependencies:
    pandas
    numpy
    scikit-learn
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_CUSTOMERS = 8
MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
RANDOM_STATE = 42
N_INIT = 20

SEGMENT_FEATURES = [
    "revenue",
    "orders",
    "quantity",
    "profit",
    "average_order_value",
    "recency_days",
    "purchase_frequency",
    "profit_margin",
    "discount_rate",
    "ltv_proxy",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _empty_result(
    warnings: Optional[list[str]] = None,
    capabilities: Optional[dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Return a consistent unavailable result."""

    return {
        "available": False,
        "model_ready": False,
        "model": None,
        "segments": pd.DataFrame(),
        "segment_summary": pd.DataFrame(),
        "segment_distribution": pd.DataFrame(),
        "segment_profiles": pd.DataFrame(),
        "customer_assignments": pd.DataFrame(),
        "metrics": {},
        "recommendations": [],
        "capabilities": capabilities or {},
        "training": {},
        "warnings": warnings or [],
    }


def _safe_numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return requested numeric columns, creating missing ones."""

    result = pd.DataFrame(index=frame.index)

    for column in columns:
        if column in frame.columns:
            result[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
        else:
            result[column] = np.nan

    return result


def _prepare_features(
    customer_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Prepare customer-level features for clustering.

    Returns:
        original_customer_data,
        transformed_input,
        usable_feature_names
    """

    data = customer_features.copy()

    usable_features = [
        column
        for column in SEGMENT_FEATURES
        if column in data.columns
    ]

    if len(usable_features) < 3:
        return data, pd.DataFrame(), usable_features

    numeric = _safe_numeric(data, usable_features)

    # Business metrics such as revenue, profit and orders can be highly
    # skewed. Log transformation makes clustering more stable.
    transformed = numeric.copy()

    skewed_columns = [
        "revenue",
        "orders",
        "quantity",
        "profit",
        "average_order_value",
        "purchase_frequency",
        "ltv_proxy",
    ]

    for column in skewed_columns:
        if column in transformed.columns:
            values = transformed[column]

            # Shift only when negative values exist.
            minimum = values.min(skipna=True)

            if pd.notna(minimum) and minimum < 0:
                values = values - minimum

            transformed[column] = np.log1p(
                values.clip(lower=0)
            )

    # Recency should remain interpretable and should not be log-transformed.
    return data, transformed, usable_features


def _select_cluster_count(
    scaled_features: np.ndarray,
) -> tuple[int, dict[int, float]]:
    """
    Select K using silhouette score.

    For small datasets, the available K range is automatically reduced.
    """

    customer_count = len(scaled_features)

    max_k = min(
        MAX_CLUSTERS,
        customer_count - 1,
    )

    if max_k < MIN_CLUSTERS:
        return MIN_CLUSTERS, {}

    scores: dict[int, float] = {}

    for k in range(MIN_CLUSTERS, max_k + 1):
        try:
            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=N_INIT,
            )

            labels = model.fit_predict(scaled_features)

            # Silhouette requires at least two actual clusters.
            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(
                scaled_features,
                labels,
            )

            scores[k] = float(score)

        except Exception:
            continue

    if not scores:
        return min(MIN_CLUSTERS, max_k), {}

    best_k = max(
        scores,
        key=scores.get,
    )

    return int(best_k), scores


def _segment_name(
    profile: pd.Series,
    overall: pd.Series,
) -> str:
    """Create a business-friendly segment label."""

    revenue = profile.get("revenue", np.nan)
    orders = profile.get("orders", np.nan)
    recency = profile.get("recency_days", np.nan)
    profit_margin = profile.get("profit_margin", np.nan)

    revenue_median = overall.get("revenue", np.nan)
    orders_median = overall.get("orders", np.nan)
    recency_median = overall.get("recency_days", np.nan)
    margin_median = overall.get("profit_margin", np.nan)

    high_value = (
        pd.notna(revenue)
        and pd.notna(revenue_median)
        and revenue >= revenue_median
    )

    frequent = (
        pd.notna(orders)
        and pd.notna(orders_median)
        and orders >= orders_median
    )

    recent = (
        pd.notna(recency)
        and pd.notna(recency_median)
        and recency <= recency_median
    )

    profitable = (
        pd.notna(profit_margin)
        and pd.notna(margin_median)
        and profit_margin >= margin_median
    )

    if high_value and frequent and recent:
        return "High-Value Loyal"

    if high_value and recent:
        return "High-Value Active"

    if high_value and not recent:
        return "High-Value At Risk"

    if frequent and recent:
        return "Active Repeat"

    if profitable and recent:
        return "Profitable Active"

    if not recent:
        return "At Risk"

    if not frequent:
        return "Low Engagement"

    return "Emerging Customers"


def _build_profiles(
    customer_data: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build customer segment profiles."""

    data = customer_data.copy()
    data["segment_id"] = labels

    numeric_columns = [
        column
        for column in SEGMENT_FEATURES
        if column in data.columns
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    overall = data[numeric_columns].median(
        numeric_only=True
    )

    grouped = (
        data.groupby("segment_id")[numeric_columns]
        .median()
        .reset_index()
    )

    counts = (
        data.groupby("segment_id")
        .size()
        .reset_index(name="customer_count")
    )

    profiles = grouped.merge(
        counts,
        on="segment_id",
        how="left",
    )

    profiles["customer_share_pct"] = (
        profiles["customer_count"]
        / max(len(data), 1)
        * 100
    )

    profiles["segment_name"] = profiles.apply(
        lambda row: _segment_name(row, overall),
        axis=1,
    )

    return profiles


def _build_assignments(
    customer_data: pd.DataFrame,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Create customer-level segment assignments."""

    assignments = pd.DataFrame(index=customer_data.index)

    if "customer_id" in customer_data.columns:
        assignments["customer_id"] = (
            customer_data["customer_id"].astype(str)
        )
    else:
        assignments["customer_id"] = customer_data.index.astype(str)

    assignments["segment_id"] = labels.astype(int)

    if probabilities is not None:
        assignments["segment_confidence"] = probabilities

    for column in [
        "revenue",
        "orders",
        "profit",
        "average_order_value",
        "recency_days",
        "ltv_proxy",
    ]:
        if column in customer_data.columns:
            assignments[column] = customer_data[column].values

    return assignments


def _attach_segment_names(
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Attach business segment names to customer assignments."""

    name_map = profiles.set_index(
        "segment_id"
    )["segment_name"].to_dict()

    result = assignments.copy()

    result["segment_name"] = (
        result["segment_id"]
        .map(name_map)
        .fillna("Unclassified")
    )

    return result


def _build_recommendations(
    profiles: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate business recommendations from segment profiles."""

    recommendations: list[dict[str, Any]] = []

    for _, row in profiles.iterrows():

        name = str(
            row.get(
                "segment_name",
                "Customer Segment",
            )
        )

        share = float(
            row.get(
                "customer_share_pct",
                0,
            )
        )

        revenue = row.get("revenue", np.nan)
        orders = row.get("orders", np.nan)
        recency = row.get("recency_days", np.nan)

        if name == "High-Value Loyal":
            action = (
                "Protect this segment with loyalty benefits, "
                "personalized offers and retention programs."
            )
            priority = "High"

        elif name == "High-Value At Risk":
            action = (
                "Prioritize win-back campaigns and personalized "
                "outreach because these customers have high value "
                "but weakening recency."
            )
            priority = "Critical"

        elif name == "High-Value Active":
            action = (
                "Increase cross-sell and upsell opportunities "
                "while engagement is strong."
            )
            priority = "High"

        elif name == "Active Repeat":
            action = (
                "Use loyalty incentives and product recommendations "
                "to move repeat customers toward higher lifetime value."
            )
            priority = "Medium"

        elif name == "At Risk":
            action = (
                "Run reactivation campaigns and identify the "
                "products or service issues contributing to inactivity."
            )
            priority = "High"

        elif name == "Low Engagement":
            action = (
                "Use targeted onboarding, personalized recommendations "
                "and low-friction promotions to increase engagement."
            )
            priority = "Medium"

        elif name == "Profitable Active":
            action = (
                "Protect profitability while increasing order frequency "
                "through relevant cross-sell opportunities."
            )
            priority = "Medium"

        else:
            action = (
                "Nurture this segment with personalized recommendations "
                "and monitor its movement toward higher-value segments."
            )
            priority = "Low"

        recommendations.append(
            {
                "segment_name": name,
                "customer_share_pct": round(share, 2),
                "median_revenue": (
                    round(float(revenue), 2)
                    if pd.notna(revenue)
                    else None
                ),
                "median_orders": (
                    round(float(orders), 2)
                    if pd.notna(orders)
                    else None
                ),
                "median_recency_days": (
                    round(float(recency), 2)
                    if pd.notna(recency)
                    else None
                ),
                "priority": priority,
                "recommendation": action,
            }
        )

    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }

    recommendations.sort(
        key=lambda item: priority_order.get(
            item["priority"],
            99,
        )
    )

    return recommendations


# ---------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------

def run_customer_segmentation(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    feature_result: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run customer segmentation using K-Means clustering.

    Parameters
    ----------
    df:
        Original analysis-ready dataset.

    mapping:
        Schema mapping produced by schema_mapper.py.

    feature_result:
        Output from feature_engineering.build_ml_features().

    Returns
    -------
    dict
        Stable DataPulse ML result contract.
    """

    del df
    del mapping

    capabilities = {
        "customer_features": False,
        "rfm_clustering": False,
        "kmeans": False,
        "silhouette_selection": False,
        "segment_profiles": False,
        "customer_assignments": False,
        "recommendations": False,
    }

    if not feature_result:
        return _empty_result(
            warnings=[
                "ML feature engineering results were not provided."
            ],
            capabilities=capabilities,
        )

    customer_features = feature_result.get(
        "customer_features"
    )

    if (
        customer_features is None
        or not isinstance(customer_features, pd.DataFrame)
        or customer_features.empty
    ):
        return _empty_result(
            warnings=[
                "Customer-level features are unavailable."
            ],
            capabilities=capabilities,
        )

    if "customer_id" not in customer_features.columns:
        return _empty_result(
            warnings=[
                "Customer segmentation requires a customer_id feature."
            ],
            capabilities=capabilities,
        )

    customer_count = len(customer_features)

    if customer_count < MIN_CUSTOMERS:
        return _empty_result(
            warnings=[
                f"Customer segmentation requires at least "
                f"{MIN_CUSTOMERS} customers; "
                f"only {customer_count} were available."
            ],
            capabilities=capabilities,
        )

    customer_data, transformed, usable_features = _prepare_features(
        customer_features
    )

    if len(usable_features) < 3:
        return _empty_result(
            warnings=[
                "Not enough usable customer features were available "
                "for meaningful segmentation."
            ],
            capabilities=capabilities,
        )

    # Remove features that contain no usable information.
    valid_features = []

    for column in usable_features:
        if (
            transformed[column].notna().sum() > 0
            and transformed[column].nunique(dropna=True) > 1
        ):
            valid_features.append(column)

    if len(valid_features) < 3:
        return _empty_result(
            warnings=[
                "Customer features do not contain enough variation "
                "for meaningful clustering."
            ],
            capabilities=capabilities,
        )

    transformed = transformed[valid_features]

    # Imputation.
    imputer = SimpleImputer(
        strategy="median"
    )

    imputed = imputer.fit_transform(
        transformed
    )

    # Scaling is essential for distance-based clustering.
    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        imputed
    )

    # Select K using silhouette score.
    best_k, silhouette_scores = _select_cluster_count(
        scaled
    )

    capabilities["customer_features"] = True
    capabilities["rfm_clustering"] = True
    capabilities["silhouette_selection"] = bool(
        silhouette_scores
    )

    try:
        model = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )

        labels = model.fit_predict(
            scaled
        )

    except Exception as exc:
        return _empty_result(
            warnings=[
                f"Customer segmentation failed: {exc}"
            ],
            capabilities=capabilities,
        )

    capabilities["kmeans"] = True

    # -----------------------------------------------------------------
    # Cluster quality
    # -----------------------------------------------------------------

    metrics: dict[str, Any] = {
        "customer_count": int(customer_count),
        "cluster_count": int(best_k),
        "features_used": valid_features,
        "silhouette_scores": {
            str(k): round(score, 4)
            for k, score in silhouette_scores.items()
        },
    }

    if (
        len(np.unique(labels)) >= 2
        and len(customer_data) > best_k
    ):
        try:
            final_silhouette = silhouette_score(
                scaled,
                labels,
            )

            metrics["silhouette_score"] = round(
                float(final_silhouette),
                4,
            )

        except Exception:
            metrics["silhouette_score"] = None
    else:
        metrics["silhouette_score"] = None

    # -----------------------------------------------------------------
    # Segment profiles
    # -----------------------------------------------------------------

    profiles = _build_profiles(
        customer_data,
        labels,
    )

    capabilities["segment_profiles"] = True

    # Segment IDs are model-generated and not naturally ordered.
    # Keep a stable presentation order based on median revenue.
    if "revenue" in profiles.columns:
        profiles = profiles.sort_values(
            "revenue",
            ascending=False,
        ).reset_index(drop=True)

    # Re-number presentation segment IDs.
    segment_id_map = {
        old_id: new_id
        for new_id, old_id in enumerate(
            profiles["segment_id"].tolist(),
            start=1,
        )
    }

    customer_labels = np.array(
        [
            segment_id_map.get(
                int(label),
                int(label),
            )
            for label in labels
        ]
    )

    profiles["segment_id"] = profiles[
        "segment_id"
    ].map(segment_id_map)

    profiles = profiles.sort_values(
        "segment_id"
    ).reset_index(drop=True)

    assignments = _build_assignments(
        customer_data,
        customer_labels,
    )

    assignments = _attach_segment_names(
        assignments,
        profiles,
    )

    capabilities["customer_assignments"] = True

    # -----------------------------------------------------------------
    # Distribution
    # -----------------------------------------------------------------

    distribution = (
        profiles[
            [
                "segment_id",
                "segment_name",
                "customer_count",
                "customer_share_pct",
            ]
        ]
        .copy()
    )

    distribution["customer_share_pct"] = (
        distribution["customer_share_pct"]
        .round(2)
    )

    # -----------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------

    recommendations = _build_recommendations(
        profiles
    )

    capabilities["recommendations"] = True

    # -----------------------------------------------------------------
    # Training metadata
    # -----------------------------------------------------------------

    training = {
        "algorithm": "K-Means",
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "selected_clusters": int(best_k),
        "selection_method": (
            "Silhouette score"
            if silhouette_scores
            else "Fallback minimum cluster count"
        ),
        "feature_count": len(valid_features),
        "features": valid_features,
        "scaling": "StandardScaler",
        "missing_value_strategy": "Median imputation",
        "customer_count": int(customer_count),
    }

    return {
        "available": True,
        "model_ready": True,
        "model": model,
        "segments": assignments.copy(),
        "segment_summary": profiles.copy(),
        "segment_distribution": distribution.copy(),
        "segment_profiles": profiles.copy(),
        "customer_assignments": assignments.copy(),
        "metrics": metrics,
        "recommendations": recommendations,
        "capabilities": capabilities,
        "training": training,
        "warnings": [],
    }


# ---------------------------------------------------------------------
# Compatibility aliases / helpers
# ---------------------------------------------------------------------

def get_customer_segments(
    result: dict[str, Any],
) -> pd.DataFrame:
    """Return customer-level segment assignments."""

    return result.get(
        "customer_assignments",
        pd.DataFrame(),
    )


def get_segment_profiles(
    result: dict[str, Any],
) -> pd.DataFrame:
    """Return segment profiles."""

    return result.get(
        "segment_profiles",
        pd.DataFrame(),
    )


def get_segment_recommendations(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return business recommendations."""

    return result.get(
        "recommendations",
        [],
    )


# Backward-compatible alias.
run_segmentation = run_customer_segmentation