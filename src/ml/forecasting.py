"""
DataPulse - Sales Forecasting
-----------------------------
Business-focused sales forecasting using historical aggregated sales
time series.

Public interface:
    run_sales_forecast(df, mapping, feature_result=None)

Expected feature_result:
    Output from src.feature_engineering.build_ml_features()

Models:
    - Baseline seasonal-naive forecast
    - Random Forest regression
    - Optional XGBoost regression when available

The engine automatically chooses the best model based on historical
time-series validation.

Dependencies:
    pandas
    numpy
    scikit-learn
    xgboost (optional at runtime)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MIN_PERIODS = 12
DEFAULT_FORECAST_PERIODS = 6
MAX_FORECAST_PERIODS = 12

MIN_TRAIN_PERIODS = 8

RANDOM_STATE = 42

LAG_FEATURES = [
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
]

ROLLING_FEATURES = [
    "revenue_rolling_3",
    "revenue_rolling_6",
    "revenue_rolling_std_3",
]

CALENDAR_FEATURES = [
    "time_index",
    "month_number",
    "quarter",
]

FORECAST_FEATURES = (
    LAG_FEATURES
    + ROLLING_FEATURES
    + CALENDAR_FEATURES
)


# ---------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------

def _empty_result(
    warnings: Optional[list[str]] = None,
    capabilities: Optional[dict[str, bool]] = None,
) -> dict[str, Any]:
    """Return a consistent unavailable forecasting result."""

    return {
        "available": False,
        "model_ready": False,
        "model": None,
        "forecast": pd.DataFrame(),
        "historical": pd.DataFrame(),
        "metrics": {},
        "model_comparison": pd.DataFrame(),
        "trend": {},
        "recommendations": [],
        "capabilities": capabilities or {},
        "training": {},
        "warnings": warnings or [],
    }


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def _prepare_time_series(
    feature_result: dict[str, Any],
) -> pd.DataFrame:
    """Extract and clean the engineered sales time series."""

    time_series = feature_result.get(
        "sales_time_series"
    )

    if (
        time_series is None
        or not isinstance(time_series, pd.DataFrame)
        or time_series.empty
    ):
        return pd.DataFrame()

    data = time_series.copy()

    if "period" not in data.columns:
        return pd.DataFrame()

    data["period"] = pd.to_datetime(
        data["period"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["period"]
    )

    if "revenue" not in data.columns:
        return pd.DataFrame()

    data["revenue"] = pd.to_numeric(
        data["revenue"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["revenue"]
    )

    data = (
        data.sort_values("period")
        .drop_duplicates(
            subset=["period"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return data


def _infer_frequency(
    periods: pd.Series,
) -> str:
    """Infer a human-readable time frequency."""

    if len(periods) < 2:
        return "unknown"

    differences = (
        periods.sort_values()
        .diff()
        .dropna()
        .dt.days
    )

    if differences.empty:
        return "unknown"

    median_days = float(
        differences.median()
    )

    if median_days <= 2:
        return "daily"

    if median_days <= 10:
        return "weekly"

    if median_days <= 45:
        return "monthly"

    if median_days <= 120:
        return "quarterly"

    return "irregular"


def _frequency_offset(
    frequency: str,
) -> pd.DateOffset:
    """Return an appropriate future-period offset."""

    if frequency == "daily":
        return pd.DateOffset(days=1)

    if frequency == "weekly":
        return pd.DateOffset(weeks=1)

    if frequency == "quarterly":
        return pd.DateOffset(months=3)

    # Monthly is the standard fallback.
    return pd.DateOffset(months=1)


# ---------------------------------------------------------------------
# Historical model features
# ---------------------------------------------------------------------

def _ensure_forecast_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure all forecasting features exist.

    feature_engineering.py already creates these features, but this
    defensive layer prevents a forecasting failure if an older cached
    feature result is encountered.
    """

    result = data.copy()

    result = result.sort_values(
        "period"
    ).reset_index(drop=True)

    if "time_index" not in result.columns:
        result["time_index"] = np.arange(
            len(result)
        )

    if "month_number" not in result.columns:
        result["month_number"] = (
            result["period"].dt.month
        )

    if "quarter" not in result.columns:
        result["quarter"] = (
            result["period"].dt.quarter
        )

    for lag in [1, 2, 3]:
        column = f"revenue_lag_{lag}"

        if column not in result.columns:
            result[column] = (
                result["revenue"]
                .shift(lag)
            )

    if "revenue_rolling_3" not in result.columns:
        result["revenue_rolling_3"] = (
            result["revenue"]
            .shift(1)
            .rolling(3)
            .mean()
        )

    if "revenue_rolling_6" not in result.columns:
        result["revenue_rolling_6"] = (
            result["revenue"]
            .shift(1)
            .rolling(6)
            .mean()
        )

    if "revenue_rolling_std_3" not in result.columns:
        result["revenue_rolling_std_3"] = (
            result["revenue"]
            .shift(1)
            .rolling(3)
            .std()
        )

    return result


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def _build_random_forest() -> Pipeline:
    """Build the Random Forest forecasting pipeline."""

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
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _build_xgboost():
    """
    Build XGBoost when installed.

    Returns None when unavailable so DataPulse remains deployable
    without depending on XGBoost-specific behavior.
    """

    try:
        from xgboost import XGBRegressor

        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=2,
                    ),
                ),
            ]
        )

        return model

    except Exception:
        return None


# ---------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------

def _seasonal_naive_predictions(
    train: pd.Series,
    test_length: int,
    seasonal_period: int = 1,
) -> np.ndarray:
    """
    Generate a simple seasonal-naive forecast.

    For monthly data, one period is deliberately used as the primary
    short-history baseline because many uploaded business datasets do
    not contain enough years for reliable 12-month seasonality.
    """

    values = (
        pd.to_numeric(
            train,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    if len(values) == 0:
        return np.zeros(
            test_length
        )

    seasonal_period = max(
        1,
        min(
            seasonal_period,
            len(values),
        ),
    )

    repeated = []

    for i in range(test_length):
        index = (
            len(values)
            - seasonal_period
            + (i % seasonal_period)
        )

        index = max(
            0,
            min(
                index,
                len(values) - 1,
            ),
        )

        repeated.append(
            values[index]
        )

    return np.asarray(
        repeated,
        dtype=float,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Optional[float]]:
    """Calculate regression forecasting metrics."""

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
        }

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    non_zero = (
        np.abs(actual) > 1e-9
    )

    if non_zero.any():
        mape = (
            np.mean(
                np.abs(
                    (
                        actual[non_zero]
                        - predicted[non_zero]
                    )
                    / actual[non_zero]
                )
            )
            * 100
        )
    else:
        mape = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": (
            float(mape)
            if mape is not None
            else None
        ),
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _time_series_validation(
    data: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    model,
) -> tuple[dict[str, Any], Any]:
    """
    Validate a forecasting model using the final historical periods.

    No random train/test split is used because this is a time-series
    problem.
    """

    clean = data.copy()

    clean = clean.dropna(
        subset=["revenue"]
    )

    if len(clean) < MIN_TRAIN_PERIODS + 2:
        return (
            {
                "model": model_name,
                "mae": None,
                "rmse": None,
                "mape": None,
                "validation_periods": 0,
            },
            model,
        )

    validation_size = max(
        2,
        min(
            4,
            len(clean) // 4,
        ),
    )

    train = clean.iloc[
        :-validation_size
    ].copy()

    test = clean.iloc[
        -validation_size:
    ].copy()

    if len(train) < MIN_TRAIN_PERIODS:
        return (
            {
                "model": model_name,
                "mae": None,
                "rmse": None,
                "mape": None,
                "validation_periods": 0,
            },
            model,
        )

    X_train = train[
        feature_columns
    ]

    y_train = train[
        "revenue"
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        "revenue"
    ]

    try:
        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        predictions = np.maximum(
            predictions,
            0,
        )

        metrics = _calculate_metrics(
            y_test.to_numpy(),
            predictions,
        )

        return (
            {
                "model": model_name,
                **metrics,
                "validation_periods": int(
                    validation_size
                ),
            },
            model,
        )

    except Exception as exc:
        return (
            {
                "model": model_name,
                "mae": None,
                "rmse": None,
                "mape": None,
                "validation_periods": 0,
                "error": str(exc),
            },
            model,
        )


# ---------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------

def _build_future_row(
    history: pd.DataFrame,
    future_period: pd.Timestamp,
) -> dict[str, Any]:
    """Construct one recursive future feature row."""

    revenue_values = (
        history["revenue"]
        .astype(float)
        .tolist()
    )

    row: dict[str, Any] = {}

    row["period"] = future_period

    row["time_index"] = len(history)

    row["month_number"] = (
        future_period.month
    )

    row["quarter"] = (
        future_period.quarter
    )

    for lag in [1, 2, 3]:
        if len(revenue_values) >= lag:
            row[
                f"revenue_lag_{lag}"
            ] = revenue_values[-lag]
        else:
            row[
                f"revenue_lag_{lag}"
            ] = np.nan

    last_values_3 = revenue_values[-3:]

    if last_values_3:
        row["revenue_rolling_3"] = float(
            np.mean(last_values_3)
        )
        row["revenue_rolling_std_3"] = float(
            np.std(
                last_values_3,
                ddof=1,
            )
        ) if len(last_values_3) > 1 else 0.0
    else:
        row["revenue_rolling_3"] = np.nan
        row["revenue_rolling_std_3"] = np.nan

    last_values_6 = revenue_values[-6:]

    if last_values_6:
        row["revenue_rolling_6"] = float(
            np.mean(last_values_6)
        )
    else:
        row["revenue_rolling_6"] = np.nan

    return row


def _recursive_forecast(
    model,
    history: pd.DataFrame,
    frequency: str,
    periods: int,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate recursive multi-period forecasts."""

    working = history.copy()

    offset = _frequency_offset(
        frequency
    )

    forecasts = []

    last_period = working[
        "period"
    ].max()

    for step in range(
        1,
        periods + 1,
    ):
        future_period = (
            last_period + offset
        )

        row = _build_future_row(
            working,
            future_period,
        )

        feature_row = pd.DataFrame(
            [row]
        )

        X_future = feature_row[
            feature_columns
        ]

        try:
            prediction = float(
                model.predict(
                    X_future
                )[0]
            )
        except Exception:
            prediction = float(
                working["revenue"]
                .tail(3)
                .mean()
            )

        # Revenue cannot be negative.
        prediction = max(
            0.0,
            prediction,
        )

        row["forecast_revenue"] = prediction

        forecasts.append(
            row
        )

        # Feed the prediction into future lags/rolling features.
        append_row = row.copy()

        append_row["revenue"] = prediction

        working = pd.concat(
            [
                working,
                pd.DataFrame(
                    [append_row]
                ),
            ],
            ignore_index=True,
        )

        last_period = future_period

    return pd.DataFrame(
        forecasts
    )


# ---------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------

def _calculate_trend(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate business-friendly historical and forecast trends."""

    history = historical.copy()

    result: dict[str, Any] = {
        "historical_direction": "Stable",
        "forecast_direction": "Stable",
        "historical_growth_pct": None,
        "forecast_growth_pct": None,
        "last_revenue": None,
        "forecast_total_revenue": None,
    }

    if history.empty:
        return result

    revenue = pd.to_numeric(
        history["revenue"],
        errors="coerce",
    ).dropna()

    if not revenue.empty:
        result["last_revenue"] = float(
            revenue.iloc[-1]
        )

    if len(revenue) >= 2:
        first = float(
            revenue.iloc[0]
        )

        last = float(
            revenue.iloc[-1]
        )

        if abs(first) > 1e-9:
            growth = (
                (last - first)
                / abs(first)
                * 100
            )

            result[
                "historical_growth_pct"
            ] = float(growth)

            if growth > 5:
                result[
                    "historical_direction"
                ] = "Growing"

            elif growth < -5:
                result[
                    "historical_direction"
                ] = "Declining"

    if not forecast.empty:
        forecast_values = pd.to_numeric(
            forecast[
                "forecast_revenue"
            ],
            errors="coerce",
        ).dropna()

        if not forecast_values.empty:
            result[
                "forecast_total_revenue"
            ] = float(
                forecast_values.sum()
            )

            first_forecast = float(
                forecast_values.iloc[0]
            )

            last_forecast = float(
                forecast_values.iloc[-1]
            )

            if abs(first_forecast) > 1e-9:
                growth = (
                    (
                        last_forecast
                        - first_forecast
                    )
                    / abs(first_forecast)
                    * 100
                )

                result[
                    "forecast_growth_pct"
                ] = float(growth)

                if growth > 5:
                    result[
                        "forecast_direction"
                    ] = "Growing"

                elif growth < -5:
                    result[
                        "forecast_direction"
                    ] = "Declining"

    return result


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

def _build_recommendations(
    trend: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate actionable forecasting recommendations."""

    recommendations = []

    historical_direction = trend.get(
        "historical_direction",
        "Stable",
    )

    forecast_direction = trend.get(
        "forecast_direction",
        "Stable",
    )

    mape = metrics.get(
        "mape"
    )

    if forecast_direction == "Growing":
        recommendations.append(
            {
                "priority": "High",
                "area": "Growth",
                "recommendation": (
                    "Prepare inventory, fulfillment capacity and "
                    "customer acquisition efforts for expected sales growth."
                ),
            }
        )

    elif forecast_direction == "Declining":
        recommendations.append(
            {
                "priority": "High",
                "area": "Revenue",
                "recommendation": (
                    "Investigate declining demand and prioritize "
                    "retention, product and campaign actions before "
                    "reducing inventory commitments."
                ),
            }
        )

    else:
        recommendations.append(
            {
                "priority": "Medium",
                "area": "Planning",
                "recommendation": (
                    "Maintain balanced inventory and monitor upcoming "
                    "periods for changes in demand."
                ),
            }
        )

    if historical_direction == "Declining":
        recommendations.append(
            {
                "priority": "High",
                "area": "Trend",
                "recommendation": (
                    "Review the drivers behind the historical revenue "
                    "decline, including products, regions, pricing and campaigns."
                ),
            }
        )

    if (
        mape is not None
        and mape > 25
    ):
        recommendations.append(
            {
                "priority": "Medium",
                "area": "Forecast Quality",
                "recommendation": (
                    "Forecast error is relatively high. Add more "
                    "historical periods or richer demand drivers before "
                    "using the forecast for high-stakes planning."
                ),
            }
        )

    elif (
        mape is not None
        and mape <= 15
    ):
        recommendations.append(
            {
                "priority": "Low",
                "area": "Forecast Quality",
                "recommendation": (
                    "Historical validation indicates relatively strong "
                    "forecast accuracy for the available data."
                ),
            }
        )

    return recommendations


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def run_sales_forecast(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    feature_result: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Run DataPulse sales forecasting.

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
        Stable forecasting result consumed by DataPulse.
    """

    del df
    del mapping

    capabilities = {
        "historical_time_series": False,
        "feature_engineering": False,
        "time_series_validation": False,
        "random_forest": False,
        "xgboost": False,
        "multi_period_forecast": False,
        "trend_analysis": False,
        "recommendations": False,
    }

    if not feature_result:
        return _empty_result(
            warnings=[
                "ML feature engineering results were not provided."
            ],
            capabilities=capabilities,
        )

    data = _prepare_time_series(
        feature_result
    )

    if data.empty:
        return _empty_result(
            warnings=[
                "Historical sales time-series data is unavailable."
            ],
            capabilities=capabilities,
        )

    if len(data) < MIN_PERIODS:
        return _empty_result(
            warnings=[
                f"Sales forecasting requires at least "
                f"{MIN_PERIODS} historical periods; "
                f"only {len(data)} were available."
            ],
            capabilities=capabilities,
        )

    capabilities[
        "historical_time_series"
    ] = True

    data = _ensure_forecast_features(
        data
    )

    feature_columns = [
        column
        for column in FORECAST_FEATURES
        if column in data.columns
    ]

    if len(feature_columns) < 4:
        return _empty_result(
            warnings=[
                "Not enough forecasting features are available."
            ],
            capabilities=capabilities,
        )

    # -----------------------------------------------------------------
    # Remove early rows where the lag/rolling structure cannot be used.
    # -----------------------------------------------------------------

    modeling_data = data.copy()

    modeling_data = modeling_data.dropna(
        subset=["revenue"]
    ).reset_index(drop=True)

    if len(modeling_data) < MIN_TRAIN_PERIODS + 2:
        return _empty_result(
            warnings=[
                "Not enough historical observations remain after "
                "forecast feature preparation."
            ],
            capabilities=capabilities,
        )

    capabilities[
        "feature_engineering"
    ] = True

    # -----------------------------------------------------------------
    # Candidate models
    # -----------------------------------------------------------------

    candidates: list[
        tuple[str, Any]
    ] = []

    random_forest = _build_random_forest()

    candidates.append(
        (
            "Random Forest",
            random_forest,
        )
    )

    capabilities[
        "random_forest"
    ] = True

    xgb_model = _build_xgboost()

    if xgb_model is not None:
        candidates.append(
            (
                "XGBoost",
                xgb_model,
            )
        )

        capabilities[
            "xgboost"
        ] = True

    # -----------------------------------------------------------------
    # Validate candidate models.
    # -----------------------------------------------------------------

    validation_results = []

    fitted_models = {}

    for model_name, candidate in candidates:

        result, fitted = (
            _time_series_validation(
                modeling_data,
                feature_columns,
                model_name,
                candidate,
            )
        )

        validation_results.append(
            result
        )

        fitted_models[
            model_name
        ] = fitted

    # -----------------------------------------------------------------
    # Select best model.
    # -----------------------------------------------------------------

    valid_results = [
        result
        for result in validation_results
        if result.get("mae") is not None
    ]

    if valid_results:
        best_result = min(
            valid_results,
            key=lambda item: item["mae"],
        )

        selected_model_name = best_result[
            "model"
        ]

    else:
        # Fall back to Random Forest.
        selected_model_name = "Random Forest"

        best_result = {
            "model": selected_model_name,
            "mae": None,
            "rmse": None,
            "mape": None,
            "validation_periods": 0,
        }

    selected_model = fitted_models[
        selected_model_name
    ]

    capabilities[
        "time_series_validation"
    ] = bool(valid_results)

    # -----------------------------------------------------------------
    # Fit selected model on all available history.
    # -----------------------------------------------------------------

    X_all = modeling_data[
        feature_columns
    ]

    y_all = modeling_data[
        "revenue"
    ]

    try:
        selected_model.fit(
            X_all,
            y_all,
        )
    except Exception as exc:
        return _empty_result(
            warnings=[
                f"Final forecasting model training failed: {exc}"
            ],
            capabilities=capabilities,
        )

    # -----------------------------------------------------------------
    # Generate future forecast.
    # -----------------------------------------------------------------

    frequency = _infer_frequency(
        modeling_data["period"]
    )

    if frequency == "unknown":
        frequency = "monthly"

    forecast_periods = min(
        DEFAULT_FORECAST_PERIODS,
        MAX_FORECAST_PERIODS,
    )

    forecast = _recursive_forecast(
        selected_model,
        modeling_data,
        frequency,
        forecast_periods,
        feature_columns,
    )

    if forecast.empty:
        return _empty_result(
            warnings=[
                "The forecasting model could not generate future periods."
            ],
            capabilities=capabilities,
        )

    capabilities[
        "multi_period_forecast"
    ] = True

    # -----------------------------------------------------------------
    # Trend analysis
    # -----------------------------------------------------------------

    trend = _calculate_trend(
        modeling_data,
        forecast,
    )

    capabilities[
        "trend_analysis"
    ] = True

    # -----------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------

    comparison = pd.DataFrame(
        validation_results
    )

    if not comparison.empty:
        comparison = comparison[
            [
                column
                for column in [
                    "model",
                    "mae",
                    "rmse",
                    "mape",
                    "validation_periods",
                ]
                if column in comparison.columns
            ]
        ]

        for column in [
            "mae",
            "rmse",
            "mape",
        ]:
            if column in comparison.columns:
                comparison[column] = comparison[
                    column
                ].round(2)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    metrics = {
        "selected_model": selected_model_name,
        "mae": best_result.get(
            "mae"
        ),
        "rmse": best_result.get(
            "rmse"
        ),
        "mape": best_result.get(
            "mape"
        ),
        "validation_periods": best_result.get(
            "validation_periods",
            0,
        ),
        "historical_periods": int(
            len(modeling_data)
        ),
        "forecast_periods": int(
            len(forecast)
        ),
        "frequency": frequency,
    }

    # Round metric values.
    for key in [
        "mae",
        "rmse",
        "mape",
    ]:
        if metrics[key] is not None:
            metrics[key] = round(
                float(metrics[key]),
                2,
            )

    # -----------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------

    recommendations = _build_recommendations(
        trend,
        metrics,
    )

    capabilities[
        "recommendations"
    ] = True

    # -----------------------------------------------------------------
    # Forecast output cleanup
    # -----------------------------------------------------------------

    forecast_output = forecast[
        [
            "period",
            "forecast_revenue",
        ]
    ].copy()

    forecast_output[
        "forecast_revenue"
    ] = forecast_output[
        "forecast_revenue"
    ].round(2)

    forecast_output[
        "forecast_period"
    ] = np.arange(
        1,
        len(forecast_output) + 1,
    )

    forecast_output = forecast_output[
        [
            "forecast_period",
            "period",
            "forecast_revenue",
        ]
    ]

    # Historical output.
    historical_output = modeling_data[
        [
            column
            for column in [
                "period",
                "revenue",
                "orders",
                "quantity",
                "profit",
            ]
            if column in modeling_data.columns
        ]
    ].copy()

    # -----------------------------------------------------------------
    # Training metadata
    # -----------------------------------------------------------------

    training = {
        "selected_model": selected_model_name,
        "candidate_models": [
            result.get("model")
            for result in validation_results
        ],
        "feature_columns": feature_columns,
        "historical_periods": int(
            len(modeling_data)
        ),
        "forecast_periods": int(
            len(forecast_output)
        ),
        "frequency": frequency,
        "validation_method": (
            "Time-ordered holdout validation"
        ),
        "random_state": RANDOM_STATE,
    }

    return {
        "available": True,
        "model_ready": True,
        "model": selected_model,
        "forecast": forecast_output,
        "historical": historical_output,
        "metrics": metrics,
        "model_comparison": comparison,
        "trend": trend,
        "recommendations": recommendations,
        "capabilities": capabilities,
        "training": training,
        "warnings": [],
    }


# ---------------------------------------------------------------------
# Compatibility aliases / helpers
# ---------------------------------------------------------------------

def get_forecast(
    result: dict[str, Any],
) -> pd.DataFrame:
    """Return the generated forecast."""

    return result.get(
        "forecast",
        pd.DataFrame(),
    )


def get_forecast_metrics(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Return forecast model metrics."""

    return result.get(
        "metrics",
        {},
    )


def get_forecast_recommendations(
    result: dict[str, Any],
) -> list[dict[str, str]]:
    """Return business recommendations."""

    return result.get(
        "recommendations",
        [],
    )


# Backward-compatible alias.
run_forecasting = run_sales_forecast