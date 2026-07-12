"""Model training, evaluation, and prediction functions.

The app trains four model families:

1. Linear Regression for next-day price. Direction is inferred from the price.
2. Random Forest Regressor + Random Forest Classifier.
3. XGBoost Regressor + XGBoost Classifier, when xgboost is installed.
4. PyTorch LSTM for next-day price, when torch is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from features import (
    FEATURE_COLUMNS,
    LSTM_FEATURE_COLUMNS,
    TARGET_CLOSE,
    TARGET_DATE,
    TARGET_DIRECTION,
)

try:  # Optional dependency, included in requirements.txt.
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - handled in the UI.
    XGBClassifier = None
    XGBRegressor = None

try:  # Optional dependency, included in requirements.txt.
    import torch
    from torch import nn

    # Keep Streamlit reruns responsive on small CPU instances. Some hosted
    # environments make PyTorch LSTM layers slow when too many CPU threads or
    # MKLDNN kernels are used for tiny datasets.
    torch.set_num_threads(1)
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass
except Exception:  # pragma: no cover - handled in the UI.
    torch = None
    nn = None


MODEL_ORDER = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
MIN_CLEAN_ROWS = 60


@dataclass
class ModelResult:
    """Evaluation results and fitted objects for one model family."""

    name: str
    metrics: Dict[str, float]
    predictions: pd.DataFrame
    confusion_matrix: np.ndarray
    error_std: float
    price_model: object | None = None
    direction_model: object | None = None
    extra: Dict[str, object] = field(default_factory=dict)
    unavailable_reason: Optional[str] = None


@dataclass
class TrainedBundle:
    """Container for all successfully trained model families."""

    results: Dict[str, ModelResult]
    split_index: int
    unavailable: Dict[str, str]


def train_and_evaluate_all(
    model_data: pd.DataFrame,
    include_lstm: bool = True,
    lstm_epochs: int = 20,
) -> TrainedBundle:
    """Train all requested models using a time-based 80/20 split."""

    if len(model_data) < MIN_CLEAN_ROWS:
        raise ValueError(
            "Not enough clean rows after feature engineering. Use a longer date range. "
            f"Clean rows available: {len(model_data)}; minimum required: {MIN_CLEAN_ROWS}."
        )

    X = model_data[FEATURE_COLUMNS]
    y_price = model_data[TARGET_CLOSE]
    y_direction = model_data[TARGET_DIRECTION]
    current_close = model_data["Close"]

    split_index = int(len(model_data) * 0.80)
    if split_index <= 0 or split_index >= len(model_data):
        raise ValueError("The selected date range is too small for an 80/20 split.")

    target_index = _target_index(model_data.iloc[split_index:])

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_price_train = y_price.iloc[:split_index]
    y_price_test = _series_with_index(y_price.iloc[split_index:], target_index)
    y_direction_train = y_direction.iloc[:split_index]
    y_direction_test = _series_with_index(y_direction.iloc[split_index:], target_index)
    current_close_test = _series_with_index(current_close.iloc[split_index:], target_index)

    results: Dict[str, ModelResult] = {}
    unavailable: Dict[str, str] = {}

    # 1) Linear Regression price model. Direction is inferred from predicted price.
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_price_train)
    linear_price_pred = linear_model.predict(X_test)
    linear_dir_pred = (linear_price_pred > current_close_test.to_numpy(dtype=float)).astype(int)
    results["Linear Regression"] = _build_result(
        name="Linear Regression",
        y_price_test=y_price_test,
        price_predictions=linear_price_pred,
        y_direction_test=y_direction_test,
        direction_predictions=linear_dir_pred,
        current_close_test=current_close_test,
        price_model=linear_model,
    )

    # 2) Random Forest model family.
    rf_regressor = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    rf_classifier = _fit_direction_classifier(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
        ),
        X_train,
        y_direction_train,
    )
    rf_regressor.fit(X_train, y_price_train)
    rf_price_pred = rf_regressor.predict(X_test)
    rf_dir_pred = rf_classifier.predict(X_test)
    rf_probability_up = _probability_for_class(rf_classifier, X_test, positive_class=1)
    results["Random Forest"] = _build_result(
        name="Random Forest",
        y_price_test=y_price_test,
        price_predictions=rf_price_pred,
        y_direction_test=y_direction_test,
        direction_predictions=rf_dir_pred,
        current_close_test=current_close_test,
        price_model=rf_regressor,
        direction_model=rf_classifier,
        probability_up=rf_probability_up,
    )

    # 3) XGBoost model family.
    if XGBRegressor is None or XGBClassifier is None:
        unavailable["XGBoost"] = "xgboost is not installed. Run: pip install xgboost"
    else:
        try:
            xgb_regressor = XGBRegressor(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
            xgb_classifier_base = XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            )
            xgb_regressor.fit(X_train, y_price_train)
            xgb_classifier = _fit_direction_classifier(
                xgb_classifier_base,
                X_train,
                y_direction_train,
            )
            xgb_price_pred = xgb_regressor.predict(X_test)
            xgb_dir_pred = xgb_classifier.predict(X_test)
            xgb_probability_up = _probability_for_class(
                xgb_classifier, X_test, positive_class=1
            )
            results["XGBoost"] = _build_result(
                name="XGBoost",
                y_price_test=y_price_test,
                price_predictions=xgb_price_pred,
                y_direction_test=y_direction_test,
                direction_predictions=xgb_dir_pred,
                current_close_test=current_close_test,
                price_model=xgb_regressor,
                direction_model=xgb_classifier,
                probability_up=xgb_probability_up,
            )
        except Exception as exc:  # pragma: no cover - shown to the user.
            unavailable["XGBoost"] = str(exc)

    # 4) LSTM price model with 60-day sequences.
    if include_lstm:
        if torch is None or nn is None:
            unavailable["LSTM"] = "torch is not installed. Run: pip install torch"
        else:
            try:
                results["LSTM"] = _train_lstm_result(model_data, epochs=lstm_epochs)
            except Exception as exc:  # pragma: no cover - shown to the user.
                unavailable["LSTM"] = str(exc)
    else:
        unavailable["LSTM"] = "LSTM training was disabled for this run."

    return TrainedBundle(results=results, split_index=split_index, unavailable=unavailable)


def predict_tomorrow(
    bundle: TrainedBundle,
    latest_features: pd.DataFrame,
    model_data: pd.DataFrame,
    selected_model: str,
    latest_close: Optional[float] = None,
    latest_lstm_sequence: Optional[pd.DataFrame] = None,
) -> Dict[str, float | int | str]:
    """Predict next close and next direction for the selected model family.

    ``model_data`` intentionally drops the most recent raw row because that row
    has no known next-day target. For true tomorrow predictions, pass
    ``latest_close`` from the raw price data and ``latest_lstm_sequence`` from
    the raw price data so the direction and LSTM sequence include the latest day.
    """

    if selected_model not in bundle.results:
        raise ValueError(f"{selected_model} is not available for prediction.")

    result = bundle.results[selected_model]
    if latest_close is None:
        latest_close = float(model_data["Close"].iloc[-1])
    latest_close = float(latest_close)

    if selected_model == "LSTM":
        predicted_price = _predict_lstm_tomorrow(
            result,
            latest_lstm_sequence if latest_lstm_sequence is not None else model_data[LSTM_FEATURE_COLUMNS],
        )
        predicted_direction = int(predicted_price > latest_close)
        probability_up = np.nan
        confidence = float(result.metrics.get("Accuracy", 0.5))
    else:
        predicted_price = float(result.price_model.predict(latest_features)[0])
        predicted_direction = int(predicted_price > latest_close)

        probability_up = np.nan
        confidence = float(result.metrics.get("Accuracy", 0.0))
        if result.direction_model is not None and hasattr(result.direction_model, "predict"):
            predicted_direction = int(result.direction_model.predict(latest_features)[0])
            proba = _probability_for_class(
                result.direction_model, latest_features, positive_class=1
            )
            if proba is not None and len(proba) > 0 and not np.isnan(proba[0]):
                probability_up = float(proba[0])
                confidence = probability_up if predicted_direction == 1 else 1 - probability_up

    return {
        "model": selected_model,
        "predicted_price": float(predicted_price),
        "predicted_direction": int(predicted_direction),
        "confidence": float(np.clip(confidence, 0, 1)),
        "probability_up": float(probability_up) if not np.isnan(probability_up) else np.nan,
        "error_std": float(result.error_std),
    }


def make_metrics_table(bundle: TrainedBundle) -> pd.DataFrame:
    """Create a side-by-side metrics table for all trained models."""

    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in bundle.results:
            rows.append(
                {
                    "Model": model_name,
                    "Status": bundle.unavailable.get(model_name, "Unavailable"),
                    "RMSE": np.nan,
                    "R²": np.nan,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Error Std": np.nan,
                }
            )
            continue

        result = bundle.results[model_name]
        rows.append(
            {
                "Model": model_name,
                "Status": "Trained",
                "RMSE": result.metrics["RMSE"],
                "R²": result.metrics["R2"],
                "Accuracy": result.metrics["Accuracy"],
                "Precision": result.metrics["Precision"],
                "Recall": result.metrics["Recall"],
                "Error Std": result.error_std,
            }
        )

    return pd.DataFrame(rows)


def _target_index(rows: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the actual next-day date index for prediction rows."""

    if TARGET_DATE in rows.columns:
        return pd.DatetimeIndex(pd.to_datetime(rows[TARGET_DATE]))
    return pd.DatetimeIndex(rows.index)


def _series_with_index(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Copy a Series and replace its index without losing the original name."""

    copied = series.copy()
    copied.index = index
    return copied


def _fit_direction_classifier(model: object, X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """Fit a classifier, falling back to DummyClassifier for one-class targets."""

    if y_train.nunique(dropna=True) < 2:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        return dummy

    model.fit(X_train, y_train)
    return model


def _build_result(
    name: str,
    y_price_test: pd.Series,
    price_predictions: np.ndarray,
    y_direction_test: pd.Series,
    direction_predictions: np.ndarray,
    current_close_test: pd.Series,
    price_model: object | None = None,
    direction_model: object | None = None,
    probability_up: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, object]] = None,
) -> ModelResult:
    """Build metrics, prediction table, and confusion matrix for one model."""

    price_predictions = np.asarray(price_predictions, dtype=float)
    direction_predictions = np.asarray(direction_predictions, dtype=int)

    rmse = float(np.sqrt(mean_squared_error(y_price_test, price_predictions)))
    r2 = float(r2_score(y_price_test, price_predictions)) if len(y_price_test) > 1 else np.nan
    accuracy = float(accuracy_score(y_direction_test, direction_predictions))
    precision = float(
        precision_score(y_direction_test, direction_predictions, zero_division=0)
    )
    recall = float(recall_score(y_direction_test, direction_predictions, zero_division=0))

    errors = price_predictions - y_price_test.to_numpy(dtype=float)
    error_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0

    predictions = pd.DataFrame(
        {
            "Current_Close": current_close_test,
            "Actual_Close": y_price_test,
            "Predicted_Close": price_predictions,
            "Actual_Direction": y_direction_test,
            "Predicted_Direction": direction_predictions,
            "Prediction_Error": errors,
            "CI_Lower": price_predictions - error_std,
            "CI_Upper": price_predictions + error_std,
        },
        index=y_price_test.index,
    )
    predictions.index.name = "Prediction_Date"

    if probability_up is not None:
        predictions["Probability_Up"] = probability_up

    cm = confusion_matrix(y_direction_test, direction_predictions, labels=[0, 1])

    return ModelResult(
        name=name,
        metrics={
            "RMSE": rmse,
            "R2": r2,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
        },
        predictions=predictions,
        confusion_matrix=cm,
        error_std=error_std,
        price_model=price_model,
        direction_model=direction_model,
        extra=extra or {},
    )


def _probability_for_class(model: object, X: pd.DataFrame, positive_class: int = 1):
    """Return probability for a class if the model supports predict_proba."""

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if positive_class not in classes:
        return np.full(len(X), np.nan)
    return probabilities[:, classes.index(positive_class)]


if nn is not None:

    class _PriceLSTM(nn.Module):  # type: ignore[misc]
        """Small LSTM regressor for next-day close prediction."""

        def __init__(self, input_size: int, hidden_size: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.15)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):  # noqa: D401 - PyTorch forward method.
            output, _ = self.lstm(x)
            last_hidden = output[:, -1, :]
            return self.linear(self.dropout(last_hidden))

else:

    class _PriceLSTM:  # type: ignore[no-redef]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for the LSTM model.")


def _make_lstm_sequences(
    model_data: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Create rolling sequences ending on the current row and targeting next day."""

    features = model_data[LSTM_FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = model_data[TARGET_CLOSE].to_numpy(dtype=float)
    current_close = model_data["Close"].to_numpy(dtype=float)
    directions = model_data[TARGET_DIRECTION].to_numpy(dtype=int)
    target_dates = _target_index(model_data)

    X_seq, y_seq, idx_seq, close_seq, dir_seq = [], [], [], [], []
    for current_pos in range(sequence_length - 1, len(model_data)):
        start_pos = current_pos - sequence_length + 1
        X_seq.append(features[start_pos : current_pos + 1])
        y_seq.append(targets[current_pos])
        idx_seq.append(target_dates[current_pos])
        close_seq.append(current_close[current_pos])
        dir_seq.append(directions[current_pos])

    return (
        np.asarray(X_seq, dtype=float),
        np.asarray(y_seq, dtype=float),
        pd.DatetimeIndex(idx_seq),
        np.asarray(close_seq, dtype=float),
        np.asarray(dir_seq, dtype=int),
    )


def _train_lstm_result(
    model_data: pd.DataFrame,
    sequence_length: int = 60,
    epochs: int = 20,
) -> ModelResult:
    """Train a small PyTorch LSTM and return metrics/predictions."""

    if len(model_data) < sequence_length + 5:
        raise ValueError(
            f"Need more data for LSTM. Try 1 year or 2 years; current clean rows: {len(model_data)}."
        )

    X_seq, y_seq, idx_seq, current_close_seq, actual_dir_seq = _make_lstm_sequences(
        model_data, sequence_length
    )

    if len(X_seq) < 10:
        raise ValueError("Not enough LSTM sequences after feature engineering.")

    split_index = int(len(X_seq) * 0.80)
    if split_index <= 0 or split_index >= len(X_seq):
        raise ValueError("Not enough LSTM sequences for an 80/20 time split.")

    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]
    idx_test = idx_seq[split_index:]
    current_close_test = current_close_seq[split_index:]
    y_direction_test = actual_dir_seq[split_index:]

    feature_scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    feature_scaler.fit(X_train_2d)
    X_train_scaled = feature_scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    net = _PriceLSTM(input_size=X_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = net(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        pred_scaled = net(X_test_tensor).cpu().numpy().reshape(-1, 1)

    price_predictions = target_scaler.inverse_transform(pred_scaled).ravel()
    direction_predictions = (price_predictions > current_close_test).astype(int)

    return _build_result(
        name="LSTM",
        y_price_test=pd.Series(y_test, index=idx_test, name=TARGET_CLOSE),
        price_predictions=price_predictions,
        y_direction_test=pd.Series(y_direction_test, index=idx_test, name=TARGET_DIRECTION),
        direction_predictions=direction_predictions,
        current_close_test=pd.Series(current_close_test, index=idx_test, name="Close"),
        price_model=net,
        extra={
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "sequence_length": sequence_length,
        },
    )


def _predict_lstm_tomorrow(result: ModelResult, latest_lstm_data: pd.DataFrame) -> float:
    """Predict tomorrow's close using a fitted LSTM result."""

    net = result.price_model
    feature_scaler = result.extra["feature_scaler"]
    target_scaler = result.extra["target_scaler"]
    sequence_length = int(result.extra.get("sequence_length", 60))

    sequence = latest_lstm_data[LSTM_FEATURE_COLUMNS].tail(sequence_length).to_numpy(dtype=float)
    if len(sequence) < sequence_length:
        raise ValueError("Not enough rows for tomorrow's LSTM sequence.")

    sequence_scaled = feature_scaler.transform(sequence).reshape(1, sequence_length, -1)
    tensor = torch.tensor(sequence_scaled, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        pred_scaled = net(tensor).cpu().numpy().reshape(-1, 1)
    predicted_price = target_scaler.inverse_transform(pred_scaled).ravel()[0]
    return float(predicted_price)
