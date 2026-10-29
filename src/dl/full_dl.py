from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# VERSION / METHOD
# =============================================================================

SCRIPT_VERSION = "2026-10-01_FULL_DL_V9_FIXED_LOPO_CONVERGENCE_15X15"

# Fixed a priori — no nested Optuna/HPO.
DEFAULT_HIDDEN_LAYERS = (16, 8)
DEFAULT_DROPOUT = 0.10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_EPOCHS = 250
DEFAULT_GRADIENT_CLIP = 1.0
ENSEMBLE_SEEDS_DEFAULT = (42, 123, 2026, 31415, 27182)


HYPERPARAMETER_INTERPRETATION = (
    "FIXED/PREDECLARED FOR THIS CODE VERSION; NOT HPO-OPTIMIZED"
)

HYPERPARAMETER_SELECTION_METHOD = (
    "FIXED_PREDECLARED_ANALYSIS_SPECIFICATION_NOT_HPO_OPTIMIZED"
)

HYPERPARAMETER_RATIONALE = (
    "Compact small-sample DNN specification fixed for this revised analysis; "
    "not selected by LOPO outcomes and not optimized by Optuna/HPO."
)

PREDECLARED_MEANING = (
    "Predeclared here means fixed in this code/version before this evaluation "
    "run; it does not imply external preregistration."
)

CONVERGENCE_INTERPRETATION = (
    "TRAINING-LOSS OPTIMIZATION DIAGNOSTIC ONLY; "
    "NOT AN OUT-OF-SAMPLE GENERALIZATION CLAIM"
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PART1_PROJECT = PROJECT_ROOT.parent / "TAT_MS_INTERCITY"
TRAINING_DATA_DIR = (
    PART1_PROJECT / "Data" / "Results" / "Master_Summary" / "Training_Data"
)
RESULTS_ROOT = PROJECT_ROOT / "Results"
MODEL_ROOT = PROJECT_ROOT / "models"


# =============================================================================
# FROZEN DATA CONTRACT
# =============================================================================

ALL_ICE_CANDIDATES = [
    "ICE Total Distance (km)",
    "ICE Total TAT (min)",
    "ICE Average Moving Speed (km/h)",
    "ICE CHARGING_REFUELING Time (min)",
    "ICE STOP Time (min)",
    "ICE IDLE Time (min)",
    "ICE INCHING Time (min)",
    "ICE ACC Time (min)",
    "ICE DEC Time (min)",
    "ICE CRUISE Time (min)",
    "ICE CHARGING_REFUELING Distance (km)",
    "ICE STOP Distance (km)",
    "ICE IDLE Distance (km)",
    "ICE INCHING Distance (km)",
    "ICE ACC Distance (km)",
    "ICE DEC Distance (km)",
    "ICE CRUISE Distance (km)",
]

EXCLUDED_CONSTANT_ICE_FEATURES = [
    "ICE CHARGING_REFUELING Time (min)",
    "ICE CHARGING_REFUELING Distance (km)",
]

ICE_FEATURES = [
    c for c in ALL_ICE_CANDIDATES
    if c not in EXCLUDED_CONSTANT_ICE_FEATURES
]

EV_TARGETS = [
    "EV Total Distance (km)",
    "EV Total TAT (min)",
    "EV Average Moving Speed (km/h)",
    "EV STOP Time (min)",
    "EV IDLE Time (min)",
    "EV INCHING Time (min)",
    "EV ACC Time (min)",
    "EV DEC Time (min)",
    "EV CRUISE Time (min)",
    "EV STOP Distance (km)",
    "EV IDLE Distance (km)",
    "EV INCHING Distance (km)",
    "EV ACC Distance (km)",
    "EV DEC Distance (km)",
    "EV CRUISE Distance (km)",
]

PHASE_TIME_TARGETS = [
    "EV STOP Time (min)",
    "EV IDLE Time (min)",
    "EV INCHING Time (min)",
    "EV ACC Time (min)",
    "EV DEC Time (min)",
    "EV CRUISE Time (min)",
]

PAIR_COLUMN = "Pair ID"
PAIR_CLASS_COLUMN = "Pair Class"
VALID_PAIR_CLASSES = {"STRICT", "NONSTRICT"}
VALID_SCOPES = {"STRICT", "NONSTRICT", "COMBINED", "ALL"}


# =============================================================================
# REPRODUCIBILITY / DEVICE
# =============================================================================

def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def resolve_device(name: str) -> torch.device:
    name = str(name).strip().lower()

    if name in {"auto", "cpu"}:
        return torch.device("cpu")

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")

    raise ValueError("--device must be auto, cpu or cuda.")


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())

    if not seeds:
        raise ValueError("At least one ensemble seed is required.")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Ensemble seeds must be unique.")

    return seeds


def parse_architecture(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in str(text).split("-") if x.strip())

    if not values or any(v <= 0 for v in values):
        raise ValueError("Architecture must contain positive widths, e.g. 16-8.")

    return values


def arch_to_text(values: tuple[int, ...]) -> str:
    return "-".join(str(v) for v in values)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# DATA LOADING / SCOPE VALIDATION
# =============================================================================

def normalize_scope(scope: str, allow_all: bool = False) -> str:
    scope = str(scope).strip().upper()
    allowed = VALID_SCOPES if allow_all else VALID_SCOPES - {"ALL"}

    if scope not in allowed:
        raise ValueError(
            f"Invalid scope {scope!r}; allowed={sorted(allowed)}"
        )

    return scope


def training_file(pair_class: str) -> Path:
    pair_class = str(pair_class).strip().upper()

    if pair_class not in VALID_PAIR_CLASSES:
        raise ValueError("Pair Class must be STRICT or NONSTRICT.")

    path = TRAINING_DATA_DIR / f"EV_ICE_Training_{pair_class}.xlsx"

    if not path.is_file():
        raise FileNotFoundError(f"Training workbook not found:\n{path}")

    return path


def validate_numeric_nonnegative(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    for column in columns:
        converted = pd.to_numeric(dataframe[column], errors="coerce")

        if converted.isna().any():
            raise ValueError(
                f"{column!r} contains missing/non-numeric values."
            )

        values = converted.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"{column!r} contains non-finite values.")

        if (values < 0).any():
            raise ValueError(f"{column!r} contains negative source values.")


def load_source(
    pair_class: str,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    pair_class = str(pair_class).strip().upper()
    path = training_file(pair_class)

    dataframe = pd.read_excel(
        path,
        sheet_name="01_Training_Data",
        header=3,
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError(f"{pair_class} training data is empty.")

    dataframe.columns = [str(c).strip() for c in dataframe.columns]

    required = (
        ALL_ICE_CANDIDATES
        + EV_TARGETS
        + [PAIR_COLUMN, PAIR_CLASS_COLUMN]
    )

    missing = [c for c in required if c not in dataframe.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f" - {c}" for c in missing)
        )

    if len(ICE_FEATURES) != 15 or len(EV_TARGETS) != 15:
        raise RuntimeError(
            "Frozen Full-DL contract must be exactly 15 inputs -> 15 outputs."
        )

    pair_ids = dataframe[PAIR_COLUMN].astype("string").str.strip()

    if pair_ids.isna().any() or pair_ids.eq("").any():
        raise ValueError("Pair ID contains missing/blank values.")

    if pair_ids.duplicated().any():
        duplicates = pair_ids[pair_ids.duplicated(keep=False)].tolist()
        raise ValueError(f"Duplicate Pair IDs found: {duplicates}")

    pair_classes = (
        dataframe[PAIR_CLASS_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if pair_classes.isna().any() or pair_classes.eq("").any():
        raise ValueError("Pair Class contains missing/blank values.")

    observed = set(pair_classes.tolist())

    if observed != {pair_class}:
        raise ValueError(
            f"Expected authoritative Pair Class {pair_class}, "
            f"observed {sorted(observed)}."
        )

    validate_numeric_nonnegative(
        dataframe,
        ALL_ICE_CANDIDATES + EV_TARGETS,
    )

    constants: dict[str, Any] = {}

    for column in EXCLUDED_CONSTANT_ICE_FEATURES:
        values = pd.unique(dataframe[column])

        constants[column] = {
            "unique_count": int(len(values)),
            "value": float(values[0]) if len(values) == 1 else None,
        }

        if len(values) != 1:
            raise ValueError(
                f"{column!r} is no longer constant in {pair_class}. "
                "Frozen 15-input interface stopped safely."
            )

    report = {
        "rows": int(len(dataframe)),
        "unique_pairs": int(dataframe[PAIR_COLUMN].nunique()),
        "strict_rows": int((pair_classes == "STRICT").sum()),
        "nonstrict_rows": int((pair_classes == "NONSTRICT").sum()),
        "excluded_constant_report": constants,
        "pair_class_handling": "READ / VALIDATE / PRESERVE ONLY",
        "pair_class_recalculation": "NONE",
        "pair_class_overwrite": "NONE",
        "pair_class_used_as_predictor": False,
    }

    return dataframe.reset_index(drop=True), path, report


def load_scope(
    scope: str,
) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    scope = normalize_scope(scope)

    if scope in VALID_PAIR_CLASSES:
        dataframe, path, report = load_source(scope)
        report["scope"] = scope
        return dataframe, [path], report

    strict, strict_path, strict_report = load_source("STRICT")
    nonstrict, nonstrict_path, nonstrict_report = load_source("NONSTRICT")

    overlap = (
        set(strict[PAIR_COLUMN].astype(str))
        & set(nonstrict[PAIR_COLUMN].astype(str))
    )

    if overlap:
        raise ValueError(
            f"Pair IDs overlap across STRICT/NONSTRICT: {sorted(overlap)}"
        )

    if list(strict.columns) != list(nonstrict.columns):
        raise ValueError(
            "STRICT and NONSTRICT schemas differ; COMBINED stopped."
        )

    dataframe = pd.concat(
        [strict, nonstrict],
        ignore_index=True,
        sort=False,
    )

    if dataframe[PAIR_COLUMN].astype(str).duplicated().any():
        raise ValueError("Duplicate Pair IDs after COMBINED concatenation.")

    classes = set(
        dataframe[PAIR_CLASS_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if classes != {"STRICT", "NONSTRICT"}:
        raise ValueError(
            "COMBINED must preserve both authoritative Pair Classes."
        )

    report = {
        "scope": "COMBINED",
        "rows": int(len(dataframe)),
        "unique_pairs": int(dataframe[PAIR_COLUMN].nunique()),
        "strict_rows": int(strict_report["rows"]),
        "nonstrict_rows": int(nonstrict_report["rows"]),
        "overlapping_pair_ids": 0,
        "pair_class_handling": "READ / VALIDATE / PRESERVE ONLY",
        "pair_class_recalculation": "NONE",
        "pair_class_overwrite": "NONE",
        "pair_class_used_as_predictor": False,
    }

    return (
        dataframe.reset_index(drop=True),
        [strict_path, nonstrict_path],
        report,
    )


def output_dirs(scope: str) -> tuple[Path, Path]:
    scope = normalize_scope(scope)

    return (
        RESULTS_ROOT / scope / "DL" / "Full_Model",
        MODEL_ROOT / scope / "DL" / "Full_Model",
    )


# =============================================================================
# MODEL
# =============================================================================

@dataclass(frozen=True)
class HParams:
    hidden_layers: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float


class FullDNN(nn.Module):
    def __init__(self, hp: HParams) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous = len(ICE_FEATURES)

        for width in hp.hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.ReLU(),
                ]
            )

            if hp.dropout > 0:
                layers.append(nn.Dropout(hp.dropout))

            previous = width

        layers.append(nn.Linear(previous, len(EV_TARGETS)))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_fixed(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    hp: HParams,
    epochs: int,
    seed: int,
    device: torch.device,
    gradient_clip: float,
) -> tuple[FullDNN, dict[str, Any]]:
    """
    Fit one fixed DNN and return optimization diagnostics.

    Diagnostics describe training-loss behavior only. They are NOT used for
    model selection and are NOT evidence of held-out/generalization quality.
    """
    set_seed(seed)

    model = FullDNN(hp).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.learning_rate,
        weight_decay=hp.weight_decay,
    )

    loss_function = nn.MSELoss()

    X_tensor = torch.as_tensor(
        X_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_tensor = torch.as_tensor(
        y_scaled,
        dtype=torch.float32,
        device=device,
    )

    # Deterministic initial loss with dropout disabled.
    model.eval()
    with torch.no_grad():
        initial_loss = float(
            loss_function(
                model(X_tensor),
                y_tensor,
            ).detach().cpu().item()
        )

    epoch_losses: list[float] = []

    model.train()

    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)

        loss = loss_function(
            model(X_tensor),
            y_tensor,
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Full-DL training loss.")

        loss_value = float(
            loss.detach().cpu().item()
        )

        epoch_losses.append(
            loss_value
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip,
        )

        optimizer.step()

    model.eval()

    with torch.no_grad():
        final_eval_loss = float(
            loss_function(
                model(X_tensor),
                y_tensor,
            ).detach().cpu().item()
        )

    losses = np.asarray(
        epoch_losses,
        dtype=float,
    )

    if (
        losses.size != int(epochs)
        or not np.isfinite(losses).all()
        or not np.isfinite(initial_loss)
        or not np.isfinite(final_eval_loss)
    ):
        raise RuntimeError(
            "Full-DL convergence diagnostics contain non-finite values."
        )

    window = max(
        1,
        min(
            int(epochs) // 10,
            25,
        ),
    )

    first_window_mean = float(
        np.mean(
            losses[
                :window
            ]
        )
    )

    last_window_mean = float(
        np.mean(
            losses[
                -window:
            ]
        )
    )

    if len(losses) >= 2 * window:
        previous_window_mean = float(
            np.mean(
                losses[
                    -2 * window:
                    -window
                ]
            )
        )
    else:
        previous_window_mean = first_window_mean

    scale = max(
        abs(
            previous_window_mean
        ),
        1e-12,
    )

    tail_relative_change = float(
        (
            last_window_mean
            - previous_window_mean
        )
        / scale
    )

    relative_reduction = float(
        (
            initial_loss
            - final_eval_loss
        )
        / max(
            abs(
                initial_loss
            ),
            1e-12,
        )
    )

    if abs(
        tail_relative_change
    ) <= 0.02:
        tail_status = (
            "TAIL_STABLE_WITHIN_2_PERCENT"
        )
    elif tail_relative_change < 0:
        tail_status = (
            "TAIL_STILL_DECREASING"
        )
    else:
        tail_status = (
            "TAIL_INCREASED"
        )

    overall_status = (
        "TRAINING_LOSS_DECREASED"
        if final_eval_loss < initial_loss
        else "CHECK_TRAINING_LOSS"
    )

    diagnostics = {
        "Seed": int(seed),
        "Epochs": int(epochs),
        "Initial Eval Training MSE": initial_loss,
        "First Epoch Training MSE": float(
            losses[
                0
            ]
        ),
        "Minimum Epoch Training MSE": float(
            np.min(
                losses
            )
        ),
        "Final Epoch Training MSE": float(
            losses[
                -1
            ]
        ),
        "Final Eval Training MSE": final_eval_loss,
        "First Window Mean Training MSE": first_window_mean,
        "Previous Tail Window Mean Training MSE": previous_window_mean,
        "Last Window Mean Training MSE": last_window_mean,
        "Relative Initial-to-Final Loss Reduction": relative_reduction,
        "Tail Relative Loss Change": tail_relative_change,
        "Tail Diagnostic": tail_status,
        "Overall Optimization Diagnostic": overall_status,
        "Interpretation": CONVERGENCE_INTERPRETATION,
    }

    return model, diagnostics



def predict_scaled(
    model: FullDNN,
    X_scaled: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(
            X_scaled,
            dtype=torch.float32,
            device=device,
        )

        return model(tensor).detach().cpu().numpy()


# =============================================================================
# LOPO OOS EVALUATION — NO NESTED HPO
# =============================================================================

def run_lopo(
    dataframe: pd.DataFrame,
    scope: str,
    seeds: tuple[int, ...],
    device: torch.device,
    hp: HParams,
    epochs: int,
    gradient_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_ids = dataframe[PAIR_COLUMN].astype(str).to_numpy()
    unique_pairs = list(dict.fromkeys(pair_ids.tolist()))

    if len(unique_pairs) < 3:
        raise ValueError(
            "At least 3 Pair IDs are required for Full-DL LOPO."
        )

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    for fold_number, pair_id in enumerate(unique_pairs, start=1):
        started = time.monotonic()

        test_mask = pair_ids == pair_id
        train_dataframe = dataframe.loc[~test_mask].reset_index(drop=True)
        test_dataframe = dataframe.loc[test_mask].reset_index(drop=True)

        if len(test_dataframe) != 1:
            raise RuntimeError(
                f"Pair {pair_id} must correspond to exactly one row."
            )

        pair_class = str(
            test_dataframe.loc[0, PAIR_CLASS_COLUMN]
        ).strip().upper()

        X_train = train_dataframe[ICE_FEATURES].to_numpy(dtype=float)
        y_train = train_dataframe[EV_TARGETS].to_numpy(dtype=float)
        X_test = test_dataframe[ICE_FEATURES].to_numpy(dtype=float)
        y_test = test_dataframe[EV_TARGETS].to_numpy(dtype=float)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train)
        X_test_scaled = x_scaler.transform(X_test)

        seed_predictions_scaled = []

        for seed in seeds:
            model, convergence = train_fixed(
                X_scaled=X_train_scaled,
                y_scaled=y_train_scaled,
                hp=hp,
                epochs=epochs,
                seed=seed,
                device=device,
                gradient_clip=gradient_clip,
            )

            convergence_rows.append(
                {
                    "Dataset Scope": scope,
                    "Training Role": "OUTER_LOPO_TRAIN",
                    "Outer Fold": fold_number,
                    "Held-Out Pair ID": pair_id,
                    "Held-Out Pair Class": pair_class,
                    **convergence,
                }
            )

            seed_predictions_scaled.append(
                predict_scaled(
                    model,
                    X_test_scaled,
                    device,
                )
            )

        stacked_scaled = np.stack(seed_predictions_scaled, axis=0)
        mean_scaled = stacked_scaled.mean(axis=0)

        raw_prediction = y_scaler.inverse_transform(mean_scaled)
        prediction = np.maximum(raw_prediction, 0.0)

        seed_original = np.stack(
            [
                y_scaler.inverse_transform(item)
                for item in seed_predictions_scaled
            ],
            axis=0,
        )

        ensemble_sd = seed_original.std(axis=0, ddof=0)

        baseline_scaled = np.zeros_like(mean_scaled)
        baseline_prediction = y_scaler.inverse_transform(baseline_scaled)

        y_test_scaled = y_scaler.transform(y_test)

        model_standardized_rmse = float(
            np.sqrt(
                np.mean(
                    (y_test_scaled - mean_scaled) ** 2
                )
            )
        )

        baseline_standardized_rmse = float(
            np.sqrt(
                np.mean(
                    y_test_scaled ** 2
                )
            )
        )

        fold_rows.append(
            {
                "Outer Fold": fold_number,
                "Pair ID": pair_id,
                "Pair Class": pair_class,
                "Architecture": arch_to_text(hp.hidden_layers),
                "Dropout": hp.dropout,
                "Learning Rate": hp.learning_rate,
                "Weight Decay": hp.weight_decay,
                "Fixed Epochs": epochs,
                "Ensemble Seeds": " | ".join(str(s) for s in seeds),
                "Hyperparameter Interpretation": (
                    HYPERPARAMETER_INTERPRETATION
                ),
                "Model Standardized RMSE": model_standardized_rmse,
                "TrainingMean Baseline Standardized RMSE": (
                    baseline_standardized_rmse
                ),
                "Baseline Relative Standardized RMSE": (
                    model_standardized_rmse / baseline_standardized_rmse
                    if baseline_standardized_rmse > 0
                    else np.nan
                ),
                "Elapsed Seconds": time.monotonic() - started,
            }
        )

        row: dict[str, Any] = {
            "Outer Fold": fold_number,
            "Pair ID": pair_id,
            "Pair Class": pair_class,
            "Model Scope": scope,
            "Model Source": "FULL_DL",
            "Architecture": arch_to_text(hp.hidden_layers),
            "Fixed Epochs": epochs,
            "Selection Method": HYPERPARAMETER_SELECTION_METHOD,
            "Hyperparameter Interpretation": (
                HYPERPARAMETER_INTERPRETATION
            ),
        }

        for target_index, target in enumerate(EV_TARGETS):
            raw_value = float(raw_prediction[0, target_index])
            post_value = float(prediction[0, target_index])

            row[f"Actual {target}"] = float(
                y_test[0, target_index]
            )

            row[f"Raw Predicted {target}"] = raw_value
            row[f"Predicted {target}"] = post_value

            row[f"Ensemble SD {target}"] = float(
                ensemble_sd[0, target_index]
            )

            row[f"TrainingMeanBaseline {target}"] = float(
                baseline_prediction[0, target_index]
            )

            row[f"Raw Negative Flag {target}"] = bool(
                raw_value < 0
            )

        row["Predicted Six-Phase Time Sum Before Charging (min)"] = float(
            sum(
                row[f"Predicted {target}"]
                for target in PHASE_TIME_TARGETS
            )
        )

        prediction_rows.append(row)

        print(
            f"Outer {fold_number}/{len(unique_pairs)}"
            f" | {pair_id}"
            f" | {pair_class}"
            f" | {time.monotonic() - started:.2f}s"
            f" | stdRMSE={model_standardized_rmse:.4f}",
            flush=True,
        )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(convergence_rows),
    )



# =============================================================================
# METRICS / QC
# =============================================================================

def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    mask = np.abs(y_true) > 1e-8

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100.0
    )


def target_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target in EV_TARGETS:
        actual = predictions[
            f"Actual {target}"
        ].to_numpy(dtype=float)

        predicted = predictions[
            f"Predicted {target}"
        ].to_numpy(dtype=float)

        raw = predictions[
            f"Raw Predicted {target}"
        ].to_numpy(dtype=float)

        baseline = predictions[
            f"TrainingMeanBaseline {target}"
        ].to_numpy(dtype=float)

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        )

        baseline_rmse = float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    baseline,
                )
            )
        )

        rows.append(
            {
                "Target": target,
                "MAE": float(
                    mean_absolute_error(
                        actual,
                        predicted,
                    )
                ),
                "RMSE": rmse,
                "R2": (
                    float(
                        r2_score(
                            actual,
                            predicted,
                        )
                    )
                    if len(np.unique(actual)) > 1
                    else np.nan
                ),
                "Median Absolute Error": float(
                    median_absolute_error(
                        actual,
                        predicted,
                    )
                ),
                "MAPE (%)": safe_mape(
                    actual,
                    predicted,
                ),
                "Raw RMSE": float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            raw,
                        )
                    )
                ),
                "TrainingMean Baseline RMSE": baseline_rmse,
                "Baseline Relative RMSE": (
                    rmse / baseline_rmse
                    if baseline_rmse > 0
                    else np.nan
                ),
                "Raw Negative Predictions": int(
                    predictions[
                        f"Raw Negative Flag {target}"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def qc_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    phases = (
        "STOP",
        "IDLE",
        "INCHING",
        "ACC",
        "DEC",
        "CRUISE",
    )

    rows = []

    for _, record in predictions.iterrows():
        negative_targets = [
            target
            for target in EV_TARGETS
            if bool(
                record[
                    f"Raw Negative Flag {target}"
                ]
            )
        ]

        phase_distance = sum(
            float(
                record[
                    f"Predicted EV {phase} Distance (km)"
                ]
            )
            for phase in phases
        )

        phase_time = float(
            record[
                "Predicted Six-Phase Time Sum Before Charging (min)"
            ]
        )

        rows.append(
            {
                "Pair ID": record["Pair ID"],
                "Pair Class": record["Pair Class"],
                "Raw Negative Target Count": len(negative_targets),
                "Raw Negative Targets": " | ".join(negative_targets),
                "Predicted Total Distance (km)": float(
                    record["Predicted EV Total Distance (km)"]
                ),
                "Sum Predicted Phase Distances (km)": phase_distance,
                "Distance Closure Residual (km)": (
                    float(
                        record[
                            "Predicted EV Total Distance (km)"
                        ]
                    )
                    - phase_distance
                ),
                "Direct Predicted EV Total TAT (min)": float(
                    record[
                        "Predicted EV Total TAT (min)"
                    ]
                ),
                "Six-Phase Time Sum Before Charging (min)": (
                    phase_time
                ),
            }
        )

    return pd.DataFrame(rows)


def overall_summary(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    ratios = (
        metrics["Baseline Relative RMSE"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    return pd.DataFrame(
        [
            {
                "Model": "Full_DNN_Fixed_Ensemble",
                "OOS Pairs": len(predictions),
                "Inputs": len(ICE_FEATURES),
                "Outputs": len(EV_TARGETS),
                "Selection Method": "PREDECLARED_FIXED_NO_HPO",
                "Mean Target-wise Baseline-Relative RMSE": (
                    float(ratios.mean())
                    if not ratios.empty
                    else np.nan
                ),
                "Median Target-wise Baseline-Relative RMSE": (
                    float(ratios.median())
                    if not ratios.empty
                    else np.nan
                ),
                "Targets Better Than Baseline": int(
                    (
                        metrics["Baseline Relative RMSE"]
                        < 1.0
                    ).sum()
                ),
                "Mean Outer Standardized RMSE": float(
                    folds["Model Standardized RMSE"].mean()
                ),
            }
        ]
    )


# =============================================================================
# FINAL DEPLOYMENT ENSEMBLE
# =============================================================================

def save_final_models(
    dataframe: pd.DataFrame,
    scope: str,
    model_dir: Path,
    seeds: tuple[int, ...],
    device: torch.device,
    hp: HParams,
    epochs: int,
    gradient_clip: float,
) -> tuple[Path, pd.DataFrame]:
    X = dataframe[ICE_FEATURES].to_numpy(dtype=float)
    y = dataframe[EV_TARGETS].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    selected_dir = model_dir / "Selected_Final"
    selected_dir.mkdir(parents=True, exist_ok=True)

    convergence_rows: list[dict[str, Any]] = []

    for seed in seeds:
        model, convergence = train_fixed(
            X_scaled=X_scaled,
            y_scaled=y_scaled,
            hp=hp,
            epochs=epochs,
            seed=seed,
            device=device,
            gradient_clip=gradient_clip,
        )

        convergence_rows.append(
            {
                "Dataset Scope": scope,
                "Training Role": "FINAL_DEPLOYMENT_FIT",
                "Outer Fold": np.nan,
                "Held-Out Pair ID": "",
                "Held-Out Pair Class": "",
                **convergence,
            }
        )

        checkpoint = {
            "script_version": SCRIPT_VERSION,
            "model_source": "FULL_DL",
            "dataset_scope": scope,
            "seed": int(seed),
            "hidden_layers": list(hp.hidden_layers),
            "dropout": float(hp.dropout),
            "learning_rate": float(hp.learning_rate),
            "weight_decay": float(hp.weight_decay),
            "epochs": int(epochs),
            "selection_method": HYPERPARAMETER_SELECTION_METHOD,
            "hyperparameter_interpretation": (
                HYPERPARAMETER_INTERPRETATION
            ),
            "input_features": ICE_FEATURES,
            "target_features": EV_TARGETS,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }

        torch.save(
            checkpoint,
            selected_dir / f"Full_DNN_seed_{seed}.pt",
        )

    joblib.dump(
        x_scaler,
        selected_dir / "x_scaler.joblib",
    )

    joblib.dump(
        y_scaler,
        selected_dir / "y_scaler.joblib",
    )

    (
        selected_dir / "hyperparameters.json"
    ).write_text(
        json.dumps(
            {
                "selection_method": HYPERPARAMETER_SELECTION_METHOD,
                "hyperparameter_interpretation": (
                    HYPERPARAMETER_INTERPRETATION
                ),
                "hyperparameter_rationale": HYPERPARAMETER_RATIONALE,
                "predeclared_meaning": PREDECLARED_MEANING,
                "architecture": arch_to_text(hp.hidden_layers),
                "hidden_layers": list(hp.hidden_layers),
                "dropout": hp.dropout,
                "learning_rate": hp.learning_rate,
                "weight_decay": hp.weight_decay,
                "fixed_epochs": epochs,
                "ensemble_seeds": list(seeds),
                "hpo_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        selected_dir / "feature_schema.json"
    ).write_text(
        json.dumps(
            {
                "input_features": ICE_FEATURES,
                "excluded_constant_ice_features": (
                    EXCLUDED_CONSTANT_ICE_FEATURES
                ),
                "input_count": 15,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        selected_dir / "target_schema.json"
    ).write_text(
        json.dumps(
            {
                "target_features": EV_TARGETS,
                "target_count": 15,
                "physical_postprocessing": (
                    "max(raw_prediction, 0)"
                ),
                "charging": (
                    "external common charging_time.py"
                ),
                "final_assembly": (
                    "external common final_tat_assembly.py"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        selected_dir,
        pd.DataFrame(
            convergence_rows
        ),
    )



# =============================================================================
# RUN ONE SCOPE
# =============================================================================

def run_scope(
    args: argparse.Namespace,
    scope: str,
    device: torch.device,
    seeds: tuple[int, ...],
    hp: HParams,
) -> None:
    scope = normalize_scope(scope)
    results_dir, model_dir = output_dirs(scope)

    print("\n" + "=" * 92)
    print(f"FULL DL — FIXED PREDECLARED LOPO | {scope}")
    print("=" * 92)
    print(f"Version      : {SCRIPT_VERSION}")
    print(f"Device       : {device}")
    print(f"Architecture : 15 -> {arch_to_text(hp.hidden_layers)} -> 15")
    print(f"Dropout      : {hp.dropout}")
    print(f"LR           : {hp.learning_rate}")
    print(f"Weight decay : {hp.weight_decay}")
    print(f"Epochs       : {args.epochs}")
    print(f"Seeds        : {seeds}")
    print("HPO          : DISABLED")
    print(f"HP status    : {HYPERPARAMETER_INTERPRETATION}")
    print("Evaluation   : LOPO only")
    print("Pair Class   : READ / VALIDATE / PRESERVE ONLY")
    print("Convergence  : training-loss diagnostic saved; not model selection")

    dataframe, input_files, report = load_scope(scope)

    print(f"Rows         : {len(dataframe)}")
    print(f"STRICT       : {report['strict_rows']}")
    print(f"NONSTRICT    : {report['nonstrict_rows']}")

    predictions, folds, outer_convergence = run_lopo(
        dataframe=dataframe,
        scope=scope,
        seeds=seeds,
        device=device,
        hp=hp,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip_norm,
    )

    metrics = target_metrics(predictions)
    qc = qc_table(predictions)
    summary = overall_summary(predictions, metrics, folds)

    selected_dir, final_convergence = save_final_models(
        dataframe=dataframe,
        scope=scope,
        model_dir=model_dir,
        seeds=seeds,
        device=device,
        hp=hp,
        epochs=args.epochs,
        gradient_clip=args.gradient_clip_norm,
    )

    convergence = pd.concat(
        [
            outer_convergence,
            final_convergence,
        ],
        ignore_index=True,
        sort=False,
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        ("Script Version", SCRIPT_VERSION),
        ("Dataset Scope", scope),
        ("Input Files", " | ".join(str(p) for p in input_files)),
        (
            "Input SHA256",
            " | ".join(
                f"{p.name}:{file_sha256(p)}"
                for p in input_files
            ),
        ),
        ("Rows", len(dataframe)),
        ("Unique Pairs", dataframe[PAIR_COLUMN].nunique()),
        ("STRICT Rows", report["strict_rows"]),
        ("NONSTRICT Rows", report["nonstrict_rows"]),
        ("Pair Class Handling", "READ / VALIDATE / PRESERVE ONLY"),
        ("Pair Class Recalculation", "NONE"),
        ("Pair Class Overwrite", "NONE"),
        ("Pair Class Used As Predictor", "NO"),
        ("Outer Evaluation", "Leave-One-Pair-Out"),
        ("Hyperparameter Search", "DISABLED"),
        ("Selection Method", HYPERPARAMETER_SELECTION_METHOD),
        (
            "Hyperparameter Interpretation",
            HYPERPARAMETER_INTERPRETATION,
        ),
        ("Hyperparameter Rationale", HYPERPARAMETER_RATIONALE),
        ("Predeclared Meaning", PREDECLARED_MEANING),
        ("Architecture", arch_to_text(hp.hidden_layers)),
        ("Dropout", hp.dropout),
        ("Learning Rate", hp.learning_rate),
        ("Weight Decay", hp.weight_decay),
        ("Fixed Epochs", args.epochs),
        ("Activation", "ReLU"),
        ("Optimizer", "AdamW"),
        ("Loss", "MSELoss"),
        (
            "Scaler",
            "StandardScaler fit on each outer training fold only",
        ),
        ("Gradient Clipping", args.gradient_clip_norm),
        ("Ensemble Seeds", " | ".join(str(s) for s in seeds)),
        (
            "Physical Constraint",
            "max(raw_prediction, 0), identical in OOS and inference",
        ),
        (
            "Convergence Evidence",
            "Per-seed training-loss diagnostics saved separately",
        ),
        (
            "Convergence Interpretation",
            CONVERGENCE_INTERPRETATION,
        ),
        ("Charging", "NOT calculated in training code"),
        ("Final Assembly", "NOT calculated in training code"),
        ("Python", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("NumPy", np.__version__),
        ("Pandas", pd.__version__),
        ("PyTorch", torch.__version__),
        ("Device", str(device)),
    ]

    manifest = pd.DataFrame(
        manifest_rows,
        columns=["Parameter", "Value"],
    )

    predictions.to_excel(
        results_dir / "Full_DL_LOPO_OOS_Predictions.xlsx",
        index=False,
    )

    metrics.to_excel(
        results_dir / "Full_DL_Target_Wise_OOS_Metrics.xlsx",
        index=False,
    )

    summary.to_excel(
        results_dir / "Full_DL_OOS_Summary.xlsx",
        index=False,
    )

    qc.to_excel(
        results_dir / "Full_DL_Physical_QC.xlsx",
        index=False,
    )

    folds.to_excel(
        results_dir / "Full_DL_Outer_Folds.xlsx",
        index=False,
    )

    convergence.to_excel(
        results_dir / "Full_DL_Training_Convergence.xlsx",
        index=False,
    )

    manifest.to_excel(
        results_dir / "Full_DL_Run_Manifest.xlsx",
        index=False,
    )

    with pd.ExcelWriter(
        results_dir / "Full_DL_Detailed_Evaluation.xlsx",
        engine="openpyxl",
    ) as writer:
        summary.to_excel(
            writer,
            sheet_name="01_Overall",
            index=False,
        )

        metrics.to_excel(
            writer,
            sheet_name="02_Target_Metrics",
            index=False,
        )

        folds.to_excel(
            writer,
            sheet_name="03_Outer_Folds",
            index=False,
        )

        qc.to_excel(
            writer,
            sheet_name="04_Physical_QC",
            index=False,
        )

        convergence.to_excel(
            writer,
            sheet_name="05_Convergence",
            index=False,
        )

        manifest.to_excel(
            writer,
            sheet_name="06_Manifest",
            index=False,
        )

    (
        selected_dir / "training_manifest.json"
    ).write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "dataset_scope": scope,
                "input_files": [str(p) for p in input_files],
                "input_sha256": {
                    p.name: file_sha256(p)
                    for p in input_files
                },
                "rows": len(dataframe),
                "unique_pairs": int(
                    dataframe[PAIR_COLUMN].nunique()
                ),
                "pair_class_handling": (
                    "READ / VALIDATE / PRESERVE ONLY"
                ),
                "pair_class_recalculation": "NONE",
                "pair_class_overwrite": "NONE",
                "pair_class_used_as_predictor": False,
                "selection_method": HYPERPARAMETER_SELECTION_METHOD,
                "hyperparameter_interpretation": (
                    HYPERPARAMETER_INTERPRETATION
                ),
                "hyperparameter_rationale": HYPERPARAMETER_RATIONALE,
                "predeclared_meaning": PREDECLARED_MEANING,
                "architecture": arch_to_text(hp.hidden_layers),
                "dropout": hp.dropout,
                "learning_rate": hp.learning_rate,
                "weight_decay": hp.weight_decay,
                "fixed_epochs": args.epochs,
                "ensemble_seeds": list(seeds),
                "hpo_used": False,
                "convergence_diagnostic": (
                    "Full_DL_Training_Convergence.xlsx; "
                    + CONVERGENCE_INTERPRETATION
                ),
                "physical_postprocessing": (
                    "max(raw_prediction, 0)"
                ),
                "charging": (
                    "external common charging_time.py"
                ),
                "final_assembly": (
                    "external common final_tat_assembly.py"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print(f"FULL DL COMPLETED | {scope}")
    print("=" * 92)
    print(f"OOS pairs       : {len(predictions)}")
    print(f"Final models    : {selected_dir}")
    print(f"Results         : {results_dir}")
    print("Nested HPO      : REMOVED")
    print(f"Hyperparameters : {HYPERPARAMETER_INTERPRETATION}")
    print("Convergence     : Full_DL_Training_Convergence.xlsx")
    print("Pair Class      : NOT recalculated or overwritten")



# =============================================================================
# CLI
# =============================================================================

def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-DL fixed predeclared LOPO research pipeline. "
            "No nested Optuna/HPO."
        )
    )

    parser.add_argument(
        "--scope",
        choices=[
            "STRICT",
            "NONSTRICT",
            "COMBINED",
            "ALL",
        ],
        default="ALL",
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--ensemble-seeds",
        default=",".join(
            str(seed)
            for seed in ENSEMBLE_SEEDS_DEFAULT
        ),
    )

    parser.add_argument(
        "--architecture",
        default=arch_to_text(DEFAULT_HIDDEN_LAYERS),
        help="Predeclared hidden-layer widths, e.g. 16-8.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--epochs",
        "--max-epochs",
        dest="epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "Fixed training epochs. --max-epochs is retained "
            "as a backward-compatible alias."
        ),
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP,
    )

    return parser


def main() -> None:
    args = parser().parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if not (0 <= args.dropout < 1):
        raise ValueError("--dropout must be in [0, 1).")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")

    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be > 0.")

    requested_scope = normalize_scope(
        args.scope,
        allow_all=True,
    )

    scopes = (
        ["STRICT", "NONSTRICT", "COMBINED"]
        if requested_scope == "ALL"
        else [requested_scope]
    )

    seeds = parse_seeds(args.ensemble_seeds)
    device = resolve_device(args.device)

    hp = HParams(
        hidden_layers=parse_architecture(args.architecture),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    print("=" * 92)
    print("FULL DL — FIXED PREDECLARED LOPO")
    print("=" * 92)
    print(f"Version : {SCRIPT_VERSION}")
    print(f"Scopes  : {scopes}")
    print(f"Device  : {device}")
    print(f"Seeds   : {seeds}")
    print("HPO     : DISABLED")

    for scope in scopes:
        run_scope(
            args=args,
            scope=scope,
            device=device,
            seeds=seeds,
            hp=hp,
        )

    print("\n" + "=" * 92)
    print("ALL REQUESTED FULL-DL RUNS COMPLETED")
    print("=" * 92)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
