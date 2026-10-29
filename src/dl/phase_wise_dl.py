from __future__ import annotations

import argparse
import hashlib
import json
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

SCRIPT_VERSION = "2026-10-01_PHASE_WISE_DL_V8_FIXED_LOPO_CONVERGENCE"

DEFAULT_HIDDEN_LAYERS = (4, 2)
DEFAULT_DROPOUT = 0.10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_EPOCHS = 250
DEFAULT_GRADIENT_CLIP = 1.0

DEFAULT_ENSEMBLE_SEEDS = (
    42,
    123,
    2026,
    31415,
    27182,
)


HYPERPARAMETER_INTERPRETATION = (
    "FIXED/PREDECLARED FOR THIS CODE VERSION; NOT HPO-OPTIMIZED"
)

HYPERPARAMETER_SELECTION_METHOD = (
    "FIXED_PREDECLARED_ANALYSIS_SPECIFICATION_NOT_HPO_OPTIMIZED"
)

HYPERPARAMETER_RATIONALE = (
    "Compact phase DNN specification fixed for this revised small-sample "
    "analysis; not selected by LOPO outcomes and not optimized by Optuna/HPO."
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
    PART1_PROJECT
    / "Data"
    / "Results"
    / "Master_Summary"
    / "Training_Data"
)

RESULTS_ROOT = PROJECT_ROOT / "Results"
MODEL_ROOT = PROJECT_ROOT / "models"


# =============================================================================
# FROZEN PHASE CONTRACT
# =============================================================================

PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
]

PHASE_FEATURES = {
    phase: [
        f"ICE {phase} Time (min)",
        f"ICE {phase} Distance (km)",
    ]
    for phase in PHASES
}

PHASE_TARGETS = {
    phase: [
        f"EV {phase} Time (min)",
        f"EV {phase} Distance (km)",
    ]
    for phase in PHASES
}

ALL_PHASE_MODEL_COLUMNS = [
    column
    for phase in PHASES
    for column in (
        PHASE_FEATURES[phase]
        + PHASE_TARGETS[phase]
    )
]

PAIR_COLUMN = "Pair ID"
PAIR_CLASS_COLUMN = "Pair Class"

VALID_PAIR_CLASSES = {
    "STRICT",
    "NONSTRICT",
}

VALID_SCOPES = {
    "STRICT",
    "NONSTRICT",
    "COMBINED",
    "ALL",
}


# =============================================================================
# HELPERS / DEVICE / REPRODUCIBILITY
# =============================================================================

def set_reproducibility(seed: int) -> None:
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        ":4096:8",
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()

    if requested in {"auto", "cpu"}:
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but CUDA is unavailable."
            )

        return torch.device("cuda")

    raise ValueError("--device must be auto, cpu or cuda.")


def normalize_scope(
    value: str,
    *,
    allow_all: bool = False,
) -> str:
    normalized = str(value).strip().upper()
    allowed = (
        VALID_SCOPES
        if allow_all
        else VALID_SCOPES - {"ALL"}
    )

    if normalized not in allowed:
        raise ValueError(
            "Dataset scope must be one of: "
            + ", ".join(sorted(allowed))
            + f". Received: {value!r}"
        )

    return normalized


def parse_seed_list(text: str) -> tuple[int, ...]:
    seeds = tuple(
        int(item.strip())
        for item in str(text).split(",")
        if item.strip()
    )

    if not seeds:
        raise ValueError(
            "At least one ensemble seed is required."
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Ensemble seeds must be unique."
        )

    return seeds


def parse_architecture(text: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in str(text).split("-")
        if item.strip()
    )

    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "Architecture must contain positive widths, e.g. 4-2."
        )

    return values


def architecture_to_text(
    architecture: tuple[int, ...],
) -> str:
    return "-".join(
        str(int(value))
        for value in architecture
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# DATA LOADING / VALIDATION
# =============================================================================

def get_training_file(pair_class: str) -> Path:
    pair_class = str(pair_class).strip().upper()

    if pair_class not in VALID_PAIR_CLASSES:
        raise ValueError(
            "Training workbook class must be STRICT or NONSTRICT."
        )

    path = (
        TRAINING_DATA_DIR
        / f"EV_ICE_Training_{pair_class}.xlsx"
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Training workbook was not found:\n"
            f"{path}"
        )

    return path


def validate_single_scope_dataframe(
    dataframe: pd.DataFrame,
    expected_pair_class: str,
) -> dict[str, Any]:
    expected_pair_class = str(
        expected_pair_class
    ).strip().upper()

    required = [
        PAIR_COLUMN,
        PAIR_CLASS_COLUMN,
        *ALL_PHASE_MODEL_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Missing required phase-DL columns:\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )

    pair_ids = (
        dataframe[PAIR_COLUMN]
        .astype("string")
        .str.strip()
    )

    if pair_ids.isna().any() or pair_ids.eq("").any():
        raise ValueError(
            "Pair ID contains missing/blank values."
        )

    if pair_ids.duplicated().any():
        duplicate_pairs = (
            pair_ids[
                pair_ids.duplicated(
                    keep=False
                )
            ].tolist()
        )

        raise ValueError(
            f"Duplicate Pair IDs detected: {duplicate_pairs}"
        )

    pair_classes = (
        dataframe[PAIR_CLASS_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if pair_classes.isna().any() or pair_classes.eq("").any():
        raise ValueError(
            "Pair Class contains missing/blank values."
        )

    observed = set(
        pair_classes.tolist()
    )

    if observed != {
        expected_pair_class
    }:
        raise ValueError(
            "Authoritative Pair Class mismatch. "
            f"Expected {expected_pair_class}; observed {sorted(observed)}."
        )

    for column in ALL_PHASE_MODEL_COLUMNS:
        converted = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if converted.isna().any():
            raise ValueError(
                f"{column!r} contains missing/non-numeric values."
            )

        values = converted.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column!r} contains non-finite values."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column!r} contains negative source values."
            )

    return {
        "rows": int(len(dataframe)),
        "unique_pairs": int(
            dataframe[PAIR_COLUMN].nunique()
        ),
        "strict_rows": int(
            (pair_classes == "STRICT").sum()
        ),
        "nonstrict_rows": int(
            (pair_classes == "NONSTRICT").sum()
        ),
        "pair_class_handling": (
            "READ / VALIDATE / PRESERVE ONLY"
        ),
        "pair_class_recalculation": "NONE",
        "pair_class_overwrite": "NONE",
        "pair_class_used_as_predictor": False,
    }


def load_single_scope(
    pair_class: str,
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    path = get_training_file(
        pair_class
    )

    dataframe = pd.read_excel(
        path,
        sheet_name="01_Training_Data",
        header=3,
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError(
            f"{pair_class} training workbook is empty."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    report = validate_single_scope_dataframe(
        dataframe,
        expected_pair_class=pair_class,
    )

    return (
        dataframe.reset_index(drop=True),
        path,
        report,
    )


def load_dataset_scope(
    scope: str,
) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    scope = normalize_scope(scope)

    if scope in VALID_PAIR_CLASSES:
        dataframe, path, report = load_single_scope(scope)
        report["scope"] = scope
        return dataframe, [path], report

    strict, strict_path, strict_report = load_single_scope(
        "STRICT"
    )

    nonstrict, nonstrict_path, nonstrict_report = load_single_scope(
        "NONSTRICT"
    )

    overlap = (
        set(strict[PAIR_COLUMN].astype(str))
        & set(nonstrict[PAIR_COLUMN].astype(str))
    )

    if overlap:
        raise ValueError(
            "Pair IDs overlap between STRICT and NONSTRICT:\n"
            + "\n".join(
                f" - {value}"
                for value in sorted(overlap)
            )
        )

    if list(strict.columns) != list(nonstrict.columns):
        raise ValueError(
            "STRICT/NONSTRICT schemas differ; COMBINED stopped."
        )

    dataframe = pd.concat(
        [strict, nonstrict],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if dataframe[PAIR_COLUMN].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate Pair IDs after COMBINED concatenation."
        )

    observed_classes = set(
        dataframe[PAIR_CLASS_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if observed_classes != {
        "STRICT",
        "NONSTRICT",
    }:
        raise ValueError(
            "COMBINED must preserve STRICT and NONSTRICT Pair Class."
        )

    report = {
        "scope": "COMBINED",
        "rows": int(len(dataframe)),
        "unique_pairs": int(
            dataframe[PAIR_COLUMN].nunique()
        ),
        "strict_rows": int(
            strict_report["rows"]
        ),
        "nonstrict_rows": int(
            nonstrict_report["rows"]
        ),
        "pair_class_handling": (
            "READ / VALIDATE / PRESERVE ONLY"
        ),
        "pair_class_recalculation": "NONE",
        "pair_class_overwrite": "NONE",
        "pair_class_used_as_predictor": False,
    }

    return (
        dataframe.reset_index(drop=True),
        [strict_path, nonstrict_path],
        report,
    )


def get_output_directories(
    scope: str,
) -> tuple[Path, Path]:
    scope = normalize_scope(scope)

    return (
        RESULTS_ROOT
        / scope
        / "DL"
        / "Phase_Wise",
        MODEL_ROOT
        / scope
        / "DL"
        / "Phase_Wise",
    )


# =============================================================================
# MODEL
# =============================================================================

@dataclass(frozen=True)
class DNNHyperparameters:
    hidden_layers: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float


class PhaseDNNRegressor(nn.Module):
    def __init__(
        self,
        hidden_layers: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = 2

        for hidden_dim in hidden_layers:
            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                )
            )

            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(
                    nn.Dropout(
                        p=dropout
                    )
                )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                2,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


def build_model(
    hyperparameters: DNNHyperparameters,
) -> PhaseDNNRegressor:
    return PhaseDNNRegressor(
        hidden_layers=hyperparameters.hidden_layers,
        dropout=hyperparameters.dropout,
    )


def train_fixed_epochs(
    *,
    X_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    hyperparameters: DNNHyperparameters,
    device: torch.device,
    seed: int,
    epochs: int,
    gradient_clip_norm: float,
) -> tuple[PhaseDNNRegressor, dict[str, Any]]:
    """
    Fit one fixed phase DNN and return training-loss diagnostics.

    Diagnostics are optimization evidence only. They are not used for model
    selection and do not replace held-out LOPO performance.
    """
    set_reproducibility(seed)

    model = build_model(
        hyperparameters
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
    )

    loss_function = nn.MSELoss()

    X_tensor = torch.as_tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=device,
    )

    y_tensor = torch.as_tensor(
        y_train_scaled,
        dtype=torch.float32,
        device=device,
    )

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
        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            X_tensor
        )

        loss = loss_function(
            prediction,
            y_tensor,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Phase-DL training loss became non-finite."
            )

        epoch_losses.append(
            float(
                loss.detach().cpu().item()
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
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
            "Phase-DL convergence diagnostics contain non-finite values."
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

    tail_relative_change = float(
        (
            last_window_mean
            - previous_window_mean
        )
        / max(
            abs(
                previous_window_mean
            ),
            1e-12,
        )
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
    *,
    model: PhaseDNNRegressor,
    X_scaled: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(
            X_scaled,
            dtype=torch.float32,
            device=device,
        )

        return (
            model(tensor)
            .detach()
            .cpu()
            .numpy()
        )


# =============================================================================
# LOPO FOR ONE PHASE — NO NESTED HPO
# =============================================================================

def run_lopo_phase(
    *,
    dataframe: pd.DataFrame,
    scope: str,
    phase: str,
    ensemble_seeds: tuple[int, ...],
    device: torch.device,
    hyperparameters: DNNHyperparameters,
    epochs: int,
    gradient_clip_norm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = PHASE_FEATURES[phase]
    targets = PHASE_TARGETS[phase]

    pair_ids = (
        dataframe[PAIR_COLUMN]
        .astype(str)
        .to_numpy()
    )

    unique_pairs = list(
        dict.fromkeys(
            pair_ids.tolist()
        )
    )

    if len(unique_pairs) < 3:
        raise ValueError(
            f"LOPO for {phase} requires at least 3 unique pairs."
        )

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    print("\n" + "=" * 92)
    print(f"PHASE: {phase} | FIXED PREDECLARED LOPO")
    print("=" * 92)

    for fold_number, test_pair_id in enumerate(
        unique_pairs,
        start=1,
    ):
        fold_started = time.monotonic()

        test_mask = (
            pair_ids
            == test_pair_id
        )

        outer_train = (
            dataframe.loc[
                ~test_mask
            ]
            .reset_index(
                drop=True
            )
        )

        outer_test = (
            dataframe.loc[
                test_mask
            ]
            .reset_index(
                drop=True
            )
        )

        if len(outer_test) != 1:
            raise RuntimeError(
                "Outer LOPO requires one row per Pair ID. "
                f"{test_pair_id} produced {len(outer_test)} rows."
            )

        pair_class = str(
            outer_test.loc[
                0,
                PAIR_CLASS_COLUMN,
            ]
        ).strip().upper()

        X_train = (
            outer_train[
                features
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_train = (
            outer_train[
                targets
            ]
            .to_numpy(
                dtype=float
            )
        )

        X_test = (
            outer_test[
                features
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_test = (
            outer_test[
                targets
            ]
            .to_numpy(
                dtype=float
            )
        )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = (
            x_scaler.fit_transform(
                X_train
            )
        )

        y_train_scaled = (
            y_scaler.fit_transform(
                y_train
            )
        )

        X_test_scaled = (
            x_scaler.transform(
                X_test
            )
        )

        seed_predictions_scaled = []

        for seed in ensemble_seeds:
            model, convergence = train_fixed_epochs(
                X_train_scaled=X_train_scaled,
                y_train_scaled=y_train_scaled,
                hyperparameters=hyperparameters,
                device=device,
                seed=seed,
                epochs=epochs,
                gradient_clip_norm=gradient_clip_norm,
            )

            convergence_rows.append(
                {
                    "Dataset Scope": scope,
                    "Phase": phase,
                    "Training Role": "OUTER_LOPO_TRAIN",
                    "Outer Fold": fold_number,
                    "Held-Out Pair ID": test_pair_id,
                    "Held-Out Pair Class": pair_class,
                    **convergence,
                }
            )

            seed_predictions_scaled.append(
                predict_scaled(
                    model=model,
                    X_scaled=X_test_scaled,
                    device=device,
                )
            )

        stacked_scaled = np.stack(
            seed_predictions_scaled,
            axis=0,
        )

        mean_scaled = np.mean(
            stacked_scaled,
            axis=0,
        )

        raw_prediction = (
            y_scaler.inverse_transform(
                mean_scaled
            )
        )

        prediction = np.maximum(
            raw_prediction,
            0.0,
        )

        seed_predictions_original = np.stack(
            [
                y_scaler.inverse_transform(
                    item
                )
                for item in seed_predictions_scaled
            ],
            axis=0,
        )

        ensemble_std = np.std(
            seed_predictions_original,
            axis=0,
            ddof=0,
        )

        baseline_scaled = np.zeros_like(
            mean_scaled
        )

        baseline_prediction = (
            y_scaler.inverse_transform(
                baseline_scaled
            )
        )

        y_test_scaled = (
            y_scaler.transform(
                y_test
            )
        )

        model_standardized_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        y_test_scaled
                        - mean_scaled
                    )
                    ** 2
                )
            )
        )

        baseline_standardized_rmse = float(
            np.sqrt(
                np.mean(
                    y_test_scaled
                    ** 2
                )
            )
        )

        fold_rows.append(
            {
                "Phase": phase,
                "Outer Fold": fold_number,
                PAIR_COLUMN: test_pair_id,
                PAIR_CLASS_COLUMN: pair_class,
                "Outer Training Rows": int(
                    len(
                        outer_train
                    )
                ),
                "Architecture": (
                    architecture_to_text(
                        hyperparameters.hidden_layers
                    )
                ),
                "Dropout": hyperparameters.dropout,
                "Learning Rate": (
                    hyperparameters.learning_rate
                ),
                "Weight Decay": (
                    hyperparameters.weight_decay
                ),
                "Fixed Epochs": epochs,
                "Ensemble Seeds": (
                    " | ".join(
                        str(seed)
                        for seed in ensemble_seeds
                    )
                ),
                "Hyperparameter Interpretation": (
                    HYPERPARAMETER_INTERPRETATION
                ),
                "Model Standardized RMSE": (
                    model_standardized_rmse
                ),
                "TrainingMean Baseline Standardized RMSE": (
                    baseline_standardized_rmse
                ),
                "Baseline Relative Standardized RMSE": (
                    model_standardized_rmse
                    / baseline_standardized_rmse
                    if baseline_standardized_rmse > 0
                    else np.nan
                ),
                "Elapsed Seconds": (
                    time.monotonic()
                    - fold_started
                ),
            }
        )

        row: dict[str, Any] = {
            "Phase": phase,
            "Outer Fold": fold_number,
            PAIR_COLUMN: test_pair_id,
            PAIR_CLASS_COLUMN: pair_class,
            "Model Scope": scope,
            "Model Source": "PHASE_DL",
            "Architecture": architecture_to_text(
                hyperparameters.hidden_layers
            ),
            "Fixed Epochs": epochs,
            "Selection Method": HYPERPARAMETER_SELECTION_METHOD,
            "Hyperparameter Interpretation": (
                HYPERPARAMETER_INTERPRETATION
            ),
        }

        for target_index, target in enumerate(
            targets
        ):
            raw_value = float(
                raw_prediction[
                    0,
                    target_index,
                ]
            )

            row[
                f"Actual {target}"
            ] = float(
                y_test[
                    0,
                    target_index,
                ]
            )

            row[
                f"Raw Predicted {target}"
            ] = raw_value

            row[
                f"Predicted {target}"
            ] = float(
                prediction[
                    0,
                    target_index,
                ]
            )

            row[
                f"Ensemble SD {target}"
            ] = float(
                ensemble_std[
                    0,
                    target_index,
                ]
            )

            row[
                f"TrainingMeanBaseline {target}"
            ] = float(
                baseline_prediction[
                    0,
                    target_index,
                ]
            )

            row[
                f"Raw Negative Flag {target}"
            ] = bool(
                raw_value
                < 0
            )

        prediction_rows.append(row)

        print(
            f"[{fold_number}/{len(unique_pairs)}]"
            f" {phase}"
            f" | {test_pair_id}"
            f" | {pair_class}"
            f" | {time.monotonic() - fold_started:.2f}s"
            f" | stdRMSE={model_standardized_rmse:.4f}",
            flush=True,
        )

    return (
        pd.DataFrame(
            prediction_rows
        ),
        pd.DataFrame(
            fold_rows
        ),
        pd.DataFrame(
            convergence_rows
        ),
    )



# =============================================================================
# METRICS
# =============================================================================

def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    mask = np.abs(
        y_true
    ) > 1e-8

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    - y_pred[mask]
                )
                / y_true[mask]
            )
        )
        * 100.0
    )


def target_wise_metrics(
    *,
    predictions: pd.DataFrame,
    phase: str,
) -> pd.DataFrame:
    rows = []

    for target in PHASE_TARGETS[
        phase
    ]:
        actual = predictions[
            f"Actual {target}"
        ].to_numpy(
            dtype=float
        )

        predicted = predictions[
            f"Predicted {target}"
        ].to_numpy(
            dtype=float
        )

        raw = predictions[
            f"Raw Predicted {target}"
        ].to_numpy(
            dtype=float
        )

        baseline = predictions[
            f"TrainingMeanBaseline {target}"
        ].to_numpy(
            dtype=float
        )

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
                "Phase": phase,
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
                    if len(
                        np.unique(
                            actual
                        )
                    ) >= 2
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
                "TrainingMean Baseline RMSE": (
                    baseline_rmse
                ),
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

    return pd.DataFrame(
        rows
    )


def phase_summary(
    *,
    phase: str,
    predictions: pd.DataFrame,
    target_metrics: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    relative = (
        target_metrics[
            "Baseline Relative RMSE"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    return pd.DataFrame(
        [
            {
                "Phase": phase,
                "Model": "Phase_DNN_Fixed_Ensemble",
                "OOS Pairs": len(predictions),
                "Inputs": 2,
                "Outputs": 2,
                "Selection Method": HYPERPARAMETER_SELECTION_METHOD,
            "Hyperparameter Interpretation": HYPERPARAMETER_INTERPRETATION,
                "Mean Target-wise Baseline-Relative RMSE": (
                    float(relative.mean())
                    if not relative.empty
                    else np.nan
                ),
                "Targets Better Than TrainingMean Baseline": int(
                    (
                        target_metrics[
                            "Baseline Relative RMSE"
                        ]
                        < 1.0
                    ).sum()
                ),
                "Mean Outer Standardized RMSE": float(
                    folds[
                        "Model Standardized RMSE"
                    ].mean()
                ),
            }
        ]
    )


# =============================================================================
# FINAL MODEL PER PHASE
# =============================================================================

def save_final_phase_ensemble(
    *,
    dataframe: pd.DataFrame,
    scope: str,
    phase: str,
    model_dir: Path,
    ensemble_seeds: tuple[int, ...],
    device: torch.device,
    hyperparameters: DNNHyperparameters,
    epochs: int,
    gradient_clip_norm: float,
) -> tuple[Path, pd.DataFrame]:
    features = PHASE_FEATURES[
        phase
    ]

    targets = PHASE_TARGETS[
        phase
    ]

    X = (
        dataframe[
            features
        ]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        dataframe[
            targets
        ]
        .to_numpy(
            dtype=float
        )
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = (
        x_scaler.fit_transform(
            X
        )
    )

    y_scaled = (
        y_scaler.fit_transform(
            y
        )
    )

    selected_dir = (
        model_dir
        / phase
        / "Selected_Final"
    )

    selected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    convergence_rows: list[dict[str, Any]] = []

    for seed in ensemble_seeds:
        model, convergence = train_fixed_epochs(
            X_train_scaled=X_scaled,
            y_train_scaled=y_scaled,
            hyperparameters=hyperparameters,
            device=device,
            seed=seed,
            epochs=epochs,
            gradient_clip_norm=gradient_clip_norm,
        )

        convergence_rows.append(
            {
                "Dataset Scope": scope,
                "Phase": phase,
                "Training Role": "FINAL_DEPLOYMENT_FIT",
                "Outer Fold": np.nan,
                "Held-Out Pair ID": "",
                "Held-Out Pair Class": "",
                **convergence,
            }
        )

        checkpoint = {
            "script_version": SCRIPT_VERSION,
            "model_source": "PHASE_DL",
            "dataset_scope": scope,
            "phase": phase,
            "seed": int(seed),
            "input_features": features,
            "target_features": targets,
            "hidden_layers": list(
                hyperparameters.hidden_layers
            ),
            "dropout": float(
                hyperparameters.dropout
            ),
            "learning_rate": float(
                hyperparameters.learning_rate
            ),
            "weight_decay": float(
                hyperparameters.weight_decay
            ),
            "epochs": int(epochs),
            "selection_method": HYPERPARAMETER_SELECTION_METHOD,
            "hyperparameter_interpretation": (
                HYPERPARAMETER_INTERPRETATION
            ),
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
        }

        torch.save(
            checkpoint,
            selected_dir
            / f"{phase}_DNN_seed_{seed}.pt",
        )

    joblib.dump(
        x_scaler,
        selected_dir
        / "x_scaler.joblib",
    )

    joblib.dump(
        y_scaler,
        selected_dir
        / "y_scaler.joblib",
    )

    (
        selected_dir
        / "hyperparameters.json"
    ).write_text(
        json.dumps(
            {
                "phase": phase,
                "selection_method": (
                    HYPERPARAMETER_SELECTION_METHOD
                ),
                "hyperparameter_interpretation": (
                    HYPERPARAMETER_INTERPRETATION
                ),
                "hyperparameter_rationale": (
                    HYPERPARAMETER_RATIONALE
                ),
                "predeclared_meaning": (
                    PREDECLARED_MEANING
                ),
                "architecture": (
                    architecture_to_text(
                        hyperparameters.hidden_layers
                    )
                ),
                "hidden_layers": list(
                    hyperparameters.hidden_layers
                ),
                "dropout": float(
                    hyperparameters.dropout
                ),
                "learning_rate": float(
                    hyperparameters.learning_rate
                ),
                "weight_decay": float(
                    hyperparameters.weight_decay
                ),
                "fixed_epochs": int(
                    epochs
                ),
                "ensemble_seeds": list(
                    ensemble_seeds
                ),
                "hpo_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        selected_dir
        / "feature_schema.json"
    ).write_text(
        json.dumps(
            {
                "phase": phase,
                "input_features": features,
                "input_count": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        selected_dir
        / "target_schema.json"
    ).write_text(
        json.dumps(
            {
                "phase": phase,
                "target_features": targets,
                "target_count": 2,
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
# COMBINE SIX PHASE OOS PREDICTIONS
# =============================================================================

def combine_phase_predictions(
    phase_prediction_frames: dict[
        str,
        pd.DataFrame,
    ],
    *,
    scope: str,
) -> pd.DataFrame:
    combined = None

    identity = [
        PAIR_COLUMN,
        PAIR_CLASS_COLUMN,
        "Model Scope",
        "Model Source",
    ]

    for phase in PHASES:
        frame = phase_prediction_frames[
            phase
        ].copy()

        keep = (
            identity
            + [
                column
                for column in frame.columns
                if (
                    column.startswith("Actual EV ")
                    or column.startswith("Raw Predicted EV ")
                    or column.startswith("Predicted EV ")
                    or column.startswith("Ensemble SD EV ")
                    or column.startswith("TrainingMeanBaseline EV ")
                    or column.startswith("Raw Negative Flag EV ")
                )
            ]
        )

        frame = frame[
            [
                column
                for column in keep
                if column in frame.columns
            ]
        ]

        if combined is None:
            combined = frame
        else:
            combined = combined.merge(
                frame,
                on=identity,
                how="inner",
                validate="one_to_one",
            )

    if combined is None:
        raise RuntimeError(
            "No phase prediction frames were available."
        )

    combined[
        "Model Scope"
    ] = scope

    combined[
        "Model Source"
    ] = "PHASE_DL"

    return combined


# =============================================================================
# RUN ONE SCOPE
# =============================================================================

def run_phase_wise_dl_scope(
    *,
    scope: str,
    ensemble_seeds: tuple[int, ...],
    device: torch.device,
    hyperparameters: DNNHyperparameters,
    epochs: int,
    gradient_clip_norm: float,
) -> None:
    scope = normalize_scope(scope)

    results_dir, model_dir = get_output_directories(
        scope
    )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "TAT MS INTERCITY — PHASE-WISE DL FIXED PREDECLARED LOPO"
    )

    print(
        "=" * 92
    )

    print(f"Version       : {SCRIPT_VERSION}")
    print(f"Dataset Scope : {scope}")
    print(f"Device        : {device}")
    print(
        "Architecture  : "
        f"2 -> {architecture_to_text(hyperparameters.hidden_layers)} -> 2"
    )
    print(f"Dropout       : {hyperparameters.dropout}")
    print(f"LR            : {hyperparameters.learning_rate}")
    print(f"Weight decay  : {hyperparameters.weight_decay}")
    print(f"Epochs        : {epochs}")
    print(f"Seeds         : {ensemble_seeds}")
    print("HPO           : DISABLED")
    print(f"HP status     : {HYPERPARAMETER_INTERPRETATION}")
    print("Convergence   : training-loss diagnostic saved; not model selection")
    print("Pair Class    : READ / VALIDATE / PRESERVE ONLY")

    (
        dataframe,
        input_files,
        validation_report,
    ) = load_dataset_scope(
        scope
    )

    print(f"Rows          : {len(dataframe)}")
    print(
        f"STRICT rows   : {validation_report['strict_rows']}"
    )
    print(
        f"NONSTRICT rows: {validation_report['nonstrict_rows']}"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_phase_summaries = []
    all_target_metrics = []
    all_outer_folds = []
    all_convergence = []
    phase_prediction_frames = {}

    final_model_dirs = {}

    for phase in PHASES:
        predictions, folds, convergence = run_lopo_phase(
            dataframe=dataframe,
            scope=scope,
            phase=phase,
            ensemble_seeds=ensemble_seeds,
            device=device,
            hyperparameters=hyperparameters,
            epochs=epochs,
            gradient_clip_norm=gradient_clip_norm,
        )

        metrics = target_wise_metrics(
            predictions=predictions,
            phase=phase,
        )

        summary = phase_summary(
            phase=phase,
            predictions=predictions,
            target_metrics=metrics,
            folds=folds,
        )

        selected_dir, final_convergence = save_final_phase_ensemble(
            dataframe=dataframe,
            scope=scope,
            phase=phase,
            model_dir=model_dir,
            ensemble_seeds=ensemble_seeds,
            device=device,
            hyperparameters=hyperparameters,
            epochs=epochs,
            gradient_clip_norm=gradient_clip_norm,
        )

        phase_prediction_frames[
            phase
        ] = predictions

        all_phase_summaries.append(
            summary
        )

        all_target_metrics.append(
            metrics
        )

        all_outer_folds.append(
            folds
        )

        all_convergence.append(
            convergence
        )

        all_convergence.append(
            final_convergence
        )

        final_model_dirs[
            phase
        ] = str(
            selected_dir
        )

        phase_results_dir = (
            results_dir
            / phase
        )

        phase_results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        predictions.to_excel(
            phase_results_dir
            / f"{phase}_DL_LOPO_OOS_Predictions.xlsx",
            index=False,
        )

        metrics.to_excel(
            phase_results_dir
            / f"{phase}_DL_Target_Wise_OOS_Metrics.xlsx",
            index=False,
        )

        folds.to_excel(
            phase_results_dir
            / f"{phase}_DL_Outer_Folds.xlsx",
            index=False,
        )

        summary.to_excel(
            phase_results_dir
            / f"{phase}_DL_OOS_Summary.xlsx",
            index=False,
        )

    summaries = pd.concat(
        all_phase_summaries,
        ignore_index=True,
    )

    target_metrics = pd.concat(
        all_target_metrics,
        ignore_index=True,
    )

    outer_folds = pd.concat(
        all_outer_folds,
        ignore_index=True,
    )

    convergence = pd.concat(
        all_convergence,
        ignore_index=True,
        sort=False,
    )

    combined_predictions = combine_phase_predictions(
        phase_prediction_frames,
        scope=scope,
    )

    manifest_rows = [
        ("Script Version", SCRIPT_VERSION),
        ("Dataset Scope", scope),
        (
            "Input Files",
            " | ".join(
                str(path)
                for path in input_files
            ),
        ),
        (
            "Input SHA256",
            " | ".join(
                f"{path.name}:{sha256_file(path)}"
                for path in input_files
            ),
        ),
        ("Rows", len(dataframe)),
        (
            "Unique Pairs",
            dataframe[PAIR_COLUMN].nunique(),
        ),
        (
            "STRICT Rows",
            validation_report["strict_rows"],
        ),
        (
            "NONSTRICT Rows",
            validation_report["nonstrict_rows"],
        ),
        (
            "Pair Class Handling",
            "READ / VALIDATE / PRESERVE ONLY",
        ),
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
        (
            "Hyperparameter Rationale",
            HYPERPARAMETER_RATIONALE,
        ),
        (
            "Predeclared Meaning",
            PREDECLARED_MEANING,
        ),
        (
            "Architecture",
            architecture_to_text(
                hyperparameters.hidden_layers
            ),
        ),
        ("Dropout", hyperparameters.dropout),
        (
            "Learning Rate",
            hyperparameters.learning_rate,
        ),
        (
            "Weight Decay",
            hyperparameters.weight_decay,
        ),
        ("Fixed Epochs", epochs),
        (
            "Ensemble Seeds",
            " | ".join(
                str(seed)
                for seed in ensemble_seeds
            ),
        ),
        (
            "Scaler",
            "StandardScaler fit on each outer training fold only",
        ),
        ("Optimizer", "AdamW"),
        ("Loss", "MSELoss"),
        (
            "Convergence Evidence",
            "Per-seed training-loss diagnostics saved separately",
        ),
        (
            "Convergence Interpretation",
            CONVERGENCE_INTERPRETATION,
        ),
        (
            "Physical Constraint",
            "max(raw_prediction, 0), identical in OOS and inference",
        ),
        ("Charging", "NOT calculated in training code"),
        (
            "Final Assembly",
            "NOT calculated in training code",
        ),
        ("Python", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("NumPy", np.__version__),
        ("Pandas", pd.__version__),
        ("PyTorch", torch.__version__),
        ("Device", str(device)),
    ]

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "Parameter",
            "Value",
        ],
    )

    summaries.to_excel(
        results_dir
        / "Phase_Wise_DL_OOS_Summary.xlsx",
        index=False,
    )

    target_metrics.to_excel(
        results_dir
        / "Phase_Wise_DL_Target_Wise_OOS_Metrics.xlsx",
        index=False,
    )

    outer_folds.to_excel(
        results_dir
        / "Phase_Wise_DL_Outer_Folds.xlsx",
        index=False,
    )

    convergence.to_excel(
        results_dir
        / "Phase_Wise_DL_Training_Convergence.xlsx",
        index=False,
    )

    combined_predictions.to_excel(
        results_dir
        / "Phase_Wise_DL_All_Phases_OOS_Predictions.xlsx",
        index=False,
    )

    manifest.to_excel(
        results_dir
        / "Phase_Wise_DL_Run_Manifest.xlsx",
        index=False,
    )

    with pd.ExcelWriter(
        results_dir
        / "Phase_Wise_DL_Detailed_Evaluation.xlsx",
        engine="openpyxl",
    ) as writer:
        summaries.to_excel(
            writer,
            sheet_name="01_Phase_Summary",
            index=False,
        )

        target_metrics.to_excel(
            writer,
            sheet_name="02_Target_Metrics",
            index=False,
        )

        outer_folds.to_excel(
            writer,
            sheet_name="03_Outer_Folds",
            index=False,
        )

        convergence.to_excel(
            writer,
            sheet_name="04_Convergence",
            index=False,
        )

        combined_predictions.to_excel(
            writer,
            sheet_name="05_All_Phase_Pred",
            index=False,
        )

        manifest.to_excel(
            writer,
            sheet_name="06_Manifest",
            index=False,
        )

    (
        model_dir
        / "training_manifest.json"
    ).write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "dataset_scope": scope,
                "input_files": [
                    str(path)
                    for path in input_files
                ],
                "input_sha256": {
                    path.name: sha256_file(path)
                    for path in input_files
                },
                "rows": int(
                    len(
                        dataframe
                    )
                ),
                "unique_pairs": int(
                    dataframe[
                        PAIR_COLUMN
                    ].nunique()
                ),
                "strict_rows": int(
                    validation_report[
                        "strict_rows"
                    ]
                ),
                "nonstrict_rows": int(
                    validation_report[
                        "nonstrict_rows"
                    ]
                ),
                "pair_class_handling": (
                    "READ / VALIDATE / PRESERVE ONLY"
                ),
                "pair_class_recalculation": "NONE",
                "pair_class_overwrite": "NONE",
                "pair_class_used_as_predictor": False,
                "model_source": "PHASE_DL",
                "phases": PHASES,
                "selection_method": (
                    HYPERPARAMETER_SELECTION_METHOD
                ),
                "hyperparameter_interpretation": (
                    HYPERPARAMETER_INTERPRETATION
                ),
                "hyperparameter_rationale": (
                    HYPERPARAMETER_RATIONALE
                ),
                "predeclared_meaning": (
                    PREDECLARED_MEANING
                ),
                "hpo_used": False,
                "architecture": (
                    architecture_to_text(
                        hyperparameters.hidden_layers
                    )
                ),
                "dropout": hyperparameters.dropout,
                "learning_rate": (
                    hyperparameters.learning_rate
                ),
                "weight_decay": (
                    hyperparameters.weight_decay
                ),
                "fixed_epochs": epochs,
                "ensemble_seeds": list(
                    ensemble_seeds
                ),
                "convergence_diagnostic": (
                    "Phase_Wise_DL_Training_Convergence.xlsx; "
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
                "selected_final_directories": (
                    final_model_dirs
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "PHASE-WISE DL SCOPE COMPLETED"
    )

    print(
        "=" * 92
    )

    print(f"Scope    : {scope}")
    print(f"Pairs    : {len(combined_predictions)}")
    print(f"Phases   : {len(PHASES)}")
    print(f"Results  : {results_dir}")
    print(f"Models   : {model_dir}")
    print("Nested HPO: REMOVED")
    print(f"Hyperparameters: {HYPERPARAMETER_INTERPRETATION}")
    print("Convergence: Phase_Wise_DL_Training_Convergence.xlsx")
    print(
        "Pair Class: NOT recalculated, overwritten, "
        "or used as a DNN predictor."
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-wise DNN fixed predeclared LOPO pipeline. "
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
            for seed in DEFAULT_ENSEMBLE_SEEDS
        ),
    )

    parser.add_argument(
        "--architecture",
        default=architecture_to_text(
            DEFAULT_HIDDEN_LAYERS
        ),
        help=(
            "Predeclared hidden-layer widths, e.g. 4-2."
        ),
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
            "Fixed training epochs. --max-epochs remains "
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
    args = build_parser().parse_args()

    if args.epochs < 1:
        raise ValueError(
            "--epochs must be >= 1."
        )

    if not (
        0
        <= args.dropout
        < 1
    ):
        raise ValueError(
            "--dropout must be in [0, 1)."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay must be >= 0."
        )

    if args.gradient_clip_norm <= 0:
        raise ValueError(
            "--gradient-clip-norm must be > 0."
        )

    requested_scope = normalize_scope(
        args.scope,
        allow_all=True,
    )

    scopes = (
        [
            "STRICT",
            "NONSTRICT",
            "COMBINED",
        ]
        if requested_scope
        == "ALL"
        else [
            requested_scope
        ]
    )

    ensemble_seeds = parse_seed_list(
        args.ensemble_seeds
    )

    device = resolve_device(
        args.device
    )

    hyperparameters = DNNHyperparameters(
        hidden_layers=parse_architecture(
            args.architecture
        ),
        dropout=float(
            args.dropout
        ),
        learning_rate=float(
            args.learning_rate
        ),
        weight_decay=float(
            args.weight_decay
        ),
    )

    print(
        "=" * 92
    )

    print(
        "PHASE-WISE DL — FIXED PREDECLARED LOPO"
    )

    print(
        "=" * 92
    )

    print(f"Version : {SCRIPT_VERSION}")
    print(f"Scopes  : {scopes}")
    print(f"Device  : {device}")
    print(f"Seeds   : {ensemble_seeds}")
    print("HPO     : DISABLED")

    for scope in scopes:
        run_phase_wise_dl_scope(
            scope=scope,
            ensemble_seeds=ensemble_seeds,
            device=device,
            hyperparameters=hyperparameters,
            epochs=args.epochs,
            gradient_clip_norm=args.gradient_clip_norm,
        )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "ALL REQUESTED PHASE-WISE DL RUNS COMPLETED"
    )

    print(
        "=" * 92
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
