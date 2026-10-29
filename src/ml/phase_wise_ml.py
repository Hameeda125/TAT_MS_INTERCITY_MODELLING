"""
TAT MS INTERCITY
PHASE-WISE ML MODELLING

Scientific design
-----------------
Six non-charging phases are modelled independently:

    STOP
    IDLE
    INCHING
    ACC
    DEC
    CRUISE

For every phase:

    ICE phase time + ICE phase distance
                    ↓
              ML model
                    ↓
    EV phase time + EV phase distance

CHARGING_REFUELING is intentionally excluded from this phase-wise
module and remains handled by the separate charging/refuelling
prediction workflow.

Data scopes
-----------
STRICT      : authoritative STRICT training workbook
NONSTRICT   : authoritative NONSTRICT training workbook
COMBINED    : STRICT + NONSTRICT concatenated in memory only, after
              independent validation and Pair-ID overlap checks

Pair Class is READ / VALIDATED / PRESERVED ONLY.
It is never recalculated, overwritten, or used as a model input.

Validation
----------
Leave-One-Pair-Out Cross-Validation (LOPO-CV).

Model selection
---------------
The two phase targets have different physical units (minutes and km),
so raw pooled MAE/RMSE are diagnostic only and are NOT used for
selection.

For each phase and model:
    target-wise LOPO RMSE
        ÷
    target-wise LOPO training-mean-baseline RMSE

is calculated separately for EV phase time and EV phase distance.

The primary phase selection score is the mean of those two
dimensionless baseline-relative RMSE values.  Mean baseline-relative
MAE is the secondary criterion.

No held-out prediction clipping is performed. Negative predictions,
when present, are reported diagnostically rather than silently altered.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================================
# VERSION / REPRODUCIBILITY
# ============================================================================

SCRIPT_VERSION = "2026-09-29_PHASE_WISE_ML_V4_STRICT_NONSTRICT_COMBINED"
RANDOM_STATE = 42


# ============================================================================
# PROJECT PATHS
# ============================================================================

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


# ============================================================================
# DATA SCOPES
# ============================================================================

VALID_SINGLE_CLASSES = ("STRICT", "NONSTRICT")
VALID_SCOPES = ("STRICT", "NONSTRICT", "COMBINED")


def normalize_scope(scope: str) -> str:
    normalized = str(scope).strip().upper()

    if normalized not in VALID_SCOPES:
        raise ValueError(
            "Scope must be STRICT, NONSTRICT, or COMBINED. "
            f"Received: {scope!r}"
        )

    return normalized


def get_training_file(pair_class: str) -> Path:
    pair_class = str(pair_class).strip().upper()

    if pair_class not in VALID_SINGLE_CLASSES:
        raise ValueError(
            "Training workbook class must be STRICT or NONSTRICT."
        )

    path = (
        TRAINING_DATA_DIR
        / f"EV_ICE_Training_{pair_class}.xlsx"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Training workbook was not found:\n"
            f"{path}"
        )

    if not path.is_file():
        raise ValueError(
            "Training workbook path is not a file:\n"
            f"{path}"
        )

    return path


def get_output_directories(
    scope: str,
) -> tuple[Path, Path]:
    """
    Class-first output layout.

    Results/STRICT/ML/Phase_Wise
    Results/NONSTRICT/ML/Phase_Wise
    Results/COMBINED/ML/Phase_Wise

    and the same hierarchy under models/.
    """

    scope = normalize_scope(scope)

    results_dir = (
        RESULTS_ROOT
        / scope
        / "ML"
        / "Phase_Wise"
    )

    model_dir = (
        MODEL_ROOT
        / scope
        / "ML"
        / "Phase_Wise"
    )

    return results_dir, model_dir


# ============================================================================
# PHASE DEFINITIONS
# ============================================================================

PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
]

PAIR_COLUMN = "Pair ID"
PAIR_CLASS_COLUMN = "Pair Class"


def phase_column_prefix(phase: str) -> str:
    """
    Map canonical modelling phase to the training-workbook name.
    """

    mapping = {
        "STOP": "STOP",
        "IDLE": "IDLE",
        "INCHING": "INCHING",
        "ACC": "ACC",
        "DEC": "DEC",
        "CRUISE": "CRUISE",
    }

    try:
        return mapping[phase]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported phase: {phase}"
        ) from exc


def expected_columns_for_phase(
    phase: str,
) -> tuple[list[str], list[str]]:

    prefix = phase_column_prefix(
        phase
    )

    x_columns = [
        f"ICE {prefix} Time (min)",
        f"ICE {prefix} Distance (km)",
    ]

    y_columns = [
        f"EV {prefix} Time (min)",
        f"EV {prefix} Distance (km)",
    ]

    return x_columns, y_columns


def all_phase_modelling_columns() -> list[str]:
    columns: list[str] = []

    for phase in PHASES:
        x_columns, y_columns = (
            expected_columns_for_phase(
                phase
            )
        )

        columns.extend(
            x_columns
        )
        columns.extend(
            y_columns
        )

    return columns


# ============================================================================
# MODEL REGISTRY
# ============================================================================

def build_models() -> dict[str, object]:
    """
    Candidate family aligned with the existing phase-wise design.

    Scaling stays inside each estimator/pipeline so preprocessing is
    fitted using training-fold data only.
    """

    return {
        "LinearRegression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),

        "Ridge": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0
                    ),
                ),
            ]
        ),

        "ElasticNet": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    MultiOutputRegressor(
                        ElasticNet(
                            alpha=0.01,
                            l1_ratio=0.5,
                            max_iter=10000,
                            random_state=RANDOM_STATE,
                        )
                    ),
                ),
            ]
        ),

        # Preserve the original conservative one-component PLS design.
        "PLS": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    PLSRegression(
                        n_components=1,
                        scale=False,
                        max_iter=500,
                    ),
                ),
            ]
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_features=1.0,
            min_samples_leaf=1,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_features=1.0,
            min_samples_leaf=1,
            n_jobs=-1,
        ),

        "SVR": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    MultiOutputRegressor(
                        SVR(
                            C=10.0,
                            epsilon=0.1,
                            kernel="rbf",
                        )
                    ),
                ),
            ]
        ),
    }


# ============================================================================
# DATA LOADING / CLASS VALIDATION
# ============================================================================

def load_single_workbook(
    pair_class: str,
) -> tuple[pd.DataFrame, Path]:

    pair_class = str(
        pair_class
    ).strip().upper()

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

    return dataframe, path


def validate_single_class_data(
    dataframe: pd.DataFrame,
    expected_pair_class: str,
) -> dict:
    """
    Validate one authoritative upstream class.

    No row/column/value is modified.
    Pair Class is never derived here.
    """

    expected_pair_class = str(
        expected_pair_class
    ).strip().upper()

    if expected_pair_class not in VALID_SINGLE_CLASSES:
        raise ValueError(
            "Expected class must be STRICT or NONSTRICT."
        )

    required_columns = (
        [
            PAIR_COLUMN,
            PAIR_CLASS_COLUMN,
        ]
        + all_phase_modelling_columns()
    )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Missing required phase-wise modelling columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # Pair ID.
    pair_ids = dataframe[
        PAIR_COLUMN
    ]

    blank_pair_ids = (
        pair_ids.isna()
        | pair_ids.astype(
            "string"
        ).str.strip().eq("")
    )

    if blank_pair_ids.any():
        raise ValueError(
            "Pair ID contains missing/blank values."
        )

    if pair_ids.duplicated().any():
        duplicates = (
            dataframe.loc[
                pair_ids.duplicated(
                    keep=False
                ),
                PAIR_COLUMN,
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate Pair IDs detected:\n"
            + "\n".join(
                f"  - {value}"
                for value in duplicates
            )
        )

    if len(dataframe) < 3:
        raise ValueError(
            "At least three paired observations are required "
            "for LOPO model evaluation."
        )

    # Pair Class.
    observed_class = (
        dataframe[
            PAIR_CLASS_COLUMN
        ]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    missing_class = (
        observed_class.isna()
        | observed_class.eq("")
    )

    if missing_class.any():
        raise ValueError(
            "Pair Class contains missing/blank values."
        )

    invalid_class = sorted(
        set(
            observed_class.dropna()
        )
        - set(
            VALID_SINGLE_CLASSES
        )
    )

    if invalid_class:
        raise ValueError(
            "Invalid Pair Class values found:\n"
            + "\n".join(
                f"  - {value}"
                for value in invalid_class
            )
        )

    mismatch = (
        observed_class
        != expected_pair_class
    )

    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} row(s) do not match "
            f"expected Pair Class {expected_pair_class}."
        )

    # Numeric, finite, non-negative.
    modelling_columns = (
        all_phase_modelling_columns()
    )

    constant_columns: list[str] = []

    for column in modelling_columns:

        original = dataframe[
            column
        ]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        nonblank_original = (
            original.notna()
            & original.astype(
                "string"
            ).str.strip().ne("")
        )

        conversion_failure = (
            nonblank_original
            & converted.isna()
        )

        if conversion_failure.any():
            raise ValueError(
                "Non-numeric value(s) found in modelling column:\n"
                f"{column}"
            )

        if converted.isna().any():
            raise ValueError(
                "Missing value(s) found in modelling column:\n"
                f"{column}"
            )

        values = converted.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Infinite/non-finite value(s) found in modelling column:\n"
                f"{column}"
            )

        if (
            values
            < 0
        ).any():
            raise ValueError(
                "Negative physical value(s) found in modelling column:\n"
                f"{column}"
            )

        if converted.nunique(
            dropna=True
        ) <= 1:
            constant_columns.append(
                column
            )

    return {
        "pair_class": expected_pair_class,
        "rows": int(
            len(
                dataframe
            )
        ),
        "columns": int(
            len(
                dataframe.columns
            )
        ),
        "unique_pairs": int(
            pair_ids.nunique()
        ),
        "constant_modelling_columns": (
            constant_columns
        ),
    }


def load_scope_data(
    scope: str,
) -> tuple[
    pd.DataFrame,
    list[Path],
    dict,
]:
    """
    Load STRICT, NONSTRICT, or COMBINED.

    COMBINED is constructed in memory only after:
      1) independent class validation
      2) exact schema comparison
      3) cross-class Pair ID overlap check
    """

    scope = normalize_scope(
        scope
    )

    if scope in VALID_SINGLE_CLASSES:

        dataframe, path = (
            load_single_workbook(
                scope
            )
        )

        validation = (
            validate_single_class_data(
                dataframe,
                scope,
            )
        )

        return (
            dataframe,
            [path],
            {
                scope: validation,
            },
        )

    strict_df, strict_path = (
        load_single_workbook(
            "STRICT"
        )
    )

    nonstrict_df, nonstrict_path = (
        load_single_workbook(
            "NONSTRICT"
        )
    )

    strict_validation = (
        validate_single_class_data(
            strict_df,
            "STRICT",
        )
    )

    nonstrict_validation = (
        validate_single_class_data(
            nonstrict_df,
            "NONSTRICT",
        )
    )

    # Exact schema guard.
    if list(
        strict_df.columns
    ) != list(
        nonstrict_df.columns
    ):
        raise ValueError(
            "STRICT and NONSTRICT workbooks do not have the "
            "same ordered column schema. COMBINED modelling "
            "has been stopped."
        )

    strict_ids = set(
        strict_df[
            PAIR_COLUMN
        ]
        .astype("string")
        .str.strip()
        .tolist()
    )

    nonstrict_ids = set(
        nonstrict_df[
            PAIR_COLUMN
        ]
        .astype("string")
        .str.strip()
        .tolist()
    )

    overlap = sorted(
        strict_ids
        & nonstrict_ids
    )

    if overlap:
        raise ValueError(
            "Pair ID overlap was found between STRICT and "
            "NONSTRICT datasets. COMBINED modelling has been "
            "stopped:\n"
            + "\n".join(
                f"  - {pair_id}"
                for pair_id in overlap
            )
        )

    combined = pd.concat(
        [
            strict_df,
            nonstrict_df,
        ],
        ignore_index=True,
        sort=False,
    )

    if combined[
        PAIR_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Duplicate Pair IDs exist after COMBINED concatenation."
        )

    observed_counts = (
        combined[
            PAIR_CLASS_COLUMN
        ]
        .astype("string")
        .str.strip()
        .str.upper()
        .value_counts()
        .to_dict()
    )

    expected_counts = {
        "STRICT": int(
            len(
                strict_df
            )
        ),
        "NONSTRICT": int(
            len(
                nonstrict_df
            )
        ),
    }

    if observed_counts != expected_counts:
        raise ValueError(
            "Pair Class preservation check failed after "
            "COMBINED concatenation."
        )

    return (
        combined,
        [
            strict_path,
            nonstrict_path,
        ],
        {
            "STRICT": strict_validation,
            "NONSTRICT": (
                nonstrict_validation
            ),
            "COMBINED": {
                "rows": int(
                    len(
                        combined
                    )
                ),
                "unique_pairs": int(
                    combined[
                        PAIR_COLUMN
                    ].nunique()
                ),
                "observed_pair_class_counts": (
                    observed_counts
                ),
                "cross_class_pair_id_overlap": 0,
            },
        },
    )


# ============================================================================
# METRICS
# ============================================================================

def safe_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(
            actual
        )
        & np.isfinite(
            predicted
        )
        & (
            actual
            != 0
        )
    )

    if not np.any(
        mask
    ):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    actual[
                        mask
                    ]
                    - predicted[
                        mask
                    ]
                )
                / actual[
                    mask
                ]
            )
        )
        * 100.0
    )


def safe_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(
            actual
        )
        & np.isfinite(
            predicted
        )
    )

    actual = actual[
        mask
    ]

    predicted = predicted[
        mask
    ]

    if len(
        actual
    ) < 2:
        return np.nan

    if np.isclose(
        np.var(
            actual
        ),
        0.0,
    ):
        return np.nan

    return float(
        r2_score(
            actual,
            predicted,
        )
    )


def vector_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(
            actual
        )
        & np.isfinite(
            predicted
        )
    )

    actual = actual[
        mask
    ]

    predicted = predicted[
        mask
    ]

    if len(
        actual
    ) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE (%)": np.nan,
        }

    return {
        "MAE": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "R2": safe_r2(
            actual,
            predicted,
        ),
        "MAPE (%)": safe_mape(
            actual,
            predicted,
        ),
    }


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if (
        not np.isfinite(
            numerator
        )
        or not np.isfinite(
            denominator
        )
        or denominator <= 0
    ):
        return np.nan

    return float(
        numerator
        / denominator
    )


# ============================================================================
# WARNING CAPTURE
# ============================================================================

def warning_records_to_text(
    warning_records: list,
) -> str:

    messages: list[str] = []

    for record in warning_records:

        category = (
            record.category.__name__
            if hasattr(
                record,
                "category",
            )
            else "Warning"
        )

        message = (
            f"{category}: "
            f"{record.message}"
        )

        if message not in messages:
            messages.append(
                message
            )

    return " | ".join(
        messages
    )


# ============================================================================
# PHASE-WISE LOPO-CV
# ============================================================================

def evaluate_phase(
    dataframe: pd.DataFrame,
    phase: str,
    scope: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate every candidate model for one phase with LOPO-CV.

    Returns
    -------
    phase_summary
        One row per model, with dimensionless baseline-relative
        selection metrics and mixed-unit diagnostic metrics.

    target_metrics
        One row per model/target.

    held_out_predictions
        Long-format actual/predicted rows including Pair Class.

    fold_status
        Success/failure/warning information for every model/fold.
    """

    scope = normalize_scope(
        scope
    )

    x_columns, y_columns = (
        expected_columns_for_phase(
            phase
        )
    )

    x = (
        dataframe[
            x_columns
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .copy()
    )

    y = (
        dataframe[
            y_columns
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .copy()
    )

    pair_ids = (
        dataframe[
            PAIR_COLUMN
        ]
        .astype(str)
        .to_numpy()
    )

    pair_classes = (
        dataframe[
            PAIR_CLASS_COLUMN
        ]
        .astype("string")
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    unique_pairs = list(
        dict.fromkeys(
            pair_ids
        )
    )

    expected_pair_count = len(
        unique_pairs
    )

    all_predictions: list[
        dict
    ] = []

    fold_status: list[
        dict
    ] = []

    candidate_names = list(
        build_models().keys()
    )

    for (
        fold_number,
        test_pair,
    ) in enumerate(
        unique_pairs,
        start=1,
    ):

        test_mask = (
            pair_ids
            == test_pair
        )

        train_mask = (
            ~test_mask
        )

        x_train = x.loc[
            train_mask
        ]

        y_train = y.loc[
            train_mask
        ]

        x_test = x.loc[
            test_mask
        ]

        y_test = y.loc[
            test_mask
        ]

        test_classes = (
            pair_classes[
                test_mask
            ]
        )

        if len(
            x_test
        ) != 1:
            raise ValueError(
                "LOPO expected exactly one row for "
                f"Pair ID {test_pair}; found {len(x_test)}."
            )

        test_class = str(
            test_classes[
                0
            ]
        )

        print(
            f"\n  Fold "
            f"{fold_number}/"
            f"{expected_pair_count}"
            f" | Test Pair: "
            f"{test_pair}"
            f" | Class: "
            f"{test_class}"
        )

        # ------------------------------------------------------------------
        # Training-mean baseline.
        # ------------------------------------------------------------------

        baseline_prediction = (
            y_train.mean(
                axis=0
            )
            .to_numpy(
                dtype=float
            )
        )

        for (
            target_index,
            target_name,
        ) in enumerate(
            y_columns
        ):

            all_predictions.append(
                {
                    "Scope": scope,
                    "Phase": phase,
                    "Fold": (
                        fold_number
                    ),
                    "Pair ID": (
                        test_pair
                    ),
                    "Pair Class": (
                        test_class
                    ),
                    "Model": (
                        "TrainingMeanBaseline"
                    ),
                    "Target": (
                        target_name
                    ),
                    "Actual": float(
                        y_test.iloc[
                            0,
                            target_index,
                        ]
                    ),
                    "Predicted": float(
                        baseline_prediction[
                            target_index
                        ]
                    ),
                }
            )

        # ------------------------------------------------------------------
        # Candidate models.
        # ------------------------------------------------------------------

        for model_name in (
            candidate_names
        ):

            model = (
                build_models()[
                    model_name
                ]
            )

            status_record = {
                "Scope": scope,
                "Phase": phase,
                "Fold": fold_number,
                "Test Pair": test_pair,
                "Pair Class": (
                    test_class
                ),
                "Model": (
                    model_name
                ),
                "Status": "FAILED",
                "Warnings Count": 0,
                "Warnings": "",
                "Error": "",
            }

            try:

                with warnings.catch_warnings(
                    record=True
                ) as caught:

                    warnings.simplefilter(
                        "always"
                    )

                    model.fit(
                        x_train,
                        y_train,
                    )

                    prediction = (
                        np.asarray(
                            model.predict(
                                x_test
                            ),
                            dtype=float,
                        )
                    )

                if prediction.ndim == 1:
                    prediction = (
                        prediction.reshape(
                            1,
                            -1,
                        )
                    )

                if (
                    prediction.shape
                    != y_test.shape
                ):
                    raise ValueError(
                        "Unexpected prediction shape "
                        f"{prediction.shape}; expected "
                        f"{y_test.shape}."
                    )

                if not np.isfinite(
                    prediction
                ).all():
                    raise ValueError(
                        "Model produced non-finite held-out predictions."
                    )

                warning_text = (
                    warning_records_to_text(
                        caught
                    )
                )

                status_record[
                    "Warnings Count"
                ] = len(
                    caught
                )

                status_record[
                    "Warnings"
                ] = warning_text

                status_record[
                    "Status"
                ] = (
                    "SUCCESS_WITH_WARNINGS"
                    if caught
                    else "SUCCESS"
                )

                for (
                    target_index,
                    target_name,
                ) in enumerate(
                    y_columns
                ):

                    all_predictions.append(
                        {
                            "Scope": scope,
                            "Phase": phase,
                            "Fold": (
                                fold_number
                            ),
                            "Pair ID": (
                                test_pair
                            ),
                            "Pair Class": (
                                test_class
                            ),
                            "Model": (
                                model_name
                            ),
                            "Target": (
                                target_name
                            ),
                            "Actual": float(
                                y_test.iloc[
                                    0,
                                    target_index,
                                ]
                            ),
                            "Predicted": float(
                                prediction[
                                    0,
                                    target_index,
                                ]
                            ),
                        }
                    )

                print(
                    f"    "
                    f"{model_name:<18} "
                    f"{status_record['Status']}"
                )

                if warning_text:
                    print(
                        "        Warning: "
                        f"{warning_text}"
                    )

            except Exception as error:

                status_record[
                    "Error"
                ] = str(
                    error
                )

                print(
                    f"    "
                    f"{model_name:<18} "
                    "FAILED: "
                    f"{error}"
                )

            fold_status.append(
                status_record
            )

    prediction_df = pd.DataFrame(
        all_predictions
    )

    fold_status_df = pd.DataFrame(
        fold_status
    )

    phase_summary_df, target_df = (
        summarize_phase_predictions(
            prediction_df=prediction_df,
            phase=phase,
            expected_pair_count=(
                expected_pair_count
            ),
        )
    )

    return (
        phase_summary_df,
        target_df,
        prediction_df,
        fold_status_df,
    )


# ============================================================================
# PHASE METRIC SUMMARIZATION
# ============================================================================

def summarize_phase_predictions(
    *,
    prediction_df: pd.DataFrame,
    phase: str,
    expected_pair_count: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    _, y_columns = (
        expected_columns_for_phase(
            phase
        )
    )

    baseline_df = (
        prediction_df[
            prediction_df[
                "Model"
            ]
            == "TrainingMeanBaseline"
        ]
    )

    baseline_metrics: dict[
        str,
        dict[str, float],
    ] = {}

    for target_name in y_columns:

        target_rows = (
            baseline_df[
                baseline_df[
                    "Target"
                ]
                == target_name
            ]
        )

        baseline_metrics[
            target_name
        ] = vector_metrics(
            target_rows[
                "Actual"
            ].to_numpy(
                dtype=float
            ),
            target_rows[
                "Predicted"
            ].to_numpy(
                dtype=float
            ),
        )

    model_order = [
        "TrainingMeanBaseline",
        *build_models().keys(),
    ]

    target_rows_out: list[
        dict
    ] = []

    phase_rows_out: list[
        dict
    ] = []

    for model_name in model_order:

        model_rows = (
            prediction_df[
                prediction_df[
                    "Model"
                ]
                == model_name
            ]
        )

        if model_rows.empty:
            continue

        target_relative_rmse: list[
            float
        ] = []

        target_relative_mae: list[
            float
        ] = []

        targets_better_than_baseline = 0

        complete_target_count = 0

        negative_prediction_count = int(
            (
                model_rows[
                    "Predicted"
                ]
                < 0
            ).sum()
        )

        for target_name in y_columns:

            target_rows = (
                model_rows[
                    model_rows[
                        "Target"
                    ]
                    == target_name
                ]
            )

            metrics = vector_metrics(
                target_rows[
                    "Actual"
                ].to_numpy(
                    dtype=float
                ),
                target_rows[
                    "Predicted"
                ].to_numpy(
                    dtype=float
                ),
            )

            baseline_target = (
                baseline_metrics[
                    target_name
                ]
            )

            relative_rmse = safe_ratio(
                metrics[
                    "RMSE"
                ],
                baseline_target[
                    "RMSE"
                ],
            )

            relative_mae = safe_ratio(
                metrics[
                    "MAE"
                ],
                baseline_target[
                    "MAE"
                ],
            )

            if np.isfinite(
                relative_rmse
            ):
                target_relative_rmse.append(
                    relative_rmse
                )

            if np.isfinite(
                relative_mae
            ):
                target_relative_mae.append(
                    relative_mae
                )

            if (
                np.isfinite(
                    relative_rmse
                )
                and relative_rmse < 1.0
            ):
                targets_better_than_baseline += 1

            complete_target = (
                len(
                    target_rows
                )
                == expected_pair_count
                and target_rows[
                    "Pair ID"
                ].nunique()
                == expected_pair_count
            )

            if complete_target:
                complete_target_count += 1

            target_rows_out.append(
                {
                    "Phase": phase,
                    "Model": model_name,
                    "Target": target_name,
                    "MAE": (
                        metrics[
                            "MAE"
                        ]
                    ),
                    "RMSE": (
                        metrics[
                            "RMSE"
                        ]
                    ),
                    "R2": (
                        metrics[
                            "R2"
                        ]
                    ),
                    "MAPE (%)": (
                        metrics[
                            "MAPE (%)"
                        ]
                    ),
                    "Baseline MAE": (
                        baseline_target[
                            "MAE"
                        ]
                    ),
                    "Baseline RMSE": (
                        baseline_target[
                            "RMSE"
                        ]
                    ),
                    "MAE / Baseline MAE": (
                        relative_mae
                    ),
                    "RMSE / Baseline RMSE": (
                        relative_rmse
                    ),
                    "Better Than Baseline": (
                        bool(
                            np.isfinite(
                                relative_rmse
                            )
                            and relative_rmse < 1.0
                        )
                        if model_name
                        != "TrainingMeanBaseline"
                        else False
                    ),
                    "N": int(
                        len(
                            target_rows
                        )
                    ),
                    "Expected N": (
                        expected_pair_count
                    ),
                    "Complete OOS Coverage": (
                        complete_target
                    ),
                    "Negative Predictions": int(
                        (
                            target_rows[
                                "Predicted"
                            ]
                            < 0
                        ).sum()
                    ),
                }
            )

        pooled_metrics = vector_metrics(
            model_rows[
                "Actual"
            ].to_numpy(
                dtype=float
            ),
            model_rows[
                "Predicted"
            ].to_numpy(
                dtype=float
            ),
        )

        complete_coverage = (
            complete_target_count
            == len(
                y_columns
            )
        )

        if (
            model_name
            == "TrainingMeanBaseline"
        ):
            mean_relative_rmse = 1.0
            median_relative_rmse = 1.0
            mean_relative_mae = 1.0
            targets_better_than_baseline = 0
        else:
            mean_relative_rmse = (
                float(
                    np.mean(
                        target_relative_rmse
                    )
                )
                if target_relative_rmse
                else np.nan
            )

            median_relative_rmse = (
                float(
                    np.median(
                        target_relative_rmse
                    )
                )
                if target_relative_rmse
                else np.nan
            )

            mean_relative_mae = (
                float(
                    np.mean(
                        target_relative_mae
                    )
                )
                if target_relative_mae
                else np.nan
            )

        phase_rows_out.append(
            {
                "Phase": phase,
                "Model": model_name,
                "Mean Baseline-Relative RMSE": (
                    mean_relative_rmse
                ),
                "Median Baseline-Relative RMSE": (
                    median_relative_rmse
                ),
                "Mean Baseline-Relative MAE": (
                    mean_relative_mae
                ),
                "Targets Better Than Baseline": (
                    targets_better_than_baseline
                ),
                "Targets Evaluated": int(
                    len(
                        target_relative_rmse
                    )
                ),
                "Complete OOS Coverage": (
                    complete_coverage
                ),
                "Negative Predictions": (
                    negative_prediction_count
                ),
                "Pooled MAE": (
                    pooled_metrics[
                        "MAE"
                    ]
                ),
                "Pooled RMSE": (
                    pooled_metrics[
                        "RMSE"
                    ]
                ),
                "Pooled R2": (
                    pooled_metrics[
                        "R2"
                    ]
                ),
                "Pooled MAPE (%)": (
                    pooled_metrics[
                        "MAPE (%)"
                    ]
                ),
                "Pooled Metric Scope": (
                    "MIXED_UNIT_DIAGNOSTIC_ONLY"
                ),
                "N Predictions": int(
                    len(
                        model_rows
                    )
                ),
                "Expected Predictions": int(
                    expected_pair_count
                    * len(
                        y_columns
                    )
                ),
            }
        )

    phase_summary_df = pd.DataFrame(
        phase_rows_out
    )

    target_df = pd.DataFrame(
        target_rows_out
    )

    return (
        phase_summary_df,
        target_df,
    )


# ============================================================================
# MODEL RANKING / SELECTION
# ============================================================================

def rank_and_select_models(
    phase_summary: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Rank complete-coverage models independently within each phase.

    Primary:
        Mean Baseline-Relative RMSE

    Secondary:
        Mean Baseline-Relative MAE
    """

    ranking_frames: list[
        pd.DataFrame
    ] = []

    selections: list[
        dict
    ] = []

    for phase in PHASES:

        phase_results = (
            phase_summary[
                (
                    phase_summary[
                        "Phase"
                    ]
                    == phase
                )
                & (
                    phase_summary[
                        "Model"
                    ]
                    != "TrainingMeanBaseline"
                )
                & (
                    phase_summary[
                        "Complete OOS Coverage"
                    ]
                )
            ]
            .copy()
        )

        if phase_results.empty:
            raise RuntimeError(
                "No candidate model achieved complete "
                f"LOPO coverage for phase {phase}."
            )

        phase_results = (
            phase_results.sort_values(
                by=[
                    "Mean Baseline-Relative RMSE",
                    "Mean Baseline-Relative MAE",
                ],
                ascending=[
                    True,
                    True,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        phase_results[
            "Phase Rank"
        ] = np.arange(
            1,
            len(
                phase_results
            )
            + 1,
        )

        ranking_frames.append(
            phase_results
        )

        best = (
            phase_results.iloc[
                0
            ]
        )

        selections.append(
            {
                "Phase": phase,
                "Best Model": (
                    best[
                        "Model"
                    ]
                ),
                "Mean Baseline-Relative RMSE": (
                    best[
                        "Mean Baseline-Relative RMSE"
                    ]
                ),
                "Mean Baseline-Relative MAE": (
                    best[
                        "Mean Baseline-Relative MAE"
                    ]
                ),
                "Targets Better Than Baseline": (
                    best[
                        "Targets Better Than Baseline"
                    ]
                ),
                "Negative Predictions": (
                    best[
                        "Negative Predictions"
                    ]
                ),
                "Complete OOS Coverage": (
                    best[
                        "Complete OOS Coverage"
                    ]
                ),
                "Selection Basis": (
                    "Lowest mean target-wise LOPO RMSE / "
                    "training-mean-baseline RMSE; "
                    "mean baseline-relative MAE as tie-breaker"
                ),
            }
        )

    rankings = pd.concat(
        ranking_frames,
        ignore_index=True,
    )

    selections_df = pd.DataFrame(
        selections
    )

    return (
        rankings,
        selections_df,
    )


# ============================================================================
# COMBINED CLASS-WISE OOS DIAGNOSTICS
# ============================================================================

def create_combined_classwise_evaluation(
    held_out_predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate the COMBINED model predictions separately on held-out
    STRICT and NONSTRICT observations.

    This is diagnostic only and does not alter COMBINED model selection.
    """

    target_rows_out: list[
        dict
    ] = []

    summary_rows_out: list[
        dict
    ] = []

    for pair_class in VALID_SINGLE_CLASSES:

        class_df = (
            held_out_predictions[
                held_out_predictions[
                    "Pair Class"
                ]
                == pair_class
            ]
            .copy()
        )

        if class_df.empty:
            continue

        for phase in PHASES:

            phase_df = (
                class_df[
                    class_df[
                        "Phase"
                    ]
                    == phase
                ]
            )

            if phase_df.empty:
                continue

            _, y_columns = (
                expected_columns_for_phase(
                    phase
                )
            )

            baseline_df = (
                phase_df[
                    phase_df[
                        "Model"
                    ]
                    == "TrainingMeanBaseline"
                ]
            )

            baseline_by_target = {}

            for target_name in y_columns:

                rows = (
                    baseline_df[
                        baseline_df[
                            "Target"
                        ]
                        == target_name
                    ]
                )

                baseline_by_target[
                    target_name
                ] = vector_metrics(
                    rows[
                        "Actual"
                    ].to_numpy(
                        dtype=float
                    ),
                    rows[
                        "Predicted"
                    ].to_numpy(
                        dtype=float
                    ),
                )

            for model_name in (
                phase_df[
                    "Model"
                ].drop_duplicates()
            ):

                model_df = (
                    phase_df[
                        phase_df[
                            "Model"
                        ]
                        == model_name
                    ]
                )

                relative_values = []

                targets_better = 0

                for target_name in y_columns:

                    rows = (
                        model_df[
                            model_df[
                                "Target"
                            ]
                            == target_name
                        ]
                    )

                    metrics = vector_metrics(
                        rows[
                            "Actual"
                        ].to_numpy(
                            dtype=float
                        ),
                        rows[
                            "Predicted"
                        ].to_numpy(
                            dtype=float
                        ),
                    )

                    baseline_metrics = (
                        baseline_by_target[
                            target_name
                        ]
                    )

                    relative_rmse = safe_ratio(
                        metrics[
                            "RMSE"
                        ],
                        baseline_metrics[
                            "RMSE"
                        ],
                    )

                    if np.isfinite(
                        relative_rmse
                    ):
                        relative_values.append(
                            relative_rmse
                        )

                    if (
                        np.isfinite(
                            relative_rmse
                        )
                        and relative_rmse < 1.0
                    ):
                        targets_better += 1

                    target_rows_out.append(
                        {
                            "Pair Class": (
                                pair_class
                            ),
                            "Phase": phase,
                            "Model": (
                                model_name
                            ),
                            "Target": (
                                target_name
                            ),
                            "MAE": (
                                metrics[
                                    "MAE"
                                ]
                            ),
                            "RMSE": (
                                metrics[
                                    "RMSE"
                                ]
                            ),
                            "R2": (
                                metrics[
                                    "R2"
                                ]
                            ),
                            "MAPE (%)": (
                                metrics[
                                    "MAPE (%)"
                                ]
                            ),
                            "Baseline RMSE": (
                                baseline_metrics[
                                    "RMSE"
                                ]
                            ),
                            "RMSE / Baseline RMSE": (
                                relative_rmse
                            ),
                            "N": int(
                                len(
                                    rows
                                )
                            ),
                            "Negative Predictions": int(
                                (
                                    rows[
                                        "Predicted"
                                    ]
                                    < 0
                                ).sum()
                            ),
                        }
                    )

                summary_rows_out.append(
                    {
                        "Pair Class": (
                            pair_class
                        ),
                        "Phase": phase,
                        "Model": (
                            model_name
                        ),
                        "Mean Baseline-Relative RMSE": (
                            float(
                                np.mean(
                                    relative_values
                                )
                            )
                            if relative_values
                            else np.nan
                        ),
                        "Targets Better Than Baseline": (
                            targets_better
                        ),
                        "Targets Evaluated": int(
                            len(
                                relative_values
                            )
                        ),
                        "N Held-Out Pairs": int(
                            model_df[
                                "Pair ID"
                            ].nunique()
                        ),
                    }
                )

    return (
        pd.DataFrame(
            summary_rows_out
        ),
        pd.DataFrame(
            target_rows_out
        ),
    )


# ============================================================================
# FINAL MODEL FIT
# ============================================================================

def fit_final_models(
    *,
    dataframe: pd.DataFrame,
    selections_df: pd.DataFrame,
    model_dir: Path,
    scope: str,
) -> pd.DataFrame:
    """
    Fit one selected model per phase using all pairs in the scope.

    These final fits are NOT used to calculate LOPO performance.
    """

    scope = normalize_scope(
        scope
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_records: list[
        dict
    ] = []

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL PHASE-WISE MODEL FIT "
        f"ON ALL {scope} PAIRS"
    )

    print(
        "=" * 72
    )

    for _, selection in (
        selections_df.iterrows()
    ):

        phase = str(
            selection[
                "Phase"
            ]
        )

        model_name = str(
            selection[
                "Best Model"
            ]
        )

        x_columns, y_columns = (
            expected_columns_for_phase(
                phase
            )
        )

        x = (
            dataframe[
                x_columns
            ]
            .apply(
                pd.to_numeric,
                errors="raise",
            )
        )

        y = (
            dataframe[
                y_columns
            ]
            .apply(
                pd.to_numeric,
                errors="raise",
            )
        )

        model = (
            build_models()[
                model_name
            ]
        )

        status = {
            "Scope": scope,
            "Phase": phase,
            "Model": model_name,
            "Status": "FAILED",
            "Warnings Count": 0,
            "Warnings": "",
            "Error": "",
        }

        try:

            with warnings.catch_warnings(
                record=True
            ) as caught:

                warnings.simplefilter(
                    "always"
                )

                model.fit(
                    x,
                    y,
                )

            warning_text = (
                warning_records_to_text(
                    caught
                )
            )

            status[
                "Warnings Count"
            ] = len(
                caught
            )

            status[
                "Warnings"
            ] = warning_text

            status[
                "Status"
            ] = (
                "SUCCESS_WITH_WARNINGS"
                if caught
                else "SUCCESS"
            )

            phase_dir = (
                model_dir
                / phase
            )

            phase_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_path = (
                phase_dir
                / (
                    f"{model_name}_"
                    f"{phase}_Phase.pkl"
                )
            )

            joblib.dump(
                model,
                model_path,
            )

            metadata = {
                "script_version": (
                    SCRIPT_VERSION
                ),
                "scope": scope,
                "phase": phase,
                "model": model_name,
                "input_columns": (
                    x_columns
                ),
                "target_columns": (
                    y_columns
                ),
                "pair_class_used_as_predictor": (
                    False
                ),
                "pair_class_handling": (
                    "READ / VALIDATE / PRESERVE ONLY"
                ),
                "pair_class_recalculation": (
                    "NONE"
                ),
                "pair_class_overwrite": (
                    "NONE"
                ),
                "training_pairs": (
                    dataframe[
                        PAIR_COLUMN
                    ]
                    .astype(str)
                    .tolist()
                ),
                "training_pair_classes": (
                    dataframe[
                        PAIR_CLASS_COLUMN
                    ]
                    .astype(str)
                    .tolist()
                ),
                "selection_score_mean_baseline_relative_rmse": (
                    float(
                        selection[
                            "Mean Baseline-Relative RMSE"
                        ]
                    )
                ),
            }

            joblib.dump(
                metadata,
                (
                    phase_dir
                    / f"{phase}_metadata.pkl"
                ),
            )

            print(
                f"{phase:<10} "
                f"{model_name:<18} "
                f"{status['Status']}"
            )

            print(
                "           Saved: "
                f"{model_path}"
            )

            if warning_text:
                print(
                    "           Warning: "
                    f"{warning_text}"
                )

        except Exception as error:

            status[
                "Error"
            ] = str(
                error
            )

            print(
                f"{phase:<10} "
                f"{model_name:<18} "
                "FAILED: "
                f"{error}"
            )

        status_records.append(
            status
        )

    status_df = pd.DataFrame(
        status_records
    )

    status_df.to_excel(
        model_dir
        / "Final_Phase_Model_Fit_Status.xlsx",
        index=False,
    )

    return status_df


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    *,
    results_dir: Path,
    scope: str,
    input_paths: list[Path],
    validation_info: dict,
    phase_summary: pd.DataFrame,
    target_results: pd.DataFrame,
    held_out_predictions: pd.DataFrame,
    fold_status: pd.DataFrame,
    rankings: pd.DataFrame,
    selections: pd.DataFrame,
    combined_class_summary: pd.DataFrame | None,
    combined_class_targets: pd.DataFrame | None,
) -> None:

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        results_dir
        / "Phase_Wise_ML_Model_Comparison.xlsx"
    )

    target_path = (
        results_dir
        / "Phase_Wise_ML_Per_Target_Metrics.xlsx"
    )

    predictions_path = (
        results_dir
        / "Phase_Wise_ML_Held_Out_Predictions.xlsx"
    )

    detailed_path = (
        results_dir
        / "Phase_Wise_ML_Detailed_Evaluation.xlsx"
    )

    selection_path = (
        results_dir
        / "Phase_Wise_Best_Model_By_Phase.xlsx"
    )

    phase_summary.to_excel(
        comparison_path,
        index=False,
    )

    target_results.to_excel(
        target_path,
        index=False,
    )

    held_out_predictions.to_excel(
        predictions_path,
        index=False,
    )

    selections.to_excel(
        selection_path,
        index=False,
    )

    manifest_rows = []

    for input_path in input_paths:

        manifest_rows.append(
            {
                "Script Version": (
                    SCRIPT_VERSION
                ),
                "Scope": scope,
                "Input File": str(
                    input_path
                ),
                "Phases": (
                    ", ".join(
                        PHASES
                    )
                ),
                "Inputs Per Phase": (
                    "ICE phase Time + Distance"
                ),
                "Targets Per Phase": (
                    "EV phase Time + Distance"
                ),
                "Charging/Refuelling": (
                    "Excluded; separate pipeline"
                ),
                "Validation": (
                    "Leave-One-Pair-Out CV"
                ),
                "Primary Selection Metric": (
                    "Mean target-wise RMSE / "
                    "training-mean-baseline RMSE"
                ),
                "Secondary Selection Metric": (
                    "Mean target-wise MAE / "
                    "training-mean-baseline MAE"
                ),
                "Pair Class as Predictor": (
                    "NO"
                ),
                "Pair Class Handling": (
                    "READ / VALIDATE / PRESERVE ONLY"
                ),
                "Pair Class Recalculation": (
                    "NONE"
                ),
                "Pair Class Overwrite": (
                    "NONE"
                ),
                "Prediction Clipping": (
                    "NONE"
                ),
                "Random State": (
                    RANDOM_STATE
                ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    validation_rows = []

    for key, value in (
        validation_info.items()
    ):

        row = {
            "Dataset": key,
        }

        if isinstance(
            value,
            dict,
        ):
            for field, field_value in (
                value.items()
            ):
                row[
                    str(
                        field
                    )
                ] = (
                    str(
                        field_value
                    )
                    if isinstance(
                        field_value,
                        (
                            dict,
                            list,
                            tuple,
                            set,
                        ),
                    )
                    else field_value
                )

        validation_rows.append(
            row
        )

    with pd.ExcelWriter(
        detailed_path,
        engine="openpyxl",
    ) as writer:

        manifest.to_excel(
            writer,
            sheet_name="Run_Manifest",
            index=False,
        )

        pd.DataFrame(
            validation_rows
        ).to_excel(
            writer,
            sheet_name="Validation_Summary",
            index=False,
        )

        phase_summary.to_excel(
            writer,
            sheet_name="Phase_Model_Summary",
            index=False,
        )

        rankings.to_excel(
            writer,
            sheet_name="Phase_Model_Rankings",
            index=False,
        )

        target_results.to_excel(
            writer,
            sheet_name="Target_Wise_OOS",
            index=False,
        )

        selections.to_excel(
            writer,
            sheet_name="Best_Model_By_Phase",
            index=False,
        )

        held_out_predictions.to_excel(
            writer,
            sheet_name="Held_Out_Predictions",
            index=False,
        )

        fold_status.to_excel(
            writer,
            sheet_name="Fold_Model_Status",
            index=False,
        )

        if (
            combined_class_summary
            is not None
            and not combined_class_summary.empty
        ):
            combined_class_summary.to_excel(
                writer,
                sheet_name="Combined_Class_Summary",
                index=False,
            )

        if (
            combined_class_targets
            is not None
            and not combined_class_targets.empty
        ):
            combined_class_targets.to_excel(
                writer,
                sheet_name="Combined_Class_Targets",
                index=False,
            )

    print(
        "\nSaved model comparison:\n"
        f"{comparison_path}"
    )

    print(
        "\nSaved target-wise metrics:\n"
        f"{target_path}"
    )

    print(
        "\nSaved held-out predictions:\n"
        f"{predictions_path}"
    )

    print(
        "\nSaved detailed evaluation:\n"
        f"{detailed_path}"
    )

    print(
        "\nSaved best-model selection:\n"
        f"{selection_path}"
    )


# ============================================================================
# RUN ONE SCOPE
# ============================================================================

def run_phase_wise_ml(
    scope: str,
) -> None:

    scope = normalize_scope(
        scope
    )

    results_dir, model_dir = (
        get_output_directories(
            scope
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TAT MS INTERCITY"
    )

    print(
        "PHASE-WISE ML MODELLING"
    )

    print(
        "=" * 72
    )

    print(
        f"Version : "
        f"{SCRIPT_VERSION}"
    )

    print(
        f"Scope   : "
        f"{scope}"
    )

    dataframe, input_paths, validation_info = (
        load_scope_data(
            scope
        )
    )

    print(
        "\nTraining data loaded."
    )

    for path in input_paths:
        print(
            f"Input   : "
            f"{path}"
        )

    print(
        f"Rows    : "
        f"{len(dataframe)}"
    )

    print(
        f"Columns : "
        f"{len(dataframe.columns)}"
    )

    class_counts = (
        dataframe[
            PAIR_CLASS_COLUMN
        ]
        .astype("string")
        .str.strip()
        .str.upper()
        .value_counts()
        .to_dict()
    )

    print(
        "\nSTRICT / NONSTRICT validation:"
    )

    print(
        f"  Observed counts : "
        f"{class_counts}"
    )

    print(
        "  Handling        : "
        "READ / VALIDATE / PRESERVE ONLY"
    )

    print(
        "  Recalculation   : NONE"
    )

    print(
        "  Overwrite       : NONE"
    )

    print(
        "  Used in X       : NO"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE-WISE ML CONFIGURATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Training scope   : "
        f"{scope}"
    )

    print(
        "Phases           : "
        "STOP, IDLE, INCHING, ACC, DEC, CRUISE"
    )

    print(
        "Inputs / phase   : "
        "ICE time + ICE distance"
    )

    print(
        "Targets / phase  : "
        "EV time + EV distance"
    )

    print(
        "Charging         : "
        "Separate pipeline"
    )

    print(
        "Validation       : "
        "Leave-One-Pair-Out CV"
    )

    print(
        "Models           : "
        f"{len(build_models())} candidate ML models"
    )

    print(
        "Selection metric : "
        "Mean target-wise baseline-relative RMSE"
    )

    if len(
        dataframe
    ) < 30:
        print(
            "\nSCIENTIFIC NOTE:"
        )

        print(
            "This scope has fewer than 30 paired observations."
        )

        print(
            "Phase-wise model comparisons are exploratory "
            "and should be interpreted cautiously."
        )

    all_phase_summary = []
    all_target = []
    all_predictions = []
    all_fold_status = []

    for phase in PHASES:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"PHASE: "
            f"{phase}"
        )

        print(
            "=" * 72
        )

        (
            phase_summary_df,
            target_df,
            prediction_df,
            fold_status_df,
        ) = evaluate_phase(
            dataframe=(
                dataframe
            ),
            phase=(
                phase
            ),
            scope=(
                scope
            ),
        )

        all_phase_summary.append(
            phase_summary_df
        )

        all_target.append(
            target_df
        )

        all_predictions.append(
            prediction_df
        )

        all_fold_status.append(
            fold_status_df
        )

        print(
            "\nPhase OOS comparison:"
        )

        display = (
            phase_summary_df[
                phase_summary_df[
                    "Model"
                ]
                != "TrainingMeanBaseline"
            ][
                [
                    "Model",
                    "Mean Baseline-Relative RMSE",
                    "Mean Baseline-Relative MAE",
                    "Targets Better Than Baseline",
                    "Complete OOS Coverage",
                    "Negative Predictions",
                ]
            ]
            .sort_values(
                by=[
                    "Mean Baseline-Relative RMSE",
                    "Mean Baseline-Relative MAE",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
        )

        print(
            display.to_string(
                index=False
            )
        )

    phase_summary = pd.concat(
        all_phase_summary,
        ignore_index=True,
    )

    target_results = pd.concat(
        all_target,
        ignore_index=True,
    )

    held_out_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    fold_status = pd.concat(
        all_fold_status,
        ignore_index=True,
    )

    rankings, selections = (
        rank_and_select_models(
            phase_summary
        )
    )

    combined_class_summary = None
    combined_class_targets = None

    if scope == "COMBINED":
        (
            combined_class_summary,
            combined_class_targets,
        ) = create_combined_classwise_evaluation(
            held_out_predictions
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "BEST MODEL BY PHASE"
    )

    print(
        "=" * 72
    )

    print(
        selections.to_string(
            index=False
        )
    )

    save_results(
        results_dir=(
            results_dir
        ),
        scope=(
            scope
        ),
        input_paths=(
            input_paths
        ),
        validation_info=(
            validation_info
        ),
        phase_summary=(
            phase_summary
        ),
        target_results=(
            target_results
        ),
        held_out_predictions=(
            held_out_predictions
        ),
        fold_status=(
            fold_status
        ),
        rankings=(
            rankings
        ),
        selections=(
            selections
        ),
        combined_class_summary=(
            combined_class_summary
        ),
        combined_class_targets=(
            combined_class_targets
        ),
    )

    final_fit_status = (
        fit_final_models(
            dataframe=(
                dataframe
            ),
            selections_df=(
                selections
            ),
            model_dir=(
                model_dir
            ),
            scope=(
                scope
            ),
        )
    )

    selections.to_excel(
        model_dir
        / "Selected_Phase_Models.xlsx",
        index=False,
    )

    run_manifest = pd.DataFrame(
        [
            {
                "Script Version": (
                    SCRIPT_VERSION
                ),
                "Scope": scope,
                "Rows": int(
                    len(
                        dataframe
                    )
                ),
                "Unique Pairs": int(
                    dataframe[
                        PAIR_COLUMN
                    ].nunique()
                ),
                "Pair Class Counts": str(
                    class_counts
                ),
                "Pair Class as Predictor": (
                    "NO"
                ),
                "Pair Class Recalculation": (
                    "NONE"
                ),
                "Pair Class Overwrite": (
                    "NONE"
                ),
                "Phases": (
                    ", ".join(
                        PHASES
                    )
                ),
            }
        ]
    )

    run_manifest.to_excel(
        model_dir
        / "Phase_Wise_Model_Manifest.xlsx",
        index=False,
    )

    successful_final = int(
        final_fit_status[
            "Status"
        ].isin(
            [
                "SUCCESS",
                "SUCCESS_WITH_WARNINGS",
            ]
        ).sum()
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PHASE-WISE ML PIPELINE COMPLETED"
    )

    print(
        "=" * 72
    )

    print(
        f"Scope                  : "
        f"{scope}"
    )

    print(
        f"Unique training pairs  : "
        f"{dataframe[PAIR_COLUMN].nunique()}"
    )

    print(
        f"Phases evaluated       : "
        f"{len(PHASES)}"
    )

    print(
        f"Candidate ML models    : "
        f"{len(build_models())}"
    )

    print(
        f"Final phase models fit : "
        f"{successful_final}/{len(PHASES)}"
    )

    print(
        f"\nResults:\n"
        f"{results_dir}"
    )

    print(
        f"\nModels:\n"
        f"{model_dir}"
    )

    print(
        "\nCharging/refuelling is NOT part of phase-wise ML."
    )

    print(
        "Use the separate charging/refuelling prediction workflow."
    )

    print(
        "\nPair Class was read, validated, and preserved only."
    )

    print(
        "Pair Class was not recalculated, overwritten, "
        "or used as an ML predictor."
    )

    print(
        "\nBest-model selection uses dimensionless "
        "target-wise baseline-relative RMSE."
    )

    print(
        "Raw pooled mixed-unit metrics are diagnostic only."
    )


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Phase-wise ML modelling for paired "
            "Intercity EV-ICE summary data."
        )
    )

    parser.add_argument(
        "--scope",
        choices=[
            "ALL",
            "STRICT",
            "NONSTRICT",
            "COMBINED",
        ],
        default="ALL",
        help=(
            "Dataset scope. Default ALL runs "
            "STRICT, NONSTRICT, then COMBINED."
        ),
    )

    # Backward-compatible alias with the earlier Full ML interface.
    parser.add_argument(
        "--pair-class",
        choices=[
            "STRICT",
            "NONSTRICT",
            "COMBINED",
        ],
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    arguments = (
        parse_arguments()
    )

    requested_scope = (
        arguments.pair_class
        if arguments.pair_class
        is not None
        else arguments.scope
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
            normalize_scope(
                requested_scope
            )
        ]
    )

    for scope in scopes:
        run_phase_wise_ml(
            scope
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL REQUESTED PHASE-WISE ML RUNS COMPLETED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
