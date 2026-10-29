from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
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

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False


# ============================================================
# VERSION / REPRODUCIBILITY
# ============================================================

SCRIPT_VERSION = "2026-10-01_FULL_ML_V8_15X15_MODEL_ONLY"
RANDOM_STATE = 42


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PART1_PROJECT = (
    PROJECT_ROOT.parent
    / "TAT_MS_INTERCITY"
)

TRAINING_DATA_DIR = (
    PART1_PROJECT
    / "Data"
    / "Results"
    / "Master_Summary"
    / "Training_Data"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "Results"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
)


# ============================================================
# FULL UPSTREAM ICE VARIABLES — 17
# ============================================================

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


# ============================================================
# EXPLICITLY EXCLUDED CONSTANT ICE INPUTS — 2
# ============================================================

EXCLUDED_CONSTANT_ICE_FEATURES = [
    "ICE CHARGING_REFUELING Time (min)",
    "ICE CHARGING_REFUELING Distance (km)",
]


# ============================================================
# MODEL INPUTS — 15
# ============================================================

ICE_FEATURES = [
    column
    for column in ALL_ICE_CANDIDATES
    if column not in EXCLUDED_CONSTANT_ICE_FEATURES
]


# ============================================================
# FULL-ML OUTPUT DESIGN
# ============================================================

DRIVING_PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
]


# Authoritative EV columns present in the Part 1 Master Summary.
# All 17 remain source-data integrity checks.
#
# The Full-ML learned target vector contains 15 EV outputs:
#   3 trip-level outputs
#       - EV Total Distance
#       - EV Total TAT
#       - EV Average Moving Speed
#   12 non-charging phase outputs
#       - six phase times
#       - six phase distances
#
# EV charging is NOT part of Full-ML training or evaluation.
# Charging is calculated later by src/prediction/charging_time.py during
# new-ICE inference, and final derived totals are assembled downstream.
ALL_EV_SOURCE_COLUMNS = [
    "EV Total Distance (km)",
    "EV Total TAT (min)",
    "EV Average Moving Speed (km/h)",

    "EV CHARGING_REFUELING Time (min)",
    "EV STOP Time (min)",
    "EV IDLE Time (min)",
    "EV INCHING Time (min)",
    "EV ACC Time (min)",
    "EV DEC Time (min)",
    "EV CRUISE Time (min)",

    "EV CHARGING_REFUELING Distance (km)",
    "EV STOP Distance (km)",
    "EV IDLE Distance (km)",
    "EV INCHING Distance (km)",
    "EV ACC Distance (km)",
    "EV DEC Distance (km)",
    "EV CRUISE Distance (km)",
]


TRIP_LEVEL_EV_TARGETS = [
    "EV Total Distance (km)",
    "EV Total TAT (min)",
    "EV Average Moving Speed (km/h)",
]


# Primary Full-ML learned outputs = 15.
EV_TARGETS = list(
    TRIP_LEVEL_EV_TARGETS
)

for phase in DRIVING_PHASES:
    EV_TARGETS.extend(
        [
            f"EV {phase} Time (min)",
            f"EV {phase} Distance (km)",
        ]
    )


PHASE_TIME_TARGETS = [
    f"EV {phase} Time (min)"
    for phase in DRIVING_PHASES
]


PHASE_DISTANCE_TARGETS = [
    f"EV {phase} Distance (km)"
    for phase in DRIVING_PHASES
]




PAIR_COLUMN = "Pair ID"
PAIR_CLASS_COLUMN = "Pair Class"


# ============================================================
# PATH HELPERS
# ============================================================

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


def normalize_pair_class(
    pair_class: str,
) -> str:
    """
    Normalize an authoritative upstream Pair Class value.

    Only STRICT and NONSTRICT are real Pair Class values.
    COMBINED is a modelling dataset scope, not a Pair Class.
    """

    normalized = str(
        pair_class
    ).strip().upper()

    if normalized not in VALID_PAIR_CLASSES:
        raise ValueError(
            "Pair Class must be STRICT or NONSTRICT. "
            f"Received: {pair_class!r}"
        )

    return normalized


def normalize_scope(
    scope: str,
    *,
    allow_all: bool = False,
) -> str:
    """
    Normalize a modelling dataset scope.

    STRICT and NONSTRICT are source classes.
    COMBINED means the two validated source datasets are
    concatenated in memory while preserving row-level Pair Class.
    ALL means execute STRICT, NONSTRICT, then COMBINED separately.
    """

    normalized = str(
        scope
    ).strip().upper()

    allowed = (
        VALID_SCOPES
        if allow_all
        else VALID_SCOPES - {"ALL"}
    )

    if normalized not in allowed:
        allowed_text = ", ".join(
            sorted(
                allowed
            )
        )

        raise ValueError(
            "Dataset scope must be one of: "
            f"{allowed_text}. "
            f"Received: {scope!r}"
        )

    return normalized


def get_training_file(
    pair_class: str,
) -> Path:
    """
    Return one authoritative Part 1 training workbook.

    Pair Class is used only to select the upstream workbook.
    It is never derived, recalculated, or overwritten here.
    """

    normalized = normalize_pair_class(
        pair_class
    )

    path = (
        TRAINING_DATA_DIR
        / f"EV_ICE_Training_{normalized}.xlsx"
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
    Return scope-isolated results and model directories.

    Examples:
        Results/STRICT/ML/Full_Model
        Results/NONSTRICT/ML/Full_Model
        Results/COMBINED/ML/Full_Model
    """

    normalized = normalize_scope(
        scope
    )

    results_dir = (
        RESULTS_ROOT
        / normalized
        / "ML"
        / "Full_Model"
    )

    model_dir = (
        MODEL_ROOT
        / normalized
        / "ML"
        / "Full_Model"
    )

    return results_dir, model_dir

# ============================================================
# DATA LOADING
# ============================================================

def load_single_training_data(
    pair_class: str,
) -> tuple[pd.DataFrame, Path]:
    """
    Load one authoritative STRICT or NONSTRICT workbook.

    This function only loads data. It does not clean, impute,
    engineer, normalize, scale, derive Pair Class, or overwrite
    Pair Class.
    """

    pair_class = normalize_pair_class(
        pair_class
    )

    input_file = get_training_file(
        pair_class
    )

    dataframe = pd.read_excel(
        input_file,
        sheet_name="01_Training_Data",
        header=3,
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError(
            f"The {pair_class} training worksheet is empty."
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe, input_file

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_full_model_data(
    dataframe: pd.DataFrame,
    expected_pair_class: str,
) -> dict:
    """
    Validate Full Model structure and source-data integrity.

    IMPORTANT
    ---------
    This function only validates.
    It does not remove, modify, impute, scale, transform,
    recalculate, or overwrite any source values.
    """

    expected_pair_class = normalize_pair_class(
        expected_pair_class
    )

    required_columns = (
        ALL_ICE_CANDIDATES
        + ALL_EV_SOURCE_COLUMNS
        + [
            PAIR_COLUMN,
            PAIR_CLASS_COLUMN,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required Full Model columns are missing:\n"
            + "\n".join(
                f" - {column}"
                for column in missing_columns
            )
        )

    if len(ALL_ICE_CANDIDATES) != 17:
        raise ValueError(
            "Expected exactly 17 upstream ICE variables."
        )

    if len(ICE_FEATURES) != 15:
        raise ValueError(
            "The Full Model must contain exactly "
            "15 active ICE predictors."
        )

    if len(EV_TARGETS) != 15:
        raise ValueError(
            "The primary Full ML model must contain exactly "
            "15 EV outputs: 3 trip-level outputs + "
            "6 phase times + 6 phase distances."
        )

    # --------------------------------------------------------
    # Pair ID
    # --------------------------------------------------------

    pair_ids = dataframe[PAIR_COLUMN]

    pair_missing = (
        pair_ids.isna()
        | pair_ids.astype("string").str.strip().eq("")
    )

    if pair_missing.any():
        raise ValueError(
            "Pair ID contains missing/blank values."
        )

    if pair_ids.duplicated().any():
        duplicate_pairs = (
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
            "Duplicate Pair IDs were found:\n"
            + "\n".join(
                f" - {pair}"
                for pair in duplicate_pairs
            )
        )

    # --------------------------------------------------------
    # STRICT / NONSTRICT
    # --------------------------------------------------------

    pair_class = (
        dataframe[PAIR_CLASS_COLUMN]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    pair_class_missing = (
        pair_class.isna()
        | pair_class.eq("")
    )

    if pair_class_missing.any():
        raise ValueError(
            "Pair Class contains missing/blank values."
        )

    valid_classes = {
        "STRICT",
        "NONSTRICT",
    }

    invalid_classes = sorted(
        set(pair_class.dropna())
        - valid_classes
    )

    if invalid_classes:
        raise ValueError(
            "Invalid Pair Class values were found:\n"
            + "\n".join(
                f" - {value}"
                for value in invalid_classes
            )
        )

    mismatch_mask = (
        pair_class
        != expected_pair_class
    )

    if mismatch_mask.any():
        mismatch_count = int(
            mismatch_mask.sum()
        )

        raise ValueError(
            f"{mismatch_count} Pair Class value(s) "
            f"do not match expected class "
            f"{expected_pair_class}."
        )

    # --------------------------------------------------------
    # Numeric / finite / non-negative
    # --------------------------------------------------------

    numeric_columns = (
        ALL_ICE_CANDIDATES
        + ALL_EV_SOURCE_COLUMNS
    )

    for column in numeric_columns:

        original = dataframe[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        nonblank_original = (
            original.notna()
            & original.astype("string").str.strip().ne("")
        )

        conversion_failures = (
            nonblank_original
            & converted.isna()
        )

        if conversion_failures.any():
            raise ValueError(
                "Non-numeric values found in "
                f"modelling column:\n{column}"
            )

        if converted.isna().any():
            raise ValueError(
                "Missing values found in "
                f"modelling column:\n{column}"
            )

        values = converted.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Infinite/non-finite values found in "
                f"modelling column:\n{column}"
            )

        if (values < 0).any():
            raise ValueError(
                "Negative physical values found in "
                f"modelling column:\n{column}"
            )

    # --------------------------------------------------------
    # Confirm the two intentionally excluded predictors are
    # still constant in this specific dataset.
    # --------------------------------------------------------

    excluded_report = {}

    for column in EXCLUDED_CONSTANT_ICE_FEATURES:

        values = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

        unique_values = int(
            values.nunique(
                dropna=True
            )
        )

        excluded_report[column] = {
            "unique_values": unique_values,
            "constant": unique_values <= 1,
            "value": (
                float(values.iloc[0])
                if len(values) > 0
                else np.nan
            ),
        }

        if unique_values > 1:
            raise ValueError(
                "A predictor explicitly excluded because it "
                "was constant now contains variability:\n"
                f"{column}\n"
                "Reassess the Full Model feature set before "
                "continuing."
            )

    return {
        "pair_class": expected_pair_class,
        "rows": int(len(dataframe)),
        "columns": int(
            len(dataframe.columns)
        ),
        "unique_pairs": int(
            pair_ids.nunique()
        ),
        "active_ice_predictors": int(
            len(ICE_FEATURES)
        ),
        "ev_targets": int(
            len(EV_TARGETS)
        ),
        "excluded_constant_ice": (
            excluded_report
        ),
    }




def validate_combined_integrity(
    strict_dataframe: pd.DataFrame,
    nonstrict_dataframe: pd.DataFrame,
) -> dict:
    """
    Validate conditions required before in-memory concatenation.

    Both source dataframes must already have passed
    validate_full_model_data() separately.

    No Pair Class is changed. COMBINED is only a dataset scope.
    """

    strict_columns = list(
        strict_dataframe.columns
    )

    nonstrict_columns = list(
        nonstrict_dataframe.columns
    )

    strict_set = set(
        strict_columns
    )

    nonstrict_set = set(
        nonstrict_columns
    )

    if strict_set != nonstrict_set:
        strict_only = sorted(
            strict_set
            - nonstrict_set
        )

        nonstrict_only = sorted(
            nonstrict_set
            - strict_set
        )

        raise ValueError(
            "STRICT and NONSTRICT workbook schemas differ.\n"
            f"STRICT-only columns: {strict_only}\n"
            f"NONSTRICT-only columns: {nonstrict_only}"
        )

    strict_pairs = set(
        strict_dataframe[
            PAIR_COLUMN
        ].astype(str)
    )

    nonstrict_pairs = set(
        nonstrict_dataframe[
            PAIR_COLUMN
        ].astype(str)
    )

    overlap = sorted(
        strict_pairs
        & nonstrict_pairs
    )

    if overlap:
        raise ValueError(
            "Pair IDs overlap between STRICT and NONSTRICT. "
            "COMBINED modelling has been stopped to prevent "
            "duplicate/leaking pair identities:\n"
            + "\n".join(
                f" - {pair_id}"
                for pair_id in overlap
            )
        )

    combined = pd.concat(
        [
            strict_dataframe,
            nonstrict_dataframe,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if combined[
        PAIR_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Duplicate Pair IDs exist after COMBINED "
            "concatenation."
        )

    observed = (
        combined[
            PAIR_CLASS_COLUMN
        ]
        .astype("string")
        .str.strip()
        .str.upper()
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    invalid = (
        set(
            key
            for key in observed
            if pd.notna(
                key
            )
        )
        - VALID_PAIR_CLASSES
    )

    if invalid:
        raise ValueError(
            "Invalid Pair Class values exist in COMBINED data: "
            f"{sorted(invalid)}"
        )

    if not {
        "STRICT",
        "NONSTRICT",
    }.issubset(
        set(
            observed
        )
    ):
        raise ValueError(
            "COMBINED data must contain both STRICT and "
            "NONSTRICT rows."
        )

    return {
        "rows": int(
            len(
                combined
            )
        ),
        "columns": int(
            len(
                combined.columns
            )
        ),
        "unique_pairs": int(
            combined[
                PAIR_COLUMN
            ].nunique()
        ),
        "strict_rows": int(
            (
                combined[
                    PAIR_CLASS_COLUMN
                ]
                .astype("string")
                .str.strip()
                .str.upper()
                == "STRICT"
            ).sum()
        ),
        "nonstrict_rows": int(
            (
                combined[
                    PAIR_CLASS_COLUMN
                ]
                .astype("string")
                .str.strip()
                .str.upper()
                == "NONSTRICT"
            ).sum()
        ),
        "overlapping_pair_ids": 0,
        "schemas_match": True,
    }


def load_dataset_scope(
    scope: str,
) -> tuple[
    pd.DataFrame,
    list[Path],
    dict,
]:
    """
    Load and validate STRICT, NONSTRICT, or COMBINED data.

    For COMBINED:
    1. STRICT is loaded and validated independently.
    2. NONSTRICT is loaded and validated independently.
    3. Pair IDs are checked for cross-class overlap.
    4. The validated frames are concatenated in memory.
    5. Original row-level Pair Class values are preserved.

    Pair Class is never recalculated and never overwritten.
    """

    scope = normalize_scope(
        scope
    )

    if scope in VALID_PAIR_CLASSES:

        dataframe, input_file = (
            load_single_training_data(
                scope
            )
        )

        source_report = (
            validate_full_model_data(
                dataframe,
                expected_pair_class=(
                    scope
                ),
            )
        )

        report = {
            "scope": scope,
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
                dataframe[
                    PAIR_COLUMN
                ].nunique()
            ),
            "source_reports": {
                scope: (
                    source_report
                )
            },
            "strict_rows": (
                int(
                    len(
                        dataframe
                    )
                )
                if scope
                == "STRICT"
                else 0
            ),
            "nonstrict_rows": (
                int(
                    len(
                        dataframe
                    )
                )
                if scope
                == "NONSTRICT"
                else 0
            ),
            "overlapping_pair_ids": 0,
            "schemas_match": True,
        }

        return (
            dataframe,
            [
                input_file
            ],
            report,
        )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    (
        strict_dataframe,
        strict_file,
    ) = load_single_training_data(
        "STRICT"
    )

    (
        nonstrict_dataframe,
        nonstrict_file,
    ) = load_single_training_data(
        "NONSTRICT"
    )

    strict_report = (
        validate_full_model_data(
            strict_dataframe,
            expected_pair_class="STRICT",
        )
    )

    nonstrict_report = (
        validate_full_model_data(
            nonstrict_dataframe,
            expected_pair_class="NONSTRICT",
        )
    )

    combined_report = (
        validate_combined_integrity(
            strict_dataframe,
            nonstrict_dataframe,
        )
    )

    combined_dataframe = pd.concat(
        [
            strict_dataframe,
            nonstrict_dataframe,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    report = {
        "scope": "COMBINED",
        **combined_report,
        "source_reports": {
            "STRICT": (
                strict_report
            ),
            "NONSTRICT": (
                nonstrict_report
            ),
        },
    }

    return (
        combined_dataframe,
        [
            strict_file,
            nonstrict_file,
        ],
        report,
    )


# ============================================================
# PREPARE X / Y / GROUPS
# ============================================================

def prepare_full_model_data(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:

    X = (
        dataframe[
            ICE_FEATURES
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .copy()
    )

    Y = (
        dataframe[
            EV_TARGETS
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
        .copy()
    )

    groups = (
        dataframe[
            PAIR_COLUMN
        ]
        .copy()
    )

    return X, Y, groups


# ============================================================
# MODEL REGISTRY
# ============================================================

def build_candidate_models(
    n_training_samples: int,
) -> dict[str, object]:
    """
    Build conservative candidate models.

    Scaling remains inside each Pipeline so the scaler is fit
    using training-fold data only.
    """

    models: dict[
        str,
        object,
    ] = {}

    models["LinearRegression"] = Pipeline(
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
    )

    models["Ridge"] = Pipeline(
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
    )

    models["ElasticNet"] = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                MultiOutputRegressor(
                    ElasticNet(
                        alpha=0.1,
                        l1_ratio=0.5,
                        max_iter=10000,
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
        ]
    )

    pls_components = min(
        2,
        max(
            1,
            n_training_samples - 1,
        ),
        len(
            ICE_FEATURES
        ),
        len(
            EV_TARGETS
        ),
    )

    models["PLS"] = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                PLSRegression(
                    n_components=(
                        pls_components
                    )
                ),
            ),
        ]
    )

    models[
        "RandomForest"
    ] = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    models[
        "ExtraTrees"
    ] = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    models["SVR"] = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                MultiOutputRegressor(
                    SVR(
                        kernel="rbf",
                        C=1.0,
                        epsilon=0.1,
                    )
                ),
            ),
        ]
    )

    return models


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    MAPE excluding zero actual values.

    MAPE is supplementary because small actual values can
    create unstable percentages.
    """

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

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


def calculate_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Safe R² calculation over pooled held-out predictions.

    R² is not calculated with fewer than two observations or
    when the actual values have zero variance.
    """

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

    if len(actual) < 2:
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


def calculate_vector_metrics(
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

    if len(actual) == 0:
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
        "R2": calculate_r2(
            actual,
            predicted,
        ),
        "MAPE (%)": calculate_mape(
            actual,
            predicted,
        ),
    }


# ============================================================
# WARNING CAPTURE
# ============================================================

def warning_records_to_text(
    warning_records: list,
) -> str:
    """
    Convert captured warning records to compact diagnostic text.
    """

    messages = []

    for record in warning_records:

        category = (
            record.category.__name__
            if hasattr(
                record,
                "category",
            )
            else "Warning"
        )

        message = str(
            record.message
        )

        text = (
            f"{category}: "
            f"{message}"
        )

        if text not in messages:
            messages.append(
                text
            )

    return " | ".join(
        messages
    )


# ============================================================
# LOPO-CV
# ============================================================

def run_lopo_cv(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: pd.Series,
    row_pair_classes: pd.Series,
    scope: str,
) -> tuple[
    dict[str, list[dict]],
    list[dict],
    list[dict],
]:
    """
    Leave-One-Pair-Out Cross-Validation.

    Each held-out record keeps its authoritative row-level
    Pair Class. In a COMBINED run, Pair Class is NOT an input
    feature; it is retained only for audit and subgroup
    evaluation.

    For each fold:
    - one complete Pair ID is held out
    - remaining pairs are training data
    - baseline uses training-fold target means only
    - preprocessing/scaling is fit inside each model pipeline
    - warnings are captured and reported, not suppressed
    """

    scope = normalize_scope(
        scope
    )

    unique_pairs = pd.unique(
        groups
    )

    if len(unique_pairs) < 3:
        raise ValueError(
            "At least 3 unique Pair IDs are required."
        )

    if len(
        row_pair_classes
    ) != len(
        groups
    ):
        raise ValueError(
            "Pair Class series length does not match groups."
        )

    n_training_samples = (
        len(
            unique_pairs
        )
        - 1
    )

    models = build_candidate_models(
        n_training_samples
    )

    prediction_records = {
        model_name: []
        for model_name in models
    }

    baseline_records = []

    status_records = []

    print(
        "\n"
        + "=" * 75
    )

    print(
        "LEAVE-ONE-PAIR-OUT CROSS-VALIDATION"
    )

    print(
        "=" * 75
    )

    print(
        f"Dataset scope : "
        f"{scope}"
    )

    print(
        f"Unique pairs  : "
        f"{len(unique_pairs)}"
    )

    print(
        f"Train/fold    : "
        f"{n_training_samples}"
    )

    print(
        f"Predictors    : "
        f"{len(ICE_FEATURES)}"
    )

    print(
        f"Targets       : "
        f"{len(EV_TARGETS)}"
    )

    if (
        n_training_samples
        <= len(
            ICE_FEATURES
        )
    ):
        print(
            "\nSCIENTIFIC NOTE:"
        )

        print(
            "The number of training observations per fold "
            "is not greater than the number of predictors."
        )

        print(
            "Unregularized LinearRegression is retained "
            "as a diagnostic comparator only."
        )

    for (
        fold_number,
        test_pair,
    ) in enumerate(
        unique_pairs,
        start=1,
    ):

        train_mask = (
            groups
            != test_pair
        )

        test_mask = (
            groups
            == test_pair
        )

        X_train = X.loc[
            train_mask
        ]

        Y_train = Y.loc[
            train_mask
        ]

        X_test = X.loc[
            test_mask
        ]

        Y_test = Y.loc[
            test_mask
        ]

        test_classes = (
            row_pair_classes.loc[
                test_mask
            ]
            .astype("string")
            .str.strip()
            .str.upper()
            .dropna()
            .unique()
        )

        if len(
            test_classes
        ) != 1:
            raise ValueError(
                "A held-out Pair ID maps to multiple or missing "
                "Pair Class values. "
                f"Pair ID: {test_pair}"
            )

        test_pair_class = (
            normalize_pair_class(
                str(
                    test_classes[
                        0
                    ]
                )
            )
        )

        print(
            f"\nFold "
            f"{fold_number}/"
            f"{len(unique_pairs)}"
            f" | Test Pair: "
            f"{test_pair}"
            f" | Class: "
            f"{test_pair_class}"
        )

        # ----------------------------------------------------
        # Training-mean baseline
        # ----------------------------------------------------

        baseline_prediction = (
            Y_train
            .mean(
                axis=0
            )
            .to_numpy(
                dtype=float
            )
        )

        for row_number in range(
            len(
                Y_test
            )
        ):

            record = {
                "Fold": fold_number,
                "Pair ID": test_pair,
                "Pair Class": (
                    test_pair_class
                ),
                "Dataset Scope": (
                    scope
                ),
            }

            for (
                target_index,
                target,
            ) in enumerate(
                EV_TARGETS
            ):

                record[
                    f"Actual {target}"
                ] = (
                    Y_test.iloc[
                        row_number,
                        target_index,
                    ]
                )

                record[
                    f"Predicted {target}"
                ] = (
                    baseline_prediction[
                        target_index
                    ]
                )

            baseline_records.append(
                record
            )

        # ----------------------------------------------------
        # Candidate models
        # ----------------------------------------------------

        for (
            model_name,
            model,
        ) in models.items():

            status = {
                "Fold": fold_number,
                "Pair ID": test_pair,
                "Pair Class": test_pair_class,
                "Dataset Scope": scope,
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
                        X_train,
                        Y_train,
                    )

                    prediction = np.asarray(
                        model.predict(
                            X_test
                        ),
                        dtype=float,
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

                if (
                    prediction.shape
                    != Y_test.shape
                ):
                    raise ValueError(
                        "Unexpected prediction shape "
                        f"{prediction.shape}; expected "
                        f"{Y_test.shape}."
                    )

                if not np.isfinite(
                    prediction
                ).all():
                    raise ValueError(
                        "Model produced non-finite "
                        "held-out predictions."
                    )

                for row_number in range(
                    len(
                        X_test
                    )
                ):

                    record = {
                        "Fold": (
                            fold_number
                        ),
                        "Pair ID": (
                            test_pair
                        ),
                        "Pair Class": (
                            test_pair_class
                        ),
                        "Dataset Scope": (
                            scope
                        ),
                    }

                    for (
                        target_index,
                        target,
                    ) in enumerate(
                        EV_TARGETS
                    ):

                        record[
                            f"Actual {target}"
                        ] = (
                            Y_test.iloc[
                                row_number,
                                target_index,
                            ]
                        )

                        record[
                            f"Predicted {target}"
                        ] = (
                            prediction[
                                row_number,
                                target_index,
                            ]
                        )

                    prediction_records[
                        model_name
                    ].append(
                        record
                    )

                status["Status"] = (
                    "SUCCESS_WITH_WARNINGS"
                    if caught
                    else "SUCCESS"
                )

                print(
                    f"  "
                    f"{model_name:<20} "
                    f"{status['Status']}"
                )

                if warning_text:
                    print(
                        "      Warning: "
                        f"{warning_text}"
                    )

            except Exception as error:

                status[
                    "Error"
                ] = str(
                    error
                )

                print(
                    f"  "
                    f"{model_name:<20} "
                    "FAILED"
                )

                print(
                    "      Reason: "
                    f"{error}"
                )

            status_records.append(
                status
            )

    return (
        prediction_records,
        baseline_records,
        status_records,
    )

# ============================================================
# COMBINE PREDICTIONS
# ============================================================

def combine_prediction_records(
    prediction_records: dict[
        str,
        list[dict],
    ],
) -> dict[
    str,
    pd.DataFrame,
]:

    return {
        model_name: pd.DataFrame(
            records
        )
        for (
            model_name,
            records,
        ) in prediction_records.items()
        if records
    }




def filter_complete_prediction_sets(
    prediction_data: dict[
        str,
        pd.DataFrame,
    ],
    *,
    expected_rows: int,
    expected_pairs: int,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Only complete OOS prediction sets are eligible for model
    comparison. A model that fails any held-out fold is not
    ranked against models evaluated on all pairs.
    """

    complete = {}

    records = []

    for (
        model_name,
        dataframe,
    ) in prediction_data.items():

        rows = int(
            len(
                dataframe
            )
        )

        unique_pairs = (
            int(
                dataframe[
                    "Pair ID"
                ].nunique()
            )
            if (
                not dataframe.empty
                and "Pair ID"
                in dataframe.columns
            )
            else 0
        )

        is_complete = (
            rows
            == expected_rows
            and unique_pairs
            == expected_pairs
        )

        records.append(
            {
                "Model": model_name,
                "Prediction Rows": rows,
                "Unique Pairs": unique_pairs,
                "Expected Rows": (
                    expected_rows
                ),
                "Expected Pairs": (
                    expected_pairs
                ),
                "Complete OOS Coverage": (
                    is_complete
                ),
                "Eligible For Ranking": (
                    is_complete
                ),
            }
        )

        if is_complete:
            complete[
                model_name
            ] = dataframe

    return (
        complete,
        pd.DataFrame(
            records
        ),
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_prediction_table(
    dataframe: pd.DataFrame,
    model_name: str,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    pd.DataFrame,
]:
    """
    Evaluate pooled held-out predictions.

    Target-wise metrics are the primary scientific metrics.

    The pooled overall and pair-wise metrics are retained only
    as mixed-unit diagnostics because the 15 primary targets contain
    minutes, kilometres, and km/h.
    """

    target_records = []

    pair_records = []

    # --------------------------------------------------------
    # Target-wise OOS metrics
    # --------------------------------------------------------

    for target in EV_TARGETS:

        actual = dataframe[
            f"Actual {target}"
        ].to_numpy(
            dtype=float
        )

        predicted = dataframe[
            f"Predicted {target}"
        ].to_numpy(
            dtype=float
        )

        metrics = (
            calculate_vector_metrics(
                actual,
                predicted,
            )
        )

        target_records.append(
            {
                "Model": model_name,
                "Target": target,
                **metrics,
                "N": int(
                    np.isfinite(
                        actual
                    ).sum()
                ),
            }
        )

    target_metrics = pd.DataFrame(
        target_records
    )

    # --------------------------------------------------------
    # Pooled mixed-unit diagnostic metrics
    # --------------------------------------------------------

    all_actual = []

    all_predicted = []

    for target in EV_TARGETS:

        all_actual.extend(
            dataframe[
                f"Actual {target}"
            ].to_numpy(
                dtype=float
            )
        )

        all_predicted.extend(
            dataframe[
                f"Predicted {target}"
            ].to_numpy(
                dtype=float
            )
        )

    pooled_metrics = (
        calculate_vector_metrics(
            np.asarray(
                all_actual
            ),
            np.asarray(
                all_predicted
            ),
        )
    )

    overall_metrics = {
        "MAE": (
            pooled_metrics[
                "MAE"
            ]
        ),
        "RMSE": (
            pooled_metrics[
                "RMSE"
            ]
        ),
        "R2": (
            pooled_metrics[
                "R2"
            ]
        ),
        "MAPE (%)": (
            pooled_metrics[
                "MAPE (%)"
            ]
        ),
        "Metric Scope": (
            "MIXED_UNIT_DIAGNOSTIC_ONLY"
        ),
    }

    # --------------------------------------------------------
    # Pair-wise mixed-unit diagnostic metrics
    # --------------------------------------------------------

    for (
        pair_id,
        pair_df,
    ) in dataframe.groupby(
        "Pair ID",
        sort=False,
    ):

        pair_actual = []

        pair_predicted = []

        for target in EV_TARGETS:

            pair_actual.extend(
                pair_df[
                    f"Actual {target}"
                ].to_numpy(
                    dtype=float
                )
            )

            pair_predicted.extend(
                pair_df[
                    f"Predicted {target}"
                ].to_numpy(
                    dtype=float
                )
            )

        metrics = (
            calculate_vector_metrics(
                np.asarray(
                    pair_actual
                ),
                np.asarray(
                    pair_predicted
                ),
            )
        )

        pair_records.append(
            {
                "Model": model_name,
                "Pair ID": pair_id,
                **metrics,
                "Metric Scope": (
                    "MIXED_UNIT_DIAGNOSTIC_ONLY"
                ),
            }
        )

    pair_metrics = pd.DataFrame(
        pair_records
    )

    return (
        target_metrics,
        overall_metrics,
        pair_metrics,
    )


# ============================================================
# DIMENSIONLESS MODEL COMPARISON
# ============================================================

def add_baseline_relative_metrics(
    target_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add target-wise RMSE relative to the training-mean baseline.

    This produces a dimensionless comparison:
        model RMSE / baseline RMSE

    < 1 means lower RMSE than the LOPO training-mean baseline.
    """

    output = target_results.copy()

    baseline = (
        output[
            output[
                "Model"
            ]
            == "TrainingMeanBaseline"
        ][
            [
                "Target",
                "RMSE",
            ]
        ]
        .rename(
            columns={
                "RMSE": (
                    "Baseline RMSE"
                )
            }
        )
    )

    output = output.merge(
        baseline,
        on="Target",
        how="left",
    )

    denominator = output[
        "Baseline RMSE"
    ].to_numpy(
        dtype=float
    )

    numerator = output[
        "RMSE"
    ].to_numpy(
        dtype=float
    )

    relative = np.full(
        len(
            output
        ),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            numerator
        )
        & np.isfinite(
            denominator
        )
        & (
            denominator
            > 0
        )
    )

    relative[
        valid
    ] = (
        numerator[
            valid
        ]
        / denominator[
            valid
        ]
    )

    output[
        "RMSE / Baseline RMSE"
    ] = relative

    output[
        "Better Than Baseline"
    ] = (
        output[
            "RMSE / Baseline RMSE"
        ]
        < 1.0
    )

    output.loc[
        output[
            "Model"
        ]
        == "TrainingMeanBaseline",
        "Better Than Baseline",
    ] = False

    return output


def evaluate_all_models(
    prediction_data: dict[
        str,
        pd.DataFrame,
    ],
    baseline_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate baseline and all successfully predicted models.

    Model ranking uses the mean target-wise baseline-relative
    RMSE, not pooled raw RMSE across mixed physical units.
    """

    overall_records = []

    target_frames = []

    pair_frames = []

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    (
        baseline_target,
        baseline_overall,
        baseline_pair,
    ) = evaluate_prediction_table(
        baseline_data,
        "TrainingMeanBaseline",
    )

    overall_records.append(
        {
            "Model": (
                "TrainingMeanBaseline"
            ),
            **baseline_overall,
        }
    )

    target_frames.append(
        baseline_target
    )

    pair_frames.append(
        baseline_pair
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for (
        model_name,
        dataframe,
    ) in prediction_data.items():

        (
            target_metrics,
            overall_metrics,
            pair_metrics,
        ) = evaluate_prediction_table(
            dataframe,
            model_name,
        )

        overall_records.append(
            {
                "Model": model_name,
                **overall_metrics,
            }
        )

        target_frames.append(
            target_metrics
        )

        pair_frames.append(
            pair_metrics
        )

    target_results = pd.concat(
        target_frames,
        ignore_index=True,
    )

    target_results = (
        add_baseline_relative_metrics(
            target_results
        )
    )

    pair_results = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    overall_results = pd.DataFrame(
        overall_records
    )

    # --------------------------------------------------------
    # Dimensionless aggregate ranking
    # --------------------------------------------------------

    model_summary_records = []

    for model_name in (
        overall_results[
            "Model"
        ].tolist()
    ):

        subset = target_results[
            target_results[
                "Model"
            ]
            == model_name
        ]

        relative_values = pd.to_numeric(
            subset[
                "RMSE / Baseline RMSE"
            ],
            errors="coerce",
        )

        finite_relative = (
            relative_values[
                np.isfinite(
                    relative_values
                )
            ]
        )

        if (
            model_name
            == "TrainingMeanBaseline"
        ):
            mean_relative = 1.0
            median_relative = 1.0
            targets_better = 0

        elif len(
            finite_relative
        ) > 0:
            mean_relative = float(
                finite_relative.mean()
            )
            median_relative = float(
                finite_relative.median()
            )
            targets_better = int(
                (
                    finite_relative
                    < 1.0
                ).sum()
            )

        else:
            mean_relative = np.nan
            median_relative = np.nan
            targets_better = 0

        model_summary_records.append(
            {
                "Model": model_name,
                "Mean Baseline-Relative RMSE": (
                    mean_relative
                ),
                "Median Baseline-Relative RMSE": (
                    median_relative
                ),
                "Targets Better Than Baseline": (
                    targets_better
                ),
                "Targets Evaluated": int(
                    len(
                        finite_relative
                    )
                ),
            }
        )

    dimensionless_summary = (
        pd.DataFrame(
            model_summary_records
        )
    )

    overall_results = (
        overall_results.merge(
            dimensionless_summary,
            on="Model",
            how="left",
        )
    )

    ml_only = overall_results[
        overall_results[
            "Model"
        ]
        != "TrainingMeanBaseline"
    ].copy()

    ml_only = ml_only.sort_values(
        by=[
            "Mean Baseline-Relative RMSE",
            "Median Baseline-Relative RMSE",
        ],
        ascending=[
            True,
            True,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    ml_only[
        "Overall Rank"
    ] = np.arange(
        1,
        len(
            ml_only
        )
        + 1,
    )

    overall_results = (
        overall_results.merge(
            ml_only[
                [
                    "Model",
                    "Overall Rank",
                ]
            ],
            on="Model",
            how="left",
        )
    )

    overall_results[
        "Better Than Baseline"
    ] = (
        overall_results[
            "Mean Baseline-Relative RMSE"
        ]
        < 1.0
    )

    overall_results.loc[
        overall_results[
            "Model"
        ]
        == "TrainingMeanBaseline",
        "Better Than Baseline",
    ] = False

    overall_results = (
        overall_results.sort_values(
            by="Overall Rank",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        overall_results,
        target_results,
        pair_results,
    )


# ============================================================
# TARGET-WISE MODEL RANKING
# ============================================================

def create_target_ranking(
    target_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create model-level counts of target-wise best MAE/RMSE.
    """

    ml_targets = target_results[
        target_results[
            "Model"
        ]
        != "TrainingMeanBaseline"
    ].copy()

    ml_targets[
        "MAE Rank"
    ] = (
        ml_targets.groupby(
            "Target"
        )[
            "MAE"
        ]
        .rank(
            method="min",
            ascending=True,
        )
    )

    ml_targets[
        "RMSE Rank"
    ] = (
        ml_targets.groupby(
            "Target"
        )[
            "RMSE"
        ]
        .rank(
            method="min",
            ascending=True,
        )
    )

    summary = (
        ml_targets.groupby(
            "Model"
        )
        .agg(
            **{
                "Targets with Best MAE": (
                    "MAE Rank",
                    lambda values: int(
                        (
                            values
                            == 1
                        ).sum()
                    ),
                ),
                "Targets with Best RMSE": (
                    "RMSE Rank",
                    lambda values: int(
                        (
                            values
                            == 1
                        ).sum()
                    ),
                ),
                "Mean MAE Rank": (
                    "MAE Rank",
                    "mean",
                ),
                "Mean RMSE Rank": (
                    "RMSE Rank",
                    "mean",
                ),
            }
        )
        .reset_index()
        .sort_values(
            by=[
                "Mean RMSE Rank",
                "Mean MAE Rank",
            ],
            ascending=[
                True,
                True,
            ],
        )
    )

    return summary


# ============================================================
# ERROR ANALYSIS
# ============================================================

def create_error_analysis(
    prediction_data: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    records = []

    for (
        model_name,
        dataframe,
    ) in prediction_data.items():

        for target in EV_TARGETS:

            actual = dataframe[
                f"Actual {target}"
            ].to_numpy(
                dtype=float
            )

            predicted = dataframe[
                f"Predicted {target}"
            ].to_numpy(
                dtype=float
            )

            error = (
                predicted
                - actual
            )

            records.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "Mean Signed Error": float(
                        np.mean(
                            error
                        )
                    ),
                    "Mean Absolute Error": float(
                        np.mean(
                            np.abs(
                                error
                            )
                        )
                    ),
                    "Maximum Absolute Error": float(
                        np.max(
                            np.abs(
                                error
                            )
                        )
                    ),
                    "Overprediction Count": int(
                        np.sum(
                            error
                            > 0
                        )
                    ),
                    "Underprediction Count": int(
                        np.sum(
                            error
                            < 0
                        )
                    ),
                }
            )

    return pd.DataFrame(
        records
    )




def evaluate_by_pair_class(
    prediction_data: dict[
        str,
        pd.DataFrame,
    ],
    baseline_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Evaluate the same OOS predictions separately for held-out
    STRICT and NONSTRICT observations.

    This is especially important for COMBINED modelling so
    pooling does not hide class-specific performance.
    """

    overall_frames = []

    target_frames = []

    for pair_class in [
        "STRICT",
        "NONSTRICT",
    ]:

        baseline_subset = (
            baseline_data[
                baseline_data[
                    "Pair Class"
                ]
                == pair_class
            ]
            .copy()
        )

        if baseline_subset.empty:
            continue

        prediction_subset = {}

        for (
            model_name,
            dataframe,
        ) in prediction_data.items():

            subset = (
                dataframe[
                    dataframe[
                        "Pair Class"
                    ]
                    == pair_class
                ]
                .copy()
            )

            if not subset.empty:
                prediction_subset[
                    model_name
                ] = subset

        if not prediction_subset:
            continue

        (
            overall,
            target,
            _pair,
        ) = evaluate_all_models(
            prediction_subset,
            baseline_subset,
        )

        overall.insert(
            0,
            "Evaluation Pair Class",
            pair_class,
        )

        target.insert(
            0,
            "Evaluation Pair Class",
            pair_class,
        )

        overall_frames.append(
            overall
        )

        target_frames.append(
            target
        )

    overall_output = (
        pd.concat(
            overall_frames,
            ignore_index=True,
        )
        if overall_frames
        else pd.DataFrame()
    )

    target_output = (
        pd.concat(
            target_frames,
            ignore_index=True,
        )
        if target_frames
        else pd.DataFrame()
    )

    return (
        overall_output,
        target_output,
    )



# ============================================================
# SELECTED PRIMARY MODEL
# ============================================================

def save_selected_primary_model(
    *,
    final_models: dict[
        str,
        object,
    ],
    overall_results: pd.DataFrame,
    model_dir: Path,
    scope: str,
) -> None:
    """
    Save the primary 15-output Full-ML model selected from OOS ranking.

    This function saves model/schema metadata only.
    It does NOT calculate charging, SOC, or final EV TAT.
    """

    ranked = (
        overall_results[
            overall_results[
                "Model"
            ]
            != "TrainingMeanBaseline"
        ]
        .dropna(
            subset=[
                "Overall Rank"
            ]
        )
        .sort_values(
            "Overall Rank"
        )
    )

    if ranked.empty:
        raise RuntimeError(
            "No eligible Full-ML model was available for primary selection."
        )

    selected_model_name = str(
        ranked.iloc[0]["Model"]
    )

    if selected_model_name not in final_models:
        raise RuntimeError(
            "Selected Full-ML model was not available in final fitted models: "
            f"{selected_model_name}"
        )

    selected_dir = (
        model_dir
        / "Selected_Primary"
    )

    selected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_models[
            selected_model_name
        ],
        selected_dir
        / "Selected_Full_ML_Physics_Model.pkl",
    )

    metadata = {
        "script_version": SCRIPT_VERSION,
        "dataset_scope": scope,
        "selected_model": selected_model_name,
        "selection_basis": (
            "Mean target-wise baseline-relative RMSE "
            "across all 15 learned EV outputs"
        ),
        "input_columns": ICE_FEATURES,
        "output_columns": EV_TARGETS,
        "trip_level_output_columns": TRIP_LEVEL_EV_TARGETS,
        "phase_time_output_columns": PHASE_TIME_TARGETS,
        "phase_distance_output_columns": PHASE_DISTANCE_TARGETS,
        "charging_calculated_in_full_ml": False,
        "soc_used_in_full_ml": False,
        "final_tat_calculated_in_full_ml": False,
        "pair_class_used_as_predictor": False,
        "pair_class_recalculation": "NONE",
        "pair_class_overwrite": "NONE",
    }

    joblib.dump(
        metadata,
        selected_dir
        / "Selected_Full_ML_Physics_Model_Metadata.pkl",
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    *,
    results_dir: Path,
    scope: str,
    input_files: list[Path],
    validation_report: dict,
    prediction_data: dict[
        str,
        pd.DataFrame,
    ],
    baseline_data: pd.DataFrame,
    status_data: pd.DataFrame,
    coverage_report: pd.DataFrame,
    overall_results: pd.DataFrame,
    target_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    target_ranking: pd.DataFrame,
    error_analysis: pd.DataFrame,
    class_overall_results: pd.DataFrame,
    class_target_results: pd.DataFrame,
) -> None:

    scope = normalize_scope(
        scope
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        results_dir
        / "Full_ML_Primary_Model_Comparison.xlsx"
    )

    overall_results.to_excel(
        overall_path,
        index=False,
    )

    analysis_path = (
        results_dir
        / "Full_ML_Detailed_Evaluation.xlsx"
    )

    input_text = " | ".join(
        str(
            path
        )
        for path in input_files
    )

    manifest = pd.DataFrame(
        [
            {
                "Script Version": (
                    SCRIPT_VERSION
                ),
                "Dataset Scope": (
                    scope
                ),
                "Input Files": (
                    input_text
                ),
                "Rows": (
                    validation_report[
                        "rows"
                    ]
                ),
                "Strict Rows": (
                    validation_report[
                        "strict_rows"
                    ]
                ),
                "Nonstrict Rows": (
                    validation_report[
                        "nonstrict_rows"
                    ]
                ),
                "Unique Pairs": (
                    validation_report[
                        "unique_pairs"
                    ]
                ),
                "Active ICE Predictors": (
                    len(
                        ICE_FEATURES
                    )
                ),
                "Primary EV Targets": (
                    len(
                        EV_TARGETS
                    )
                ),
                "Primary Target Meaning": (
                    "3 trip-level EV outputs + six non-charging "
                    "EV phases × time/distance"
                ),
                "EV Total TAT Role": (
                    "ONE OF THE 15 LEARNED OUTPUTS"
                ),
                "EV Total Distance Role": (
                    "ONE OF THE 15 LEARNED OUTPUTS"
                ),
                "EV Average Moving Speed Role": (
                    "ONE OF THE 15 LEARNED OUTPUTS"
                ),
                "Charging Calculated In Full ML": (
                    "NO"
                ),
                "SOC Used In Full ML": (
                    "NO"
                ),
                "Final Derived Totals Calculated In Full ML": (
                    "NO"
                ),
                "CV Method": (
                    "Leave-One-Pair-Out"
                ),
                "Primary Ranking Metric": (
                    "Mean Baseline-Relative RMSE"
                ),
                "Random State": (
                    RANDOM_STATE
                ),
                "Pair Class Used As Predictor": (
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
                "COMBINED Construction": (
                    "Validated STRICT + validated NONSTRICT "
                    "concatenated in memory"
                    if scope
                    == "COMBINED"
                    else "NOT APPLICABLE"
                ),
                "Cross-Class Pair ID Overlap": (
                    validation_report[
                        "overlapping_pair_ids"
                    ]
                ),
                "Source Data Cleaning": (
                    "NONE"
                ),
                "Source Data Imputation": (
                    "NONE"
                ),
                "Source Data Scaling": (
                    "NONE"
                ),
                "Excluded ICE Predictor 1": (
                    EXCLUDED_CONSTANT_ICE_FEATURES[
                        0
                    ]
                ),
                "Excluded ICE Predictor 2": (
                    EXCLUDED_CONSTANT_ICE_FEATURES[
                        1
                    ]
                ),
                "Exclusion Reason": (
                    "Constant in separately validated "
                    "source dataset(s)"
                ),
            }
        ]
    )

    with pd.ExcelWriter(
        analysis_path,
        engine="openpyxl",
    ) as writer:

        manifest.to_excel(
            writer,
            sheet_name="Run_Manifest",
            index=False,
        )

        overall_results.to_excel(
            writer,
            sheet_name="Overall_OOS_Metrics",
            index=False,
        )

        target_results.to_excel(
            writer,
            sheet_name="Target_Wise_OOS",
            index=False,
        )

        pair_results.to_excel(
            writer,
            sheet_name="Pair_Wise_OOS",
            index=False,
        )

        target_ranking.to_excel(
            writer,
            sheet_name="Target_Ranking_Summary",
            index=False,
        )

        error_analysis.to_excel(
            writer,
            sheet_name="Signed_Error_Analysis",
            index=False,
        )

        status_data.to_excel(
            writer,
            sheet_name="Fold_Model_Status",
            index=False,
        )

        coverage_report.to_excel(
            writer,
            sheet_name="OOS_Coverage",
            index=False,
        )

        baseline_data.to_excel(
            writer,
            sheet_name="Baseline_Predictions",
            index=False,
        )

        if not class_overall_results.empty:

            class_overall_results.to_excel(
                writer,
                sheet_name="Class_Overall_OOS",
                index=False,
            )

        if not class_target_results.empty:

            class_target_results.to_excel(
                writer,
                sheet_name="Class_Target_OOS",
                index=False,
            )

        pd.DataFrame(
            {
                "Active ICE Predictor": (
                    ICE_FEATURES
                )
            }
        ).to_excel(
            writer,
            sheet_name="Active_ICE_Features",
            index=False,
        )

        pd.DataFrame(
            {
                "EV Target": (
                    EV_TARGETS
                )
            }
        ).to_excel(
            writer,
            sheet_name="EV_Targets",
            index=False,
        )

    per_target_path = (
        results_dir
        / "Full_ML_Primary_Per_Target_Metrics.xlsx"
    )

    target_results.to_excel(
        per_target_path,
        index=False,
    )

    prediction_path = (
        results_dir
        / "Full_ML_Held_Out_15_Target_Predictions.xlsx"
    )

    with pd.ExcelWriter(
        prediction_path,
        engine="openpyxl",
    ) as writer:

        baseline_data.to_excel(
            writer,
            sheet_name="TrainingMeanBaseline",
            index=False,
        )

        for (
            model_name,
            dataframe,
        ) in prediction_data.items():

            dataframe.to_excel(
                writer,
                sheet_name=model_name[
                    :31
                ],
                index=False,
            )

    print(
        "\nSaved model comparison:\n"
        f"{overall_path}"
    )

    print(
        "\nSaved detailed evaluation:\n"
        f"{analysis_path}"
    )

    print(
        "\nSaved per-target metrics:\n"
        f"{per_target_path}"
    )

    print(
        "\nSaved held-out predictions:\n"
        f"{prediction_path}"
    )

# ============================================================
# FINAL MODEL FIT
# ============================================================

def fit_final_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    scope: str,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
]:
    """
    Fit candidate models on all data in the selected scope.

    These fits are not used to calculate LOPO evaluation
    metrics. Warnings are captured and reported.
    """

    scope = normalize_scope(
        scope
    )

    models = build_candidate_models(
        n_training_samples=len(
            X
        )
    )

    final_models = {}

    status_records = []

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL MODEL FIT ON ALL "
        f"{scope} PAIRS"
    )

    print(
        "=" * 75
    )

    for (
        model_name,
        model,
    ) in models.items():

        status = {
            "Dataset Scope": (
                scope
            ),
            "Model": (
                model_name
            ),
            "Status": (
                "FAILED"
            ),
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
                    X,
                    Y,
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

            status["Status"] = (
                "SUCCESS_WITH_WARNINGS"
                if caught
                else "SUCCESS"
            )

            final_models[
                model_name
            ] = model

            print(
                f"{model_name:<20} "
                f"{status['Status']}"
            )

            if warning_text:
                print(
                    "    Warning: "
                    f"{warning_text}"
                )

        except Exception as error:

            status[
                "Error"
            ] = str(
                error
            )

            print(
                f"{model_name:<20} "
                "FAILED"
            )

            print(
                "    Reason: "
                f"{error}"
            )

        status_records.append(
            status
        )

    return (
        final_models,
        pd.DataFrame(
            status_records
        ),
    )

# ============================================================
# SAVE FINAL MODELS
# ============================================================

def save_final_models(
    *,
    final_models: dict[
        str,
        object,
    ],
    final_status: pd.DataFrame,
    model_dir: Path,
    scope: str,
    input_files: list[Path],
) -> None:

    scope = normalize_scope(
        scope
    )

    primary_model_dir = (
        model_dir
        / "Primary_Physics"
    )

    primary_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for (
        model_name,
        model,
    ) in final_models.items():

        model_path = (
            primary_model_dir
            / f"{model_name}_Full_15_Targets.pkl"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            "Saved final model: "
            f"{model_path.name}"
        )

    final_status.to_excel(
        primary_model_dir
        / "Final_Primary_Model_Fit_Status.xlsx",
        index=False,
    )

    manifest = pd.DataFrame(
        [
            {
                "Script Version": (
                    SCRIPT_VERSION
                ),
                "Dataset Scope": (
                    scope
                ),
                "Input Files": (
                    " | ".join(
                        str(
                            path
                        )
                        for path
                        in input_files
                    )
                ),
                "Active ICE Predictors": (
                    len(
                        ICE_FEATURES
                    )
                ),
                "Primary EV Targets": (
                    len(
                        EV_TARGETS
                    )
                ),
                "Primary Target Meaning": (
                    "3 trip-level EV outputs + six non-charging "
                    "EV phases × time/distance"
                ),
                "EV Total TAT Role": (
                    "ONE OF THE 15 LEARNED OUTPUTS"
                ),
                "Charging Calculated In Full ML": (
                    "NO"
                ),
                "SOC Used In Full ML": (
                    "NO"
                ),
                "Final Derived Totals Calculated In Full ML": (
                    "NO"
                ),
                "Pair Class Used As Predictor": (
                    "NO"
                ),
                "Random State": (
                    RANDOM_STATE
                ),
            }
        ]
    )

    manifest.to_excel(
        model_dir
        / "Model_Manifest.xlsx",
        index=False,
    )

# ============================================================
# RUN ONE DATASET SCOPE
# ============================================================

def run_full_ml(
    scope: str = "STRICT",
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
        + "=" * 78
    )

    print(
        "TAT MS INTERCITY"
    )

    print(
        "FULL ML — 15 ICE INPUTS -> 15 EV OUTPUTS"
    )

    print(
        "=" * 78
    )

    print(
        f"Version       : {SCRIPT_VERSION}"
    )

    print(
        f"Dataset Scope : {scope}"
    )

    (
        dataframe,
        input_files,
        validation_report,
    ) = load_dataset_scope(
        scope
    )

    print(
        "\nTraining data loaded and validated."
    )

    for path in input_files:
        print(
            f"Input   : {path}"
        )

    print(
        f"Rows    : {len(dataframe)}"
    )

    print(
        f"Columns : {len(dataframe.columns)}"
    )

    print(
        "STRICT rows    : "
        f"{validation_report['strict_rows']}"
    )

    print(
        "NONSTRICT rows : "
        f"{validation_report['nonstrict_rows']}"
    )

    print(
        "\nSTRICT/NONSTRICT handling       : "
        "READ / VALIDATE / PRESERVE ONLY"
    )

    print(
        "STRICT/NONSTRICT recalculation  : NONE"
    )

    print(
        "STRICT/NONSTRICT overwrite      : NONE"
    )

    print(
        "Pair Class used as ML predictor : NO"
    )

    if scope == "COMBINED":

        print(
            "\nCOMBINED construction:"
        )

        print(
            "  STRICT validated separately   : PASS"
        )

        print(
            "  NONSTRICT validated separately: PASS"
        )

        print(
            "  Cross-class Pair ID overlap   : 0"
        )

        print(
            "  Concatenation                  : IN MEMORY ONLY"
        )

    print(
        "\nFull-ML model contract:"
    )

    print(
        f"  ICE inputs   : {len(ICE_FEATURES)}"
    )

    print(
        "  EV outputs   : "
        "15 = Total Distance + Total TAT + Average Moving Speed "
        "+ 6 phase times + 6 phase distances"
    )

    print(
        "  Charging     : NOT calculated here"
    )

    print(
        "  SOC          : NOT used here"
    )

    print(
        "  Final totals : NOT reconstructed here"
    )

    print(
        "\nExcluded constant ICE predictors:"
    )

    for source_class, source_report in (
        validation_report[
            "source_reports"
        ].items()
    ):

        print(
            f"  Source {source_class}:"
        )

        for column in (
            EXCLUDED_CONSTANT_ICE_FEATURES
        ):

            report = (
                source_report[
                    "excluded_constant_ice"
                ][
                    column
                ]
            )

            print(
                f"    - {column}"
                f" | unique={report['unique_values']}"
                f" | constant={report['constant']}"
            )

    print(
        "\nThese columns remain in the source workbook(s) "
        "and are excluded only from X."
    )

    X, Y, groups = (
        prepare_full_model_data(
            dataframe
        )
    )

    row_pair_classes = (
        dataframe[
            PAIR_CLASS_COLUMN
        ]
        .astype("string")
        .str.strip()
        .str.upper()
        .copy()
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FULL MODEL DATA"
    )

    print(
        "=" * 78
    )

    print(
        f"Unique Pair IDs : {groups.nunique()}"
    )

    print(
        f"ICE input shape : {X.shape}"
    )

    print(
        f"EV target shape : {Y.shape}"
    )

    (
        prediction_records,
        baseline_records,
        status_records,
    ) = run_lopo_cv(
        X,
        Y,
        groups,
        row_pair_classes,
        scope,
    )

    all_prediction_data = (
        combine_prediction_records(
            prediction_records
        )
    )

    (
        prediction_data,
        coverage_report,
    ) = filter_complete_prediction_sets(
        all_prediction_data,
        expected_rows=(
            len(
                dataframe
            )
        ),
        expected_pairs=(
            groups.nunique()
        ),
    )

    if not prediction_data:
        raise RuntimeError(
            "No Full-ML model produced complete "
            "held-out predictions for every pair."
        )

    incomplete_models = (
        coverage_report[
            ~coverage_report[
                "Complete OOS Coverage"
            ]
        ][
            "Model"
        ]
        .tolist()
    )

    if incomplete_models:

        print(
            "\nModels excluded from ranking because "
            "held-out coverage is incomplete:"
        )

        for model_name in incomplete_models:
            print(
                f"  - {model_name}"
            )

    baseline_data = pd.DataFrame(
        baseline_records
    )

    status_data = pd.DataFrame(
        status_records
    )

    (
        overall_results,
        target_results,
        pair_results,
    ) = evaluate_all_models(
        prediction_data,
        baseline_data,
    )

    target_ranking = (
        create_target_ranking(
            target_results
        )
    )

    error_analysis = (
        create_error_analysis(
            prediction_data
        )
    )

    (
        class_overall_results,
        class_target_results,
    ) = evaluate_by_pair_class(
        prediction_data,
        baseline_data,
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FULL-ML OOS COMPARISON — 15 EV OUTPUTS"
    )

    print(
        "=" * 78
    )

    display_columns = [
        "Model",
        "Mean Baseline-Relative RMSE",
        "Median Baseline-Relative RMSE",
        "Targets Better Than Baseline",
        "Targets Evaluated",
        "Overall Rank",
        "Better Than Baseline",
    ]

    print(
        overall_results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nNOTE:"
    )

    print(
        "Primary ranking is based on all 15 learned EV outputs: "
        "3 trip-level outputs + 12 phase outputs."
    )

    print(
        "Raw pooled mixed-unit metrics remain diagnostic only."
    )

    save_results(
        results_dir=(
            results_dir
        ),
        scope=(
            scope
        ),
        input_files=(
            input_files
        ),
        validation_report=(
            validation_report
        ),
        prediction_data=(
            prediction_data
        ),
        baseline_data=(
            baseline_data
        ),
        status_data=(
            status_data
        ),
        coverage_report=(
            coverage_report
        ),
        overall_results=(
            overall_results
        ),
        target_results=(
            target_results
        ),
        pair_results=(
            pair_results
        ),
        target_ranking=(
            target_ranking
        ),
        error_analysis=(
            error_analysis
        ),
        class_overall_results=(
            class_overall_results
        ),
        class_target_results=(
            class_target_results
        ),
    )

    (
        final_models,
        final_status,
    ) = fit_final_models(
        X,
        Y,
        scope,
    )

    save_final_models(
        final_models=(
            final_models
        ),
        final_status=(
            final_status
        ),
        model_dir=(
            model_dir
        ),
        scope=(
            scope
        ),
        input_files=(
            input_files
        ),
    )

    save_selected_primary_model(
        final_models=(
            final_models
        ),
        overall_results=(
            overall_results
        ),
        model_dir=(
            model_dir
        ),
        scope=(
            scope
        ),
    )

    successful_statuses = {
        "SUCCESS",
        "SUCCESS_WITH_WARNINGS",
    }

    successful_models = (
        status_data[
            status_data[
                "Status"
            ].isin(
                successful_statuses
            )
        ][
            "Model"
        ]
        .nunique()
        if not status_data.empty
        else 0
    )

    total_models = len(
        build_candidate_models(
            len(
                groups
            )
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FULL ML PIPELINE COMPLETED"
    )

    print(
        "=" * 78
    )

    print(
        f"Dataset Scope       : {scope}"
    )

    print(
        f"LOPO-CV pairs       : {groups.nunique()}"
    )

    print(
        f"Candidate ML models : {total_models}"
    )

    print(
        f"Model inputs        : {len(ICE_FEATURES)}"
    )

    print(
        f"Model outputs       : {len(EV_TARGETS)}"
    )

    print(
        "Models with successful folds : "
        f"{successful_models}"
    )

    print(
        "Models eligible for ranking  : "
        f"{len(prediction_data)}"
    )

    print(
        "\nFull ML ends here:"
    )

    print(
        "  - no charging calculation"
    )

    print(
        "  - no SOC assumption"
    )

    print(
        "  - no final TAT/distance/speed reconstruction"
    )

    print(
        "Downstream new-ICE inference will use model_prediction.py, "
        "charging_time.py, and final_tat_assembly.py."
    )

    print(
        "\nPair Class was read, validated, and preserved only."
    )

    print(
        "Pair Class was not recalculated, overwritten, "
        "or used as an ML predictor."
    )

    print(
        f"\nResults:\n{results_dir}"
    )

    print(
        f"\nModels:\n{model_dir}"
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Full ML modelling for paired Intercity EV-ICE data. "
            "Trains/evaluates 15 ICE inputs -> 15 EV outputs only."
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
        help=(
            "Dataset scope. Default ALL runs STRICT, "
            "NONSTRICT, then COMBINED."
        ),
    )

    parser.add_argument(
        "--pair-class",
        choices=[
            "STRICT",
            "NONSTRICT",
            "COMBINED",
        ],
        default=None,
        help=(
            "Backward-compatible single-scope alias. "
            "If supplied, it overrides --scope."
        ),
    )

    return parser.parse_args()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    arguments = parse_arguments()

    requested_scope = (
        arguments.pair_class
        if arguments.pair_class
        is not None
        else arguments.scope
    )

    requested_scope = normalize_scope(
        requested_scope,
        allow_all=True,
    )

    if requested_scope == "ALL":

        scopes_to_run = [
            "STRICT",
            "NONSTRICT",
            "COMBINED",
        ]

    else:

        scopes_to_run = [
            requested_scope
        ]

    for scope_name in scopes_to_run:

        run_full_ml(
            scope=(
                scope_name
            ),
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ALL REQUESTED FULL ML RUNS COMPLETED"
    )

    print(
        "=" * 78
    )
