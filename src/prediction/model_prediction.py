"""
TAT MS INTERCITY MODELLING
NEW RAW ICE -> TRAINED MODEL PREDICTION ADAPTER

Version:
    2026-09-30_MODEL_PREDICTION_ADAPTER_V1

ROLE IN THE FINAL PIPELINE
--------------------------
This is the model-testing / inference layer for a NEW raw ICE trip.

Upstream:
    Raw ICE workbook
        -> src/ice_raw_data/ice_processing.py
        -> Results/Prediction/ICE_17_Parameters_Table.xlsx

This module:
    ICE 17-parameter table
        -> selected trained model(s)
        -> predicted EV parameters
        -> Results/Prediction/Model_Predictions.xlsx

Downstream:
    model predictions
        + common charging_time.py output
        -> final_tat_assembly.py
        -> Calculated EV Total TAT

SUPPORTED MODEL SOURCES
-----------------------
    FULL_ML
    PHASE_ML
    FULL_DL
    PHASE_DL

Convenience selections:
    ML   = FULL_ML + PHASE_ML
    DL   = FULL_DL + PHASE_DL
    ALL  = all four

SUPPORTED TRAINING SCOPES
-------------------------
    STRICT
    NONSTRICT
    COMBINED
    ALL = run the selected model(s) trained under all three scopes

IMPORTANT PAIR-CLASS RULE
-------------------------
A new raw ICE trip is NOT STRICT or NONSTRICT.

For every new-ICE prediction row:
    Pair Class = NOT_APPLICABLE_NEW_ICE_TRIP

Model Scope is separate and records which historical training population
produced the model:
    STRICT / NONSTRICT / COMBINED

This adapter:
    - never derives Pair Class,
    - never recalculates Pair Class,
    - never overwrites historical Pair Class,
    - never uses Pair Class as a predictor.

MODEL-SPECIFIC INFERENCE RULES
------------------------------
FULL_ML
    Loads:
        models/<scope>/ML/Full_Model/Selected_Primary/
            Selected_Full_ML_Physics_Model.pkl
            Selected_Full_ML_Physics_Model_Metadata.pkl

    Uses the exact input/output schema saved by full_ml.py.

    No new clipping is introduced here because the current Full-ML historical
    evaluation did not declare clipping as its inference post-processing rule.
    Negative predictions are therefore retained and explicitly QC-flagged.

    If the matching direct Total-TAT benchmark model exists, its result is
    retained only as:
        Direct Predicted EV Total TAT Reference (min)

PHASE_ML
    Loads one selected fitted model per phase from:
        models/<scope>/ML/Phase_Wise/<PHASE>/

    Uses the exact saved phase metadata for input and target columns.

    No new clipping is introduced. Negative predictions are retained and
    explicitly QC-flagged.

FULL_DL
    Loads:
        models/<scope>/DL/Full_Model/Selected_Final/
            Full_DNN_seed_*.pt
            x_scaler.joblib
            y_scaler.joblib
            feature_schema.json
            target_schema.json

    Performs the same declared inference rule as historical evaluation:
        seed ensemble in standardized target space
        -> ensemble mean
        -> inverse target scaling
        -> max(raw_prediction, 0)

PHASE_DL
    Loads six phase ensembles from:
        models/<scope>/DL/Phase_Wise/<PHASE>/Selected_Final/

    Performs the same declared phase-DL inference rule:
        seed ensemble
        -> inverse target scaling
        -> max(raw_prediction, 0)

STANDARDIZED OUTPUT CONTRACT
----------------------------
Every output row contains:
    Trip ID
    Source ICE File
    Model Source
    Model Scope
    Pair Class

and at minimum:
    Predicted EV STOP Time (min)
    Predicted EV IDLE Time (min)
    Predicted EV INCHING Time (min)
    Predicted EV ACC Time (min)
    Predicted EV DEC Time (min)
    Predicted EV CRUISE Time (min)

The final TAT assembler consumes those six time columns.

DEFAULT COMMANDS
----------------
Full ML only, all scopes:
    python .\\src\\prediction\\model_prediction.py --models FULL_ML --scope ALL

All ML:
    python .\\src\\prediction\\model_prediction.py --models ML --scope ALL

All DL:
    python .\\src\\prediction\\model_prediction.py --models DL --scope ALL

Everything:
    python .\\src\\prediction\\model_prediction.py --models ALL --scope ALL
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_VERSION = "2026-10-01_MODEL_PREDICTION_V3_FIXED_DL_ARTIFACT_COMPAT"

NEW_ICE_PAIR_CLASS = "NOT_APPLICABLE_NEW_ICE_TRIP"

VALID_MODEL_SOURCES = (
    "FULL_ML",
    "PHASE_ML",
    "FULL_DL",
    "PHASE_DL",
)

VALID_SCOPES = (
    "STRICT",
    "NONSTRICT",
    "COMBINED",
)

PHASES = (
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
)

PHASE_TIME_OUTPUT_COLUMNS = [
    f"Predicted EV {phase} Time (min)"
    for phase in PHASES
]

PHASE_DISTANCE_OUTPUT_COLUMNS = [
    f"Predicted EV {phase} Distance (km)"
    for phase in PHASES
]

# Canonical NEW-ICE preprocessing schema.
# ice_processing.py correctly emits ICE REFUELING for an ICE vehicle.
ALL_ICE_COLUMNS_17 = [
    "ICE Total Distance (km)",
    "ICE Total TAT (min)",
    "ICE Average Moving Speed (km/h)",
    "ICE REFUELING Time (min)",
    "ICE STOP Time (min)",
    "ICE IDLE Time (min)",
    "ICE INCHING Time (min)",
    "ICE ACC Time (min)",
    "ICE DEC Time (min)",
    "ICE CRUISE Time (min)",
    "ICE REFUELING Distance (km)",
    "ICE STOP Distance (km)",
    "ICE IDLE Distance (km)",
    "ICE INCHING Distance (km)",
    "ICE ACC Distance (km)",
    "ICE DEC Distance (km)",
    "ICE CRUISE Distance (km)",
]

# Historical training workbooks used the generic replenishment label
# CHARGING_REFUELING.  The adapter bridges only this naming difference.
# Values are copied exactly; nothing is recalculated.
HISTORICAL_REPLENISHMENT_ALIASES = {
    "ICE CHARGING_REFUELING Time (min)": "ICE REFUELING Time (min)",
    "ICE CHARGING_REFUELING Distance (km)": "ICE REFUELING Distance (km)",
}

DIRECT_TAT_REFERENCE_COLUMN = (
    "Direct Predicted EV Total TAT Reference (min)"
)


FIXED_DL_SELECTION_METHOD = (
    "FIXED_PREDECLARED_ANALYSIS_SPECIFICATION_NOT_HPO_OPTIMIZED"
)

FULL_DL_EXPECTED_TARGETS = [
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

PHASE_DL_EXPECTED_FEATURES = {
    phase: [
        f"ICE {phase} Time (min)",
        f"ICE {phase} Distance (km)",
    ]
    for phase in PHASES
}

PHASE_DL_EXPECTED_TARGETS = {
    phase: [
        f"EV {phase} Time (min)",
        f"EV {phase} Distance (km)",
    ]
    for phase in PHASES
}


# =============================================================================
# GENERIC DL ARCHITECTURE
# =============================================================================

class DenseRegressor(nn.Module):
    """
    Generic fully-connected ReLU/dropout regressor.

    The attribute name `network` deliberately matches both current training
    implementations, so their saved state_dict keys load without translation.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_layers: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous = int(input_dim)

        for width in hidden_layers:
            width = int(width)

            layers.append(
                nn.Linear(
                    previous,
                    width,
                )
            )
            layers.append(
                nn.ReLU()
            )

            if float(dropout) > 0:
                layers.append(
                    nn.Dropout(
                        float(dropout)
                    )
                )

            previous = width

        layers.append(
            nn.Linear(
                previous,
                int(output_dim),
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            x
        )


# =============================================================================
# PATH / SELECTION HELPERS
# =============================================================================

def default_project_root() -> Path:
    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
    )


def default_ice_input(
    project_root: Path,
) -> Path:
    return (
        project_root
        / "Results"
        / "Prediction"
        / "ICE_17_Parameters_Table.xlsx"
    )


def default_output_path(
    project_root: Path,
) -> Path:
    return (
        project_root
        / "Results"
        / "Prediction"
        / "Model_Predictions.xlsx"
    )


def normalize_scope_selection(
    value: str,
) -> list[str]:
    normalized = str(
        value
    ).strip().upper()

    if normalized == "ALL":
        return list(
            VALID_SCOPES
        )

    if normalized not in VALID_SCOPES:
        raise ValueError(
            "Scope must be STRICT, NONSTRICT, COMBINED, or ALL. "
            f"Received: {value!r}"
        )

    return [
        normalized
    ]


def normalize_model_selection(
    value: str,
) -> list[str]:
    normalized = str(
        value
    ).strip().upper()

    mapping = {
        "FULL_ML": [
            "FULL_ML"
        ],
        "PHASE_ML": [
            "PHASE_ML"
        ],
        "ML": [
            "FULL_ML",
            "PHASE_ML",
        ],
        "FULL_DL": [
            "FULL_DL"
        ],
        "PHASE_DL": [
            "PHASE_DL"
        ],
        "DL": [
            "FULL_DL",
            "PHASE_DL",
        ],
        "ALL": list(
            VALID_MODEL_SOURCES
        ),
    }

    if normalized not in mapping:
        raise ValueError(
            "Models must be FULL_ML, PHASE_ML, ML, FULL_DL, "
            "PHASE_DL, DL, or ALL. "
            f"Received: {value!r}"
        )

    return mapping[
        normalized
    ]


def resolve_device(
    requested: str,
) -> torch.device:
    value = str(
        requested
    ).strip().lower()

    if value == "cpu":
        return torch.device(
            "cpu"
        )

    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )

        return torch.device(
            "cuda"
        )

    if value == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    raise ValueError(
        "Device must be cpu, cuda, or auto."
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def _clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(
            value
        ):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(
        value
    ).strip()


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required column(s):\n"
            + "\n".join(
                f" - {column}"
                for column in missing
            )
        )


def _numeric_frame(
    dataframe: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
    require_nonnegative: bool,
) -> pd.DataFrame:
    result = (
        dataframe[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if result.isna().any().any():
        bad = [
            column
            for column in columns
            if result[
                column
            ].isna().any()
        ]

        raise ValueError(
            f"{label} contains missing/non-numeric values in:\n"
            + "\n".join(
                f" - {column}"
                for column in bad
            )
        )

    values = result.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{label} contains non-finite numeric values."
        )

    if (
        require_nonnegative
        and (
            values
            < 0
        ).any()
    ):
        raise ValueError(
            f"{label} contains negative source values."
        )

    return result


def _add_replenishment_schema_aliases(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bridge the only intentional naming difference between:
        new raw ICE preprocessing: ICE REFUELING ...
        historical modelling schema: ICE CHARGING_REFUELING ...

    No value is recalculated, transformed, imputed, or overwritten.

    If both names are present, their numeric values must agree.
    """
    output = dataframe.copy()

    for historical_name, raw_name in (
        HISTORICAL_REPLENISHMENT_ALIASES.items()
    ):
        raw_present = (
            raw_name
            in output.columns
        )

        historical_present = (
            historical_name
            in output.columns
        )

        if (
            not raw_present
            and not historical_present
        ):
            continue

        if (
            raw_present
            and historical_present
        ):
            raw_values = pd.to_numeric(
                output[
                    raw_name
                ],
                errors="coerce",
            )

            historical_values = pd.to_numeric(
                output[
                    historical_name
                ],
                errors="coerce",
            )

            mismatch = (
                raw_values.notna()
                & historical_values.notna()
                & ~np.isclose(
                    raw_values.to_numpy(
                        dtype=float
                    ),
                    historical_values.to_numpy(
                        dtype=float
                    ),
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            )

            if mismatch.any():
                raise ValueError(
                    "Conflicting replenishment columns were found: "
                    f"{raw_name!r} and {historical_name!r}. "
                    "The adapter will not choose between inconsistent values."
                )

        elif raw_present:
            output[
                historical_name
            ] = output[
                raw_name
            ]

        else:
            output[
                raw_name
            ] = output[
                historical_name
            ]

    return output


def load_ice_parameter_table(
    input_path: Path,
    sheet_name: str,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(
            "ICE prediction-input table was not found:\n"
            f"{input_path}\n"
            "Run src\\ice_raw_data\\ice_processing.py first."
        )

    dataframe = pd.read_excel(
        input_path,
        sheet_name=(
            sheet_name
        ),
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError(
            "ICE prediction-input table contains no trips."
        )

    dataframe.columns = [
        str(
            column
        ).strip()
        for column in dataframe.columns
    ]

    # New ICE preprocessing uses REFUELING; historical model metadata can
    # still contain CHARGING_REFUELING.  Preserve values and expose both
    # schema names so inference remains compatible with the trained models.
    dataframe = (
        _add_replenishment_schema_aliases(
            dataframe
        )
    )

    _require_columns(
        dataframe,
        [
            "Trip ID",
            "Source ICE File",
            *ALL_ICE_COLUMNS_17,
        ],
        label=(
            "ICE prediction-input table"
        ),
    )

    dataframe[
        "Trip ID"
    ] = (
        dataframe[
            "Trip ID"
        ]
        .map(
            _clean_text
        )
    )

    dataframe[
        "Source ICE File"
    ] = (
        dataframe[
            "Source ICE File"
        ]
        .map(
            _clean_text
        )
    )

    if (
        dataframe[
            "Trip ID"
        ]
        .eq(
            ""
        )
        .any()
    ):
        raise ValueError(
            "ICE prediction-input table contains blank Trip ID values."
        )

    if (
        dataframe[
            "Trip ID"
        ]
        .duplicated()
        .any()
    ):
        duplicates = (
            dataframe.loc[
                dataframe[
                    "Trip ID"
                ].duplicated(
                    keep=False
                ),
                "Trip ID",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "ICE prediction-input table contains duplicate Trip IDs:\n"
            + "\n".join(
                f" - {item}"
                for item in duplicates
            )
        )

    numeric = _numeric_frame(
        dataframe,
        ALL_ICE_COLUMNS_17,
        label=(
            "ICE prediction-input table"
        ),
        require_nonnegative=True,
    )

    dataframe.loc[
        :,
        ALL_ICE_COLUMNS_17,
    ] = numeric

    return dataframe


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def _base_output(
    ice: pd.DataFrame,
    *,
    model_source: str,
    scope: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Trip ID": (
                ice[
                    "Trip ID"
                ].astype(
                    str
                )
            ),
            "Source ICE File": (
                ice[
                    "Source ICE File"
                ].astype(
                    str
                )
            ),
            "Model Source": (
                model_source
            ),
            "Model Scope": (
                scope
            ),
            "Pair Class": (
                NEW_ICE_PAIR_CLASS
            ),
        }
    )


def _standard_prediction_column(
    target: str,
) -> str:
    target = str(
        target
    ).strip()

    if target.startswith(
        "Predicted "
    ):
        return target

    return (
        f"Predicted {target}"
    )


def _ensure_prediction_shape(
    prediction: np.ndarray,
    *,
    rows: int,
    outputs: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(
        prediction,
        dtype=float,
    )

    if array.ndim == 1:
        if (
            rows == 1
            and len(
                array
            )
            == outputs
        ):
            array = array.reshape(
                1,
                outputs,
            )
        elif (
            outputs == 1
            and len(
                array
            )
            == rows
        ):
            array = array.reshape(
                rows,
                1,
            )

    if array.shape != (
        rows,
        outputs,
    ):
        raise ValueError(
            f"{label} produced shape {array.shape}; "
            f"expected {(rows, outputs)}."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"{label} produced non-finite predictions."
        )

    return array


def _load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file was not found:\n{path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _load_torch_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"DL checkpoint was not found:\n{path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=(
                device
            ),
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=(
                device
            ),
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            f"Checkpoint is not a dictionary:\n{path}"
        )

    if "state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain state_dict:\n{path}"
        )

    return checkpoint


def _prediction_qc_for_rows(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for _, row in predictions.iterrows():
        missing_columns = [
            column
            for column in PHASE_TIME_OUTPUT_COLUMNS
            if column not in predictions.columns
        ]

        if missing_columns:
            phase_values = np.asarray(
                [],
                dtype=float,
            )
            finite = False
            nonnegative = False
            negative_count = np.nan
            status = (
                "FAIL_MISSING_PHASE_TIME_COLUMNS"
            )

        else:
            phase_values = np.asarray(
                [
                    pd.to_numeric(
                        row[
                            column
                        ],
                        errors="coerce",
                    )
                    for column in PHASE_TIME_OUTPUT_COLUMNS
                ],
                dtype=float,
            )

            finite = bool(
                np.isfinite(
                    phase_values
                ).all()
            )

            nonnegative = bool(
                finite
                and (
                    phase_values
                    >= 0
                ).all()
            )

            negative_count = int(
                (
                    phase_values
                    < 0
                ).sum()
            ) if finite else np.nan

            if not finite:
                status = (
                    "FAIL_NONFINITE_PHASE_TIME"
                )

            elif not nonnegative:
                status = (
                    "FAIL_NEGATIVE_PHASE_TIME"
                )

            else:
                status = (
                    "PASS"
                )

        records.append(
            {
                "Trip ID": (
                    row[
                        "Trip ID"
                    ]
                ),
                "Model Source": (
                    row[
                        "Model Source"
                    ]
                ),
                "Model Scope": (
                    row[
                        "Model Scope"
                    ]
                ),
                "Pair Class": (
                    row[
                        "Pair Class"
                    ]
                ),
                "Six Required Phase Times Present": (
                    not bool(
                        missing_columns
                    )
                ),
                "Six Phase Times Finite": (
                    finite
                ),
                "Six Phase Times Non-Negative": (
                    nonnegative
                ),
                "Negative Phase-Time Count": (
                    negative_count
                ),
                "Direct TAT Used For Final TAT": (
                    False
                ),
                "Prediction Physical QC Status": (
                    status
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =============================================================================
# FULL ML ADAPTER
# =============================================================================

def predict_full_ml(
    *,
    ice: pd.DataFrame,
    project_root: Path,
    scope: str,
) -> pd.DataFrame:
    full_model_dir = (
        project_root
        / "models"
        / scope
        / "ML"
        / "Full_Model"
    )

    selected_dir = (
        full_model_dir
        / "Selected_Primary"
    )

    model_path = (
        selected_dir
        / "Selected_Full_ML_Physics_Model.pkl"
    )

    metadata_path = (
        selected_dir
        / "Selected_Full_ML_Physics_Model_Metadata.pkl"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Selected Full-ML physics model was not found:\n"
            f"{model_path}\n"
            f"Run Full ML training for scope {scope} first."
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Selected Full-ML metadata was not found:\n"
            f"{metadata_path}"
        )

    model = joblib.load(
        model_path
    )

    metadata = joblib.load(
        metadata_path
    )

    if not isinstance(
        metadata,
        dict,
    ):
        raise ValueError(
            "Selected Full-ML metadata is not a dictionary."
        )

    input_columns = [
        str(
            item
        )
        for item in metadata.get(
            "input_columns",
            [],
        )
    ]

    output_columns = [
        str(
            item
        )
        for item in metadata.get(
            "output_columns",
            [],
        )
    ]

    if len(
        input_columns
    ) != 15:
        raise ValueError(
            "Full-ML artifact must contain exactly 15 active ICE inputs. "
            f"Observed {len(input_columns)}."
        )

    if len(
        output_columns
    ) != 15:
        raise ValueError(
            "Full-ML artifact must contain exactly 15 EV outputs. "
            f"Observed {len(output_columns)}. Retrain Full ML with V8+."
        )

    missing_full_outputs = [
        target
        for target in FULL_DL_EXPECTED_TARGETS
        if target not in output_columns
    ]

    if missing_full_outputs:
        raise ValueError(
            "Full-ML artifact is missing frozen 15-output targets:\n"
            + "\n".join(
                f" - {target}"
                for target in missing_full_outputs
            )
        )

    _require_columns(
        ice,
        input_columns,
        label=(
            "Full-ML ICE input"
        ),
    )

    X = _numeric_frame(
        ice,
        input_columns,
        label=(
            "Full-ML model input"
        ),
        require_nonnegative=True,
    )

    raw = _ensure_prediction_shape(
        model.predict(
            X
        ),
        rows=len(
            ice
        ),
        outputs=len(
            output_columns
        ),
        label=(
            f"FULL_ML {scope}"
        ),
    )

    output = _base_output(
        ice,
        model_source=(
            "FULL_ML"
        ),
        scope=(
            scope
        ),
    )

    for index, target in enumerate(
        output_columns
    ):
        output[
            _standard_prediction_column(
                target
            )
        ] = raw[
            :,
            index
        ]

    output[
        "Adapter Postprocessing"
    ] = (
        "NONE; preserve Full-ML inference scale"
    )

    output[
        "Selected Model"
    ] = str(
        metadata.get(
            "selected_model",
            "",
        )
    )

    output[
        "Training Script Version"
    ] = str(
        metadata.get(
            "script_version",
            "",
        )
    )

    # In the frozen 15-output architecture, EV Total TAT is a direct model
    # output. There is no separate Direct_TAT_Benchmark inference model.
    output[
        DIRECT_TAT_REFERENCE_COLUMN
    ] = output[
        "Predicted EV Total TAT (min)"
    ]

    output[
        "Direct TAT Reference Status"
    ] = (
        "AVAILABLE_FROM_15_OUTPUT_FULL_ML_MODEL"
    )

    output[
        "Artifact Compatibility Status"
    ] = (
        "PASS_FULL_ML_15X15_METADATA_CONTRACT"
    )

    output[
        "Training Hyperparameter Interpretation"
    ] = (
        "SCIKIT-LEARN MODEL FAMILY; SEE SAVED FULL-ML METADATA"
    )

    return output



# =============================================================================
# PHASE ML ADAPTER
# =============================================================================

def predict_phase_ml(
    *,
    ice: pd.DataFrame,
    project_root: Path,
    scope: str,
) -> pd.DataFrame:
    model_dir = (
        project_root
        / "models"
        / scope
        / "ML"
        / "Phase_Wise"
    )

    output = _base_output(
        ice,
        model_source=(
            "PHASE_ML"
        ),
        scope=(
            scope
        ),
    )

    selected_models: list[
        str
    ] = []

    training_versions: list[
        str
    ] = []

    for phase in PHASES:
        phase_dir = (
            model_dir
            / phase
        )

        metadata_path = (
            phase_dir
            / f"{phase}_metadata.pkl"
        )

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Phase-ML metadata was not found for {phase}:\n"
                f"{metadata_path}\n"
                f"Run Phase-wise ML training for scope {scope} first."
            )

        metadata = joblib.load(
            metadata_path
        )

        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(
                f"{phase} Phase-ML metadata is not a dictionary."
            )

        model_name = _clean_text(
            metadata.get(
                "model"
            )
        )

        input_columns = [
            str(
                item
            )
            for item in metadata.get(
                "input_columns",
                [],
            )
        ]

        target_columns = [
            str(
                item
            )
            for item in metadata.get(
                "target_columns",
                [],
            )
        ]

        if not model_name:
            raise ValueError(
                f"{phase} Phase-ML metadata has no selected model name."
            )

        if len(
            input_columns
        ) != 2:
            raise ValueError(
                f"{phase} Phase-ML metadata must contain exactly 2 inputs."
            )

        if len(
            target_columns
        ) != 2:
            raise ValueError(
                f"{phase} Phase-ML metadata must contain exactly 2 targets."
            )

        model_path = (
            phase_dir
            / (
                f"{model_name}_"
                f"{phase}_Phase.pkl"
            )
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Selected Phase-ML model was not found:\n{model_path}"
            )

        _require_columns(
            ice,
            input_columns,
            label=(
                f"{phase} Phase-ML input"
            ),
        )

        X = _numeric_frame(
            ice,
            input_columns,
            label=(
                f"{phase} Phase-ML model input"
            ),
            require_nonnegative=True,
        )

        model = joblib.load(
            model_path
        )

        prediction = _ensure_prediction_shape(
            model.predict(
                X
            ),
            rows=len(
                ice
            ),
            outputs=2,
            label=(
                f"PHASE_ML {scope} {phase}"
            ),
        )

        for index, target in enumerate(
            target_columns
        ):
            output[
                _standard_prediction_column(
                    target
                )
            ] = prediction[
                :,
                index
            ]

        selected_models.append(
            f"{phase}:{model_name}"
        )

        version = _clean_text(
            metadata.get(
                "script_version"
            )
        )

        if version:
            training_versions.append(
                version
            )

    output[
        "Adapter Postprocessing"
    ] = (
        "NONE; preserve current Phase-ML inference scale"
    )

    output[
        "Selected Model"
    ] = (
        " | ".join(
            selected_models
        )
    )

    output[
        "Training Script Version"
    ] = (
        " | ".join(
            sorted(
                set(
                    training_versions
                )
            )
        )
    )

    output[
        DIRECT_TAT_REFERENCE_COLUMN
    ] = np.nan

    output[
        "Direct TAT Reference Status"
    ] = (
        "NOT_APPLICABLE_FOR_PHASE_ML"
    )

    output[
        "Artifact Compatibility Status"
    ] = (
        "PASS_PHASE_ML_2X2_METADATA_CONTRACT"
    )

    output[
        "Training Hyperparameter Interpretation"
    ] = (
        "SCIKIT-LEARN MODEL FAMILY; SEE SAVED PHASE-ML METADATA"
    )

    return output


# =============================================================================
# DL ENSEMBLE CORE
# =============================================================================

@dataclass(frozen=True)
class DLEnsemblePrediction:
    raw_mean: np.ndarray
    postprocessed_mean: np.ndarray
    ensemble_sd: np.ndarray
    seeds: tuple[int, ...]


def _predict_dl_ensemble(
    *,
    X: pd.DataFrame,
    x_scaler: Any,
    y_scaler: Any,
    checkpoint_paths: list[Path],
    input_count: int,
    output_count: int,
    device: torch.device,
) -> DLEnsemblePrediction:
    X_values = X.to_numpy(
        dtype=float
    )

    X_scaled = x_scaler.transform(
        X_values
    )

    X_tensor = torch.as_tensor(
        X_scaled,
        dtype=torch.float32,
        device=(
            device
        ),
    )

    seed_predictions_scaled: list[
        np.ndarray
    ] = []

    seeds: list[
        int
    ] = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = (
            _load_torch_checkpoint(
                checkpoint_path,
                device,
            )
        )

        hidden_layers = tuple(
            int(
                item
            )
            for item in checkpoint.get(
                "hidden_layers",
                [],
            )
        )

        if not hidden_layers:
            raise ValueError(
                "Checkpoint contains no hidden_layers:\n"
                f"{checkpoint_path}"
            )

        dropout = float(
            checkpoint.get(
                "dropout",
                0.0,
            )
        )

        model = DenseRegressor(
            input_dim=(
                input_count
            ),
            output_dim=(
                output_count
            ),
            hidden_layers=(
                hidden_layers
            ),
            dropout=(
                dropout
            ),
        ).to(
            device
        )

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
        )

        model.eval()

        with torch.no_grad():
            prediction_scaled = (
                model(
                    X_tensor
                )
                .detach()
                .cpu()
                .numpy()
            )

        seed_predictions_scaled.append(
            prediction_scaled
        )

        seeds.append(
            int(
                checkpoint.get(
                    "seed",
                    len(
                        seeds
                    ),
                )
            )
        )

    if not seed_predictions_scaled:
        raise RuntimeError(
            "No DL checkpoints were available for ensemble inference."
        )

    stacked_scaled = np.stack(
        seed_predictions_scaled,
        axis=0,
    )

    mean_scaled = np.mean(
        stacked_scaled,
        axis=0,
    )

    raw_mean = y_scaler.inverse_transform(
        mean_scaled
    )

    postprocessed_mean = np.maximum(
        raw_mean,
        0.0,
    )

    seed_original = np.stack(
        [
            y_scaler.inverse_transform(
                item
            )
            for item in (
                seed_predictions_scaled
            )
        ],
        axis=0,
    )

    ensemble_sd = np.std(
        seed_original,
        axis=0,
        ddof=0,
    )

    if not np.isfinite(
        raw_mean
    ).all():
        raise ValueError(
            "DL ensemble produced non-finite raw predictions."
        )

    if not np.isfinite(
        postprocessed_mean
    ).all():
        raise ValueError(
            "DL ensemble produced non-finite postprocessed predictions."
        )

    if not np.isfinite(
        ensemble_sd
    ).all():
        raise ValueError(
            "DL ensemble produced non-finite ensemble SD values."
        )

    return DLEnsemblePrediction(
        raw_mean=(
            raw_mean
        ),
        postprocessed_mean=(
            postprocessed_mean
        ),
        ensemble_sd=(
            ensemble_sd
        ),
        seeds=tuple(
            seeds
        ),
    )



def _validate_fixed_dl_artifact_bundle(
    *,
    selected_dir: Path,
    input_features: list[str],
    target_features: list[str],
    checkpoint_paths: list[Path],
    x_scaler: Any,
    y_scaler: Any,
    device: torch.device,
    expected_model_source: str,
    expected_scope: str,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    """
    Validate that inference artifacts exactly match the revised fixed-DL
    training contract before any prediction is accepted.

    This deliberately rejects stale nested-HPO artifacts, mixed seed bundles,
    schema/checkpoint mismatches, and incompatible scaler dimensions.
    """
    hyperparameters_path = (
        selected_dir
        / "hyperparameters.json"
    )

    hyperparameters = _load_json(
        hyperparameters_path
    )

    selection_method = _clean_text(
        hyperparameters.get(
            "selection_method"
        )
    )

    if selection_method != FIXED_DL_SELECTION_METHOD:
        raise ValueError(
            "DL artifact bundle is not from the revised fixed/predeclared "
            "training pipeline.\n"
            f"Expected selection_method={FIXED_DL_SELECTION_METHOD!r}\n"
            f"Observed={selection_method!r}\n"
            f"Artifact folder: {selected_dir}\n"
            "Retrain this scope with the revised DL code before inference."
        )

    if bool(
        hyperparameters.get(
            "hpo_used",
            False,
        )
    ):
        raise ValueError(
            "DL hyperparameters.json reports hpo_used=True; "
            "stale HPO artifacts are not accepted."
        )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No DL checkpoints found under:\n{selected_dir}"
        )

    if hasattr(
        x_scaler,
        "n_features_in_",
    ):
        if int(
            x_scaler.n_features_in_
        ) != len(
            input_features
        ):
            raise ValueError(
                "DL x_scaler dimensionality does not match feature schema."
            )

    if hasattr(
        y_scaler,
        "n_features_in_",
    ):
        if int(
            y_scaler.n_features_in_
        ) != len(
            target_features
        ):
            raise ValueError(
                "DL y_scaler dimensionality does not match target schema."
            )

    expected_seeds = tuple(
        int(
            item
        )
        for item in hyperparameters.get(
            "ensemble_seeds",
            [],
        )
    )

    observed_seeds: list[int] = []
    observed_versions: list[str] = []
    observed_architectures: list[tuple[int, ...]] = []
    observed_dropouts: list[float] = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = _load_torch_checkpoint(
            checkpoint_path,
            device,
        )

        for required_key in (
            "state_dict",
            "hidden_layers",
            "dropout",
            "seed",
            "input_features",
            "target_features",
        ):
            if required_key not in checkpoint:
                raise ValueError(
                    f"Checkpoint missing {required_key!r}:\n"
                    f"{checkpoint_path}"
                )

        checkpoint_source = _clean_text(
            checkpoint.get(
                "model_source"
            )
        )

        if (
            checkpoint_source
            and checkpoint_source
            != expected_model_source
        ):
            raise ValueError(
                "DL checkpoint model_source mismatch:\n"
                f"{checkpoint_path}\n"
                f"Expected {expected_model_source}; "
                f"observed {checkpoint_source}."
            )

        checkpoint_scope = _clean_text(
            checkpoint.get(
                "dataset_scope"
            )
        )

        if (
            checkpoint_scope
            and checkpoint_scope
            != expected_scope
        ):
            raise ValueError(
                "DL checkpoint dataset_scope mismatch:\n"
                f"{checkpoint_path}\n"
                f"Expected {expected_scope}; observed {checkpoint_scope}."
            )

        checkpoint_phase = _clean_text(
            checkpoint.get(
                "phase"
            )
        )

        if (
            expected_phase is not None
            and checkpoint_phase
            and checkpoint_phase
            != expected_phase
        ):
            raise ValueError(
                "Phase-DL checkpoint phase mismatch:\n"
                f"{checkpoint_path}\n"
                f"Expected {expected_phase}; observed {checkpoint_phase}."
            )

        checkpoint_inputs = [
            str(
                item
            )
            for item in checkpoint.get(
                "input_features",
                [],
            )
        ]

        checkpoint_targets = [
            str(
                item
            )
            for item in checkpoint.get(
                "target_features",
                [],
            )
        ]

        if checkpoint_inputs != input_features:
            raise ValueError(
                "DL checkpoint input_features do not match "
                "feature_schema.json:\n"
                f"{checkpoint_path}"
            )

        if checkpoint_targets != target_features:
            raise ValueError(
                "DL checkpoint target_features do not match "
                "target_schema.json:\n"
                f"{checkpoint_path}"
            )

        checkpoint_selection = _clean_text(
            checkpoint.get(
                "selection_method"
            )
        )

        if (
            checkpoint_selection
            and checkpoint_selection
            != FIXED_DL_SELECTION_METHOD
        ):
            raise ValueError(
                "DL checkpoint selection_method does not match revised "
                "fixed-DL contract:\n"
                f"{checkpoint_path}"
            )

        observed_seeds.append(
            int(
                checkpoint[
                    "seed"
                ]
            )
        )

        observed_versions.append(
            _clean_text(
                checkpoint.get(
                    "script_version"
                )
            )
        )

        observed_architectures.append(
            tuple(
                int(
                    item
                )
                for item in checkpoint[
                    "hidden_layers"
                ]
            )
        )

        observed_dropouts.append(
            float(
                checkpoint[
                    "dropout"
                ]
            )
        )

    if len(
        observed_seeds
    ) != len(
        set(
            observed_seeds
        )
    ):
        raise ValueError(
            "DL artifact bundle contains duplicate checkpoint seeds."
        )

    if expected_seeds:
        if set(
            observed_seeds
        ) != set(
            expected_seeds
        ):
            raise ValueError(
                "DL checkpoint seeds do not match hyperparameters.json.\n"
                f"Expected: {expected_seeds}\n"
                f"Observed: {tuple(observed_seeds)}"
            )

    if len(
        set(
            observed_architectures
        )
    ) != 1:
        raise ValueError(
            "DL checkpoints contain mixed architectures."
        )

    if len(
        set(
            observed_dropouts
        )
    ) != 1:
        raise ValueError(
            "DL checkpoints contain mixed dropout values."
        )

    interpretation = _clean_text(
        hyperparameters.get(
            "hyperparameter_interpretation"
        )
    )

    if not interpretation:
        raise ValueError(
            "DL hyperparameters.json contains no "
            "hyperparameter_interpretation."
        )

    return {
        "status": "PASS_FIXED_DL_ARTIFACT_CONTRACT",
        "selection_method": selection_method,
        "hyperparameter_interpretation": interpretation,
        "hyperparameter_rationale": _clean_text(
            hyperparameters.get(
                "hyperparameter_rationale"
            )
        ),
        "architecture": "-".join(
            str(
                value
            )
            for value in observed_architectures[
                0
            ]
        ),
        "dropout": observed_dropouts[
            0
        ],
        "seeds": tuple(
            observed_seeds
        ),
        "training_versions": tuple(
            sorted(
                {
                    value
                    for value in observed_versions
                    if value
                }
            )
        ),
    }


# =============================================================================
# FULL DL ADAPTER
# =============================================================================

def predict_full_dl(
    *,
    ice: pd.DataFrame,
    project_root: Path,
    scope: str,
    device: torch.device,
) -> pd.DataFrame:
    selected_dir = (
        project_root
        / "models"
        / scope
        / "DL"
        / "Full_Model"
        / "Selected_Final"
    )

    feature_schema = _load_json(
        selected_dir
        / "feature_schema.json"
    )

    target_schema = _load_json(
        selected_dir
        / "target_schema.json"
    )

    input_features = [
        str(
            item
        )
        for item in feature_schema.get(
            "input_features",
            [],
        )
    ]

    target_features = [
        str(
            item
        )
        for item in target_schema.get(
            "target_features",
            [],
        )
    ]

    if len(
        input_features
    ) != 15:
        raise ValueError(
            "Full-DL feature schema must contain exactly 15 inputs."
        )

    if len(
        target_features
    ) != 15:
        raise ValueError(
            "Full-DL target schema must contain exactly 15 targets."
        )

    missing_targets = [
        target
        for target in FULL_DL_EXPECTED_TARGETS
        if target not in target_features
    ]

    if missing_targets:
        raise ValueError(
            "Full-DL target schema is missing frozen outputs:\n"
            + "\n".join(
                f" - {target}"
                for target in missing_targets
            )
        )

    _require_columns(
        ice,
        input_features,
        label=(
            "Full-DL ICE input"
        ),
    )

    X = _numeric_frame(
        ice,
        input_features,
        label=(
            "Full-DL model input"
        ),
        require_nonnegative=True,
    )

    x_scaler_path = (
        selected_dir
        / "x_scaler.joblib"
    )

    y_scaler_path = (
        selected_dir
        / "y_scaler.joblib"
    )

    if not x_scaler_path.is_file():
        raise FileNotFoundError(
            f"Full-DL x_scaler was not found:\n{x_scaler_path}"
        )

    if not y_scaler_path.is_file():
        raise FileNotFoundError(
            f"Full-DL y_scaler was not found:\n{y_scaler_path}"
        )

    x_scaler = joblib.load(
        x_scaler_path
    )

    y_scaler = joblib.load(
        y_scaler_path
    )

    checkpoint_paths = sorted(
        selected_dir.glob(
            "Full_DNN_seed_*.pt"
        )
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            "No Full-DL ensemble checkpoints were found under:\n"
            f"{selected_dir}"
        )

    compatibility = _validate_fixed_dl_artifact_bundle(
        selected_dir=selected_dir,
        input_features=input_features,
        target_features=target_features,
        checkpoint_paths=checkpoint_paths,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        expected_model_source="FULL_DL",
        expected_scope=scope,
    )

    ensemble = _predict_dl_ensemble(
        X=X,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        checkpoint_paths=checkpoint_paths,
        input_count=len(
            input_features
        ),
        output_count=len(
            target_features
        ),
        device=device,
    )

    output = _base_output(
        ice,
        model_source=(
            "FULL_DL"
        ),
        scope=(
            scope
        ),
    )

    for index, target in enumerate(
        target_features
    ):
        output[
            f"Raw Predicted {target}"
        ] = ensemble.raw_mean[
            :,
            index
        ]

        output[
            _standard_prediction_column(
                target
            )
        ] = (
            ensemble.postprocessed_mean[
                :,
                index
            ]
        )

        output[
            f"Ensemble SD {target}"
        ] = ensemble.ensemble_sd[
            :,
            index
        ]

    output[
        DIRECT_TAT_REFERENCE_COLUMN
    ] = output[
        "Predicted EV Total TAT (min)"
    ]

    output[
        "Direct TAT Reference Status"
    ] = (
        "AVAILABLE_FROM_15_OUTPUT_FULL_DL_MODEL"
    )

    output[
        "Adapter Postprocessing"
    ] = (
        "max(raw_prediction, 0); matches Full-DL OOS/inference rule"
    )

    output[
        "Selected Model"
    ] = (
        f"{len(checkpoint_paths)}-seed fixed DNN ensemble"
    )

    output[
        "Ensemble Seeds"
    ] = (
        ",".join(
            str(
                seed
            )
            for seed in ensemble.seeds
        )
    )

    output[
        "Training Script Version"
    ] = (
        " | ".join(
            compatibility[
                "training_versions"
            ]
        )
    )

    output[
        "Artifact Compatibility Status"
    ] = compatibility[
        "status"
    ]

    output[
        "Training Hyperparameter Interpretation"
    ] = compatibility[
        "hyperparameter_interpretation"
    ]

    output[
        "Training Selection Method"
    ] = compatibility[
        "selection_method"
    ]

    return output



# =============================================================================
# PHASE DL ADAPTER
# =============================================================================

def predict_phase_dl(
    *,
    ice: pd.DataFrame,
    project_root: Path,
    scope: str,
    device: torch.device,
) -> pd.DataFrame:
    model_dir = (
        project_root
        / "models"
        / scope
        / "DL"
        / "Phase_Wise"
    )

    output = _base_output(
        ice,
        model_source=(
            "PHASE_DL"
        ),
        scope=(
            scope
        ),
    )

    phase_seed_text: list[str] = []
    training_versions: list[str] = []
    compatibility_statuses: list[str] = []
    interpretations: list[str] = []
    selection_methods: list[str] = []

    for phase in PHASES:
        selected_dir = (
            model_dir
            / phase
            / "Selected_Final"
        )

        feature_schema = _load_json(
            selected_dir
            / "feature_schema.json"
        )

        target_schema = _load_json(
            selected_dir
            / "target_schema.json"
        )

        input_features = [
            str(
                item
            )
            for item in feature_schema.get(
                "input_features",
                [],
            )
        ]

        target_features = [
            str(
                item
            )
            for item in target_schema.get(
                "target_features",
                [],
            )
        ]

        expected_inputs = PHASE_DL_EXPECTED_FEATURES[
            phase
        ]

        expected_targets = PHASE_DL_EXPECTED_TARGETS[
            phase
        ]

        if input_features != expected_inputs:
            raise ValueError(
                f"{phase} Phase-DL input schema mismatch.\n"
                f"Expected: {expected_inputs}\n"
                f"Observed: {input_features}"
            )

        if target_features != expected_targets:
            raise ValueError(
                f"{phase} Phase-DL target schema mismatch.\n"
                f"Expected: {expected_targets}\n"
                f"Observed: {target_features}"
            )

        _require_columns(
            ice,
            input_features,
            label=(
                f"{phase} Phase-DL ICE input"
            ),
        )

        X = _numeric_frame(
            ice,
            input_features,
            label=(
                f"{phase} Phase-DL model input"
            ),
            require_nonnegative=True,
        )

        x_scaler_path = (
            selected_dir
            / "x_scaler.joblib"
        )

        y_scaler_path = (
            selected_dir
            / "y_scaler.joblib"
        )

        if not x_scaler_path.is_file():
            raise FileNotFoundError(
                f"{phase} Phase-DL x_scaler was not found:\n"
                f"{x_scaler_path}"
            )

        if not y_scaler_path.is_file():
            raise FileNotFoundError(
                f"{phase} Phase-DL y_scaler was not found:\n"
                f"{y_scaler_path}"
            )

        x_scaler = joblib.load(
            x_scaler_path
        )

        y_scaler = joblib.load(
            y_scaler_path
        )

        checkpoint_paths = sorted(
            selected_dir.glob(
                f"{phase}_DNN_seed_*.pt"
            )
        )

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No Phase-DL checkpoints were found for {phase}:\n"
                f"{selected_dir}"
            )

        compatibility = _validate_fixed_dl_artifact_bundle(
            selected_dir=selected_dir,
            input_features=input_features,
            target_features=target_features,
            checkpoint_paths=checkpoint_paths,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            device=device,
            expected_model_source="PHASE_DL",
            expected_scope=scope,
            expected_phase=phase,
        )

        ensemble = _predict_dl_ensemble(
            X=X,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            checkpoint_paths=checkpoint_paths,
            input_count=2,
            output_count=2,
            device=device,
        )

        for index, target in enumerate(
            target_features
        ):
            output[
                f"Raw Predicted {target}"
            ] = ensemble.raw_mean[
                :,
                index
            ]

            output[
                _standard_prediction_column(
                    target
                )
            ] = (
                ensemble.postprocessed_mean[
                    :,
                    index
                ]
            )

            output[
                f"Ensemble SD {target}"
            ] = ensemble.ensemble_sd[
                :,
                index
            ]

        phase_seed_text.append(
            f"{phase}:"
            + ",".join(
                str(
                    seed
                )
                for seed in ensemble.seeds
            )
        )

        training_versions.extend(
            compatibility[
                "training_versions"
            ]
        )

        compatibility_statuses.append(
            f"{phase}:{compatibility['status']}"
        )

        interpretations.append(
            compatibility[
                "hyperparameter_interpretation"
            ]
        )

        selection_methods.append(
            compatibility[
                "selection_method"
            ]
        )

    output[
        DIRECT_TAT_REFERENCE_COLUMN
    ] = np.nan

    output[
        "Direct TAT Reference Status"
    ] = (
        "NOT_APPLICABLE_FOR_PHASE_DL"
    )

    output[
        "Adapter Postprocessing"
    ] = (
        "max(raw_prediction, 0); matches Phase-DL OOS/inference rule"
    )

    output[
        "Selected Model"
    ] = (
        "Independent fixed phase DNN ensembles"
    )

    output[
        "Ensemble Seeds"
    ] = (
        " | ".join(
            phase_seed_text
        )
    )

    output[
        "Training Script Version"
    ] = (
        " | ".join(
            sorted(
                set(
                    training_versions
                )
            )
        )
    )

    output[
        "Artifact Compatibility Status"
    ] = (
        " | ".join(
            compatibility_statuses
        )
    )

    output[
        "Training Hyperparameter Interpretation"
    ] = (
        " | ".join(
            sorted(
                set(
                    interpretations
                )
            )
        )
    )

    output[
        "Training Selection Method"
    ] = (
        " | ".join(
            sorted(
                set(
                    selection_methods
                )
            )
        )
    )

    return output



# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_one_adapter(
    *,
    model_source: str,
    scope: str,
    ice: pd.DataFrame,
    project_root: Path,
    device: torch.device,
) -> pd.DataFrame:
    if model_source == "FULL_ML":
        return predict_full_ml(
            ice=(
                ice
            ),
            project_root=(
                project_root
            ),
            scope=(
                scope
            ),
        )

    if model_source == "PHASE_ML":
        return predict_phase_ml(
            ice=(
                ice
            ),
            project_root=(
                project_root
            ),
            scope=(
                scope
            ),
        )

    if model_source == "FULL_DL":
        return predict_full_dl(
            ice=(
                ice
            ),
            project_root=(
                project_root
            ),
            scope=(
                scope
            ),
            device=(
                device
            ),
        )

    if model_source == "PHASE_DL":
        return predict_phase_dl(
            ice=(
                ice
            ),
            project_root=(
                project_root
            ),
            scope=(
                scope
            ),
            device=(
                device
            ),
        )

    raise ValueError(
        f"Unsupported model source: {model_source}"
    )


def build_manifest(
    *,
    project_root: Path,
    input_path: Path,
    output_path: Path,
    model_sources: list[str],
    scopes: list[str],
    ice: pd.DataFrame,
    predictions: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Script Version": (
                    SCRIPT_VERSION
                ),
                "Generated": (
                    datetime.now().isoformat(
                        timespec="seconds"
                    )
                ),
                "Project Root": str(
                    project_root
                ),
                "ICE Input": str(
                    input_path
                ),
                "Output": str(
                    output_path
                ),
                "New ICE Trips": int(
                    len(
                        ice
                    )
                ),
                "Requested Model Sources": (
                    " | ".join(
                        model_sources
                    )
                ),
                "Requested Model Scopes": (
                    " | ".join(
                        scopes
                    )
                ),
                "Prediction Rows": int(
                    len(
                        predictions
                    )
                ),
                "Pair Class For New ICE": (
                    NEW_ICE_PAIR_CLASS
                ),
                "Pair Class Used As Predictor": (
                    "NO"
                ),
                "Pair Class Recalculation": (
                    "NONE"
                ),
                "Pair Class Overwrite": (
                    "NONE"
                ),
                "Replenishment Schema Bridge": (
                    "ICE REFUELING <-> ICE CHARGING_REFUELING; "
                    "NAME ALIAS ONLY; VALUES PRESERVED"
                ),
                "DL Device": str(
                    device
                ),
                "Final TAT Calculated Here": (
                    "NO"
                ),
                "Charging Calculated Here": (
                    "NO"
                ),
                "Downstream": (
                    "final_tat_assembly.py"
                ),
            }
        ]
    )


def save_prediction_workbook(
    *,
    output_path: Path,
    predictions: pd.DataFrame,
    qc: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        predictions.to_excel(
            writer,
            sheet_name=(
                "01_Model_Predictions"
            ),
            index=False,
        )

        qc.to_excel(
            writer,
            sheet_name=(
                "02_Prediction_QC"
            ),
            index=False,
        )

        manifest.to_excel(
            writer,
            sheet_name=(
                "03_Run_Manifest"
            ),
            index=False,
        )


def run_predictions(
    *,
    project_root: Path,
    input_path: Path,
    input_sheet: str,
    output_path: Path,
    model_selection: str,
    scope_selection: str,
    device_selection: str = "cpu",
) -> dict[
    str,
    Any,
]:
    project_root = (
        project_root.resolve()
    )

    model_sources = (
        normalize_model_selection(
            model_selection
        )
    )

    scopes = (
        normalize_scope_selection(
            scope_selection
        )
    )

    device = (
        resolve_device(
            device_selection
        )
    )

    ice = (
        load_ice_parameter_table(
            input_path,
            input_sheet,
        )
    )

    frames: list[
        pd.DataFrame
    ] = []

    run_status: list[
        dict[
            str,
            Any,
        ]
    ] = []

    print(
        "=" * 96
    )
    print(
        "NEW RAW ICE -> TRAINED MODEL PREDICTION ADAPTER"
    )
    print(
        "=" * 96
    )
    print(
        f"Version       : {SCRIPT_VERSION}"
    )
    print(
        f"ICE input     : {input_path}"
    )
    print(
        f"New ICE trips : {len(ice)}"
    )
    print(
        f"Models        : {', '.join(model_sources)}"
    )
    print(
        f"Scopes        : {', '.join(scopes)}"
    )
    print(
        f"DL device     : {device}"
    )
    print(
        "Pair Class    : NOT_APPLICABLE_NEW_ICE_TRIP"
    )
    print(
        "Final TAT     : NOT calculated here"
    )
    print(
        "=" * 96
    )

    for model_source in model_sources:
        for scope in scopes:
            print(
                f"\n[{model_source} | {scope}]"
            )

            try:
                result = (
                    run_one_adapter(
                        model_source=(
                            model_source
                        ),
                        scope=(
                            scope
                        ),
                        ice=(
                            ice
                        ),
                        project_root=(
                            project_root
                        ),
                        device=(
                            device
                        ),
                    )
                )

                frames.append(
                    result
                )

                run_status.append(
                    {
                        "Model Source": (
                            model_source
                        ),
                        "Model Scope": (
                            scope
                        ),
                        "Status": (
                            "SUCCESS"
                        ),
                        "Rows": int(
                            len(
                                result
                            )
                        ),
                        "Error": "",
                    }
                )

                print(
                    f"  SUCCESS | {len(result)} prediction row(s)"
                )

            except Exception as error:
                run_status.append(
                    {
                        "Model Source": (
                            model_source
                        ),
                        "Model Scope": (
                            scope
                        ),
                        "Status": (
                            "FAILED"
                        ),
                        "Rows": 0,
                        "Error": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )

                print(
                    f"  FAILED | {type(error).__name__}: {error}"
                )

                traceback.print_exc()

                # Fail-fast is deliberate. In the final master update pipeline,
                # a missing/invalid selected model is an upstream stage failure,
                # not something that should be silently skipped.
                raise

    if not frames:
        raise RuntimeError(
            "No model prediction rows were produced."
        )

    predictions = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    duplicate_key = [
        "Trip ID",
        "Model Source",
        "Model Scope",
    ]

    if (
        predictions[
            duplicate_key
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate prediction rows were produced for the same "
            "Trip ID + Model Source + Model Scope."
        )

    qc = (
        _prediction_qc_for_rows(
            predictions
        )
    )

    # Merge physical-QC status into the main table for downstream transparency.
    predictions = predictions.merge(
        qc[
            [
                "Trip ID",
                "Model Source",
                "Model Scope",
                "Prediction Physical QC Status",
            ]
        ],
        on=[
            "Trip ID",
            "Model Source",
            "Model Scope",
        ],
        how="left",
        validate="one_to_one",
    )

    manifest = (
        build_manifest(
            project_root=(
                project_root
            ),
            input_path=(
                input_path
            ),
            output_path=(
                output_path
            ),
            model_sources=(
                model_sources
            ),
            scopes=(
                scopes
            ),
            ice=(
                ice
            ),
            predictions=(
                predictions
            ),
            device=(
                device
            ),
        )
    )

    save_prediction_workbook(
        output_path=(
            output_path
        ),
        predictions=(
            predictions
        ),
        qc=(
            qc
        ),
        manifest=(
            manifest
        ),
    )

    print(
        "\n"
        + "=" * 96
    )
    print(
        "MODEL PREDICTION ADAPTER COMPLETED"
    )
    print(
        "=" * 96
    )
    print(
        f"Prediction rows : {len(predictions)}"
    )
    print(
        f"QC PASS rows    : "
        f"{int((qc['Prediction Physical QC Status'] == 'PASS').sum())}"
    )
    print(
        f"QC FAIL rows    : "
        f"{int((qc['Prediction Physical QC Status'] != 'PASS').sum())}"
    )
    print(
        f"Output          : {output_path}"
    )
    print(
        "Next            : final_tat_assembly.py"
    )
    print(
        "=" * 96
    )

    return {
        "predictions": (
            predictions
        ),
        "qc": (
            qc
        ),
        "manifest": (
            manifest
        ),
        "run_status": pd.DataFrame(
            run_status
        ),
        "output_path": (
            output_path
        ),
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply trained Full/Phase ML/DL models to the processed "
            "new raw ICE trip."
        )
    )

    parser.add_argument(
        "--models",
        default="ALL",
        help=(
            "FULL_ML, PHASE_ML, ML, FULL_DL, PHASE_DL, DL, or ALL."
        ),
    )

    parser.add_argument(
        "--scope",
        default="ALL",
        help=(
            "STRICT, NONSTRICT, COMBINED, or ALL."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "DL inference device: cpu, cuda, or auto. Default: cpu."
        ),
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "ICE 17-parameter table. Default: "
            "Results/Prediction/ICE_17_Parameters_Table.xlsx"
        ),
    )

    parser.add_argument(
        "--input-sheet",
        default=(
            "ICE_17_Parameters"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Prediction output workbook. Default: "
            "Results/Prediction/Model_Predictions.xlsx"
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    project_root = (
        args.project_root.resolve()
        if args.project_root
        is not None
        else default_project_root()
    )

    input_path = (
        args.input.resolve()
        if args.input
        is not None
        else default_ice_input(
            project_root
        )
    )

    output_path = (
        args.output.resolve()
        if args.output
        is not None
        else default_output_path(
            project_root
        )
    )

    run_predictions(
        project_root=(
            project_root
        ),
        input_path=(
            input_path
        ),
        input_sheet=(
            args.input_sheet
        ),
        output_path=(
            output_path
        ),
        model_selection=(
            args.models
        ),
        scope_selection=(
            args.scope
        ),
        device_selection=(
            args.device
        ),
    )


if __name__ == "__main__":
    main()
