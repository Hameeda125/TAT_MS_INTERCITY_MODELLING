"""
TAT MS INTERCITY MODELLING
FINAL EV PREDICTION / TAT ASSEMBLY

Version:
    2026-10-02_FINAL_EV_ASSEMBLY_V3_MODEL_SELECTABLE

PURPOSE
-------
This is the single downstream physical-assembly stage for new-ICE inference.

It does NOT:
    - train ML/DL models,
    - load ML/DL model checkpoints,
    - scale model inputs,
    - calculate charging physics,
    - derive/recalculate STRICT/NONSTRICT,
    - alter Pair Class,
    - tune any model.

It reads:
    Results/Prediction/Model_Predictions.xlsx
and:
    Results/Prediction/Charging/All_ICE_Charging_Results.xlsx

It can assemble:
    FULL_ML
    PHASE_ML
    FULL_DL
    PHASE_DL
    ML
    DL
    ALL

and:
    STRICT
    NONSTRICT
    COMBINED
    ALL

Every requested Trip ID must have every requested Model Source x Model Scope
combination. Missing trip/model/scope combinations are rejected before the
final workbook is written.

AUTHORITATIVE RECONSTRUCTIONS
-----------------------------

1. Calculated EV Total TAT (min)

    STOP Time
    + IDLE Time
    + INCHING Time
    + ACC Time
    + DEC Time
    + CRUISE Time
    + Charging Time

2. Calculated EV Total Distance (km)

    STOP Distance
    + IDLE Distance
    + INCHING Distance
    + ACC Distance
    + DEC Distance
    + CRUISE Distance
    + Charging Distance

where:
    Charging Distance (km) = 0.0

3. Calculated EV Average Moving Speed (km/h)

    (
        INCHING Distance
        + ACC Distance
        + DEC Distance
        + CRUISE Distance
    )
    /
    (
        INCHING Time
        + ACC Time
        + DEC Time
        + CRUISE Time
    )
    * 60

STOP, IDLE and charging are excluded from Average Moving Speed.

DIRECT FULL-MODEL OUTPUTS
-------------------------
FULL_ML and FULL_DL directly predict:

    Predicted EV Total TAT (min)
    Predicted EV Total Distance (km)
    Predicted EV Average Moving Speed (km/h)

These are preserved exactly as model outputs. They are NOT used to construct
the calculated/reconstructed quantities and charging is NEVER added to the
direct predicted Total TAT.

PHASE_ML and PHASE_DL have no direct trip-level model outputs. Their three
direct fields are therefore N/A/NaN by design, while calculated trip-level
quantities are reconstructed from their phase predictions.

PAIR CLASS
----------
Pair Class is upstream authority and is preserved only.

New raw ICE inference:
    NOT_APPLICABLE_NEW_ICE_TRIP

Historical values, if this module is reused:
    STRICT
    NONSTRICT

No Pair Class is derived, recalculated or overwritten here.

DEFAULT OUTPUT
--------------
When no explicit model/scope selector is supplied, AUTO mode infers the
model sources/scopes present in Model_Predictions.xlsx. This preserves
compatibility with update_pipeline.py.

Default AUTO output:
    Results/Prediction/Final_TAT/Final_EV_TAT_Results.xlsx

Explicit example:
    --model FULL_DL --scope ALL

writes by default:
    Final_EV_TAT_FULL_DL_ALL.xlsx

Use --output to override the filename.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# VERSION / PATHS
# =============================================================================

SCRIPT_VERSION = "2026-10-02_FINAL_EV_ASSEMBLY_V3_MODEL_SELECTABLE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREDICTION_FILE = (
    PROJECT_ROOT
    / "Results"
    / "Prediction"
    / "Model_Predictions.xlsx"
)

DEFAULT_CHARGING_FILE = (
    PROJECT_ROOT
    / "Results"
    / "Prediction"
    / "Charging"
    / "All_ICE_Charging_Results.xlsx"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "Results"
    / "Prediction"
    / "Final_TAT"
)

DEFAULT_OUTPUT_FILE = (
    DEFAULT_OUTPUT_DIR
    / "Final_EV_TAT_Results.xlsx"
)


# =============================================================================
# AUTHORITATIVE CONTRACT
# =============================================================================

TRIP_ID_COLUMN = "Trip ID"
SOURCE_ICE_FILE_COLUMN = "Source ICE File"
MODEL_SOURCE_COLUMN = "Model Source"
MODEL_SCOPE_COLUMN = "Model Scope"
PAIR_CLASS_COLUMN = "Pair Class"

MODEL_SOURCE_ORDER = (
    "FULL_ML",
    "PHASE_ML",
    "FULL_DL",
    "PHASE_DL",
)

MODEL_SCOPE_ORDER = (
    "STRICT",
    "NONSTRICT",
    "COMBINED",
)

VALID_PAIR_CLASSES = {
    "STRICT",
    "NONSTRICT",
    "NOT_APPLICABLE_NEW_ICE_TRIP",
}

MODEL_SELECTION_MAP = {
    "FULL_ML": ("FULL_ML",),
    "PHASE_ML": ("PHASE_ML",),
    "ML": ("FULL_ML", "PHASE_ML"),
    "FULL_DL": ("FULL_DL",),
    "PHASE_DL": ("PHASE_DL",),
    "DL": ("FULL_DL", "PHASE_DL"),
    "ALL": MODEL_SOURCE_ORDER,
}

SCOPE_SELECTION_MAP = {
    "STRICT": ("STRICT",),
    "NONSTRICT": ("NONSTRICT",),
    "COMBINED": ("COMBINED",),
    "ALL": MODEL_SCOPE_ORDER,
}

PHASES = (
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
)

MOVING_PHASES = (
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
)

PHASE_TIME_COLUMNS = [
    f"Predicted EV {phase} Time (min)"
    for phase in PHASES
]

PHASE_DISTANCE_COLUMNS = [
    f"Predicted EV {phase} Distance (km)"
    for phase in PHASES
]

MOVING_TIME_COLUMNS = [
    f"Predicted EV {phase} Time (min)"
    for phase in MOVING_PHASES
]

MOVING_DISTANCE_COLUMNS = [
    f"Predicted EV {phase} Distance (km)"
    for phase in MOVING_PHASES
]

DIRECT_TOTAL_TAT_COLUMN = "Predicted EV Total TAT (min)"
DIRECT_TOTAL_DISTANCE_COLUMN = "Predicted EV Total Distance (km)"
DIRECT_MOVING_SPEED_COLUMN = "Predicted EV Average Moving Speed (km/h)"

DIRECT_TRIP_COLUMNS = [
    DIRECT_TOTAL_TAT_COLUMN,
    DIRECT_TOTAL_DISTANCE_COLUMN,
    DIRECT_MOVING_SPEED_COLUMN,
]

PHASE_TIME_SUM_COLUMN = (
    "Predicted Six-Phase Time Sum Before Charging (min)"
)

PHASE_DISTANCE_SUM_COLUMN = (
    "Predicted Six-Phase Distance Sum (km)"
)

MOVING_TIME_SUM_COLUMN = (
    "Predicted Moving-Phase Time Sum (min)"
)

MOVING_DISTANCE_SUM_COLUMN = (
    "Predicted Moving-Phase Distance Sum (km)"
)

CHARGING_TIME_COLUMN = "Charging Time (min)"
CHARGING_DISTANCE_COLUMN = "Charging Distance (km)"

CALCULATED_TOTAL_TAT_COLUMN = (
    "Calculated EV Total TAT (min)"
)

CALCULATED_TOTAL_DISTANCE_COLUMN = (
    "Calculated EV Total Distance (km)"
)

CALCULATED_MOVING_SPEED_COLUMN = (
    "Calculated EV Average Moving Speed (km/h)"
)

CALCULATED_OVERALL_SPEED_COLUMN = (
    "Calculated EV Overall Average Speed (km/h)"
)

FULL_MODEL_SOURCES = {
    "FULL_ML",
    "FULL_DL",
}

PHASE_MODEL_SOURCES = {
    "PHASE_ML",
    "PHASE_DL",
}

CLOSURE_TOLERANCE = 1e-9


# =============================================================================
# GENERIC HELPERS
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


def _normalize_upper(
    value: Any,
) -> str:
    return (
        _clean_text(
            value
        )
        .upper()
    )


def _require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    *,
    label: str,
) -> None:
    missing = [
        column
        for column in required
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


def _read_excel(
    path: Path,
    sheet_name: str | int,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Excel file was not found:\n{path}"
        )

    dataframe = pd.read_excel(
        path,
        sheet_name=sheet_name,
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError(
            f"Excel input contains no rows:\n{path}"
        )

    dataframe.columns = [
        str(
            column
        ).strip()
        for column in dataframe.columns
    ]

    return dataframe


def _numeric_series(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(
        float
    )


def _finite_nonnegative_mask(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    values = (
        dataframe[
            columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    mask = (
        np.isfinite(
            values
        ).all(
            axis=1
        )
        & (
            values
            >= 0
        ).all(
            axis=1
        )
    )

    return pd.Series(
        mask,
        index=dataframe.index,
        dtype=bool,
    )


def _safe_sum(
    dataframe: pd.DataFrame,
    columns: list[str],
    valid_mask: pd.Series,
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=dataframe.index,
        dtype=float,
    )

    if bool(
        valid_mask.any()
    ):
        result.loc[
            valid_mask
        ] = (
            dataframe.loc[
                valid_mask,
                columns,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .sum(
                axis=1
            )
            .to_numpy(
                dtype=float
            )
        )

    return result


def normalize_model_selection(
    value: str,
) -> str:
    normalized = str(
        value
    ).strip().upper()

    valid = {
        "AUTO",
        *MODEL_SELECTION_MAP.keys(),
    }

    if normalized not in valid:
        raise ValueError(
            "Model selection must be AUTO, FULL_ML, PHASE_ML, ML, "
            "FULL_DL, PHASE_DL, DL, or ALL. "
            f"Received: {value!r}"
        )

    return normalized


def normalize_scope_selection(
    value: str,
) -> str:
    normalized = str(
        value
    ).strip().upper()

    valid = {
        "AUTO",
        *SCOPE_SELECTION_MAP.keys(),
    }

    if normalized not in valid:
        raise ValueError(
            "Scope selection must be AUTO, STRICT, NONSTRICT, "
            "COMBINED, or ALL. "
            f"Received: {value!r}"
        )

    return normalized


def resolve_requested_models(
    selection: str,
    available: set[str],
) -> list[str]:
    selection = normalize_model_selection(
        selection
    )

    if selection == "AUTO":
        selected = [
            source
            for source in MODEL_SOURCE_ORDER
            if source in available
        ]
    else:
        selected = list(
            MODEL_SELECTION_MAP[
                selection
            ]
        )

    missing = [
        source
        for source in selected
        if source not in available
    ]

    if missing:
        raise ValueError(
            "Requested model source(s) are absent from the prediction file:\n"
            + "\n".join(
                f" - {source}"
                for source in missing
            )
        )

    if not selected:
        raise ValueError(
            "No supported Model Source is available for final assembly."
        )

    return selected


def resolve_requested_scopes(
    selection: str,
    available: set[str],
) -> list[str]:
    selection = normalize_scope_selection(
        selection
    )

    if selection == "AUTO":
        selected = [
            scope
            for scope in MODEL_SCOPE_ORDER
            if scope in available
        ]
    else:
        selected = list(
            SCOPE_SELECTION_MAP[
                selection
            ]
        )

    missing = [
        scope
        for scope in selected
        if scope not in available
    ]

    if missing:
        raise ValueError(
            "Requested model scope(s) are absent from the prediction file:\n"
            + "\n".join(
                f" - {scope}"
                for scope in missing
            )
        )

    if not selected:
        raise ValueError(
            "No supported Model Scope is available for final assembly."
        )

    return selected


def dynamic_default_output(
    *,
    model_selection: str,
    scope_selection: str,
) -> Path:
    model_selection = normalize_model_selection(
        model_selection
    )

    scope_selection = normalize_scope_selection(
        scope_selection
    )

    if (
        model_selection in {
            "AUTO",
            "ALL",
        }
        and scope_selection in {
            "AUTO",
            "ALL",
        }
    ):
        return DEFAULT_OUTPUT_FILE

    model_token = (
        "AUTO"
        if model_selection == "AUTO"
        else model_selection
    )

    scope_token = (
        "AUTO"
        if scope_selection == "AUTO"
        else scope_selection
    )

    return (
        DEFAULT_OUTPUT_DIR
        / (
            f"Final_EV_TAT_"
            f"{model_token}_"
            f"{scope_token}.xlsx"
        )
    )


# =============================================================================
# PREDICTION INPUT VALIDATION
# =============================================================================

def validate_prediction_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = predictions.copy()

    _require_columns(
        dataframe,
        [
            TRIP_ID_COLUMN,
            MODEL_SOURCE_COLUMN,
            MODEL_SCOPE_COLUMN,
            PAIR_CLASS_COLUMN,
            *PHASE_TIME_COLUMNS,
            *PHASE_DISTANCE_COLUMNS,
        ],
        label="Prediction table",
    )

    dataframe[
        TRIP_ID_COLUMN
    ] = (
        dataframe[
            TRIP_ID_COLUMN
        ]
        .map(
            _clean_text
        )
    )

    if (
        dataframe[
            TRIP_ID_COLUMN
        ]
        .eq(
            ""
        )
        .any()
    ):
        raise ValueError(
            "Prediction table contains blank Trip ID values."
        )

    dataframe[
        MODEL_SOURCE_COLUMN
    ] = (
        dataframe[
            MODEL_SOURCE_COLUMN
        ]
        .map(
            _normalize_upper
        )
    )

    invalid_sources = sorted(
        set(
            dataframe[
                MODEL_SOURCE_COLUMN
            ]
        )
        - set(
            MODEL_SOURCE_ORDER
        )
    )

    if invalid_sources:
        raise ValueError(
            "Invalid Model Source value(s): "
            f"{invalid_sources}"
        )

    dataframe[
        MODEL_SCOPE_COLUMN
    ] = (
        dataframe[
            MODEL_SCOPE_COLUMN
        ]
        .map(
            _normalize_upper
        )
    )

    invalid_scopes = sorted(
        set(
            dataframe[
                MODEL_SCOPE_COLUMN
            ]
        )
        - set(
            MODEL_SCOPE_ORDER
        )
    )

    if invalid_scopes:
        raise ValueError(
            "Invalid Model Scope value(s): "
            f"{invalid_scopes}"
        )

    dataframe[
        PAIR_CLASS_COLUMN
    ] = (
        dataframe[
            PAIR_CLASS_COLUMN
        ]
        .map(
            _normalize_upper
        )
    )

    invalid_pair_classes = sorted(
        set(
            dataframe[
                PAIR_CLASS_COLUMN
            ]
        )
        - VALID_PAIR_CLASSES
    )

    if invalid_pair_classes:
        raise ValueError(
            "Invalid Pair Class value(s): "
            f"{invalid_pair_classes}. "
            "Pair Class is upstream authority and is not redefined here."
        )

    duplicate_key = [
        TRIP_ID_COLUMN,
        MODEL_SOURCE_COLUMN,
        MODEL_SCOPE_COLUMN,
    ]

    duplicated = (
        dataframe[
            duplicate_key
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():
        duplicate_rows = (
            dataframe.loc[
                duplicated,
                duplicate_key,
            ]
            .drop_duplicates()
        )

        raise ValueError(
            "Duplicate prediction keys were found:\n"
            + duplicate_rows.to_string(
                index=False
            )
        )

    # A given new trip should carry one upstream Pair Class across every
    # model/scope branch. Historical reuse also benefits from this check.
    pair_class_counts = (
        dataframe.groupby(
            TRIP_ID_COLUMN,
            dropna=False,
        )[
            PAIR_CLASS_COLUMN
        ]
        .nunique(
            dropna=False
        )
    )

    inconsistent_pair_class = (
        pair_class_counts[
            pair_class_counts
            > 1
        ]
    )

    if not inconsistent_pair_class.empty:
        raise ValueError(
            "Pair Class is inconsistent across model branches for Trip ID(s):\n"
            + "\n".join(
                f" - {trip_id}"
                for trip_id in inconsistent_pair_class.index
            )
        )

    # Preserve direct model outputs if present. They are required for FULL
    # model rows later, but not for PHASE rows.
    for column in DIRECT_TRIP_COLUMNS:
        if column not in dataframe.columns:
            dataframe[
                column
            ] = np.nan
        else:
            dataframe[
                column
            ] = _numeric_series(
                dataframe[
                    column
                ]
            )

    # Convert phase columns to numeric but DO NOT clip or impute. Physical
    # validity is assessed row-wise later.
    for column in (
        PHASE_TIME_COLUMNS
        + PHASE_DISTANCE_COLUMNS
    ):
        dataframe[
            column
        ] = _numeric_series(
            dataframe[
                column
            ]
        )

    return dataframe


# =============================================================================
# CHARGING INPUT VALIDATION
# =============================================================================

def validate_charging_table(
    charging: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = charging.copy()

    _require_columns(
        dataframe,
        [
            TRIP_ID_COLUMN,
            CHARGING_TIME_COLUMN,
        ],
        label="Charging table",
    )

    dataframe[
        TRIP_ID_COLUMN
    ] = (
        dataframe[
            TRIP_ID_COLUMN
        ]
        .map(
            _clean_text
        )
    )

    if (
        dataframe[
            TRIP_ID_COLUMN
        ]
        .eq(
            ""
        )
        .any()
    ):
        raise ValueError(
            "Charging table contains blank Trip ID values."
        )

    if (
        dataframe[
            TRIP_ID_COLUMN
        ]
        .duplicated()
        .any()
    ):
        duplicate_ids = (
            dataframe.loc[
                dataframe[
                    TRIP_ID_COLUMN
                ]
                .duplicated(
                    keep=False
                ),
                TRIP_ID_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Charging table contains duplicate Trip IDs:\n"
            + "\n".join(
                f" - {trip_id}"
                for trip_id in duplicate_ids
            )
        )

    dataframe[
        CHARGING_TIME_COLUMN
    ] = _numeric_series(
        dataframe[
            CHARGING_TIME_COLUMN
        ]
    )

    charging_values = (
        dataframe[
            CHARGING_TIME_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        charging_values
    ).all():
        raise ValueError(
            "Charging table contains missing/non-finite Charging Time (min)."
        )

    if (
        charging_values
        < 0
    ).any():
        raise ValueError(
            "Charging table contains negative Charging Time (min)."
        )

    return dataframe


# =============================================================================
# COVERAGE / SELECTION
# =============================================================================

def select_prediction_rows(
    predictions: pd.DataFrame,
    *,
    model_selection: str,
    scope_selection: str,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    pd.DataFrame,
]:
    available_models = set(
        predictions[
            MODEL_SOURCE_COLUMN
        ]
        .astype(
            str
        )
        .tolist()
    )

    available_scopes = set(
        predictions[
            MODEL_SCOPE_COLUMN
        ]
        .astype(
            str
        )
        .tolist()
    )

    selected_models = resolve_requested_models(
        model_selection,
        available_models,
    )

    selected_scopes = resolve_requested_scopes(
        scope_selection,
        available_scopes,
    )

    # Every trip in the prediction workbook must be represented for every
    # explicitly/inferred requested model x scope combination.
    trip_ids = (
        predictions[
            TRIP_ID_COLUMN
        ]
        .drop_duplicates()
        .tolist()
    )

    existing_keys = {
        (
            str(
                row[
                    TRIP_ID_COLUMN
                ]
            ),
            str(
                row[
                    MODEL_SOURCE_COLUMN
                ]
            ),
            str(
                row[
                    MODEL_SCOPE_COLUMN
                ]
            ),
        )
        for _,
        row in predictions.iterrows()
    }

    coverage_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    missing_keys: list[
        tuple[
            str,
            str,
            str,
        ]
    ] = []

    for trip_id in trip_ids:
        for model_source in selected_models:
            for model_scope in selected_scopes:
                key = (
                    str(
                        trip_id
                    ),
                    model_source,
                    model_scope,
                )

                present = (
                    key
                    in existing_keys
                )

                coverage_rows.append(
                    {
                        TRIP_ID_COLUMN: trip_id,
                        MODEL_SOURCE_COLUMN: model_source,
                        MODEL_SCOPE_COLUMN: model_scope,
                        "Prediction Row Present": bool(
                            present
                        ),
                        "Coverage Status": (
                            "PASS"
                            if present
                            else "FAIL_MISSING_PREDICTION"
                        ),
                    }
                )

                if not present:
                    missing_keys.append(
                        key
                    )

    coverage = pd.DataFrame(
        coverage_rows
    )

    if missing_keys:
        preview = "\n".join(
            (
                f" - Trip ID={trip_id} | "
                f"Model Source={model_source} | "
                f"Model Scope={model_scope}"
            )
            for (
                trip_id,
                model_source,
                model_scope,
            ) in missing_keys[
                :50
            ]
        )

        more = (
            ""
            if len(
                missing_keys
            )
            <= 50
            else (
                f"\n ... plus "
                f"{len(missing_keys) - 50} additional missing combination(s)"
            )
        )

        raise ValueError(
            "Not every trip has every requested model/scope prediction.\n"
            + preview
            + more
        )

    selected = (
        predictions.loc[
            predictions[
                MODEL_SOURCE_COLUMN
            ]
            .isin(
                selected_models
            )
            & predictions[
                MODEL_SCOPE_COLUMN
            ]
            .isin(
                selected_scopes
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    expected_rows = (
        len(
            trip_ids
        )
        * len(
            selected_models
        )
        * len(
            selected_scopes
        )
    )

    if len(
        selected
    ) != expected_rows:
        raise RuntimeError(
            "Selected prediction row count does not equal the expected "
            "Trip ID x Model Source x Model Scope coverage."
        )

    return (
        selected,
        selected_models,
        selected_scopes,
        coverage,
    )


# =============================================================================
# FINAL PHYSICAL ASSEMBLY
# =============================================================================

def assemble_final_predictions(
    predictions: pd.DataFrame,
    charging: pd.DataFrame,
    *,
    model_selection: str = "AUTO",
    scope_selection: str = "AUTO",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:
    prediction_table = (
        validate_prediction_table(
            predictions
        )
    )

    charging_table = (
        validate_charging_table(
            charging
        )
    )

    (
        selected,
        selected_models,
        selected_scopes,
        coverage,
    ) = select_prediction_rows(
        prediction_table,
        model_selection=model_selection,
        scope_selection=scope_selection,
    )

    charging_keep = [
        column
        for column in charging_table.columns
        if (
            column
            == TRIP_ID_COLUMN
            or column
            not in selected.columns
        )
    ]

    charging_for_merge = (
        charging_table[
            charging_keep
        ]
        .copy()
    )

    merged = (
        selected.merge(
            charging_for_merge,
            on=TRIP_ID_COLUMN,
            how="left",
            validate="many_to_one",
            indicator=True,
        )
    )

    missing_charging = (
        merged[
            "_merge"
        ]
        != "both"
    )

    if missing_charging.any():
        missing_ids = (
            merged.loc[
                missing_charging,
                TRIP_ID_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Charging result is missing for prediction Trip ID(s):\n"
            + "\n".join(
                f" - {trip_id}"
                for trip_id in missing_ids
            )
        )

    merged.drop(
        columns=[
            "_merge"
        ],
        inplace=True,
    )

    merged[
        CHARGING_DISTANCE_COLUMN
    ] = 0.0

    time_valid = (
        _finite_nonnegative_mask(
            merged,
            PHASE_TIME_COLUMNS,
        )
    )

    distance_valid = (
        _finite_nonnegative_mask(
            merged,
            PHASE_DISTANCE_COLUMNS,
        )
    )

    charging_time = _numeric_series(
        merged[
            CHARGING_TIME_COLUMN
        ]
    )

    charging_valid = pd.Series(
        (
            np.isfinite(
                charging_time.to_numpy(
                    dtype=float
                )
            )
            & (
                charging_time.to_numpy(
                    dtype=float
                )
                >= 0
            )
        ),
        index=merged.index,
        dtype=bool,
    )

    merged[
        PHASE_TIME_SUM_COLUMN
    ] = _safe_sum(
        merged,
        PHASE_TIME_COLUMNS,
        time_valid,
    )

    merged[
        PHASE_DISTANCE_SUM_COLUMN
    ] = _safe_sum(
        merged,
        PHASE_DISTANCE_COLUMNS,
        distance_valid,
    )

    merged[
        MOVING_TIME_SUM_COLUMN
    ] = _safe_sum(
        merged,
        MOVING_TIME_COLUMNS,
        time_valid,
    )

    merged[
        MOVING_DISTANCE_SUM_COLUMN
    ] = _safe_sum(
        merged,
        MOVING_DISTANCE_COLUMNS,
        distance_valid,
    )

    tat_valid = (
        time_valid
        & charging_valid
    )

    merged[
        CALCULATED_TOTAL_TAT_COLUMN
    ] = np.where(
        tat_valid,
        (
            merged[
                PHASE_TIME_SUM_COLUMN
            ]
            + charging_time
        ),
        np.nan,
    )

    merged[
        CALCULATED_TOTAL_DISTANCE_COLUMN
    ] = np.where(
        distance_valid,
        (
            merged[
                PHASE_DISTANCE_SUM_COLUMN
            ]
            + merged[
                CHARGING_DISTANCE_COLUMN
            ]
        ),
        np.nan,
    )

    moving_time = _numeric_series(
        merged[
            MOVING_TIME_SUM_COLUMN
        ]
    )

    moving_distance = _numeric_series(
        merged[
            MOVING_DISTANCE_SUM_COLUMN
        ]
    )

    moving_speed_valid = (
        time_valid
        & distance_valid
        & np.isfinite(
            moving_time.to_numpy(
                dtype=float
            )
        )
        & np.isfinite(
            moving_distance.to_numpy(
                dtype=float
            )
        )
        & (
            moving_time
            > 0
        )
    )

    merged[
        CALCULATED_MOVING_SPEED_COLUMN
    ] = np.where(
        moving_speed_valid,
        (
            moving_distance
            / moving_time
            * 60.0
        ),
        np.nan,
    )

    total_tat = _numeric_series(
        merged[
            CALCULATED_TOTAL_TAT_COLUMN
        ]
    )

    total_distance = _numeric_series(
        merged[
            CALCULATED_TOTAL_DISTANCE_COLUMN
        ]
    )

    overall_speed_valid = (
        np.isfinite(
            total_tat.to_numpy(
                dtype=float
            )
        )
        & np.isfinite(
            total_distance.to_numpy(
                dtype=float
            )
        )
        & (
            total_tat
            > 0
        )
    )

    merged[
        CALCULATED_OVERALL_SPEED_COLUMN
    ] = np.where(
        overall_speed_valid,
        (
            total_distance
            / total_tat
            * 60.0
        ),
        np.nan,
    )

    merged[
        "Charging Contribution (%)"
    ] = np.where(
        tat_valid
        & (
            total_tat
            > 0
        ),
        (
            charging_time
            / total_tat
            * 100.0
        ),
        np.nan,
    )

    # ---------------------------------------------------------------------
    # Direct trip-level predictions:
    # FULL models -> required and preserved exactly.
    # PHASE models -> explicitly N/A.
    # ---------------------------------------------------------------------
    model_source = (
        merged[
            MODEL_SOURCE_COLUMN
        ]
        .astype(
            str
        )
    )

    full_mask = (
        model_source.isin(
            FULL_MODEL_SOURCES
        )
    )

    phase_mask = (
        model_source.isin(
            PHASE_MODEL_SOURCES
        )
    )

    for column in DIRECT_TRIP_COLUMNS:
        merged[
            column
        ] = _numeric_series(
            merged[
                column
            ]
        )

        merged.loc[
            phase_mask,
            column,
        ] = np.nan

    direct_values = (
        merged[
            DIRECT_TRIP_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    direct_finite = pd.Series(
        np.isfinite(
            direct_values
        ).all(
            axis=1
        ),
        index=merged.index,
        dtype=bool,
    )

    direct_nonnegative = pd.Series(
        (
            direct_values
            >= 0
        ).all(
            axis=1
        ),
        index=merged.index,
        dtype=bool,
    )

    direct_status = pd.Series(
        "NOT_APPLICABLE_PHASE_MODEL",
        index=merged.index,
        dtype=object,
    )

    direct_status.loc[
        full_mask
        & ~direct_finite
    ] = (
        "FAIL_MISSING_OR_NONFINITE_DIRECT_OUTPUT"
    )

    direct_status.loc[
        full_mask
        & direct_finite
        & ~direct_nonnegative
    ] = (
        "FAIL_NEGATIVE_DIRECT_OUTPUT"
    )

    direct_status.loc[
        full_mask
        & direct_finite
        & direct_nonnegative
    ] = "PASS"

    merged[
        "Direct Trip-Level Prediction QC Status"
    ] = direct_status

    merged[
        "Direct Trip-Level Prediction Role"
    ] = np.where(
        full_mask,
        (
            "INDEPENDENT FULL-MODEL OUTPUT; "
            "NOT USED IN PHYSICAL RECONSTRUCTION"
        ),
        (
            "NOT APPLICABLE FOR PHASE-WISE MODEL"
        ),
    )

    # ---------------------------------------------------------------------
    # Closure checks.
    # ---------------------------------------------------------------------
    expected_tat = (
        merged[
            PHASE_TIME_SUM_COLUMN
        ]
        + charging_time
    )

    expected_distance = (
        merged[
            PHASE_DISTANCE_SUM_COLUMN
        ]
        + merged[
            CHARGING_DISTANCE_COLUMN
        ]
    )

    expected_moving_speed = (
        moving_distance
        / moving_time
        * 60.0
    )

    tat_closure = (
        total_tat
        - expected_tat
    )

    distance_closure = (
        total_distance
        - expected_distance
    )

    moving_speed_closure = (
        _numeric_series(
            merged[
                CALCULATED_MOVING_SPEED_COLUMN
            ]
        )
        - expected_moving_speed
    )

    merged[
        "TAT Closure Residual (min)"
    ] = np.where(
        tat_valid,
        tat_closure,
        np.nan,
    )

    merged[
        "Distance Closure Residual (km)"
    ] = np.where(
        distance_valid,
        distance_closure,
        np.nan,
    )

    merged[
        "Moving Speed Closure Residual (km/h)"
    ] = np.where(
        moving_speed_valid,
        moving_speed_closure,
        np.nan,
    )

    tat_status = np.where(
        (
            tat_valid
            & np.isfinite(
                total_tat.to_numpy(
                    dtype=float
                )
            )
            & (
                total_tat
                >= 0
            )
            & (
                np.abs(
                    tat_closure
                )
                <= CLOSURE_TOLERANCE
            )
        ),
        "PASS",
        "FAIL",
    )

    distance_status = np.where(
        (
            distance_valid
            & np.isfinite(
                total_distance.to_numpy(
                    dtype=float
                )
            )
            & (
                total_distance
                >= 0
            )
            & (
                np.abs(
                    distance_closure
                )
                <= CLOSURE_TOLERANCE
            )
        ),
        "PASS",
        "FAIL",
    )

    moving_status = np.where(
        (
            moving_speed_valid
            & np.isfinite(
                merged[
                    CALCULATED_MOVING_SPEED_COLUMN
                ].to_numpy(
                    dtype=float
                )
            )
            & (
                merged[
                    CALCULATED_MOVING_SPEED_COLUMN
                ]
                >= 0
            )
            & (
                np.abs(
                    moving_speed_closure
                )
                <= CLOSURE_TOLERANCE
            )
        ),
        "PASS",
        "FAIL",
    )

    merged[
        "TAT Assembly Status"
    ] = tat_status

    merged[
        "Distance Assembly Status"
    ] = distance_status

    merged[
        "Moving Speed Assembly Status"
    ] = moving_status

    reconstruction_pass = (
        (
            merged[
                "TAT Assembly Status"
            ]
            == "PASS"
        )
        & (
            merged[
                "Distance Assembly Status"
            ]
            == "PASS"
        )
        & (
            merged[
                "Moving Speed Assembly Status"
            ]
            == "PASS"
        )
    )

    merged[
        "Reconstruction QC Status"
    ] = np.where(
        reconstruction_pass,
        "PASS",
        "FAIL",
    )

    # Direct-output QC is reported but does NOT invalidate an otherwise valid
    # physical reconstruction because direct outputs are independent reference
    # model predictions.
    merged[
        "Overall Physical QC Status"
    ] = merged[
        "Reconstruction QC Status"
    ]

    merged[
        "TAT Formula"
    ] = (
        "STOP + IDLE + INCHING + ACC + DEC + CRUISE + Charging Time"
    )

    merged[
        "Distance Formula"
    ] = (
        "STOP + IDLE + INCHING + ACC + DEC + CRUISE + Charging Distance"
    )

    merged[
        "Average Moving Speed Formula"
    ] = (
        "(INCHING + ACC + DEC + CRUISE distance) / "
        "(INCHING + ACC + DEC + CRUISE time) * 60"
    )

    merged[
        "Charging Source"
    ] = (
        "COMMON charging_time.py OUTPUT"
    )

    merged[
        "Pair Class Handling"
    ] = (
        "PRESERVED ONLY; NOT RECALCULATED / NOT OVERWRITTEN"
    )

    merged[
        "Final Assembly Version"
    ] = (
        SCRIPT_VERSION
    )

    qc = build_qc_table(
        merged
    )

    front = [
        TRIP_ID_COLUMN,
        SOURCE_ICE_FILE_COLUMN,
        MODEL_SOURCE_COLUMN,
        MODEL_SCOPE_COLUMN,
        PAIR_CLASS_COLUMN,
        DIRECT_TOTAL_TAT_COLUMN,
        CALCULATED_TOTAL_TAT_COLUMN,
        DIRECT_TOTAL_DISTANCE_COLUMN,
        CALCULATED_TOTAL_DISTANCE_COLUMN,
        DIRECT_MOVING_SPEED_COLUMN,
        CALCULATED_MOVING_SPEED_COLUMN,
        CALCULATED_OVERALL_SPEED_COLUMN,
        *PHASE_TIME_COLUMNS,
        PHASE_TIME_SUM_COLUMN,
        *PHASE_DISTANCE_COLUMNS,
        PHASE_DISTANCE_SUM_COLUMN,
        CHARGING_TIME_COLUMN,
        CHARGING_DISTANCE_COLUMN,
        MOVING_TIME_SUM_COLUMN,
        MOVING_DISTANCE_SUM_COLUMN,
        "Charging Contribution (%)",
        "Direct Trip-Level Prediction Role",
        "Direct Trip-Level Prediction QC Status",
        "TAT Assembly Status",
        "Distance Assembly Status",
        "Moving Speed Assembly Status",
        "Reconstruction QC Status",
        "Overall Physical QC Status",
    ]

    front = [
        column
        for column in front
        if column in merged.columns
    ]

    remaining = [
        column
        for column in merged.columns
        if column not in front
    ]

    merged = merged[
        front
        + remaining
    ]

    return (
        merged,
        qc,
        coverage,
        selected_models,
        selected_scopes,
    )


# Backward-compatible function name used by older orchestration code.
def assemble_final_tat(
    predictions: pd.DataFrame,
    charging: pd.DataFrame,
    *,
    model_selection: str = "AUTO",
    scope_selection: str = "AUTO",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    (
        assembled,
        qc,
        _coverage,
        _models,
        _scopes,
    ) = assemble_final_predictions(
        predictions,
        charging,
        model_selection=model_selection,
        scope_selection=scope_selection,
    )

    return (
        assembled,
        qc,
    )


# =============================================================================
# QC
# =============================================================================

def build_qc_table(
    assembled: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for _, row in assembled.iterrows():
        time_values = np.asarray(
            [
                pd.to_numeric(
                    row[
                        column
                    ],
                    errors="coerce",
                )
                for column in PHASE_TIME_COLUMNS
            ],
            dtype=float,
        )

        distance_values = np.asarray(
            [
                pd.to_numeric(
                    row[
                        column
                    ],
                    errors="coerce",
                )
                for column in PHASE_DISTANCE_COLUMNS
            ],
            dtype=float,
        )

        moving_time = pd.to_numeric(
            row[
                MOVING_TIME_SUM_COLUMN
            ],
            errors="coerce",
        )

        moving_distance = pd.to_numeric(
            row[
                MOVING_DISTANCE_SUM_COLUMN
            ],
            errors="coerce",
        )

        rows.append(
            {
                TRIP_ID_COLUMN: (
                    row[
                        TRIP_ID_COLUMN
                    ]
                ),
                MODEL_SOURCE_COLUMN: (
                    row[
                        MODEL_SOURCE_COLUMN
                    ]
                ),
                MODEL_SCOPE_COLUMN: (
                    row[
                        MODEL_SCOPE_COLUMN
                    ]
                ),
                PAIR_CLASS_COLUMN: (
                    row[
                        PAIR_CLASS_COLUMN
                    ]
                ),
                "All Six Phase Times Finite": bool(
                    np.isfinite(
                        time_values
                    ).all()
                ),
                "All Six Phase Times Non-Negative": bool(
                    np.isfinite(
                        time_values
                    ).all()
                    and (
                        time_values
                        >= 0
                    ).all()
                ),
                "All Six Phase Distances Finite": bool(
                    np.isfinite(
                        distance_values
                    ).all()
                ),
                "All Six Phase Distances Non-Negative": bool(
                    np.isfinite(
                        distance_values
                    ).all()
                    and (
                        distance_values
                        >= 0
                    ).all()
                ),
                "Charging Time Finite/Non-Negative": bool(
                    np.isfinite(
                        float(
                            row[
                                CHARGING_TIME_COLUMN
                            ]
                        )
                    )
                    and float(
                        row[
                            CHARGING_TIME_COLUMN
                        ]
                    )
                    >= 0
                ),
                "Charging Distance Equals 0 km": bool(
                    np.isclose(
                        float(
                            row[
                                CHARGING_DISTANCE_COLUMN
                            ]
                        ),
                        0.0,
                    )
                ),
                "Moving Time > 0": bool(
                    np.isfinite(
                        moving_time
                    )
                    and moving_time
                    > 0
                ),
                "Moving Distance Finite/Non-Negative": bool(
                    np.isfinite(
                        moving_distance
                    )
                    and moving_distance
                    >= 0
                ),
                "TAT Closure Residual (min)": (
                    row[
                        "TAT Closure Residual (min)"
                    ]
                ),
                "Distance Closure Residual (km)": (
                    row[
                        "Distance Closure Residual (km)"
                    ]
                ),
                "Moving Speed Closure Residual (km/h)": (
                    row[
                        "Moving Speed Closure Residual (km/h)"
                    ]
                ),
                "Direct Trip-Level Prediction QC Status": (
                    row[
                        "Direct Trip-Level Prediction QC Status"
                    ]
                ),
                "TAT Assembly Status": (
                    row[
                        "TAT Assembly Status"
                    ]
                ),
                "Distance Assembly Status": (
                    row[
                        "Distance Assembly Status"
                    ]
                ),
                "Moving Speed Assembly Status": (
                    row[
                        "Moving Speed Assembly Status"
                    ]
                ),
                "Reconstruction QC Status": (
                    row[
                        "Reconstruction QC Status"
                    ]
                ),
                "Overall Physical QC Status": (
                    row[
                        "Overall Physical QC Status"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MANIFEST / SAVE
# =============================================================================

def build_manifest(
    *,
    prediction_path: Path | None,
    charging_path: Path | None,
    output_path: Path | None,
    assembled: pd.DataFrame,
    qc: pd.DataFrame,
    coverage: pd.DataFrame,
    requested_model_selection: str,
    requested_scope_selection: str,
    selected_models: list[str],
    selected_scopes: list[str],
) -> pd.DataFrame:
    trip_count = int(
        assembled[
            TRIP_ID_COLUMN
        ]
        .nunique()
    )

    reconstruction_failures = int(
        (
            qc[
                "Reconstruction QC Status"
            ]
            != "PASS"
        )
        .sum()
    )

    direct_failures = int(
        (
            qc[
                "Direct Trip-Level Prediction QC Status"
            ]
            .astype(
                str
            )
            .str.startswith(
                "FAIL"
            )
        )
        .sum()
    )

    coverage_failures = int(
        (
            coverage[
                "Coverage Status"
            ]
            != "PASS"
        )
        .sum()
    )

    rows = [
        {
            "Parameter": "Script Version",
            "Value": SCRIPT_VERSION,
        },
        {
            "Parameter": "Generated",
            "Value": datetime.now().isoformat(
                timespec="seconds"
            ),
        },
        {
            "Parameter": "Prediction Input",
            "Value": (
                str(
                    prediction_path
                )
                if prediction_path
                is not None
                else "IN-MEMORY DATAFRAME"
            ),
        },
        {
            "Parameter": "Charging Input",
            "Value": (
                str(
                    charging_path
                )
                if charging_path
                is not None
                else "IN-MEMORY DATAFRAME"
            ),
        },
        {
            "Parameter": "Output",
            "Value": (
                str(
                    output_path
                )
                if output_path
                is not None
                else "IN-MEMORY ONLY"
            ),
        },
        {
            "Parameter": "Requested Model Selection",
            "Value": requested_model_selection,
        },
        {
            "Parameter": "Resolved Model Sources",
            "Value": " | ".join(
                selected_models
            ),
        },
        {
            "Parameter": "Requested Scope Selection",
            "Value": requested_scope_selection,
        },
        {
            "Parameter": "Resolved Model Scopes",
            "Value": " | ".join(
                selected_scopes
            ),
        },
        {
            "Parameter": "Trip Count",
            "Value": trip_count,
        },
        {
            "Parameter": "Rows Assembled",
            "Value": int(
                len(
                    assembled
                )
            ),
        },
        {
            "Parameter": "Coverage Contract",
            "Value": (
                "ONE ROW PER TRIP ID x REQUESTED MODEL SOURCE "
                "x REQUESTED MODEL SCOPE"
            ),
        },
        {
            "Parameter": "Coverage Failures",
            "Value": coverage_failures,
        },
        {
            "Parameter": "Authoritative TAT Formula",
            "Value": (
                "STOP + IDLE + INCHING + ACC + DEC + CRUISE "
                "+ Charging Time"
            ),
        },
        {
            "Parameter": "Authoritative Distance Formula",
            "Value": (
                "STOP + IDLE + INCHING + ACC + DEC + CRUISE "
                "+ Charging Distance"
            ),
        },
        {
            "Parameter": "Charging Distance",
            "Value": "0 km",
        },
        {
            "Parameter": "Authoritative Average Moving Speed Formula",
            "Value": (
                "(INCHING + ACC + DEC + CRUISE distance) / "
                "(INCHING + ACC + DEC + CRUISE time) * 60"
            ),
        },
        {
            "Parameter": "Direct Full-Model Outputs",
            "Value": (
                "PRESERVED INDEPENDENTLY; NOT USED IN RECONSTRUCTION; "
                "NO CHARGING ADDED TO DIRECT TOTAL TAT"
            ),
        },
        {
            "Parameter": "Phase-Model Direct Trip-Level Outputs",
            "Value": "N/A / NaN BY DESIGN",
        },
        {
            "Parameter": "Pair Class Handling",
            "Value": (
                "PRESERVED ONLY; NOT RECALCULATED / NOT OVERWRITTEN"
            ),
        },
        {
            "Parameter": "Reconstruction QC Failures",
            "Value": reconstruction_failures,
        },
        {
            "Parameter": "Direct Output QC Failures",
            "Value": direct_failures,
        },
        {
            "Parameter": "Historical Actual-EV Metrics",
            "Value": (
                "NOT CALCULATED HERE; MODEL EVALUATION BELONGS TO "
                "THE TRAINING/OOS SCRIPTS"
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def save_results(
    *,
    output_path: Path,
    assembled: pd.DataFrame,
    qc: pd.DataFrame,
    coverage: pd.DataFrame,
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
        assembled.to_excel(
            writer,
            sheet_name="01_Final_EV_Predictions",
            index=False,
        )

        qc.to_excel(
            writer,
            sheet_name="02_Physical_QC",
            index=False,
        )

        coverage.to_excel(
            writer,
            sheet_name="03_Coverage_QC",
            index=False,
        )

        manifest.to_excel(
            writer,
            sheet_name="04_Assembly_Manifest",
            index=False,
        )


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_tat_assembly(
    *,
    predictions: pd.DataFrame,
    charging: pd.DataFrame,
    output_path: Path | None = None,
    prediction_path_for_manifest: Path | None = None,
    charging_path_for_manifest: Path | None = None,
    model_selection: str = "AUTO",
    scope_selection: str = "AUTO",
) -> dict[
    str,
    Any,
]:
    (
        assembled,
        qc,
        coverage,
        selected_models,
        selected_scopes,
    ) = assemble_final_predictions(
        predictions,
        charging,
        model_selection=model_selection,
        scope_selection=scope_selection,
    )

    manifest = build_manifest(
        prediction_path=prediction_path_for_manifest,
        charging_path=charging_path_for_manifest,
        output_path=output_path,
        assembled=assembled,
        qc=qc,
        coverage=coverage,
        requested_model_selection=(
            normalize_model_selection(
                model_selection
            )
        ),
        requested_scope_selection=(
            normalize_scope_selection(
                scope_selection
            )
        ),
        selected_models=selected_models,
        selected_scopes=selected_scopes,
    )

    if output_path is not None:
        save_results(
            output_path=output_path,
            assembled=assembled,
            qc=qc,
            coverage=coverage,
            manifest=manifest,
        )

    reconstruction_failures = (
        qc[
            "Reconstruction QC Status"
        ]
        != "PASS"
    )

    if reconstruction_failures.any():
        failed = (
            qc.loc[
                reconstruction_failures,
                [
                    TRIP_ID_COLUMN,
                    MODEL_SOURCE_COLUMN,
                    MODEL_SCOPE_COLUMN,
                    "TAT Assembly Status",
                    "Distance Assembly Status",
                    "Moving Speed Assembly Status",
                    "Reconstruction QC Status",
                ],
            ]
        )

        message = (
            "Final EV assembly contains physically invalid reconstruction "
            "row(s). No clipping/imputation was applied."
        )

        if output_path is not None:
            message += (
                "\nThe workbook was saved for diagnosis at:\n"
                f"{output_path}"
            )

        message += (
            "\nFailed rows:\n"
            + failed.to_string(
                index=False
            )
        )

        raise RuntimeError(
            message
        )

    return {
        "assembled": assembled,
        "qc": qc,
        "coverage": coverage,
        "manifest": manifest,
        "selected_models": selected_models,
        "selected_scopes": selected_scopes,
        "output_path": output_path,
    }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble final EV TAT, distance and moving-speed predictions "
            "for every requested trip/model/scope."
        )
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTION_FILE,
        help=(
            "Prediction workbook. Default: "
            "Results/Prediction/Model_Predictions.xlsx"
        ),
    )

    parser.add_argument(
        "--prediction-sheet",
        default="01_Model_Predictions",
        help=(
            "Prediction worksheet name. "
            "Default: 01_Model_Predictions"
        ),
    )

    parser.add_argument(
        "--charging",
        type=Path,
        default=DEFAULT_CHARGING_FILE,
        help=(
            "Charging workbook. Default: "
            "Results/Prediction/Charging/All_ICE_Charging_Results.xlsx"
        ),
    )

    parser.add_argument(
        "--charging-sheet",
        default="Charging_Results",
        help=(
            "Charging worksheet name. Default: Charging_Results"
        ),
    )

    parser.add_argument(
        "--model",
        default="AUTO",
        help=(
            "AUTO, FULL_ML, PHASE_ML, ML, FULL_DL, PHASE_DL, DL, or ALL. "
            "AUTO preserves master-updater compatibility by using the model "
            "sources present in the prediction workbook."
        ),
    )

    parser.add_argument(
        "--scope",
        default="AUTO",
        help=(
            "AUTO, STRICT, NONSTRICT, COMBINED, or ALL. "
            "AUTO uses the scopes present in the prediction workbook."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output workbook. If omitted, AUTO/ALL writes "
            "Final_EV_TAT_Results.xlsx; explicit selections receive "
            "a model/scope-specific filename."
        ),
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    model_selection = (
        normalize_model_selection(
            args.model
        )
    )

    scope_selection = (
        normalize_scope_selection(
            args.scope
        )
    )

    predictions = _read_excel(
        args.predictions,
        args.prediction_sheet,
    )

    charging = _read_excel(
        args.charging,
        args.charging_sheet,
    )

    output_path = (
        args.output.resolve()
        if args.output
        is not None
        else dynamic_default_output(
            model_selection=model_selection,
            scope_selection=scope_selection,
        )
    )

    print(
        "=" * 96
    )
    print(
        "FINAL EV PREDICTION / TAT ASSEMBLY"
    )
    print(
        "=" * 96
    )
    print(
        f"Version        : {SCRIPT_VERSION}"
    )
    print(
        f"Predictions    : {args.predictions}"
    )
    print(
        f"Charging       : {args.charging}"
    )
    print(
        f"Model selector : {model_selection}"
    )
    print(
        f"Scope selector : {scope_selection}"
    )
    print(
        f"Output         : {output_path}"
    )
    print(
        "Pair Class     : PRESERVED ONLY"
    )
    print(
        "Charging dist. : 0 km"
    )
    print(
        "=" * 96
    )

    result = run_tat_assembly(
        predictions=predictions,
        charging=charging,
        output_path=output_path,
        prediction_path_for_manifest=(
            args.predictions
        ),
        charging_path_for_manifest=(
            args.charging
        ),
        model_selection=model_selection,
        scope_selection=scope_selection,
    )

    assembled = result[
        "assembled"
    ]

    qc = result[
        "qc"
    ]

    print(
        "\n"
        + "=" * 96
    )
    print(
        "FINAL EV ASSEMBLY COMPLETED"
    )
    print(
        "=" * 96
    )
    print(
        f"Trips          : "
        f"{assembled[TRIP_ID_COLUMN].nunique()}"
    )
    print(
        f"Rows           : {len(assembled)}"
    )
    print(
        "Models         : "
        + ", ".join(
            result[
                "selected_models"
            ]
        )
    )
    print(
        "Scopes         : "
        + ", ".join(
            result[
                "selected_scopes"
            ]
        )
    )
    print(
        "Reconstruction : "
        f"{int((qc['Reconstruction QC Status'] == 'PASS').sum())}"
        f"/{len(qc)} PASS"
    )
    print(
        "Direct outputs : "
        "FULL models preserved independently; PHASE models N/A"
    )
    print(
        "Calculated TAT : six phases + charging time"
    )
    print(
        "Calculated dist: six phase distances + 0 km charging distance"
    )
    print(
        "Moving speed   : moving-phase distance / moving-phase time x 60"
    )
    print(
        f"Saved          : {output_path}"
    )
    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
