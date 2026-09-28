# =============================================================================
# ts_dataset.py
# =============================================================================
# Purpose:
#   Construct the final time-series modelling dataset from:
#
#       1. Completed Part-1 STRICT paired telemetry
#       2. Completed TS analysis workbook
#
# Research design:
#
#       Historical paired ICE + EV trips
#                    |
#                    v
#       Distance-based route progression
#                    |
#             100 route bins
#                    |
#          +---------+---------+
#          |                   |
#       ICE sequence        EV sequence
#          |                   |
#          v                   v
#       ICE behaviour      EV six-phase behaviour
#                              |
#                              v
#                    EV phase time + distance
#                              |
#                              v
#                    EV TAT reconstruction
#
# Predictive EV phases:
#       STOP
#       IDLE
#       INCHING
#       ACCELERATION
#       DECELERATION
#       CRUISING
#
# Charging is handled separately.
# ICE refuelling remains part of the actual ICE TAT.
#
# IMPORTANT:
#   - No re-cleaning
#   - No re-feature engineering
#   - No re-segmentation
#   - No row-wise ICE-EV timestamp alignment
#   - No latitude/longitude as modelling features
#   - No charging time as TS prediction target
#   - No refuelling time as TS prediction target
#   - Pair ID retained for leakage-safe validation
#   - Route progress is a sequence coordinate, NOT an EV prediction target
#
# MAJOR REPRESENTATION:
#
#   A telemetry interval may cross multiple route bins.
#
#       Telemetry interval
#              |
#              v
#       Determine route-bin overlap
#              |
#              v
#       Allocate interval time/distance
#       proportionally across crossed bins
#
#   Zero-distance intervals:
#
#       STOP / IDLE / stationary interval
#              |
#              v
#       Assign duration to route position
#
# IMPORTANT PHASE-LABEL FIX:
#
#   Part-1 may use compact phase labels:
#
#       INCH
#       ACC
#       DEC
#       CRUISE
#
#   This file converts them to canonical names:
#
#       INCHING
#       ACCELERATION
#       DECELERATION
#       CRUISING
#
#   This is ONLY label normalization.
#   It does NOT change the original segmentation.
# =============================================================================


from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING"
)

PART1_DIR = Path(
    r"D:\TAT_MS_INTERCITY"
)

PAIRED_DIR = (
    PART1_DIR
    / "Data"
    / "Results"
    / "Paired"
)

ANALYSIS_FILE = (
    BASE_DIR
    / "Results"
    / "TimeSeries"
    / "TS_Complete_Analysis.xlsx"
)

OUTPUT_DIR = (
    BASE_DIR
    / "Results"
    / "TimeSeries"
    / "Dataset"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. OUTPUT FILES
# =============================================================================

DATASET_XLSX = (
    OUTPUT_DIR
    / "TS_Route_Progress_Dataset.xlsx"
)

DATASET_NPZ = (
    OUTPUT_DIR
    / "TS_Model_Dataset.npz"
)

VALIDATION_XLSX = (
    OUTPUT_DIR
    / "TS_Dataset_Validation.xlsx"
)

RUN_REPORT = (
    OUTPUT_DIR
    / "TS_Dataset_Run_Report.txt"
)


# =============================================================================
# 3. CONFIGURATION
# =============================================================================

N_BINS = 100

MIN_SPEED_KMPH = 5.0

CONSERVATION_TOLERANCE = 1e-6


# =============================================================================
# 4. PREDICTIVE PHASES
# =============================================================================

PREDICTIVE_PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACCELERATION",
    "DECELERATION",
    "CRUISING",
]


# =============================================================================
# 5. PHASE LABEL NORMALIZATION
# =============================================================================

PHASE_NORMALIZATION = {

    # -------------------------------------------------------------------------
    # STOP
    # -------------------------------------------------------------------------

    "STOP":
        "STOP",

    # -------------------------------------------------------------------------
    # IDLE
    # -------------------------------------------------------------------------

    "IDLE":
        "IDLE",

    # -------------------------------------------------------------------------
    # INCHING
    # -------------------------------------------------------------------------

    "INCH":
        "INCHING",

    "INCHING":
        "INCHING",

    # -------------------------------------------------------------------------
    # ACCELERATION
    # -------------------------------------------------------------------------

    "ACC":
        "ACCELERATION",

    "ACCELERATION":
        "ACCELERATION",

    # -------------------------------------------------------------------------
    # DECELERATION
    # -------------------------------------------------------------------------

    "DEC":
        "DECELERATION",

    "DECELERATION":
        "DECELERATION",

    # -------------------------------------------------------------------------
    # CRUISING
    # -------------------------------------------------------------------------

    "CRUISE":
        "CRUISING",

    "CRUISING":
        "CRUISING",

    # -------------------------------------------------------------------------
    # Operational phases
    # -------------------------------------------------------------------------

    "CHARGING":
        "CHARGING",

    "CHARGE":
        "CHARGING",

    "REFUELLING":
        "REFUELLING",

    "REFUELING":
        "REFUELLING",

    "REFUEL":
        "REFUELLING",
}


# =============================================================================
# 6. COLUMN ALIASES
# =============================================================================

COLUMN_ALIASES = {

    "timestamp": [
        "Timestamp",
        "timestamp",
    ],

    "speed_ms": [
        "Speed (m/s)",
        "Speed (m/sec)",
        "speed_mps",
    ],

    "speed_kmph": [
        "Speed (km/h)",
        "Speed (kmph)",
        "speed_kmph",
    ],

    "time_diff_s": [
        "Time Difference (s)",
        "Time Difference (sec)",
        "Time Difference (seconds)",
        "time_diff_s",
    ],

    "distance_m": [
        "Haversine Distance (m)",
        "Haversine Distance (meters)",
        "distance_m",
    ],

    "acceleration": [
        "Acceleration (m/s^2)",
        "Acceleration (m/s²)",
        "Acceleration (m/sec^2)",
        "Acceleration (m/sec²)",
        "acceleration",
    ],

    "phase": [
        "phase",
        "Phase",
    ],
}


# =============================================================================
# 7. NORMALIZE COLUMN NAME
# =============================================================================

def normalize_column_name(name):

    if name is None:
        return ""

    text = str(name).strip()

    replacements = {

        "²":
            "^2",

        "³":
            "^3",

        "₁":
            "1",

        "₂":
            "2",

        "₃":
            "3",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# =============================================================================
# 8. RESOLVE SOURCE COLUMN
# =============================================================================

def resolve_column(
    df,
    logical_name
):

    aliases = COLUMN_ALIASES.get(
        logical_name,
        []
    )

    normalized_actual = {

        normalize_column_name(col):
            col

        for col in df.columns
    }

    for alias in aliases:

        key = normalize_column_name(
            alias
        )

        if key in normalized_actual:

            return normalized_actual[key]

    return None


# =============================================================================
# 9. STANDARDIZE COLUMNS
# =============================================================================

def standardize_columns(df):

    df = df.copy()

    rename_map = {}

    logical_to_internal = {

        "timestamp":
            "Timestamp",

        "speed_ms":
            "Speed_ms",

        "speed_kmph":
            "Speed_kmph",

        "time_diff_s":
            "Time_Difference_s",

        "distance_m":
            "Distance_m",

        "acceleration":
            "Acceleration",

        "phase":
            "phase",
    }

    for (
        logical_name,
        internal_name
    ) in logical_to_internal.items():

        actual_col = resolve_column(
            df,
            logical_name
        )

        if actual_col is not None:

            rename_map[
                actual_col
            ] = internal_name

    return df.rename(
        columns=rename_map
    )


# =============================================================================
# 10. NORMALIZE PHASE LABELS
# =============================================================================

def normalize_phase_labels(
    df,
    vehicle_type,
    pair_id
):
    """
    Normalize already-existing Part-1 phase labels.

    IMPORTANT:
        This does NOT perform segmentation.

    It only converts equivalent labels to the canonical
    names used by the TS dataset.

    Example:

        INCH  -> INCHING
        ACC   -> ACCELERATION
        DEC   -> DECELERATION
        CRUISE -> CRUISING
    """

    df = df.copy()

    original_phases = (
        df["phase"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    normalized_phases = (
        original_phases
        .map(PHASE_NORMALIZATION)
    )

    # -------------------------------------------------------------------------
    # Detect unknown phase labels.
    # -------------------------------------------------------------------------

    unknown_mask = (
        normalized_phases.isna()
    )

    unknown_phases = sorted(
        original_phases[
            unknown_mask
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if unknown_phases:

        raise ValueError(

            f"{vehicle_type} telemetry for "
            f"{pair_id} contains unsupported phase labels:\n"
            f"{unknown_phases}\n\n"

            f"Known phase labels are:\n"
            f"{sorted(PHASE_NORMALIZATION.keys())}"
        )

    df["phase"] = (
        normalized_phases
    )

    return df


# =============================================================================
# 11. VALIDATE REQUIRED COLUMNS
# =============================================================================

def validate_required_columns(
    df,
    vehicle_type,
    pair_id
):

    required = [

        "Timestamp",

        "Speed_kmph",

        "Time_Difference_s",

        "Distance_m",

        "Acceleration",

        "phase",
    ]

    missing = [

        col

        for col in required

        if col not in df.columns
    ]

    if missing:

        raise ValueError(

            f"{vehicle_type} telemetry for "
            f"{pair_id} is missing required "
            f"columns: {missing}\n\n"

            f"Available columns:\n"
            f"{list(df.columns)}"
        )


# =============================================================================
# 12. EXTRACT PAIR ID
# =============================================================================

def extract_pair_id(path):

    match = re.search(
        r"(PAIR[_-]\d+)",
        path.stem.upper()
    )

    if match is None:

        raise ValueError(

            f"Could not extract Pair ID "
            f"from file:\n{path}"
        )

    return (
        match
        .group(1)
        .replace(
            "-",
            "_"
        )
    )


# =============================================================================
# 13. FIND STRICT FILES
# =============================================================================

def find_strict_files():

    if not PAIRED_DIR.exists():

        raise FileNotFoundError(

            "Paired directory does not exist:\n"
            f"{PAIRED_DIR}"
        )

    files = []

    for path in PAIRED_DIR.rglob(
        "*.xlsx"
    ):

        stem_upper = (
            path.stem.upper()
        )

        if "_STRICT_" in stem_upper:

            files.append(path)

    return sorted(files)


# =============================================================================
# 14. LOAD TELEMETRY
# =============================================================================

def load_telemetry(
    file_path,
    sheet_name
):

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name
    )

    original_columns = list(
        df.columns
    )

    df = standardize_columns(
        df
    )

    return (
        df,
        original_columns
    )


# =============================================================================
# 15. PREPARE TELEMETRY
# =============================================================================

def prepare_telemetry(
    df,
    vehicle_type,
    pair_id
):
    """
    Prepare already-processed Part-1 telemetry.

    No cleaning.
    No feature engineering.
    No phase re-segmentation.
    """

    df = df.copy()

    validate_required_columns(
        df,
        vehicle_type,
        pair_id
    )

    # -------------------------------------------------------------------------
    # Timestamp
    # -------------------------------------------------------------------------

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Numeric fields
    # -------------------------------------------------------------------------

    numeric_columns = [

        "Speed_kmph",

        "Time_Difference_s",

        "Distance_m",

        "Acceleration",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # -------------------------------------------------------------------------
    # Phase label normalization
    # -------------------------------------------------------------------------

    df = normalize_phase_labels(
        df,
        vehicle_type,
        pair_id
    )

    # -------------------------------------------------------------------------
    # Chronological ordering
    # -------------------------------------------------------------------------

    df = (
        df
        .sort_values(
            "Timestamp",
            kind="stable"
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------------------
    # Duration
    #
    # Part-1 Time Difference:
    #
    # previous datapoint -> current datapoint
    # -------------------------------------------------------------------------

    df["Duration_min"] = (
        df["Time_Difference_s"]
        .clip(lower=0)
        / 60.0
    )

    # -------------------------------------------------------------------------
    # Distance
    # -------------------------------------------------------------------------

    df["Distance_km"] = (
        df["Distance_m"]
        .clip(lower=0)
        / 1000.0
    )

    # -------------------------------------------------------------------------
    # Moving indicator
    #
    # Used only for moving-average speed.
    # -------------------------------------------------------------------------

    df["Moving"] = (
        df["Speed_kmph"]
        > MIN_SPEED_KMPH
    )

    return df


# =============================================================================
# 16. ACTUAL TRIP PARAMETERS
# =============================================================================

def calculate_actual_parameters(df):

    total_distance_km = (
        df["Distance_km"]
        .sum()
    )

    total_time_min = (
        df["Duration_min"]
        .sum()
    )

    moving_time_min = (
        df.loc[
            df["Moving"],
            "Duration_min"
        ]
        .sum()
    )

    moving_distance_km = (
        df.loc[
            df["Moving"],
            "Distance_km"
        ]
        .sum()
    )

    if moving_time_min > 0:

        moving_average_speed = (
            moving_distance_km
            /
            (moving_time_min / 60.0)
        )

    else:

        moving_average_speed = np.nan

    phase_time = {}

    phase_distance = {}

    for phase in PREDICTIVE_PHASES:

        mask = (
            df["phase"]
            == phase
        )

        phase_time[phase] = (
            df.loc[
                mask,
                "Duration_min"
            ]
            .sum()
        )

        phase_distance[phase] = (
            df.loc[
                mask,
                "Distance_km"
            ]
            .sum()
        )

    return {

        "Total_Distance_km":
            total_distance_km,

        "Total_TAT_min":
            total_time_min,

        "Moving_Average_Speed_kmh":
            moving_average_speed,

        "Phase_Time":
            phase_time,

        "Phase_Distance":
            phase_distance,
    }


# =============================================================================
# 17. EXCLUDED OPERATIONAL TIME
# =============================================================================

def calculate_excluded_operational_time(df):

    six_phase_mask = (
        df["phase"]
        .isin(
            PREDICTIVE_PHASES
        )
    )

    six_phase_time = (
        df.loc[
            six_phase_mask,
            "Duration_min"
        ]
        .sum()
    )

    total_time = (
        df["Duration_min"]
        .sum()
    )

    excluded_time = max(
        total_time
        - six_phase_time,
        0.0
    )

    return {

        "Six_Phase_Time_min":
            six_phase_time,

        "Total_TAT_min":
            total_time,

        "Excluded_Operational_Time_min":
            excluded_time,
    }


# =============================================================================
# 18. PHASE OUTPUT NAME
# =============================================================================

def phase_output_name(phase):

    return (
        phase
        .replace(
            "ACCELERATION",
            "ACC"
        )
        .replace(
            "DECELERATION",
            "DEC"
        )
        .replace(
            "CRUISING",
            "CRUISE"
        )
    )


# =============================================================================
# 19. CREATE EMPTY BIN RECORD
# =============================================================================

def create_empty_bin_record(
    vehicle_type,
    bin_id,
    n_bins
):

    route_progress = (
        (bin_id + 0.5)
        / n_bins
    )

    record = {

        "Route_Bin":
            bin_id,

        "Route_Progress":
            route_progress,

        f"{vehicle_type}_Time_Total_min":
            0.0,

        f"{vehicle_type}_Distance_Total_km":
            0.0,

        f"{vehicle_type}_Moving_Distance_km":
            0.0,

        f"{vehicle_type}_Moving_Time_min":
            0.0,

        f"{vehicle_type}_Moving_Average_Speed_kmh":
            0.0,

        f"{vehicle_type}_Acceleration_Time_Weighted_Sum":
            0.0,

        f"{vehicle_type}_Acceleration_Weight":
            0.0,

        f"{vehicle_type}_Excluded_Operational_Time_min":
            0.0,
    }

    for phase in PREDICTIVE_PHASES:

        phase_clean = phase_output_name(
            phase
        )

        record[
            f"{vehicle_type}_{phase_clean}_Time_min"
        ] = 0.0

        record[
            f"{vehicle_type}_{phase_clean}_Distance_km"
        ] = 0.0

    return record


# =============================================================================
# 20. ADD INTERVAL CONTRIBUTION
# =============================================================================

def add_interval_to_bin(
    record,
    df_row,
    vehicle_type,
    allocated_distance_km,
    allocated_time_min
):

    # -------------------------------------------------------------------------
    # Total time
    # -------------------------------------------------------------------------

    record[
        f"{vehicle_type}_Time_Total_min"
    ] += allocated_time_min

    # -------------------------------------------------------------------------
    # Total distance
    # -------------------------------------------------------------------------

    record[
        f"{vehicle_type}_Distance_Total_km"
    ] += allocated_distance_km

    # -------------------------------------------------------------------------
    # Moving contribution
    # -------------------------------------------------------------------------

    if bool(df_row["Moving"]):

        record[
            f"{vehicle_type}_Moving_Distance_km"
        ] += allocated_distance_km

        record[
            f"{vehicle_type}_Moving_Time_min"
        ] += allocated_time_min

    # -------------------------------------------------------------------------
    # Acceleration
    # -------------------------------------------------------------------------

    acceleration = df_row["Acceleration"]

    if pd.notna(acceleration):

        record[
            f"{vehicle_type}_Acceleration_Time_Weighted_Sum"
        ] += (
            float(acceleration)
            * allocated_time_min
        )

        record[
            f"{vehicle_type}_Acceleration_Weight"
        ] += allocated_time_min

    # -------------------------------------------------------------------------
    # Phase contribution
    # -------------------------------------------------------------------------

    phase = str(
        df_row["phase"]
    ).strip().upper()

    if phase in PREDICTIVE_PHASES:

        phase_clean = phase_output_name(
            phase
        )

        record[
            f"{vehicle_type}_{phase_clean}_Time_min"
        ] += allocated_time_min

        record[
            f"{vehicle_type}_{phase_clean}_Distance_km"
        ] += allocated_distance_km

    else:

        record[
            f"{vehicle_type}_Excluded_Operational_Time_min"
        ] += allocated_time_min


# =============================================================================
# 21. ROUTE-PROGRESS REPRESENTATION
# =============================================================================

def create_route_progress_representation(
    df,
    vehicle_type,
    n_bins=N_BINS
):
    """
    Convert one telemetry trip into a distance-based
    route-progress representation.

    Positive-distance intervals:
        allocated according to route-bin overlap.

    Zero-distance intervals:
        time assigned to current route position.

    Total distance and total time are conserved.
    """

    df = df.copy()

    # =========================================================================
    # Total distance
    # =========================================================================

    total_distance = (
        df["Distance_km"]
        .sum()
    )

    if total_distance <= 0:

        raise ValueError(

            f"{vehicle_type} trip has "
            f"zero total distance."
        )

    # =========================================================================
    # Cumulative distance START
    # =========================================================================

    df[
        "Cumulative_Distance_Start_km"
    ] = (
        df["Distance_km"]
        .cumsum()
        - df["Distance_km"]
    )

    # =========================================================================
    # Cumulative distance END
    # =========================================================================

    df[
        "Cumulative_Distance_End_km"
    ] = (
        df["Distance_km"]
        .cumsum()
        .clip(
            upper=total_distance
        )
    )

    # =========================================================================
    # Route-progress START
    # =========================================================================

    df[
        "Route_Progress_Start"
    ] = (
        df[
            "Cumulative_Distance_Start_km"
        ]
        /
        total_distance
    ).clip(
        0,
        1
    )

    # =========================================================================
    # Route-progress END
    # =========================================================================

    df[
        "Route_Progress_End"
    ] = (
        df[
            "Cumulative_Distance_End_km"
        ]
        /
        total_distance
    ).clip(
        0,
        1
    )

    # =========================================================================
    # Create 100 bins
    # =========================================================================

    records = [

        create_empty_bin_record(
            vehicle_type,
            bin_id,
            n_bins
        )

        for bin_id in range(
            n_bins
        )
    ]

    # =========================================================================
    # Process intervals
    # =========================================================================

    for _, row in df.iterrows():

        interval_distance = float(
            row["Distance_km"]
        )

        interval_time = float(
            row["Duration_min"]
        )

        start_distance = float(
            row[
                "Cumulative_Distance_Start_km"
            ]
        )

        end_distance = float(
            row[
                "Cumulative_Distance_End_km"
            ]
        )

        # =====================================================================
        # CASE A: ZERO-DISTANCE INTERVAL
        # =====================================================================

        if interval_distance <= 0:

            progress = (
                start_distance
                /
                total_distance
            )

            progress = float(
                np.clip(
                    progress,
                    0.0,
                    1.0
                )
            )

            bin_id = int(
                np.floor(
                    progress
                    * n_bins
                )
            )

            bin_id = int(
                np.clip(
                    bin_id,
                    0,
                    n_bins - 1
                )
            )

            add_interval_to_bin(

                records[bin_id],

                row,

                vehicle_type,

                allocated_distance_km=0.0,

                allocated_time_min=interval_time
            )

            continue

        # =====================================================================
        # CASE B: POSITIVE-DISTANCE INTERVAL
        # =====================================================================

        start_progress = (
            start_distance
            /
            total_distance
        )

        end_progress = (
            end_distance
            /
            total_distance
        )

        start_progress = float(
            np.clip(
                start_progress,
                0.0,
                1.0
            )
        )

        end_progress = float(
            np.clip(
                end_progress,
                0.0,
                1.0
            )
        )

        if end_progress < start_progress:

            start_progress, end_progress = (
                end_progress,
                start_progress
            )

        # ---------------------------------------------------------------------
        # Determine crossed bins
        # ---------------------------------------------------------------------

        first_bin = int(
            np.floor(
                start_progress
                * n_bins
            )
        )

        last_bin = int(
            np.floor(
                end_progress
                * n_bins
            )
        )

        first_bin = int(
            np.clip(
                first_bin,
                0,
                n_bins - 1
            )
        )

        last_bin = int(
            np.clip(
                last_bin,
                0,
                n_bins - 1
            )
        )

        # ---------------------------------------------------------------------
        # Allocate according to route overlap
        # ---------------------------------------------------------------------

        total_allocated_distance = 0.0

        total_allocated_time = 0.0

        interval_progress = (
            end_progress
            - start_progress
        )

        for bin_id in range(
            first_bin,
            last_bin + 1
        ):

            bin_start_progress = (
                bin_id
                /
                n_bins
            )

            bin_end_progress = (
                (bin_id + 1)
                /
                n_bins
            )

            overlap_start = max(
                start_progress,
                bin_start_progress
            )

            overlap_end = min(
                end_progress,
                bin_end_progress
            )

            overlap_progress = max(
                overlap_end
                - overlap_start,
                0.0
            )

            if overlap_progress <= 0:

                continue

            # -----------------------------------------------------------------
            # Distance allocation
            # -----------------------------------------------------------------

            allocated_distance = (
                overlap_progress
                * total_distance
            )

            # -----------------------------------------------------------------
            # Time allocation
            # -----------------------------------------------------------------

            if interval_progress > 0:

                allocation_fraction = (
                    overlap_progress
                    /
                    interval_progress
                )

            else:

                allocation_fraction = 1.0

            allocation_fraction = float(
                np.clip(
                    allocation_fraction,
                    0.0,
                    1.0
                )
            )

            allocated_time = (
                interval_time
                * allocation_fraction
            )

            # -----------------------------------------------------------------
            # Add contribution
            # -----------------------------------------------------------------

            add_interval_to_bin(

                records[bin_id],

                row,

                vehicle_type,

                allocated_distance_km=allocated_distance,

                allocated_time_min=allocated_time
            )

            total_allocated_distance += (
                allocated_distance
            )

            total_allocated_time += (
                allocated_time
            )

        # =====================================================================
        # Numerical residual correction
        # =====================================================================

        distance_error = (
            interval_distance
            - total_allocated_distance
        )

        time_error = (
            interval_time
            - total_allocated_time
        )

        # ---------------------------------------------------------------------
        # Distance residual
        # ---------------------------------------------------------------------

        if abs(distance_error) > CONSERVATION_TOLERANCE:

            records[last_bin][
                f"{vehicle_type}_Distance_Total_km"
            ] += distance_error

            if bool(row["Moving"]):

                records[last_bin][
                    f"{vehicle_type}_Moving_Distance_km"
                ] += distance_error

            phase = str(
                row["phase"]
            ).strip().upper()

            if phase in PREDICTIVE_PHASES:

                phase_clean = phase_output_name(
                    phase
                )

                records[last_bin][
                    f"{vehicle_type}_{phase_clean}_Distance_km"
                ] += distance_error

        # ---------------------------------------------------------------------
        # Time residual
        # ---------------------------------------------------------------------

        if abs(time_error) > CONSERVATION_TOLERANCE:

            records[last_bin][
                f"{vehicle_type}_Time_Total_min"
            ] += time_error

            if bool(row["Moving"]):

                records[last_bin][
                    f"{vehicle_type}_Moving_Time_min"
                ] += time_error

            phase = str(
                row["phase"]
            ).strip().upper()

            if phase in PREDICTIVE_PHASES:

                phase_clean = phase_output_name(
                    phase
                )

                records[last_bin][
                    f"{vehicle_type}_{phase_clean}_Time_min"
                ] += time_error

            else:

                records[last_bin][
                    f"{vehicle_type}_Excluded_Operational_Time_min"
                ] += time_error

    # =========================================================================
    # Convert to dataframe
    # =========================================================================

    result = pd.DataFrame(
        records
    )

    # =========================================================================
    # Moving average speed
    # =========================================================================

    moving_speed_column = (
        f"{vehicle_type}_Moving_Average_Speed_kmh"
    )

    moving_distance_column = (
        f"{vehicle_type}_Moving_Distance_km"
    )

    moving_time_column = (
        f"{vehicle_type}_Moving_Time_min"
    )

    result[moving_speed_column] = np.where(

        result[moving_time_column] > 0,

        result[moving_distance_column]
        /
        (
            result[moving_time_column]
            / 60.0
        ),

        0.0
    )

    # =========================================================================
    # Time-weighted acceleration
    # =========================================================================

    acceleration_column = (
        f"{vehicle_type}_Mean_Acceleration_ms2"
    )

    acceleration_sum_column = (
        f"{vehicle_type}_Acceleration_Time_Weighted_Sum"
    )

    acceleration_weight_column = (
        f"{vehicle_type}_Acceleration_Weight"
    )

    result[acceleration_column] = np.where(

        result[
            acceleration_weight_column
        ] > 0,

        result[
            acceleration_sum_column
        ]
        /
        result[
            acceleration_weight_column
        ],

        0.0
    )

    # =========================================================================
    # Drop internal audit columns
    # =========================================================================

    result = result.drop(

        columns=[

            acceleration_sum_column,

            acceleration_weight_column,

            moving_distance_column,

            moving_time_column,
        ],

        errors="ignore"
    )

    return result


# =============================================================================
# 22. SELECT ICE MODEL INPUT FEATURES
# =============================================================================

def select_ice_features(df):

    columns = [

        "Route_Progress",

        "ICE_Time_Total_min",

        "ICE_Distance_Total_km",

        "ICE_Moving_Average_Speed_kmh",

        "ICE_Mean_Acceleration_ms2",

        "ICE_STOP_Time_min",

        "ICE_IDLE_Time_min",

        "ICE_INCHING_Time_min",

        "ICE_ACC_Time_min",

        "ICE_DEC_Time_min",

        "ICE_CRUISE_Time_min",

        "ICE_STOP_Distance_km",

        "ICE_IDLE_Distance_km",

        "ICE_INCHING_Distance_km",

        "ICE_ACC_Distance_km",

        "ICE_DEC_Distance_km",

        "ICE_CRUISE_Distance_km",
    ]

    return df[
        columns
    ].copy()


# =============================================================================
# 23. SELECT EV MODEL TARGETS
# =============================================================================

def select_ev_targets(df):

    columns = [

        "EV_Moving_Average_Speed_kmh",

        "EV_STOP_Time_min",

        "EV_IDLE_Time_min",

        "EV_INCHING_Time_min",

        "EV_ACC_Time_min",

        "EV_DEC_Time_min",

        "EV_CRUISE_Time_min",

        "EV_STOP_Distance_km",

        "EV_IDLE_Distance_km",

        "EV_INCHING_Distance_km",

        "EV_ACC_Distance_km",

        "EV_DEC_Distance_km",

        "EV_CRUISE_Distance_km",
    ]

    return df[
        columns
    ].copy()


# =============================================================================
# 24. VALIDATE TS ANALYSIS WORKBOOK
# =============================================================================

def validate_analysis_workbook():

    if not ANALYSIS_FILE.exists():

        raise FileNotFoundError(

            "Completed TS analysis workbook "
            "not found:\n"
            f"{ANALYSIS_FILE}"
        )

    xls = pd.ExcelFile(
        ANALYSIS_FILE
    )

    sheets = set(
        xls.sheet_names
    )

    required_sheets = {

        "Master_Summary",

        "1_Telemetry",

        "2_Phase_Distribution",

        "3_Sampling",

        "4_Trip_Alignment",

        "5_Phase_Summary",

        "6_Dataset_Design",

        "7_Model_Design",

        "7_Readiness",
    }

    missing = sorted(
        required_sheets
        - sheets
    )

    if missing:

        raise ValueError(

            "TS analysis workbook "
            "validation failed.\n"

            f"Missing sheets: {missing}"
        )

    return (
        True,
        xls.sheet_names
    )


# =============================================================================
# 25. VALIDATE ROUTE PROGRESS
# =============================================================================

def validate_route_progress(
    pair_id,
    merged_df
):

    checks = []

    # -------------------------------------------------------------------------
    # Number of bins
    # -------------------------------------------------------------------------

    checks.append({

        "Pair_ID":
            pair_id,

        "Check":
            "Expected number of route bins",

        "Expected":
            N_BINS,

        "Actual":
            len(merged_df),

        "Status":
            (
                "PASS"
                if len(merged_df) == N_BINS
                else "FAIL"
            ),
    })

    # -------------------------------------------------------------------------
    # Route progress
    # -------------------------------------------------------------------------

    progress_valid = (

        merged_df[
            "Route_Progress"
        ]
        .between(
            0,
            1
        )
        .all()
    )

    checks.append({

        "Pair_ID":
            pair_id,

        "Check":
            "Route progress between 0 and 1",

        "Expected":
            True,

        "Actual":
            progress_valid,

        "Status":
            (
                "PASS"
                if progress_valid
                else "FAIL"
            ),
    })

    # -------------------------------------------------------------------------
    # Route bins
    # -------------------------------------------------------------------------

    expected_bins = np.arange(
        N_BINS
    )

    actual_bins = (
        merged_df[
            "Route_Bin"
        ]
        .to_numpy()
    )

    bins_are_complete = (
        np.array_equal(
            actual_bins,
            expected_bins
        )
    )

    checks.append({

        "Pair_ID":
            pair_id,

        "Check":
            "Route bins complete and ordered",

        "Expected":
            "0 to 99",

        "Actual":
            (
                "0 to 99"
                if bins_are_complete
                else str(actual_bins)
            ),

        "Status":
            (
                "PASS"
                if bins_are_complete
                else "FAIL"
            ),
    })

    # -------------------------------------------------------------------------
    # Duplicate bins
    # -------------------------------------------------------------------------

    duplicate_bins = (
        merged_df[
            "Route_Bin"
        ]
        .duplicated()
        .any()
    )

    checks.append({

        "Pair_ID":
            pair_id,

        "Check":
            "No duplicate route bins",

        "Expected":
            False,

        "Actual":
            duplicate_bins,

        "Status":
            (
                "PASS"
                if not duplicate_bins
                else "FAIL"
            ),
    })

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    missing_count = (
        merged_df
        .isna()
        .sum()
        .sum()
    )

    checks.append({

        "Pair_ID":
            pair_id,

        "Check":
            "Missing values",

        "Expected":
            0,

        "Actual":
            int(missing_count),

        "Status":
            (
                "PASS"
                if missing_count == 0
                else "FAIL"
            ),
    })

    return checks


# =============================================================================
# 26. CALCULATE CONSERVATION
# =============================================================================

def calculate_conservation(
    original_df,
    route_df,
    vehicle_type
):

    original_total_distance = (
        original_df["Distance_km"]
        .sum()
    )

    original_total_time = (
        original_df["Duration_min"]
        .sum()
    )

    route_total_distance = (
        route_df[
            f"{vehicle_type}_Distance_Total_km"
        ]
        .sum()
    )

    route_total_time = (
        route_df[
            f"{vehicle_type}_Time_Total_min"
        ]
        .sum()
    )

    # -------------------------------------------------------------------------
    # Six phase time
    # -------------------------------------------------------------------------

    route_six_phase_time = 0.0

    for phase in PREDICTIVE_PHASES:

        phase_clean = phase_output_name(
            phase
        )

        route_six_phase_time += (
            route_df[
                f"{vehicle_type}_{phase_clean}_Time_min"
            ]
            .sum()
        )

    # -------------------------------------------------------------------------
    # Excluded operational time
    # -------------------------------------------------------------------------

    route_excluded_time = (
        route_df[
            f"{vehicle_type}_Excluded_Operational_Time_min"
        ]
        .sum()
    )

    reconstructed_time = (
        route_six_phase_time
        +
        route_excluded_time
    )

    # -------------------------------------------------------------------------
    # Six phase distance
    # -------------------------------------------------------------------------

    route_six_phase_distance = 0.0

    for phase in PREDICTIVE_PHASES:

        phase_clean = phase_output_name(
            phase
        )

        route_six_phase_distance += (
            route_df[
                f"{vehicle_type}_{phase_clean}_Distance_km"
            ]
            .sum()
        )

    # -------------------------------------------------------------------------
    # Remaining distance
    # -------------------------------------------------------------------------

    route_excluded_distance = max(

        route_total_distance
        -
        route_six_phase_distance,

        0.0
    )

    reconstructed_distance = (
        route_six_phase_distance
        +
        route_excluded_distance
    )

    # -------------------------------------------------------------------------
    # Errors
    # -------------------------------------------------------------------------

    distance_error = (
        route_total_distance
        -
        original_total_distance
    )

    time_error = (
        route_total_time
        -
        original_total_time
    )

    phase_time_error = (
        reconstructed_time
        -
        original_total_time
    )

    distance_reconstruction_error = (
        reconstructed_distance
        -
        original_total_distance
    )

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    distance_status = (
        "PASS"
        if abs(distance_error)
        <= CONSERVATION_TOLERANCE
        else "FAIL"
    )

    time_status = (
        "PASS"
        if abs(time_error)
        <= CONSERVATION_TOLERANCE
        else "FAIL"
    )

    phase_time_status = (
        "PASS"
        if abs(phase_time_error)
        <= CONSERVATION_TOLERANCE
        else "FAIL"
    )

    phase_distance_status = (
        "PASS"
        if abs(distance_reconstruction_error)
        <= CONSERVATION_TOLERANCE
        else "FAIL"
    )

    return {

        "Vehicle":
            vehicle_type,

        "Original_Total_Distance_km":
            original_total_distance,

        "Route_Total_Distance_km":
            route_total_distance,

        "Distance_Error_km":
            distance_error,

        "Distance_Status":
            distance_status,

        "Original_Total_TAT_min":
            original_total_time,

        "Route_Total_TAT_min":
            route_total_time,

        "Time_Error_min":
            time_error,

        "Time_Status":
            time_status,

        "Route_Six_Phase_Time_min":
            route_six_phase_time,

        "Route_Excluded_Operational_Time_min":
            route_excluded_time,

        "Reconstructed_Total_Time_min":
            reconstructed_time,

        "Six_Phase_Time_Accounting_Error_min":
            phase_time_error,

        "Six_Phase_Time_Accounting_Status":
            phase_time_status,

        "Route_Six_Phase_Distance_km":
            route_six_phase_distance,

        "Route_Excluded_Distance_km":
            route_excluded_distance,

        "Reconstructed_Total_Distance_km":
            reconstructed_distance,

        "Phase_Distance_Accounting_Error_km":
            distance_reconstruction_error,

        "Phase_Distance_Accounting_Status":
            phase_distance_status,
    }


# =============================================================================
# 27. PROCESS ONE PAIR
# =============================================================================

def process_pair(file_path):

    pair_id = extract_pair_id(
        file_path
    )

    print(
        "\n"
        + "-"
        * 80
    )

    print(
        f"Pair: {pair_id}"
    )

    print(
        f"File: {file_path}"
    )

    # =========================================================================
    # Load telemetry
    # =========================================================================

    ice_raw, ice_original_columns = (
        load_telemetry(
            file_path,
            "02_ICE_Cropped_Data"
        )
    )

    ev_raw, ev_original_columns = (
        load_telemetry(
            file_path,
            "01_EV_Cropped_Data"
        )
    )

    print(
        f"ICE telemetry rows: "
        f"{len(ice_raw)}"
    )

    print(
        f"EV telemetry rows : "
        f"{len(ev_raw)}"
    )

    # =========================================================================
    # Prepare
    # =========================================================================

    ice = prepare_telemetry(
        ice_raw,
        "ICE",
        pair_id
    )

    ev = prepare_telemetry(
        ev_raw,
        "EV",
        pair_id
    )

    # =========================================================================
    # Actual parameters
    # =========================================================================

    ice_actual = (
        calculate_actual_parameters(
            ice
        )
    )

    ev_actual = (
        calculate_actual_parameters(
            ev
        )
    )

    # =========================================================================
    # Excluded operational time
    # =========================================================================

    ice_excluded = (
        calculate_excluded_operational_time(
            ice
        )
    )

    ev_excluded = (
        calculate_excluded_operational_time(
            ev
        )
    )

    # =========================================================================
    # Route representation
    # =========================================================================

    ice_route = (
        create_route_progress_representation(
            ice,
            "ICE",
            N_BINS
        )
    )

    ev_route = (
        create_route_progress_representation(
            ev,
            "EV",
            N_BINS
        )
    )

    # =========================================================================
    # Construction conservation
    # =========================================================================

    ice_conservation = (
        calculate_conservation(
            ice,
            ice_route,
            "ICE"
        )
    )

    ev_conservation = (
        calculate_conservation(
            ev,
            ev_route,
            "EV"
        )
    )

    # =========================================================================
    # Merge by route coordinate only
    # =========================================================================

    merged = pd.merge(

        ice_route,

        ev_route,

        on=[
            "Route_Bin",
            "Route_Progress",
        ],

        how="inner",

        validate="one_to_one",
    )

    # =========================================================================
    # Pair ID
    # =========================================================================

    merged.insert(
        0,
        "Pair_ID",
        pair_id
    )

    # =========================================================================
    # Structural validation
    # =========================================================================

    validation = (
        validate_route_progress(
            pair_id,
            merged
        )
    )

    validation_df = pd.DataFrame(
        validation
    )

    # =========================================================================
    # Conservation records
    # =========================================================================

    conservation_records = [

        {
            "Pair_ID":
                pair_id,

            "Check":
                "ICE distance conservation",

            "Expected":
                0.0,

            "Actual":
                ice_conservation[
                    "Distance_Error_km"
                ],

            "Status":
                ice_conservation[
                    "Distance_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "ICE total time conservation",

            "Expected":
                0.0,

            "Actual":
                ice_conservation[
                    "Time_Error_min"
                ],

            "Status":
                ice_conservation[
                    "Time_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "ICE six-phase time accounting",

            "Expected":
                0.0,

            "Actual":
                ice_conservation[
                    "Six_Phase_Time_Accounting_Error_min"
                ],

            "Status":
                ice_conservation[
                    "Six_Phase_Time_Accounting_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "ICE phase distance accounting",

            "Expected":
                0.0,

            "Actual":
                ice_conservation[
                    "Phase_Distance_Accounting_Error_km"
                ],

            "Status":
                ice_conservation[
                    "Phase_Distance_Accounting_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "EV distance conservation",

            "Expected":
                0.0,

            "Actual":
                ev_conservation[
                    "Distance_Error_km"
                ],

            "Status":
                ev_conservation[
                    "Distance_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "EV total time conservation",

            "Expected":
                0.0,

            "Actual":
                ev_conservation[
                    "Time_Error_min"
                ],

            "Status":
                ev_conservation[
                    "Time_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "EV six-phase time accounting",

            "Expected":
                0.0,

            "Actual":
                ev_conservation[
                    "Six_Phase_Time_Accounting_Error_min"
                ],

            "Status":
                ev_conservation[
                    "Six_Phase_Time_Accounting_Status"
                ],
        },

        {
            "Pair_ID":
                pair_id,

            "Check":
                "EV phase distance accounting",

            "Expected":
                0.0,

            "Actual":
                ev_conservation[
                    "Phase_Distance_Accounting_Error_km"
                ],

            "Status":
                ev_conservation[
                    "Phase_Distance_Accounting_Status"
                ],
        },
    ]

    validation_df = pd.concat(

        [
            validation_df,

            pd.DataFrame(
                conservation_records
            )
        ],

        ignore_index=True
    )

    # =========================================================================
    # Model input
    # =========================================================================

    ice_features = (
        select_ice_features(
            merged
        )
    )

    ice_features.insert(
        0,
        "Pair_ID",
        pair_id
    )

    # =========================================================================
    # Model targets
    # =========================================================================

    ev_targets = (
        select_ev_targets(
            merged
        )
    )

    ev_targets.insert(
        0,
        "Pair_ID",
        pair_id
    )

    # =========================================================================
    # Empty bins
    #
    # Here "empty" means:
    #
    #   no predictive-phase time
    #   AND
    #   no predictive-phase distance
    #
    # This does NOT imply that the original telemetry had no information.
    # =========================================================================

    ice_phase_time_total = (
        merged[
            [
                f"ICE_{phase_output_name(p)}_Time_min"
                for p in PREDICTIVE_PHASES
            ]
        ]
        .sum(axis=1)
    )

    ice_phase_distance_total = (
        merged[
            [
                f"ICE_{phase_output_name(p)}_Distance_km"
                for p in PREDICTIVE_PHASES
            ]
        ]
        .sum(axis=1)
    )

    ev_phase_time_total = (
        merged[
            [
                f"EV_{phase_output_name(p)}_Time_min"
                for p in PREDICTIVE_PHASES
            ]
        ]
        .sum(axis=1)
    )

    ev_phase_distance_total = (
        merged[
            [
                f"EV_{phase_output_name(p)}_Distance_km"
                for p in PREDICTIVE_PHASES
            ]
        ]
        .sum(axis=1)
    )

    ice_empty_bins = int(
        (
            (ice_phase_time_total <= 0)
            &
            (ice_phase_distance_total <= 0)
        ).sum()
    )

    ev_empty_bins = int(
        (
            (ev_phase_time_total <= 0)
            &
            (ev_phase_distance_total <= 0)
        ).sum()
    )

    # =========================================================================
    # Pair summary
    # =========================================================================

    summary = {

        "Pair_ID":
            pair_id,

        "ICE_Rows":
            len(ice),

        "EV_Rows":
            len(ev),

        "ICE_Total_Distance_km":
            ice_actual[
                "Total_Distance_km"
            ],

        "EV_Total_Distance_km":
            ev_actual[
                "Total_Distance_km"
            ],

        "ICE_Total_TAT_min":
            ice_actual[
                "Total_TAT_min"
            ],

        "EV_Total_TAT_min":
            ev_actual[
                "Total_TAT_min"
            ],

        "ICE_Moving_Average_Speed_kmh":
            ice_actual[
                "Moving_Average_Speed_kmh"
            ],

        "EV_Moving_Average_Speed_kmh":
            ev_actual[
                "Moving_Average_Speed_kmh"
            ],

        "Distance_Difference_km":
            (
                ev_actual[
                    "Total_Distance_km"
                ]
                -
                ice_actual[
                    "Total_Distance_km"
                ]
            ),

        "TAT_Difference_min":
            (
                ev_actual[
                    "Total_TAT_min"
                ]
                -
                ice_actual[
                    "Total_TAT_min"
                ]
            ),

        "ICE_Six_Phase_Time_min":
            ice_excluded[
                "Six_Phase_Time_min"
            ],

        "EV_Six_Phase_Time_min":
            ev_excluded[
                "Six_Phase_Time_min"
            ],

        "ICE_Excluded_Operational_Time_min":
            ice_excluded[
                "Excluded_Operational_Time_min"
            ],

        "EV_Excluded_Operational_Time_min":
            ev_excluded[
                "Excluded_Operational_Time_min"
            ],

        "ICE_Empty_Bins":
            ice_empty_bins,

        "EV_Empty_Bins":
            ev_empty_bins,

        "ICE_Empty_Bin_Percent":
            (
                ice_empty_bins
                /
                N_BINS
                * 100.0
            ),

        "EV_Empty_Bin_Percent":
            (
                ev_empty_bins
                /
                N_BINS
                * 100.0
            ),

        "ICE_Distance_Conservation":
            ice_conservation[
                "Distance_Status"
            ],

        "ICE_Time_Conservation":
            ice_conservation[
                "Time_Status"
            ],

        "ICE_Phase_Time_Accounting":
            ice_conservation[
                "Six_Phase_Time_Accounting_Status"
            ],

        "ICE_Phase_Distance_Accounting":
            ice_conservation[
                "Phase_Distance_Accounting_Status"
            ],

        "EV_Distance_Conservation":
            ev_conservation[
                "Distance_Status"
            ],

        "EV_Time_Conservation":
            ev_conservation[
                "Time_Status"
            ],

        "EV_Phase_Time_Accounting":
            ev_conservation[
                "Six_Phase_Time_Accounting_Status"
            ],

        "EV_Phase_Distance_Accounting":
            ev_conservation[
                "Phase_Distance_Accounting_Status"
            ],
    }

    return {

        "pair_id":
            pair_id,

        "merged":
            merged,

        "ice_features":
            ice_features,

        "ev_targets":
            ev_targets,

        "summary":
            summary,

        "validation":
            validation_df,

        "ice_actual":
            ice_actual,

        "ev_actual":
            ev_actual,

        "ice_excluded":
            ice_excluded,

        "ev_excluded":
            ev_excluded,

        "ice_conservation":
            ice_conservation,

        "ev_conservation":
            ev_conservation,
    }


# =============================================================================
# 28. MAIN
# =============================================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    print("=" * 80)

    print(
        "TIME-SERIES DATASET CONSTRUCTION"
    )

    print("=" * 80)

    # =========================================================================
    # STEP 1
    # =========================================================================

    print(
        "\n[1] Loading completed TS analysis..."
    )

    (
        analysis_status,
        analysis_sheets
    ) = validate_analysis_workbook()

    print(
        "Analysis validation: "
        f"{'PASS' if analysis_status else 'FAIL'}"
    )

    print(
        f"Analysis workbook:\n"
        f"{ANALYSIS_FILE}"
    )

    # =========================================================================
    # STEP 2
    # =========================================================================

    print(
        "\n[2] Discovering STRICT paired files..."
    )

    strict_files = (
        find_strict_files()
    )

    print(
        f"STRICT files found: "
        f"{len(strict_files)}"
    )

    if not strict_files:

        raise RuntimeError(
            "No STRICT paired files found."
        )

    for file_path in strict_files:

        print(
            f"  - {file_path}"
        )

    # =========================================================================
    # STEP 3
    # =========================================================================

    print(
        "\n[3] Processing STRICT paired telemetry..."
    )

    pair_results = []

    for file_path in strict_files:

        try:

            result = process_pair(
                file_path
            )

            pair_results.append(
                result
            )

            print(
                "Status: PASS"
            )

            print(
                f"  ICE empty bins: "
                f"{result['summary']['ICE_Empty_Bins']}"
                f"/{N_BINS}"
            )

            print(
                f"  EV empty bins : "
                f"{result['summary']['EV_Empty_Bins']}"
                f"/{N_BINS}"
            )

            print(
                f"  ICE distance conservation: "
                f"{result['summary']['ICE_Distance_Conservation']}"
            )

            print(
                f"  EV distance conservation : "
                f"{result['summary']['EV_Distance_Conservation']}"
            )

            print(
                f"  ICE time conservation: "
                f"{result['summary']['ICE_Time_Conservation']}"
            )

            print(
                f"  EV time conservation : "
                f"{result['summary']['EV_Time_Conservation']}"
            )

        except Exception as exc:

            print(
                "\nERROR while processing:"
            )

            print(
                f"Pair file: "
                f"{file_path}"
            )

            print(
                f"Error: {exc}"
            )

            raise

    # =========================================================================
    # STEP 4
    # =========================================================================

    print(
        "\n[4] Combining pair-level datasets..."
    )

    combined_representation = pd.concat(

        [
            result[
                "merged"
            ]

            for result in pair_results
        ],

        ignore_index=True
    )

    combined_ice = pd.concat(

        [
            result[
                "ice_features"
            ]

            for result in pair_results
        ],

        ignore_index=True
    )

    combined_ev = pd.concat(

        [
            result[
                "ev_targets"
            ]

            for result in pair_results
        ],

        ignore_index=True
    )

    # =========================================================================
    # STEP 5
    # =========================================================================

    pair_summary = pd.DataFrame(

        [
            result[
                "summary"
            ]

            for result in pair_results
        ]
    )

    # =========================================================================
    # STEP 6
    # =========================================================================

    print(
        "\n[5] Dataset validation..."
    )

    validation = pd.concat(

        [
            result[
                "validation"
            ]

            for result in pair_results
        ],

        ignore_index=True
    )

    overall_validation = (

        "PASS"

        if (
            validation[
                "Status"
            ]
            == "PASS"
        ).all()

        else "REVIEW REQUIRED"
    )

    print(
        f"Overall validation: "
        f"{overall_validation}"
    )

    # =========================================================================
    # STEP 7
    # =========================================================================

    print(
        "\n[6] Preparing dataset design information..."
    )

    dataset_design = pd.DataFrame({

        "Item": [

            "Number of STRICT pairs",

            "Number of route bins per pair",

            "Total modelling rows",

            "ICE input columns",

            "EV target columns",

            "Route representation",

            "Common reference axis",

            "Route interval allocation",

            "Zero-distance interval handling",

            "Phase label normalization",

            "Timestamp alignment",

            "Latitude/Longitude used",

            "Charging time used as TS target",

            "Refuelling time used as TS target",

            "ICE refuelling retained in actual ICE TAT",

            "EV charging retained for final TAT reconstruction",

            "Total EV TAT independently predicted per bin",

            "Total EV distance independently predicted per bin",

            "Pair ID retained",

            "Route_Bin retained",

            "Route_Progress retained",

            "Row-wise random split allowed",
        ],

        "Value": [

            len(pair_results),

            N_BINS,

            len(
                combined_representation
            ),

            len(
                combined_ice.columns
            ) - 1,

            len(
                combined_ev.columns
            ) - 1,

            "Distance-based route progression",

            "Cumulative trip distance normalized to 0-1",

            "Interval-overlap allocation",

            "Assigned to current route position",

            "Existing Part-1 labels normalized only",

            "No",

            "No",

            "No",

            "No",

            "Yes",

            "Yes",

            "No",

            "No",

            "Yes",

            "Yes",

            "Yes",

            "No",
        ],
    })

    # =========================================================================
    # STEP 8
    # =========================================================================

    print(
        "\n[7] Saving dataset workbook..."
    )

    with pd.ExcelWriter(

        DATASET_XLSX,

        engine="openpyxl"

    ) as writer:

        pair_summary.to_excel(

            writer,

            sheet_name="Pair_Summary",

            index=False
        )

        combined_representation.to_excel(

            writer,

            sheet_name="Route_Progress_All",

            index=False
        )

        combined_ice.to_excel(

            writer,

            sheet_name="ICE_Model_Input",

            index=False
        )

        combined_ev.to_excel(

            writer,

            sheet_name="EV_Model_Target",

            index=False
        )

        validation.to_excel(

            writer,

            sheet_name="Validation",

            index=False
        )

        dataset_design.to_excel(

            writer,

            sheet_name="Dataset_Design",

            index=False
        )

    print(
        f"Saved:\n"
        f"{DATASET_XLSX}"
    )

    # =========================================================================
    # STEP 9
    # =========================================================================

    print(
        "\n[8] Saving NumPy modelling dataset..."
    )

    X_columns = [

        col

        for col
        in combined_ice.columns

        if col != "Pair_ID"
    ]

    Y_columns = [

        col

        for col
        in combined_ev.columns

        if col != "Pair_ID"
    ]

    X = (
        combined_ice[
            X_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    Y = (
        combined_ev[
            Y_columns
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    pair_ids = (
        combined_ice[
            "Pair_ID"
        ]
        .astype(str)
        .to_numpy()
    )

    route_bins = (
        combined_representation[
            "Route_Bin"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    route_progress = (
        combined_representation[
            "Route_Progress"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    np.savez_compressed(

        DATASET_NPZ,

        X=X,

        Y=Y,

        pair_ids=pair_ids,

        route_bins=route_bins,

        route_progress=route_progress,

        X_columns=np.array(
            X_columns,
            dtype=object
        ),

        Y_columns=np.array(
            Y_columns,
            dtype=object
        ),
    )

    print(
        f"Saved:\n"
        f"{DATASET_NPZ}"
    )

    # =========================================================================
    # STEP 10
    # =========================================================================

    print(
        "\n[9] Saving validation workbook..."
    )

    validation_summary = pd.DataFrame({

        "Metric": [

            "Overall validation status",

            "Number of STRICT pairs",

            "Expected bins per pair",

            "Total dataset rows",

            "X rows",

            "Y rows",

            "X features",

            "Y targets",

            "Missing values in X",

            "Missing values in Y",

            "Unique Pair IDs",
        ],

        "Value": [

            overall_validation,

            len(pair_results),

            N_BINS,

            len(
                combined_representation
            ),

            X.shape[0],

            Y.shape[0],

            X.shape[1],

            Y.shape[1],

            int(
                np.isnan(X)
                .sum()
            ),

            int(
                np.isnan(Y)
                .sum()
            ),

            len(
                np.unique(
                    pair_ids
                )
            ),
        ],
    })

    with pd.ExcelWriter(

        VALIDATION_XLSX,

        engine="openpyxl"

    ) as writer:

        validation_summary.to_excel(

            writer,

            sheet_name="Validation_Summary",

            index=False
        )

        validation.to_excel(

            writer,

            sheet_name="Pair_Validation",

            index=False
        )

        pair_summary.to_excel(

            writer,

            sheet_name="Pair_Summary",

            index=False
        )

        dataset_design.to_excel(

            writer,

            sheet_name="Dataset_Design",

            index=False
        )

    print(
        f"Saved:\n"
        f"{VALIDATION_XLSX}"
    )

    # =========================================================================
    # STEP 11
    # =========================================================================

    print(
        "\n[10] Writing run report..."
    )

    report = []

    report.append(
        "TIME-SERIES DATASET CONSTRUCTION COMPLETE"
    )

    report.append(
        "=" * 80
    )

    report.append(
        f"STRICT pairs      : "
        f"{len(pair_results)}"
    )

    report.append(
        f"Bins per pair     : "
        f"{N_BINS}"
    )

    report.append(
        f"Total rows        : "
        f"{len(combined_representation)}"
    )

    report.append(
        f"ICE inputs        : "
        f"{X.shape[1]}"
    )

    report.append(
        f"EV targets        : "
        f"{Y.shape[1]}"
    )

    report.append(
        f"Validation        : "
        f"{overall_validation}"
    )

    report.append("")

    report.append(
        "PHASE LABEL NORMALIZATION"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "INCH -> INCHING"
    )

    report.append(
        "ACC -> ACCELERATION"
    )

    report.append(
        "DEC -> DECELERATION"
    )

    report.append(
        "CRUISE -> CRUISING"
    )

    report.append(
        "This is label normalization only; "
        "Part-1 phase segmentation is not repeated."
    )

    report.append("")

    report.append(
        "MODELLING REPRESENTATION"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "Common reference axis: "
        "distance-based route progression"
    )

    report.append(
        f"Route bins per pair: {N_BINS}"
    )

    report.append(
        "Telemetry interval allocation: "
        "distance-overlap based"
    )

    report.append(
        "Zero-distance interval handling: "
        "assigned to current route position"
    )

    report.append(
        "ICE-EV timestamp alignment: NOT USED"
    )

    report.append(
        "Latitude/Longitude modelling features: NOT USED"
    )

    report.append(
        "Part-1 cleaning: NOT repeated"
    )

    report.append(
        "Part-1 feature engineering: NOT repeated"
    )

    report.append(
        "Part-1 phase segmentation: NOT repeated"
    )

    report.append(
        "Charging/refuelling: handled separately"
    )

    report.append(
        "EV charging time: NOT a TS prediction target"
    )

    report.append(
        "ICE refuelling time: NOT a TS prediction target"
    )

    report.append(
        "ICE refuelling contribution: "
        "retained in actual ICE TAT"
    )

    report.append(
        "EV charging contribution: "
        "retained for final TAT reconstruction"
    )

    report.append(
        "Pair ID retained for group-aware validation"
    )

    report.append(
        "Route progress retained as sequence coordinate"
    )

    report.append(
        "Route progress is NOT an EV target"
    )

    report.append(
        "Total EV TAT is reconstructed after TS prediction"
    )

    report.append(
        "Total EV distance is reconstructed from phase distances"
    )

    # =========================================================================
    # X columns
    # =========================================================================

    report.append("")

    report.append(
        "X COLUMNS"
    )

    report.append(
        "-" * 80
    )

    for col in X_columns:

        report.append(
            str(col)
        )

    # =========================================================================
    # Y columns
    # =========================================================================

    report.append("")

    report.append(
        "Y COLUMNS"
    )

    report.append(
        "-" * 80
    )

    for col in Y_columns:

        report.append(
            str(col)
        )

    # =========================================================================
    # Pair summary
    # =========================================================================

    report.append("")

    report.append(
        "PAIR SUMMARY"
    )

    report.append(
        "-" * 80
    )

    for _, row in pair_summary.iterrows():

        report.append(

            f"{row['Pair_ID']} | "

            f"ICE distance="
            f"{row['ICE_Total_Distance_km']:.3f} km | "

            f"EV distance="
            f"{row['EV_Total_Distance_km']:.3f} km | "

            f"ICE TAT="
            f"{row['ICE_Total_TAT_min']:.3f} min | "

            f"EV TAT="
            f"{row['EV_Total_TAT_min']:.3f} min | "

            f"ICE six-phase="
            f"{row['ICE_Six_Phase_Time_min']:.3f} min | "

            f"EV six-phase="
            f"{row['EV_Six_Phase_Time_min']:.3f} min | "

            f"ICE empty="
            f"{row['ICE_Empty_Bins']}/{N_BINS} | "

            f"EV empty="
            f"{row['EV_Empty_Bins']}/{N_BINS}"
        )

    # =========================================================================
    # Interpretation
    # =========================================================================

    report.append("")

    report.append(
        "INTERPRETATION NOTES"
    )

    report.append(
        "-" * 80
    )

    report.append(
        "The constructed dataset is a "
        "distance-based route-progress representation."
    )

    report.append(
        "It is not raw telemetry-to-telemetry "
        "timestamp-aligned sequence data."
    )

    report.append(
        "Each pair contains exactly the same "
        "number of route bins."
    )

    report.append(
        "Telemetry intervals crossing multiple route bins "
        "are allocated using route-distance overlap."
    )

    report.append(
        "Zero-distance intervals are assigned to the "
        "route position where they occur."
    )

    report.append(
        "Existing Part-1 phase labels are normalized "
        "without re-segmenting telemetry."
    )

    report.append(
        "Charging/refuelling is excluded from the six-phase "
        "predictive target."
    )

    report.append(
        "Final EV TAT is reconstructed after TS prediction "
        "using separately calculated EV charging time."
    )

    report.append(
        "Conservation checks compare the route representation "
        "against the original Part-1 telemetry."
    )

    report.append(
        "100 bins remain a modelling design choice and should "
        "be evaluated during scientific validation."
    )

    RUN_REPORT.write_text(

        "\n".join(report),

        encoding="utf-8"
    )

    print(
        f"Saved:\n"
        f"{RUN_REPORT}"
    )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print(
        "\n"
        + "="
        * 80
    )

    print(
        "TIME-SERIES DATASET CONSTRUCTION COMPLETE"
    )

    print(
        "="
        * 80
    )

    print(
        f"STRICT pairs      : "
        f"{len(pair_results)}"
    )

    print(
        f"Bins per pair     : "
        f"{N_BINS}"
    )

    print(
        f"Total rows        : "
        f"{len(combined_representation)}"
    )

    print(
        f"ICE inputs        : "
        f"{X.shape[1]}"
    )

    print(
        f"EV targets        : "
        f"{Y.shape[1]}"
    )

    print(
        f"Validation        : "
        f"{overall_validation}"
    )

    print(
        "\nOutput files:"
    )

    print(
        f"1. {DATASET_XLSX}"
    )

    print(
        f"2. {DATASET_NPZ}"
    )

    print(
        f"3. {VALIDATION_XLSX}"
    )

    print(
        f"4. {RUN_REPORT}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Run the scientific validation after this dataset construction."
    )

    print(
        "Do NOT start TS model training until the empty-bin and "
        "conservation behaviour has been inspected."
    )


# =============================================================================
# 29. RUN
# =============================================================================

if __name__ == "__main__":

    main()