# ============================================================
# TS DATA LOADER
# ============================================================
#
# File:
#     src/timeseries/common/ts_data_loader.py
#
# Purpose:
#     Load and validate the scientifically validated
#     route-progress TS dataset.
#
# This module will be shared by:
#
#     - Baseline model
#     - LSTM
#     - GRU
#     - Future TS models
#
# IMPORTANT:
#     This file does NOT:
#
#       - scale the data
#       - create sequences/windows
#       - split train/test data
#       - train a model
#
# Those operations are handled separately.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING"
)

DATASET_FILE = (
    BASE_DIR
    / "Results"
    / "TimeSeries"
    / "Dataset"
    / "TS_Route_Progress_Dataset.xlsx"
)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

EXPECTED_PAIRS = 6

EXPECTED_BINS_PER_PAIR = 100


# ============================================================
# 3. PHASE TIME COLUMNS
# ============================================================

ICE_TIME_COLUMNS = [

    "ICE_STOP_Time_min",

    "ICE_IDLE_Time_min",

    "ICE_INCHING_Time_min",

    "ICE_ACC_Time_min",

    "ICE_DEC_Time_min",

    "ICE_CRUISE_Time_min",
]


EV_TIME_COLUMNS = [

    "EV_STOP_Time_min",

    "EV_IDLE_Time_min",

    "EV_INCHING_Time_min",

    "EV_ACC_Time_min",

    "EV_DEC_Time_min",

    "EV_CRUISE_Time_min",
]


# ============================================================
# 4. PHASE DISTANCE COLUMNS
# ============================================================

ICE_DISTANCE_COLUMNS = [

    "ICE_STOP_Distance_km",

    "ICE_IDLE_Distance_km",

    "ICE_INCHING_Distance_km",

    "ICE_ACC_Distance_km",

    "ICE_DEC_Distance_km",

    "ICE_CRUISE_Distance_km",
]


EV_DISTANCE_COLUMNS = [

    "EV_STOP_Distance_km",

    "EV_IDLE_Distance_km",

    "EV_INCHING_Distance_km",

    "EV_ACC_Distance_km",

    "EV_DEC_Distance_km",

    "EV_CRUISE_Distance_km",
]


# ============================================================
# 5. ICE INPUT FEATURES
# ============================================================
#
# Total = 17 ICE input features
#
# ============================================================

ICE_INPUT_COLUMNS = [

    # -------------------------
    # Route / overall
    # -------------------------

    "Route_Progress",

    "ICE_Time_Total_min",

    "ICE_Distance_Total_km",

    "ICE_Moving_Average_Speed_kmh",

    "ICE_Mean_Acceleration_ms2",

    # -------------------------
    # Phase times
    # -------------------------

    "ICE_STOP_Time_min",

    "ICE_IDLE_Time_min",

    "ICE_INCHING_Time_min",

    "ICE_ACC_Time_min",

    "ICE_DEC_Time_min",

    "ICE_CRUISE_Time_min",

    # -------------------------
    # Phase distances
    # -------------------------

    "ICE_STOP_Distance_km",

    "ICE_IDLE_Distance_km",

    "ICE_INCHING_Distance_km",

    "ICE_ACC_Distance_km",

    "ICE_DEC_Distance_km",

    "ICE_CRUISE_Distance_km",
]


# ============================================================
# 6. EV TARGET FEATURES
# ============================================================
#
# Total = 13 EV target features
#
# ============================================================

EV_TARGET_COLUMNS = [

    # -------------------------
    # Overall
    # -------------------------

    "EV_Moving_Average_Speed_kmh",

    # -------------------------
    # Phase times
    # -------------------------

    "EV_STOP_Time_min",

    "EV_IDLE_Time_min",

    "EV_INCHING_Time_min",

    "EV_ACC_Time_min",

    "EV_DEC_Time_min",

    "EV_CRUISE_Time_min",

    # -------------------------
    # Phase distances
    # -------------------------

    "EV_STOP_Distance_km",

    "EV_IDLE_Distance_km",

    "EV_INCHING_Distance_km",

    "EV_ACC_Distance_km",

    "EV_DEC_Distance_km",

    "EV_CRUISE_Distance_km",
]


# ============================================================
# 7. METADATA COLUMNS
# ============================================================

METADATA_COLUMNS = [

    "Pair_ID",

    "Route_Bin",

    "Route_Progress",
]


# ============================================================
# 8. LOAD DATASET
# ============================================================

def load_ts_dataset():

    """
    Load the route-progress TS dataset.

    Returns
    -------
    pandas.DataFrame
        Complete TS dataset.
    """

    print("=" * 70)
    print("LOADING TS DATASET")
    print("=" * 70)

    print(
        f"\nDataset:\n{DATASET_FILE}"
    )

    if not DATASET_FILE.exists():

        raise FileNotFoundError(

            "\nTS dataset not found.\n"

            f"Expected location:\n"
            f"{DATASET_FILE}\n\n"

            "Run ts_dataset.py first."
        )

    df = pd.read_excel(

        DATASET_FILE,

        sheet_name="Route_Progress_All"
    )

    print(
        f"\nRows loaded    : "
        f"{len(df)}"
    )

    print(
        f"Columns loaded : "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# 9. CHECK REQUIRED COLUMNS
# ============================================================

def validate_required_columns(df):

    """
    Check whether all required modelling columns exist.
    """

    print("\n" + "=" * 70)
    print("CHECKING REQUIRED COLUMNS")
    print("=" * 70)

    required_columns = list(
        dict.fromkeys(

            METADATA_COLUMNS

            + ICE_INPUT_COLUMNS

            + EV_TARGET_COLUMNS
        )
    )

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    found_count = (

        len(required_columns)

        - len(missing_columns)
    )

    print(
        f"\nRequired columns : "
        f"{len(required_columns)}"
    )

    print(
        f"Found columns    : "
        f"{found_count}"
    )

    print(
        f"Missing columns  : "
        f"{len(missing_columns)}"
    )

    if missing_columns:

        print(
            "\nMissing columns:"
        )

        for column in missing_columns:

            print(
                f"   - {column}"
            )

        raise ValueError(

            "\nTS dataset structure is invalid."
        )

    print(
        "\nStatus : PASS"
    )


# ============================================================
# 10. CHECK PAIR STRUCTURE
# ============================================================

def validate_pair_structure(df):

    """
    Confirm that each Pair_ID contains exactly
    the expected number of route bins.
    """

    print("\n" + "=" * 70)
    print("CHECKING PAIR STRUCTURE")
    print("=" * 70)

    pair_counts = (

        df.groupby("Pair_ID")

        .size()

        .sort_index()
    )

    unique_pairs = len(
        pair_counts
    )

    print(
        f"\nUnique pairs : "
        f"{unique_pairs}"
    )

    if unique_pairs != EXPECTED_PAIRS:

        raise ValueError(

            f"\nExpected "
            f"{EXPECTED_PAIRS} pairs "
            f"but found "
            f"{unique_pairs}."
        )

    invalid_pairs = (

        pair_counts[
            pair_counts
            != EXPECTED_BINS_PER_PAIR
        ]
    )

    if len(invalid_pairs) > 0:

        print(
            "\nInvalid pair sizes:"
        )

        for pair_id, count in (

            invalid_pairs.items()
        ):

            print(
                f"   {pair_id}: "
                f"{count} bins"
            )

        raise ValueError(

            "\nPair structure validation failed."
        )

    print(
        f"Bins per pair : "
        f"{EXPECTED_BINS_PER_PAIR}"
    )

    print(
        "Status        : PASS"
    )


# ============================================================
# 11. CHECK ROUTE ORDER
# ============================================================

def validate_route_order(df):

    """
    Confirm that route bins and route progress
    are ordered correctly within every pair.
    """

    print("\n" + "=" * 70)
    print("CHECKING ROUTE ORDER")
    print("=" * 70)

    problems = []

    for pair_id, group in (

        df.groupby("Pair_ID")
    ):

        group = group.sort_values(
            "Route_Bin"
        )

        route_bins = (

            group[
                "Route_Bin"
            ]
            .to_numpy()
        )

        route_progress = (

            group[
                "Route_Progress"
            ]
            .to_numpy()
        )

        # ----------------------------------------------------
        # Route bin ordering
        # ----------------------------------------------------

        bins_ordered = np.all(

            np.diff(route_bins)

            >= 0
        )

        # ----------------------------------------------------
        # Route progress ordering
        # ----------------------------------------------------

        progress_ordered = np.all(

            np.diff(route_progress)

            >= 0
        )

        if not bins_ordered:

            problems.append(
                (
                    pair_id,
                    "Route_Bin ordering"
                )
            )

        if not progress_ordered:

            problems.append(
                (
                    pair_id,
                    "Route_Progress ordering"
                )
            )

    if problems:

        print(
            "\nRoute-order problems:"
        )

        for pair_id, problem in problems:

            print(
                f"   {pair_id}: "
                f"{problem}"
            )

        raise ValueError(

            "\nRoute ordering validation failed."
        )

    print(
        "Route bins ordered     : PASS"
    )

    print(
        "Route progress ordered : PASS"
    )

    print(
        "Status                 : PASS"
    )


# ============================================================
# 12. CHECK MISSING VALUES
# ============================================================

def validate_missing_values(df):

    """
    Check missing values in the modelling columns.
    """

    print("\n" + "=" * 70)
    print("CHECKING MISSING VALUES")
    print("=" * 70)

    modelling_columns = (

        ICE_INPUT_COLUMNS
        + EV_TARGET_COLUMNS
    )

    missing_count = int(

        df[
            modelling_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"\nMissing modelling values : "
        f"{missing_count}"
    )

    if missing_count > 0:

        missing_by_column = (

            df[
                modelling_columns
            ]
            .isna()
            .sum()
        )

        print(
            "\nMissing values by column:"
        )

        for column, count in (

            missing_by_column[
                missing_by_column > 0
            ].items()
        ):

            print(
                f"   {column}: {count}"
            )

        raise ValueError(

            "\nMissing-value validation failed."
        )

    print(
        "Status : PASS"
    )


# ============================================================
# 13. CHECK NUMERIC DATA
# ============================================================

def validate_numeric_data(df):

    """
    Confirm all modelling variables are numeric.
    """

    print("\n" + "=" * 70)
    print("CHECKING NUMERIC DATA")
    print("=" * 70)

    modelling_columns = (

        ICE_INPUT_COLUMNS
        + EV_TARGET_COLUMNS
    )

    non_numeric = []

    for column in modelling_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            non_numeric.append(
                column
            )

    if non_numeric:

        print(
            "\nNon-numeric columns:"
        )

        for column in non_numeric:

            print(
                f"   - {column}"
            )

        raise TypeError(

            "\nNumeric-data validation failed."
        )

    print(
        f"Numeric modelling columns : "
        f"{len(modelling_columns)}"
    )

    print(
        "Status : PASS"
    )


# ============================================================
# 14. CHECK INFINITE VALUES
# ============================================================

def validate_infinite_values(df):

    """
    Check for infinite numerical values.
    """

    print("\n" + "=" * 70)
    print("CHECKING INFINITE VALUES")
    print("=" * 70)

    modelling_columns = (

        ICE_INPUT_COLUMNS
        + EV_TARGET_COLUMNS
    )

    values = (

        df[
            modelling_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    infinite_count = int(

        np.isinf(values)
        .sum()
    )

    print(
        f"\nInfinite values : "
        f"{infinite_count}"
    )

    if infinite_count > 0:

        raise ValueError(

            "\nInfinite-value validation failed."
        )

    print(
        "Status : PASS"
    )


# ============================================================
# 15. CHECK PHYSICAL VARIABLES
# ============================================================
#
# Variables such as:
#
#     time
#     distance
#     speed
#
# should not be negative.
#
# Acceleration is deliberately NOT checked here because
# negative acceleration is physically valid and represents
# deceleration.
#
# ============================================================

def validate_physical_variables(df):

    """
    Check variables that must be non-negative.
    """

    print("\n" + "=" * 70)
    print("CHECKING PHYSICAL VARIABLES")
    print("=" * 70)

    non_negative_columns = [

        "ICE_Time_Total_min",

        "ICE_Distance_Total_km",

        "ICE_Moving_Average_Speed_kmh",

        "EV_Moving_Average_Speed_kmh",
    ]

    non_negative_columns += (

        ICE_TIME_COLUMNS
        + EV_TIME_COLUMNS
        + ICE_DISTANCE_COLUMNS
        + EV_DISTANCE_COLUMNS
    )

    if "EV_Time_Total_min" in df.columns:

        non_negative_columns.append(
            "EV_Time_Total_min"
        )

    if "EV_Distance_Total_km" in df.columns:

        non_negative_columns.append(
            "EV_Distance_Total_km"
        )

    violations = []

    for column in non_negative_columns:

        if column not in df.columns:

            continue

        values = pd.to_numeric(

            df[column],

            errors="coerce"
        )

        negative_count = int(

            (
                values < -1e-9
            )
            .sum()
        )

        if negative_count > 0:

            violations.append(

                (
                    column,
                    negative_count
                )
            )

    if violations:

        print(
            "\nPhysical-variable violations:"
        )

        for column, count in violations:

            print(
                f"   {column}: "
                f"{count}"
            )

        raise ValueError(

            "\nPhysical-variable validation failed."
        )

    print(
        "Non-negative physical variables : PASS"
    )

    print(
        "Negative acceleration            : ALLOWED"
    )


# ============================================================
# 16. SORT DATASET
# ============================================================

def sort_ts_dataset(df):

    """
    Sort data by Pair_ID and route bin.

    This does not modify the observations.
    It only guarantees deterministic sequence order.
    """

    df = (

        df.sort_values(

            [
                "Pair_ID",
                "Route_Bin"
            ]
        )

        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# 17. EXTRACT X AND Y
# ============================================================

def get_X_y(df):

    """
    Extract ICE input features and EV target features.

    Pair_ID is deliberately NOT included in X or Y.

    Pair_ID is retained separately for leakage-safe
    validation and sequence construction.
    """

    X = (

        df[
            ICE_INPUT_COLUMNS
        ]
        .copy()
    )

    y = (

        df[
            EV_TARGET_COLUMNS
        ]
        .copy()
    )

    pair_ids = (

        df[
            "Pair_ID"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    route_bins = (

        df[
            "Route_Bin"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    route_progress = (

        df[
            "Route_Progress"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (

        X.reset_index(drop=True),

        y.reset_index(drop=True),

        pair_ids,

        route_bins,

        route_progress,
    )


# ============================================================
# 18. COMPLETE DATA LOADING FUNCTION
# ============================================================

def load_and_validate_ts_dataset():

    """
    Complete TS dataset loading and validation.

    Returns
    -------
    df
        Validated and sorted complete dataset.

    X
        ICE input dataframe with 17 features.

    y
        EV target dataframe with 13 targets.

    pair_ids
        Pair IDs for leakage-safe validation.

    route_bins
        Route-bin sequence positions.

    route_progress
        Route-progress positions.
    """

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_ts_dataset()

    # --------------------------------------------------------
    # Validate structure
    # --------------------------------------------------------

    validate_required_columns(
        df
    )

    # --------------------------------------------------------
    # Validate pair structure
    # --------------------------------------------------------

    validate_pair_structure(
        df
    )

    # --------------------------------------------------------
    # Validate route ordering
    # --------------------------------------------------------

    validate_route_order(
        df
    )

    # --------------------------------------------------------
    # Validate missing values
    # --------------------------------------------------------

    validate_missing_values(
        df
    )

    # --------------------------------------------------------
    # Validate numeric data
    # --------------------------------------------------------

    validate_numeric_data(
        df
    )

    # --------------------------------------------------------
    # Validate infinite values
    # --------------------------------------------------------

    validate_infinite_values(
        df
    )

    # --------------------------------------------------------
    # Validate physical variables
    # --------------------------------------------------------

    validate_physical_variables(
        df
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = sort_ts_dataset(
        df
    )

    # --------------------------------------------------------
    # Extract X and Y
    # --------------------------------------------------------

    (
        X,
        y,
        pair_ids,
        route_bins,
        route_progress,
    ) = get_X_y(
        df
    )

    # --------------------------------------------------------
    # Final feature-count checks
    # --------------------------------------------------------

    if X.shape[1] != 17:

        raise ValueError(

            f"Expected 17 ICE inputs "
            f"but found {X.shape[1]}."
        )

    if y.shape[1] != 13:

        raise ValueError(

            f"Expected 13 EV targets "
            f"but found {y.shape[1]}."
        )

    # --------------------------------------------------------
    # Final row-count check
    # --------------------------------------------------------

    expected_rows = (

        EXPECTED_PAIRS
        * EXPECTED_BINS_PER_PAIR
    )

    if len(df) != expected_rows:

        raise ValueError(

            f"Expected {expected_rows} rows "
            f"but found {len(df)}."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TS DATA LOADER COMPLETE")
    print("=" * 70)

    print(
        f"\nPairs        : "
        f"{pair_ids.nunique()}"
    )

    print(
        f"Rows         : "
        f"{len(df)}"
    )

    print(
        f"ICE inputs   : "
        f"{X.shape[1]}"
    )

    print(
        f"EV targets   : "
        f"{y.shape[1]}"
    )

    print(
        f"X shape      : "
        f"{X.shape}"
    )

    print(
        f"Y shape      : "
        f"{y.shape}"
    )

    print(
        "\nDataset loading and validation : PASS"
    )

    return (

        df,

        X,

        y,

        pair_ids,

        route_bins,

        route_progress,
    )


# ============================================================
# 19. STANDALONE TEST
# ============================================================

def main():

    (
        df,
        X,
        y,
        pair_ids,
        route_bins,
        route_progress,
    ) = load_and_validate_ts_dataset()

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    # --------------------------------------------------------
    # ICE features
    # --------------------------------------------------------

    print(
        "\nICE INPUT FEATURES "
        "(17):"
    )

    for index, column in enumerate(

        ICE_INPUT_COLUMNS,

        start=1
    ):

        print(
            f"{index:2d}. {column}"
        )

    # --------------------------------------------------------
    # EV targets
    # --------------------------------------------------------

    print(
        "\nEV TARGET FEATURES "
        "(13):"
    )

    for index, column in enumerate(

        EV_TARGET_COLUMNS,

        start=1
    ):

        print(
            f"{index:2d}. {column}"
        )

    # --------------------------------------------------------
    # Pair distribution
    # --------------------------------------------------------

    print(
        "\nPair distribution:"
    )

    print(

        pair_ids
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Route-bin range
    # --------------------------------------------------------

    print(
        "\nRoute-bin range:"
    )

    print(
        f"Minimum bin : "
        f"{route_bins.min()}"
    )

    print(
        f"Maximum bin : "
        f"{route_bins.max()}"
    )

    # --------------------------------------------------------
    # Route-progress range
    # --------------------------------------------------------

    print(
        "\nRoute-progress range:"
    )

    print(
        f"Minimum progress : "
        f"{route_progress.min():.6f}"
    )

    print(
        f"Maximum progress : "
        f"{route_progress.max():.6f}"
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    print(
        "\nModel input structure:"
    )

    print(
        "ICE sequence -> TS model -> EV sequence"
    )

    print(
        "\nPair_ID is retained separately "
        "for leakage-safe validation."
    )

    print(
        "\nNo scaling performed."
    )

    print(
        "No sequence windows created."
    )

    print(
        "No model trained."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()