# ============================================================
# TS SEQUENCE BUILDER
# ============================================================
#
# File:
#     src/timeseries/common/ts_sequence_builder.py
#
# Purpose:
#     Convert the validated route-progress TS table into
#     complete trip-level sequences for TS modelling.
#
# Current modelling representation:
#
#     X shape = (6, 100, 17)
#     Y shape = (6, 100, 13)
#
# Meaning:
#
#     6   -> independent paired trips
#     100 -> route-progress bins per trip
#     17  -> ICE input features
#     13  -> EV target features
#
# IMPORTANT:
#
#     One complete trip = one sequence.
#
#     We DO NOT create overlapping sliding windows at this
#     stage because that could artificially inflate the number
#     of samples when only 6 independent paired trips exist.
#
#     Pair_ID boundaries are strictly preserved.
#
#     No scaling is performed here.
#     No train/test split is performed here.
#     No model is trained here.
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. IMPORT COMMON DATA LOADER
# ============================================================

from ts_data_loader import (
    load_and_validate_ts_dataset,
    ICE_INPUT_COLUMNS,
    EV_TARGET_COLUMNS,
    EXPECTED_PAIRS,
    EXPECTED_BINS_PER_PAIR,
)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING"
)

OUTPUT_DIR = (
    BASE_DIR
    / "Results"
    / "TimeSeries"
    / "Dataset"
)


SEQUENCE_FILE = (
    OUTPUT_DIR
    / "TS_Sequence_Dataset.npz"
)


SEQUENCE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "TS_Sequence_Summary.xlsx"
)


SEQUENCE_REPORT_FILE = (
    OUTPUT_DIR
    / "TS_Sequence_Build_Report.txt"
)


# ============================================================
# 3. EXPECTED SHAPES
# ============================================================

EXPECTED_X_SHAPE = (
    EXPECTED_PAIRS,
    EXPECTED_BINS_PER_PAIR,
    len(ICE_INPUT_COLUMNS),
)


EXPECTED_Y_SHAPE = (
    EXPECTED_PAIRS,
    EXPECTED_BINS_PER_PAIR,
    len(EV_TARGET_COLUMNS),
)


# ============================================================
# 4. CREATE ONE SEQUENCE PER PAIR
# ============================================================

def build_trip_sequences(df):
    """
    Convert the route-progress dataframe into complete
    trip-level sequences.

    Each Pair_ID becomes exactly one sequence.

    Returns
    -------
    X_sequences : np.ndarray
        Shape:
            (number_of_pairs, 100, 17)

    Y_sequences : np.ndarray
        Shape:
            (number_of_pairs, 100, 13)

    pair_ids : np.ndarray
        Pair identifiers corresponding to sequence axis 0.

    route_bins : np.ndarray
        Route-bin positions.
        Shape:
            (100,)

    route_progress : np.ndarray
        Route-progress positions.
        Shape:
            (100,)
    """

    print("\n" + "=" * 70)
    print("BUILDING COMPLETE TRIP SEQUENCES")
    print("=" * 70)

    X_sequences = []

    Y_sequences = []

    pair_ids = []

    reference_route_bins = None

    reference_route_progress = None

    # --------------------------------------------------------
    # Process each pair independently
    # --------------------------------------------------------

    for pair_id, group in df.groupby(
        "Pair_ID",
        sort=True
    ):

        print(
            f"\nProcessing: {pair_id}"
        )

        # ----------------------------------------------------
        # Sort strictly by route bin
        # ----------------------------------------------------

        group = (
            group
            .sort_values(
                "Route_Bin"
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # Validate row count
        # ----------------------------------------------------

        if len(group) != EXPECTED_BINS_PER_PAIR:

            raise ValueError(

                f"{pair_id} contains "
                f"{len(group)} route bins. "

                f"Expected "
                f"{EXPECTED_BINS_PER_PAIR}."
            )

        # ----------------------------------------------------
        # Extract route bins
        # ----------------------------------------------------

        current_route_bins = (

            group[
                "Route_Bin"
            ]
            .to_numpy(
                dtype=int
            )
        )

        # ----------------------------------------------------
        # Expected route bins
        # ----------------------------------------------------

        expected_route_bins = np.arange(
            EXPECTED_BINS_PER_PAIR
        )

        if not np.array_equal(
            current_route_bins,
            expected_route_bins
        ):

            raise ValueError(

                f"{pair_id} does not contain "
                "the complete route-bin sequence "
                "0 to 99."
            )

        # ----------------------------------------------------
        # Extract route progress
        # ----------------------------------------------------

        current_route_progress = (

            group[
                "Route_Progress"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Route progress must be ordered
        # ----------------------------------------------------

        if not np.all(
            np.diff(
                current_route_progress
            ) >= 0
        ):

            raise ValueError(

                f"{pair_id} has non-monotonic "
                "route progress."
            )

        # ----------------------------------------------------
        # Store reference route structure
        # ----------------------------------------------------

        if reference_route_bins is None:

            reference_route_bins = (
                current_route_bins.copy()
            )

            reference_route_progress = (
                current_route_progress.copy()
            )

        else:

            # ------------------------------------------------
            # Every pair must use the same bin structure
            # ------------------------------------------------

            if not np.array_equal(
                current_route_bins,
                reference_route_bins
            ):

                raise ValueError(

                    f"{pair_id} has a different "
                    "route-bin structure."
                )

            if not np.allclose(
                current_route_progress,
                reference_route_progress
            ):

                raise ValueError(

                    f"{pair_id} has a different "
                    "route-progress structure."
                )

        # ----------------------------------------------------
        # Extract ICE sequence
        # ----------------------------------------------------

        X_pair = (

            group[
                ICE_INPUT_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # Extract EV sequence
        # ----------------------------------------------------

        Y_pair = (

            group[
                EV_TARGET_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        expected_x_pair_shape = (
            EXPECTED_BINS_PER_PAIR,
            len(ICE_INPUT_COLUMNS)
        )

        expected_y_pair_shape = (
            EXPECTED_BINS_PER_PAIR,
            len(EV_TARGET_COLUMNS)
        )

        if X_pair.shape != expected_x_pair_shape:

            raise ValueError(

                f"{pair_id} X shape is "
                f"{X_pair.shape}. "

                f"Expected "
                f"{expected_x_pair_shape}."
            )

        if Y_pair.shape != expected_y_pair_shape:

            raise ValueError(

                f"{pair_id} Y shape is "
                f"{Y_pair.shape}. "

                f"Expected "
                f"{expected_y_pair_shape}."
            )

        # ----------------------------------------------------
        # Check finite values
        # ----------------------------------------------------

        if not np.isfinite(
            X_pair
        ).all():

            raise ValueError(

                f"{pair_id} contains "
                "non-finite ICE values."
            )

        if not np.isfinite(
            Y_pair
        ).all():

            raise ValueError(

                f"{pair_id} contains "
                "non-finite EV values."
            )

        # ----------------------------------------------------
        # Append complete trip
        # ----------------------------------------------------

        X_sequences.append(
            X_pair
        )

        Y_sequences.append(
            Y_pair
        )

        pair_ids.append(
            pair_id
        )

        print(
            f"  X shape : {X_pair.shape}"
        )

        print(
            f"  Y shape : {Y_pair.shape}"
        )

        print(
            "  Status  : PASS"
        )

    # ========================================================
    # Convert list → 3D arrays
    # ========================================================

    X_sequences = np.stack(
        X_sequences,
        axis=0
    )

    Y_sequences = np.stack(
        Y_sequences,
        axis=0
    )

    pair_ids = np.asarray(
        pair_ids,
        dtype=str
    )

    # ========================================================
    # Final shape validation
    # ========================================================

    print(
        "\nFinal sequence shapes:"
    )

    print(
        f"X : {X_sequences.shape}"
    )

    print(
        f"Y : {Y_sequences.shape}"
    )

    if X_sequences.shape != EXPECTED_X_SHAPE:

        raise ValueError(

            f"Final X shape "
            f"{X_sequences.shape} "
            f"does not match expected "
            f"{EXPECTED_X_SHAPE}."
        )

    if Y_sequences.shape != EXPECTED_Y_SHAPE:

        raise ValueError(

            f"Final Y shape "
            f"{Y_sequences.shape} "
            f"does not match expected "
            f"{EXPECTED_Y_SHAPE}."
        )

    print(
        "Final shape validation : PASS"
    )

    return (
        X_sequences,
        Y_sequences,
        pair_ids,
        reference_route_bins,
        reference_route_progress,
    )


# ============================================================
# 5. VALIDATE SEQUENCE INDEPENDENCE
# ============================================================

def validate_sequence_boundaries(
    df,
    pair_ids,
):
    """
    Confirm that each sequence corresponds to exactly one
    Pair_ID and that no sequence crosses pair boundaries.
    """

    print("\n" + "=" * 70)
    print("VALIDATING SEQUENCE BOUNDARIES")
    print("=" * 70)

    dataset_pairs = (
        sorted(
            df[
                "Pair_ID"
            ]
            .astype(str)
            .unique()
        )
    )

    sequence_pairs = (
        sorted(
            pair_ids.tolist()
        )
    )

    # --------------------------------------------------------
    # Pair list must match
    # --------------------------------------------------------

    if dataset_pairs != sequence_pairs:

        raise ValueError(

            "Sequence Pair_ID list does not "
            "match the source dataset."
        )

    # --------------------------------------------------------
    # Number of sequences
    # --------------------------------------------------------

    if len(pair_ids) != EXPECTED_PAIRS:

        raise ValueError(

            f"Expected {EXPECTED_PAIRS} "
            f"sequences but found "
            f"{len(pair_ids)}."
        )

    # --------------------------------------------------------
    # No duplicate sequence IDs
    # --------------------------------------------------------

    if len(
        np.unique(pair_ids)
    ) != len(pair_ids):

        raise ValueError(

            "Duplicate Pair_ID found "
            "in sequence dataset."
        )

    print(
        f"Independent sequences : "
        f"{len(pair_ids)}"
    )

    print(
        "Pair boundaries        : PASS"
    )

    print(
        "No cross-pair sequences : PASS"
    )

    return True


# ============================================================
# 6. VALIDATE ROUTE-PROGRESS AXIS
# ============================================================

def validate_route_axis(
    route_bins,
    route_progress,
):
    """
    Validate the common route-progress axis.
    """

    print("\n" + "=" * 70)
    print("VALIDATING COMMON ROUTE AXIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Route-bin sequence
    # --------------------------------------------------------

    expected_bins = np.arange(
        EXPECTED_BINS_PER_PAIR
    )

    bins_valid = np.array_equal(
        route_bins,
        expected_bins
    )

    # --------------------------------------------------------
    # Route progress monotonicity
    # --------------------------------------------------------

    progress_valid = np.all(
        np.diff(
            route_progress
        ) >= 0
    )

    # --------------------------------------------------------
    # Expected route-progress values
    # --------------------------------------------------------

    expected_progress = (

        expected_bins + 0.5
    ) / EXPECTED_BINS_PER_PAIR

    progress_structure_valid = np.allclose(

        route_progress,

        expected_progress,

        atol=1e-9
    )

    if not bins_valid:

        raise ValueError(
            "Route-bin axis is invalid."
        )

    if not progress_valid:

        raise ValueError(
            "Route-progress axis is not monotonic."
        )

    if not progress_structure_valid:

        raise ValueError(
            "Route-progress values do not match "
            "the expected bin-center structure."
        )

    print(
        f"Route bins : "
        f"{route_bins[0]} → "
        f"{route_bins[-1]}"
    )

    print(
        f"Route progress : "
        f"{route_progress[0]:.6f} → "
        f"{route_progress[-1]:.6f}"
    )

    print(
        "Route-bin axis        : PASS"
    )

    print(
        "Route-progress axis   : PASS"
    )


# ============================================================
# 7. VALIDATE SEQUENCE CONTENT
# ============================================================

def validate_sequence_content(
    X_sequences,
    Y_sequences,
):
    """
    Validate finite values and feature dimensions.
    """

    print("\n" + "=" * 70)
    print("VALIDATING SEQUENCE CONTENT")
    print("=" * 70)

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    if not np.isfinite(
        X_sequences
    ).all():

        raise ValueError(
            "X sequences contain non-finite values."
        )

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    if not np.isfinite(
        Y_sequences
    ).all():

        raise ValueError(
            "Y sequences contain non-finite values."
        )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    if X_sequences.ndim != 3:

        raise ValueError(
            "X must be a 3-dimensional array."
        )

    if Y_sequences.ndim != 3:

        raise ValueError(
            "Y must be a 3-dimensional array."
        )

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    if (
        X_sequences.shape[1]
        != Y_sequences.shape[1]
    ):

        raise ValueError(

            "X and Y sequence lengths "
            "do not match."
        )

    # --------------------------------------------------------
    # Number of trips
    # --------------------------------------------------------

    if (
        X_sequences.shape[0]
        != Y_sequences.shape[0]
    ):

        raise ValueError(

            "X and Y contain different "
            "numbers of trip sequences."
        )

    print(
        "X finite values       : PASS"
    )

    print(
        "Y finite values       : PASS"
    )

    print(
        "X/Y sequence length   : PASS"
    )

    print(
        "X/Y trip count        : PASS"
    )


# ============================================================
# 8. BUILD SEQUENCE SUMMARY
# ============================================================

def create_sequence_summary(
    X_sequences,
    Y_sequences,
    pair_ids,
    route_bins,
    route_progress,
):
    """
    Create a human-readable sequence summary.
    """

    records = []

    for index, pair_id in enumerate(
        pair_ids
    ):

        X_pair = (
            X_sequences[index]
        )

        Y_pair = (
            Y_sequences[index]
        )

        records.append({

            "Sequence_Index":
                index,

            "Pair_ID":
                pair_id,

            "Sequence_Length":
                X_pair.shape[0],

            "ICE_Features":
                X_pair.shape[1],

            "EV_Targets":
                Y_pair.shape[1],

            "First_Route_Bin":
                route_bins[0],

            "Last_Route_Bin":
                route_bins[-1],

            "First_Route_Progress":
                route_progress[0],

            "Last_Route_Progress":
                route_progress[-1],

            "X_Contains_NaN":
                bool(
                    np.isnan(
                        X_pair
                    ).any()
                ),

            "Y_Contains_NaN":
                bool(
                    np.isnan(
                        Y_pair
                    ).any()
                ),

            "X_Contains_Inf":
                bool(
                    np.isinf(
                        X_pair
                    ).any()
                ),

            "Y_Contains_Inf":
                bool(
                    np.isinf(
                        Y_pair
                    ).any()
                ),

            "Status":
                "PASS",
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# 9. SAVE NPZ DATASET
# ============================================================

def save_sequence_dataset(
    X_sequences,
    Y_sequences,
    pair_ids,
    route_bins,
    route_progress,
):
    """
    Save the sequence representation as NPZ.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(

        SEQUENCE_FILE,

        X=X_sequences,

        Y=Y_sequences,

        Pair_ID=pair_ids,

        Route_Bin=route_bins,

        Route_Progress=route_progress,

        ICE_Features=np.asarray(
            ICE_INPUT_COLUMNS,
            dtype=str
        ),

        EV_Targets=np.asarray(
            EV_TARGET_COLUMNS,
            dtype=str
        ),
    )

    print(
        f"\nSequence dataset saved:"
    )

    print(
        SEQUENCE_FILE
    )


# ============================================================
# 10. SAVE EXCEL SUMMARY
# ============================================================

def save_sequence_summary(
    summary_df,
):
    """
    Save sequence summary information.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        SEQUENCE_SUMMARY_FILE,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Sequence summary
        # ----------------------------------------------------

        summary_df.to_excel(

            writer,

            sheet_name="Sequence_Summary",

            index=False
        )

        # ----------------------------------------------------
        # ICE features
        # ----------------------------------------------------

        pd.DataFrame({

            "Feature_Index":
                range(
                    1,
                    len(
                        ICE_INPUT_COLUMNS
                    ) + 1
                ),

            "ICE_Input_Feature":
                ICE_INPUT_COLUMNS,
        }).to_excel(

            writer,

            sheet_name="ICE_Features",

            index=False
        )

        # ----------------------------------------------------
        # EV targets
        # ----------------------------------------------------

        pd.DataFrame({

            "Target_Index":
                range(
                    1,
                    len(
                        EV_TARGET_COLUMNS
                    ) + 1
                ),

            "EV_Target":
                EV_TARGET_COLUMNS,
        }).to_excel(

            writer,

            sheet_name="EV_Targets",

            index=False
        )

        # ----------------------------------------------------
        # Shape information
        # ----------------------------------------------------

        pd.DataFrame({

            "Variable": [

                "X",

                "Y",

                "Number_of_Pairs",

                "Bins_Per_Pair",

                "ICE_Features",

                "EV_Targets",
            ],

            "Value": [

                str(
                    EXPECTED_X_SHAPE
                ),

                str(
                    EXPECTED_Y_SHAPE
                ),

                EXPECTED_PAIRS,

                EXPECTED_BINS_PER_PAIR,

                len(
                    ICE_INPUT_COLUMNS
                ),

                len(
                    EV_TARGET_COLUMNS
                ),
            ],
        }).to_excel(

            writer,

            sheet_name="Dataset_Shape",

            index=False
        )

    print(
        f"\nSequence summary saved:"
    )

    print(
        SEQUENCE_SUMMARY_FILE
    )


# ============================================================
# 11. SAVE TEXT REPORT
# ============================================================

def save_sequence_report(
    X_sequences,
    Y_sequences,
    pair_ids,
):
    """
    Save a reproducible text report.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SEQUENCE_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TS SEQUENCE BUILD REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        # ----------------------------------------------------
        # Dataset representation
        # ----------------------------------------------------

        file.write(
            "SEQUENCE REPRESENTATION\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            f"Number of independent trips: "
            f"{len(pair_ids)}\n"
        )

        file.write(
            f"Route bins per trip: "
            f"{EXPECTED_BINS_PER_PAIR}\n"
        )

        file.write(
            f"ICE input features: "
            f"{len(ICE_INPUT_COLUMNS)}\n"
        )

        file.write(
            f"EV target features: "
            f"{len(EV_TARGET_COLUMNS)}\n\n"
        )

        file.write(
            f"X shape: "
            f"{X_sequences.shape}\n"
        )

        file.write(
            f"Y shape: "
            f"{Y_sequences.shape}\n\n"
        )

        # ----------------------------------------------------
        # Pair IDs
        # ----------------------------------------------------

        file.write(
            "PAIR IDs\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        for index, pair_id in enumerate(
            pair_ids
        ):

            file.write(
                f"{index}: {pair_id}\n"
            )

        file.write(
            "\n"
        )

        # ----------------------------------------------------
        # Methodological decision
        # ----------------------------------------------------

        file.write(
            "SEQUENCE DESIGN DECISION\n"
        )

        file.write(
            "-" * 70
            + "\n"
        )

        file.write(
            "One complete paired trip is treated "
            "as one independent sequence.\n\n"
        )

        file.write(
            "No overlapping sliding windows are "
            "created at this stage.\n\n"
        )

        file.write(
            "Pair_ID boundaries are strictly preserved.\n\n"
        )

        file.write(
            "No scaling is performed in the sequence "
            "builder.\n\n"
        )

        file.write(
            "No train/test split is performed in the "
            "sequence builder.\n\n"
        )

        file.write(
            "No model is trained in the sequence builder.\n\n"
        )

        file.write(
            "This prevents the 600 route bins from "
            "being incorrectly interpreted as 600 "
            "independent trips.\n"
        )

    print(
        f"\nSequence report saved:"
    )

    print(
        SEQUENCE_REPORT_FILE
    )


# ============================================================
# 12. COMPLETE PIPELINE
# ============================================================

def build_and_validate_sequences():

    print("\n" + "=" * 70)
    print("TS SEQUENCE BUILDING PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # Load validated dataset
    # ========================================================

    (
        df,
        X,
        y,
        pair_ids_from_loader,
        route_bins_from_loader,
        route_progress_from_loader,
    ) = load_and_validate_ts_dataset()

    # ========================================================
    # STEP 2
    # Build sequences
    # ========================================================

    (
        X_sequences,
        Y_sequences,
        pair_ids,
        route_bins,
        route_progress,
    ) = build_trip_sequences(
        df
    )

    # ========================================================
    # STEP 3
    # Validate sequence boundaries
    # ========================================================

    validate_sequence_boundaries(

        df,

        pair_ids,
    )

    # ========================================================
    # STEP 4
    # Validate route axis
    # ========================================================

    validate_route_axis(

        route_bins,

        route_progress,
    )

    # ========================================================
    # STEP 5
    # Validate sequence content
    # ========================================================

    validate_sequence_content(

        X_sequences,

        Y_sequences,
    )

    # ========================================================
    # STEP 6
    # Compare loader representation
    # ========================================================

    print("\n" + "=" * 70)
    print("CROSS-CHECKING LOADER DATA")
    print("=" * 70)

    loader_X = (
        X.to_numpy(
            dtype=np.float32
        )
    )

    loader_y = (
        y.to_numpy(
            dtype=np.float32
        )
    )

    rebuilt_X = (
        X_sequences
        .reshape(
            -1,
            len(ICE_INPUT_COLUMNS)
        )
    )

    rebuilt_y = (
        Y_sequences
        .reshape(
            -1,
            len(EV_TARGET_COLUMNS)
        )
    )

    X_match = np.allclose(
        loader_X,
        rebuilt_X,
        atol=1e-6
    )

    Y_match = np.allclose(
        loader_y,
        rebuilt_y,
        atol=1e-6
    )

    if not X_match:

        raise ValueError(
            "Sequence X does not match "
            "the loader X data."
        )

    if not Y_match:

        raise ValueError(
            "Sequence Y does not match "
            "the loader Y data."
        )

    print(
        "X reconstruction match : PASS"
    )

    print(
        "Y reconstruction match : PASS"
    )

    # ========================================================
    # STEP 7
    # Create summary
    # ========================================================

    summary_df = create_sequence_summary(

        X_sequences,

        Y_sequences,

        pair_ids,

        route_bins,

        route_progress,
    )

    # ========================================================
    # STEP 8
    # Save NPZ
    # ========================================================

    save_sequence_dataset(

        X_sequences,

        Y_sequences,

        pair_ids,

        route_bins,

        route_progress,
    )

    # ========================================================
    # STEP 9
    # Save Excel summary
    # ========================================================

    save_sequence_summary(
        summary_df
    )

    # ========================================================
    # STEP 10
    # Save report
    # ========================================================

    save_sequence_report(

        X_sequences,

        Y_sequences,

        pair_ids,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 70)
    print("TS SEQUENCE BUILDING COMPLETE")
    print("=" * 70)

    print(
        f"\nNumber of sequences : "
        f"{X_sequences.shape[0]}"
    )

    print(
        f"Sequence length    : "
        f"{X_sequences.shape[1]}"
    )

    print(
        f"ICE features        : "
        f"{X_sequences.shape[2]}"
    )

    print(
        f"EV targets          : "
        f"{Y_sequences.shape[2]}"
    )

    print(
        f"\nX shape : "
        f"{X_sequences.shape}"
    )

    print(
        f"Y shape : "
        f"{Y_sequences.shape}"
    )

    print(
        "\nPair boundaries : PASS"
    )

    print(
        "Route axis      : PASS"
    )

    print(
        "Content         : PASS"
    )

    print(
        "Loader match    : PASS"
    )

    print(
        "\nOverall sequence validation : PASS"
    )

    print(
        "\nNo scaling performed."
    )

    print(
        "No train/test split performed."
    )

    print(
        "No model trained."
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    build_and_validate_sequences()