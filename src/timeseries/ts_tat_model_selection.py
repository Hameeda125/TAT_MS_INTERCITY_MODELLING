# ============================================================
# TS TAT-RELEVANT MODEL SELECTION ANALYSIS
# ============================================================
# Purpose:
#   Compare Linear Regression, LSTM and GRU specifically for the
#   six EV phase-time targets that contribute to driving TAT.
#
# IMPORTANT:
#   This script DOES NOT modify the dataset.
#   It does NOT tune models.
#   It does NOT create a new random split.
#
#   Linear Regression OOS predictions are RECONSTRUCTED here
#   using the same TS sequence dataset and the same LOPO design.
#   This is necessary because TS_Baseline_Results.xlsx stores
#   Linear Regression fold metrics, not a prediction workbook.
#
# Validation:
#   Leave-One-Pair-Out (LOPO)
#   6 complete paired trips
#   100 route bins per pair
#
# TAT phase-time targets:
#   EV_STOP_Time_min
#   EV_IDLE_Time_min
#   EV_INCHING_Time_min
#   EV_ACC_Time_min
#   EV_DEC_Time_min
#   EV_CRUISE_Time_min
#
# Primary metrics:
#   MAE
#   RMSE
#   R2
#
# MAPE is intentionally not used for model interpretation because
# phase-time targets contain zero / near-zero values.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "Results",
    "TimeSeries",
)

DATASET_DIR = os.path.join(
    RESULTS_ROOT,
    "Dataset",
)

SEQUENCE_DATASET = os.path.join(
    DATASET_DIR,
    "TS_Sequence_Dataset.npz",
)

LSTM_PREDICTIONS = os.path.join(
    RESULTS_ROOT,
    "LSTM",
    "TS_LSTM_Fold_Predictions.xlsx",
)

GRU_PREDICTIONS = os.path.join(
    RESULTS_ROOT,
    "GRU",
    "TS_GRU_Fold_Predictions.xlsx",
)

OUTPUT_DIR = os.path.join(
    RESULTS_ROOT,
    "Model_Selection",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 2. AUTHORITATIVE COLUMN DEFINITIONS
# ============================================================

TIMESERIES_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if TIMESERIES_DIR not in sys.path:
    sys.path.insert(
        0,
        TIMESERIES_DIR,
    )

from ts_data_loader import (
    ICE_INPUT_COLUMNS,
    EV_TARGET_COLUMNS,
)


TAT_TARGETS = [
    "EV_STOP_Time_min",
    "EV_IDLE_Time_min",
    "EV_INCHING_Time_min",
    "EV_ACC_Time_min",
    "EV_DEC_Time_min",
    "EV_CRUISE_Time_min",
]


# ============================================================
# 3. HELPERS
# ============================================================

def calculate_mape(
    actual,
    predicted,
):
    """
    MAPE for descriptive reporting only.

    Zero actual values are excluded.
    MAPE is NOT used for model interpretation.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
        &
        (actual != 0)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (
                    actual[valid]
                    -
                    predicted[valid]
                )
                /
                actual[valid]
            )
        )
        * 100
    )


def calculate_metrics(
    actual,
    predicted,
):
    """
    Calculate pooled OOS metrics.
    """

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE_percent": np.nan,
        }

    if np.allclose(
        actual,
        actual[0],
    ):
        r2 = np.nan
    else:
        r2 = r2_score(
            actual,
            predicted,
        )

    return {
        "N": int(len(actual)),
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
        "R2": float(r2)
        if np.isfinite(r2)
        else np.nan,
        "MAPE_percent": calculate_mape(
            actual,
            predicted,
        ),
    }


def validate_prediction_dataframe(
    df,
    model_name,
):
    """
    Validate saved LSTM/GRU prediction structure.
    """

    required = [
        "Target",
        "Actual",
        "Predicted",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{model_name} prediction file is missing "
            f"columns: {missing}"
        )

    df = df.copy()

    df["Target"] = (
        df["Target"]
        .astype(str)
        .str.strip()
    )

    df["Actual"] = pd.to_numeric(
        df["Actual"],
        errors="coerce",
    )

    df["Predicted"] = pd.to_numeric(
        df["Predicted"],
        errors="coerce",
    )

    tat_df = df[
        df["Target"].isin(TAT_TARGETS)
    ].copy()

    if len(tat_df) == 0:
        raise ValueError(
            f"No TAT phase-time targets found in "
            f"{model_name} prediction file."
        )

    return tat_df


# ============================================================
# 4. LOAD AUTHORITATIVE SEQUENCE DATA
# ============================================================

print("=" * 70)
print("TS TAT-RELEVANT MODEL SELECTION ANALYSIS")
print("=" * 70)

print("\nLoading authoritative sequence dataset...")

if not os.path.exists(SEQUENCE_DATASET):
    raise FileNotFoundError(
        "Sequence dataset not found:\n"
        f"{SEQUENCE_DATASET}"
    )

data = np.load(
    SEQUENCE_DATASET,
    allow_pickle=True,
)

X = data["X"].astype(
    np.float64
)

Y = data["Y"].astype(
    np.float64
)

pair_ids = data["Pair_ID"].astype(
    str
)

print(
    f"X shape       : {X.shape}"
)

print(
    f"Y shape       : {Y.shape}"
)

print(
    f"Pair count    : {len(pair_ids)}"
)

print(
    f"Input features: {X.shape[2]}"
)

print(
    f"EV targets    : {Y.shape[2]}"
)


# ============================================================
# 5. STRUCTURAL VALIDATION
# ============================================================

if X.ndim != 3:
    raise ValueError(
        f"Expected X to be 3D; got {X.ndim}D."
    )

if Y.ndim != 3:
    raise ValueError(
        f"Expected Y to be 3D; got {Y.ndim}D."
    )

if X.shape[0] != Y.shape[0]:
    raise ValueError(
        "X and Y pair counts do not match."
    )

if X.shape[1] != Y.shape[1]:
    raise ValueError(
        "X and Y route-bin counts do not match."
    )

if X.shape[2] != len(ICE_INPUT_COLUMNS):
    raise ValueError(
        "X feature count does not match "
        "ts_data_loader.py."
    )

if Y.shape[2] != len(EV_TARGET_COLUMNS):
    raise ValueError(
        "Y target count does not match "
        "ts_data_loader.py."
    )

if len(pair_ids) != X.shape[0]:
    raise ValueError(
        "Pair_ID count does not match X."
    )

if len(set(pair_ids)) != len(pair_ids):
    raise ValueError(
        "Duplicate Pair_ID values found in "
        "sequence dataset."
    )

if not np.isfinite(X).all():
    raise ValueError(
        "Non-finite values detected in X."
    )

if not np.isfinite(Y).all():
    raise ValueError(
        "Non-finite values detected in Y."
    )

tat_target_indices = [
    EV_TARGET_COLUMNS.index(target)
    for target in TAT_TARGETS
]


# ============================================================
# 6. RECONSTRUCT LINEAR REGRESSION OOS PREDICTIONS
# ============================================================
# This is the critical correction.
#
# We do NOT read the 36 fold rows from the workbook as if they
# were pooled results.
#
# Instead:
#   Fold 1: one complete pair held out
#   Fold 2: one complete pair held out
#   ...
#   Fold 6: one complete pair held out
#
# The predictions from all six held-out pairs are concatenated.
# Pooled metrics are then calculated from those predictions.
# ============================================================

print("\n" + "=" * 70)
print("RECONSTRUCTING LINEAR REGRESSION OOS PREDICTIONS")
print("=" * 70)

linear_prediction_rows = []
linear_fold_rows = []

unique_pairs = np.asarray(
    pair_ids
)

for fold_number, test_pair in enumerate(
    unique_pairs,
    start=1,
):

    print(
        f"\nFold {fold_number}/{len(unique_pairs)} "
        f"| Test Pair: {test_pair}"
    )

    train_mask = (
        pair_ids != test_pair
    )

    test_mask = (
        pair_ids == test_pair
    )

    X_train = X[
        train_mask
    ]

    Y_train = Y[
        train_mask
    ]

    X_test = X[
        test_mask
    ]

    Y_test = Y[
        test_mask
    ]

    # Flatten only after the pair-level split.
    # This preserves LOPO group isolation.
    X_train_flat = X_train.reshape(
        -1,
        X.shape[2],
    )

    Y_train_flat = Y_train.reshape(
        -1,
        Y.shape[2],
    )

    X_test_flat = X_test.reshape(
        -1,
        X.shape[2],
    )

    Y_test_flat = Y_test.reshape(
        -1,
        Y.shape[2],
    )

    # Same scaling principle used in the TS neural-network
    # experiments: scaler is fitted ONLY on training pairs.
    model = Pipeline(
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

    model.fit(
        X_train_flat,
        Y_train_flat,
    )

    Y_pred_flat = model.predict(
        X_test_flat
    )

    if Y_pred_flat.shape != Y_test_flat.shape:
        raise ValueError(
            "Linear Regression prediction shape mismatch."
        )

    # Store fold-level predictions.
    for row_index in range(
        len(Y_test_flat)
    ):

        route_bin = (
            row_index
            %
            X.shape[1]
        )

        for target_index, target in enumerate(
            EV_TARGET_COLUMNS
        ):

            if target not in TAT_TARGETS:
                continue

            linear_prediction_rows.append(
                {
                    "Model":
                        "Linear Regression",

                    "Fold":
                        fold_number,

                    "Pair_ID":
                        test_pair,

                    "Route_Bin":
                        route_bin,

                    "Target":
                        target,

                    "Actual":
                        float(
                            Y_test_flat[
                                row_index,
                                target_index,
                            ]
                        ),

                    "Predicted":
                        float(
                            Y_pred_flat[
                                row_index,
                                target_index,
                            ]
                        ),
                }
            )

    # Fold-level metrics for the six TAT targets.
    for target_index, target in enumerate(
        TAT_TARGETS
    ):

        actual_phase = (
            Y_test_flat[
                :,
                tat_target_indices[
                    target_index
                ],
            ]
        )

        predicted_phase = (
            Y_pred_flat[
                :,
                tat_target_indices[
                    target_index
                ],
            ]
        )

        metrics = calculate_metrics(
            actual_phase,
            predicted_phase,
        )

        linear_fold_rows.append(
            {
                "Model":
                    "Linear Regression",

                "Fold":
                    fold_number,

                "Test_Pair_ID":
                    test_pair,

                "Target":
                    target,

                **metrics,
            }
        )

    print(
        f"  Training pairs : "
        f"{int(train_mask.sum())}"
    )

    print(
        f"  Testing pairs  : "
        f"{int(test_mask.sum())}"
    )


linear_predictions = pd.DataFrame(
    linear_prediction_rows
)

linear_fold_results = pd.DataFrame(
    linear_fold_rows
)

expected_linear_rows = (
    len(unique_pairs)
    *
    X.shape[1]
    *
    len(TAT_TARGETS)
)

if len(linear_predictions) != expected_linear_rows:
    raise ValueError(
        "Unexpected Linear Regression prediction "
        f"coverage. Expected {expected_linear_rows}, "
        f"found {len(linear_predictions)}."
    )

print(
    "\nLinear Regression OOS prediction rows : "
    f"{len(linear_predictions)}"
)


# ============================================================
# 7. POOLED LINEAR REGRESSION TAT METRICS
# ============================================================

linear_tat_rows = []

for target in TAT_TARGETS:

    temp = linear_predictions[
        linear_predictions["Target"]
        ==
        target
    ]

    metrics = calculate_metrics(
        temp["Actual"].to_numpy(),
        temp["Predicted"].to_numpy(),
    )

    linear_tat_rows.append(
        {
            "Model":
                "Linear Regression",

            "Target":
                target,

            **metrics,
        }
    )


linear_tat = pd.DataFrame(
    linear_tat_rows
)


# ============================================================
# 8. LOAD LSTM / GRU SAVED OOS PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAVED LSTM / GRU OOS PREDICTIONS")
print("=" * 70)


def load_saved_predictions(
    path,
    model_name,
):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {model_name} prediction file:\n"
            f"{path}"
        )

    df = pd.read_excel(
        path
    )

    df = validate_prediction_dataframe(
        df,
        model_name,
    )

    df["Model"] = model_name

    return df


lstm_df = load_saved_predictions(
    LSTM_PREDICTIONS,
    "LSTM",
)

gru_df = load_saved_predictions(
    GRU_PREDICTIONS,
    "GRU",
)

print(
    f"LSTM TAT prediction rows : "
    f"{len(lstm_df)}"
)

print(
    f"GRU TAT prediction rows  : "
    f"{len(gru_df)}"
)


# ============================================================
# 9. VALIDATE LSTM / GRU COVERAGE
# ============================================================

expected_neural_rows = (
    len(unique_pairs)
    *
    X.shape[1]
    *
    len(TAT_TARGETS)
)

for model_name, df in [
    ("LSTM", lstm_df),
    ("GRU", gru_df),
]:

    if len(df) != expected_neural_rows:
        raise ValueError(
            f"{model_name} TAT prediction coverage "
            f"is {len(df)} rows; expected "
            f"{expected_neural_rows}."
        )

    target_counts = (
        df["Target"]
        .value_counts()
    )

    missing_targets = [
        target
        for target in TAT_TARGETS
        if target_counts.get(
            target,
            0
        )
        !=
        len(unique_pairs)
        *
        X.shape[1]
    ]

    if missing_targets:
        raise ValueError(
            f"{model_name} is missing complete "
            f"TAT-target coverage: "
            f"{missing_targets}"
        )


# ============================================================
# 10. POOLED LSTM / GRU TAT METRICS
# ============================================================

def pooled_target_metrics(
    df,
    model_name,
):

    rows = []

    for target in TAT_TARGETS:

        temp = df[
            df["Target"]
            ==
            target
        ]

        metrics = calculate_metrics(
            temp["Actual"].to_numpy(),
            temp["Predicted"].to_numpy(),
        )

        rows.append(
            {
                "Model":
                    model_name,

                "Target":
                    target,

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


lstm_tat = pooled_target_metrics(
    lstm_df,
    "LSTM",
)

gru_tat = pooled_target_metrics(
    gru_df,
    "GRU",
)


# ============================================================
# 11. COMBINE TAT PHASE RESULTS
# ============================================================

all_tat_results = pd.concat(
    [
        linear_tat,
        lstm_tat,
        gru_tat,
    ],
    ignore_index=True,
)


# Explicit uniqueness check.
duplicate_keys = all_tat_results[
    all_tat_results.duplicated(
        subset=[
            "Model",
            "Target",
        ],
        keep=False,
    )
]

if len(duplicate_keys) > 0:
    raise ValueError(
        "Duplicate Model x Target rows remain:\n"
        +
        duplicate_keys[
            [
                "Model",
                "Target",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 12. PRINT TAT PHASE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TAT-RELEVANT PHASE RESULTS")
print("=" * 70)

print(
    all_tat_results[
        [
            "Model",
            "Target",
            "N",
            "MAE",
            "RMSE",
            "R2",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 13. MODEL-LEVEL DESCRIPTIVE SUMMARY
# ============================================================
# This is descriptive only.
#
# We do NOT create an arbitrary composite score and do NOT
# claim that the model with the lowest average is universally
# "best". Phase-level behaviour remains important for TAT.
# ============================================================

model_summary_rows = []

for model_name in [
    "Linear Regression",
    "LSTM",
    "GRU",
]:

    temp = all_tat_results[
        all_tat_results["Model"]
        ==
        model_name
    ]

    valid_r2 = temp[
        temp["R2"].notna()
    ]["R2"]

    model_summary_rows.append(
        {
            "Model":
                model_name,

            "Mean_MAE":
                temp["MAE"].mean(),

            "Mean_RMSE":
                temp["RMSE"].mean(),

            "Mean_R2":
                temp["R2"].mean(),

            "Valid_R2_Count":
                int(valid_r2.count()),

            "Negative_R2_Count":
                int(
                    (valid_r2 < 0).sum()
                ),
        }
    )


model_summary = pd.DataFrame(
    model_summary_rows
)

print("\n" + "=" * 70)
print("MODEL-LEVEL TAT SUMMARY")
print("=" * 70)

print(
    model_summary.to_string(
        index=False
    )
)


# ============================================================
# 14. TAT SIX-PHASE AGGREGATE OOS ANALYSIS
# ============================================================
# For each held-out pair, sum the six predicted phase times
# across all 100 route bins.
#
# This gives a direct model-level check of the quantity that
# will later contribute to EV driving TAT.
#
# Charging is NOT included here.
# Charging is handled separately in the final EV TAT workflow.
# ============================================================

def build_pair_tat_totals(
    df,
    model_name,
):

    temp = df[
        df["Target"].isin(
            TAT_TARGETS
        )
    ].copy()

    grouped = (
        temp.groupby(
            [
                "Pair_ID",
                "Target",
            ],
            as_index=False,
        )
        .agg(
            Actual=(
                "Actual",
                "sum",
            ),
            Predicted=(
                "Predicted",
                "sum",
            ),
        )
    )

    pair_rows = []

    for pair_id, group in grouped.groupby(
        "Pair_ID"
    ):

        actual_total = (
            group["Actual"].sum()
        )

        predicted_total = (
            group["Predicted"].sum()
        )

        pair_rows.append(
            {
                "Model":
                    model_name,

                "Pair_ID":
                    pair_id,

                "Actual_Six_Phase_TAT_min":
                    actual_total,

                "Predicted_Six_Phase_TAT_min":
                    predicted_total,

                "Absolute_Error_min":
                    abs(
                        actual_total
                        -
                        predicted_total
                    ),
            }
        )

    return pd.DataFrame(
        pair_rows
    )


# Convert Linear Regression predictions to the same schema.
linear_pair_input = linear_predictions.rename(
    columns={
        "Pair_ID":
            "Pair_ID",
    }
)

lstm_pair_input = lstm_df.copy()

if "Pair_ID" not in lstm_pair_input.columns:

    if "Test_Pair_ID" in lstm_pair_input.columns:
        lstm_pair_input["Pair_ID"] = (
            lstm_pair_input[
                "Test_Pair_ID"
            ].astype(str)
        )
    else:
        raise ValueError(
            "LSTM prediction file does not contain "
            "Pair_ID or Test_Pair_ID."
        )

if "Pair_ID" not in gru_df.columns:

    if "Test_Pair_ID" in gru_df.columns:
        gru_df["Pair_ID"] = (
            gru_df[
                "Test_Pair_ID"
            ].astype(str)
        )
    else:
        raise ValueError(
            "GRU prediction file does not contain "
            "Pair_ID or Test_Pair_ID."
        )

lstm_pair_input["Pair_ID"] = (
    lstm_pair_input[
        "Pair_ID"
    ].astype(str)
)

gru_df["Pair_ID"] = (
    gru_df[
        "Pair_ID"
    ].astype(str)
)

linear_pair_tat = build_pair_tat_totals(
    linear_pair_input,
    "Linear Regression",
)

lstm_pair_tat = build_pair_tat_totals(
    lstm_pair_input,
    "LSTM",
)

gru_pair_tat = build_pair_tat_totals(
    gru_df,
    "GRU",
)

pair_tat_results = pd.concat(
    [
        linear_pair_tat,
        lstm_pair_tat,
        gru_pair_tat,
    ],
    ignore_index=True,
)


# Aggregate six-pair TAT metrics.
aggregate_tat_rows = []

for model_name in [
    "Linear Regression",
    "LSTM",
    "GRU",
]:

    temp = pair_tat_results[
        pair_tat_results["Model"]
        ==
        model_name
    ]

    actual = temp[
        "Actual_Six_Phase_TAT_min"
    ].to_numpy()

    predicted = temp[
        "Predicted_Six_Phase_TAT_min"
    ].to_numpy()

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    aggregate_tat_rows.append(
        {
            "Model":
                model_name,

            "N_Pairs":
                len(temp),

            "MAE_min":
                metrics["MAE"],

            "RMSE_min":
                metrics["RMSE"],

            "R2":
                metrics["R2"],
        }
    )


aggregate_tat = pd.DataFrame(
    aggregate_tat_rows
)

print("\n" + "=" * 70)
print("SIX-PHASE TAT AGGREGATE OOS ANALYSIS")
print("=" * 70)

print(
    aggregate_tat.to_string(
        index=False
    )
)


# ============================================================
# 15. PHASE COMPARISON TABLE
# ============================================================

phase_comparison = all_tat_results.pivot(
    index="Target",
    columns="Model",
    values=[
        "MAE",
        "RMSE",
        "R2",
    ],
)

phase_comparison.columns = [
    f"{metric}_{model}"
    for metric, model in phase_comparison.columns
]

phase_comparison = (
    phase_comparison
    .reset_index()
)


# ============================================================
# 16. MODEL-LEVEL TAT COMPARISON TABLE
# ============================================================

model_comparison = (
    model_summary.merge(
        aggregate_tat,
        on="Model",
        how="left",
    )
)


# ============================================================
# 17. SAVE LINEAR REGRESSION OOS PREDICTIONS
# ============================================================

linear_prediction_path = os.path.join(
    OUTPUT_DIR,
    "TS_LinearRegression_TAT_OOS_Predictions.xlsx",
)

linear_predictions.to_excel(
    linear_prediction_path,
    index=False,
)


# ============================================================
# 18. SAVE COMPLETE MODEL-SELECTION WORKBOOK
# ============================================================

output_excel = os.path.join(
    OUTPUT_DIR,
    "TS_TAT_Model_Selection.xlsx",
)

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl",
) as writer:

    all_tat_results.to_excel(
        writer,
        sheet_name="TAT_Phase_Results",
        index=False,
    )

    phase_comparison.to_excel(
        writer,
        sheet_name="Phase_Comparison",
        index=False,
    )

    model_summary.to_excel(
        writer,
        sheet_name="Model_Summary",
        index=False,
    )

    aggregate_tat.to_excel(
        writer,
        sheet_name="Six_Phase_TAT",
        index=False,
    )

    pair_tat_results.to_excel(
        writer,
        sheet_name="Pair_TAT",
        index=False,
    )

    linear_fold_results.to_excel(
        writer,
        sheet_name="Linear_Fold_Results",
        index=False,
    )

    linear_predictions.to_excel(
        writer,
        sheet_name="Linear_OOS_Predictions",
        index=False,
    )


# ============================================================
# 19. REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "TS_TAT_Model_Selection_Report.txt",
)

report = []

report.append(
    "TS TAT-RELEVANT MODEL SELECTION REPORT"
)

report.append(
    "=" * 80
)

report.append(
    ""
)

report.append(
    "Purpose:"
)

report.append(
    "Compare Linear Regression, LSTM and GRU "
    "for the six EV phase-time quantities "
    "that contribute to EV driving TAT."
)

report.append(
    ""
)

report.append(
    "Dataset:"
)

report.append(
    str(SEQUENCE_DATASET)
)

report.append(
    f"Pairs: {len(unique_pairs)}"
)

report.append(
    f"Route bins per pair: {X.shape[1]}"
)

report.append(
    ""
)

report.append(
    "Validation:"
)

report.append(
    "Leave-One-Pair-Out"
)

report.append(
    "Complete paired trip held out in each fold."
)

report.append(
    "No random row split."
)

report.append(
    "No cross-pair sequence leakage."
)

report.append(
    ""
)

report.append(
    "LINEAR REGRESSION:"
)

report.append(
    "OOS predictions were reconstructed from "
    "the authoritative TS sequence dataset."
)

report.append(
    "The 36 fold-level workbook rows were NOT "
    "treated as pooled target-wise results."
)

report.append(
    ""
)

report.append(
    "TAT phase targets:"
)

for target in TAT_TARGETS:
    report.append(
        f"  - {target}"
    )

report.append(
    ""
)

report.append(
    "POOLED TAT PHASE RESULTS"
)

report.append(
    "-" * 80
)

report.append(
    all_tat_results.to_string(
        index=False
    )
)

report.append(
    ""
)

report.append(
    "SIX-PHASE TAT AGGREGATE OOS RESULTS"
)

report.append(
    "-" * 80
)

report.append(
    aggregate_tat.to_string(
        index=False
    )
)

report.append(
    ""
)

report.append(
    "INTERPRETATION NOTES"
)

report.append(
    "-" * 80
)

report.append(
    "MAE, RMSE and R2 are reported phase-by-phase."
)

report.append(
    "MAPE is not used as a primary metric because "
    "phase-time targets contain zero / near-zero values."
)

report.append(
    "The six-phase aggregate represents driving-phase "
    "TAT only; EV charging time is excluded."
)

report.append(
    "EV charging time remains a separate calculation "
    "in the final EV TAT reconstruction workflow."
)

report.append(
    "Only six independent paired trips are available, "
    "so model differences remain preliminary."
)

report.append(
    "This stage performs model comparison only; "
    "no hyperparameter tuning is performed."
)

report.append(
    ""
)

report.append(
    "OUTPUT FILES"
)

report.append(
    "-" * 80
)

report.append(
    f"Model-selection workbook: {output_excel}"
)

report.append(
    f"Linear Regression OOS predictions: "
    f"{linear_prediction_path}"
)

report.append(
    f"Report: {report_path}"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "\n".join(report)
    )


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("TS TAT MODEL SELECTION COMPLETED")
print("=" * 70)

print(
    f"\nWorkbook:\n{output_excel}"
)

print(
    f"\nLinear Regression OOS predictions:\n"
    f"{linear_prediction_path}"
)

print(
    f"\nReport:\n{report_path}"
)

print(
    "\nNo dataset was modified."
)

print(
    "No hyperparameter tuning was performed."
)

print(
    "Validation: Leave-One-Pair-Out."
)

print(
    "Charging/refuelling is excluded from the "
    "six-phase predictive TAT analysis."
)
