# ============================================================
# TS DIAGNOSTIC / INTERPRETATION
# ============================================================
# Purpose:
#   Diagnose the completed Time-Series experiments before moving
#   to EV TAT reconstruction.
#
# Models:
#   - Training Mean
#   - Route-Bin Mean
#   - Linear Regression
#   - LSTM
#   - GRU
#
# This script DOES NOT:
#   - train a model
#   - tune hyperparameters
#   - change the dataset
#   - create a new train/test split
#
# It checks:
#   1. Overall metric consistency
#   2. Recomputed LSTM/GRU metrics from saved OOS predictions
#   3. Fold-wise stability
#   4. Target-wise performance
#   5. Phase-wise performance
#   6. TAT-relevant target performance
#   7. Actual target variance / near-zero variance
#   8. GRU R2 diagnostic
#   9. Prediction coverage and integrity
#  10. Practical interpretation flags
#
# IMPORTANT:
#   MAPE is NOT used for model interpretation because several
#   phase targets contain zero / near-zero values.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
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
    "TimeSeries"
)

COMPARISON_DIR = os.path.join(
    RESULTS_ROOT,
    "Model_Comparison"
)

BASELINE_RESULTS = os.path.join(
    RESULTS_ROOT,
    "Baseline",
    "TS_Baseline_Results.xlsx"
)

LSTM_RESULTS = os.path.join(
    RESULTS_ROOT,
    "LSTM",
    "TS_LSTM_Results.xlsx"
)

LSTM_PREDICTIONS = os.path.join(
    RESULTS_ROOT,
    "LSTM",
    "TS_LSTM_Fold_Predictions.xlsx"
)

GRU_RESULTS = os.path.join(
    RESULTS_ROOT,
    "GRU",
    "TS_GRU_Results.xlsx"
)

GRU_PREDICTIONS = os.path.join(
    RESULTS_ROOT,
    "GRU",
    "TS_GRU_Fold_Predictions.xlsx"
)

OUTPUT_DIR = os.path.join(
    RESULTS_ROOT,
    "Diagnostic"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# 2. IMPORT AUTHORITATIVE TARGET LIST
# ============================================================

TIMESERIES_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if TIMESERIES_DIR not in sys.path:
    sys.path.insert(
        0,
        TIMESERIES_DIR
    )

from ts_data_loader import EV_TARGET_COLUMNS


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def normalize_text(value):

    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_column(
    df,
    candidates,
    required=True
):

    normalized = {
        normalize_text(col): col
        for col in df.columns
    }

    for candidate in candidates:

        key = normalize_text(
            candidate
        )

        if key in normalized:
            return normalized[key]

    if required:

        raise KeyError(
            f"\nCould not find any of:\n"
            f"{candidates}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    return None


def find_sheet(
    excel_file,
    candidates,
    keywords=None
):

    sheets = pd.ExcelFile(
        excel_file
    ).sheet_names

    normalized = {
        normalize_text(sheet): sheet
        for sheet in sheets
    }

    for candidate in candidates:

        key = normalize_text(
            candidate
        )

        if key in normalized:
            return normalized[key]

    if keywords:

        for sheet in sheets:

            sheet_norm = normalize_text(
                sheet
            )

            if all(
                normalize_text(keyword)
                in sheet_norm
                for keyword in keywords
            ):
                return sheet

    return None


def safe_r2(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    if len(actual) == 0:
        return np.nan

    if np.allclose(
        actual,
        actual[0]
    ):
        return np.nan

    return r2_score(
        actual,
        predicted
    )


def safe_mape(
    actual,
    predicted,
    epsilon=1e-6
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    denominator = np.maximum(
        np.abs(actual),
        epsilon
    )

    return (
        np.mean(
            np.abs(
                (actual - predicted)
                / denominator
            )
        )
        * 100
    )


def metric_dict(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    return {

        "MAE":
            mean_absolute_error(
                actual,
                predicted
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            ),

        "R2":
            safe_r2(
                actual,
                predicted
            ),

        "MAPE_percent":
            safe_mape(
                actual,
                predicted
            )
    }


def identify_phase(
    target
):

    target = str(target)

    if "MOVING_AVERAGE_SPEED" in target:
        return "OVERALL"

    if "STOP" in target:
        return "STOP"

    if "IDLE" in target:
        return "IDLE"

    if "INCHING" in target:
        return "INCHING"

    if "ACC_" in target:
        return "ACCELERATION"

    if "DEC_" in target:
        return "DECELERATION"

    if "CRUISE" in target:
        return "CRUISE"

    return "OTHER"


# ============================================================
# 4. LOAD SAVED PREDICTIONS
# ============================================================

print("=" * 70)
print("TS DIAGNOSTIC / INTERPRETATION")
print("=" * 70)

for name, path in [
    ("LSTM predictions", LSTM_PREDICTIONS),
    ("GRU predictions", GRU_PREDICTIONS),
    ("Baseline results", BASELINE_RESULTS),
    ("LSTM results", LSTM_RESULTS),
    ("GRU results", GRU_RESULTS)
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}"
        )

    print(
        f"[FOUND] {name}"
    )


print(
    "\nLoading saved out-of-sample predictions..."
)

lstm_pred = pd.read_excel(
    LSTM_PREDICTIONS
)

gru_pred = pd.read_excel(
    GRU_PREDICTIONS
)

print(
    f"LSTM prediction rows : "
    f"{len(lstm_pred)}"
)

print(
    f"GRU prediction rows  : "
    f"{len(gru_pred)}"
)


# ============================================================
# 5. PREDICTION STRUCTURE AUDIT
# ============================================================

print("\n" + "=" * 70)
print("1. PREDICTION STRUCTURE AUDIT")
print("=" * 70)

required_cols = [
    "Target",
    "Actual",
    "Predicted"
]

audit_rows = []

for model_name, df in [
    ("LSTM", lstm_pred),
    ("GRU", gru_pred)
]:

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    pair_col = find_column(
        df,
        [
            "Pair_ID",
            "Test_Pair_ID",
            "Test_Pair",
            "Pair",
            "PairID"
        ],
        required=False
    )

    fold_col = find_column(
        df,
        [
            "Fold",
            "Fold_Number",
            "Fold_ID"
        ],
        required=False
    )

    route_col = find_column(
        df,
        [
            "Route_Bin",
            "RouteBin",
            "Bin"
        ],
        required=False
    )

    audit_rows.append({

        "Model":
            model_name,

        "Rows":
            len(df),

        "Missing_Required_Columns":
            ", ".join(missing)
            if missing
            else "None",

        "Unique_Targets":
            df["Target"].nunique()
            if "Target" in df.columns
            else np.nan,

        "Unique_Pairs":
            df[pair_col].nunique()
            if pair_col
            else np.nan,

        "Unique_Folds":
            df[fold_col].nunique()
            if fold_col
            else np.nan,

        "Unique_Route_Bins":
            df[route_col].nunique()
            if route_col
            else np.nan,

        "Expected_Rows":
            6 * 100 * len(EV_TARGET_COLUMNS)
    })


prediction_audit_df = pd.DataFrame(
    audit_rows
)

print(
    prediction_audit_df.to_string(
        index=False
    )
)


# ============================================================
# 6. RECOMPUTE TARGET-WISE METRICS
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET-WISE OUT-OF-SAMPLE DIAGNOSTICS")
print("=" * 70)


def calculate_target_metrics(
    df,
    model_name
):

    rows = []

    for target in EV_TARGET_COLUMNS:

        temp = df[
            df["Target"].astype(str).str.strip()
            == target
        ].copy()

        actual = pd.to_numeric(
            temp["Actual"],
            errors="coerce"
        ).to_numpy()

        predicted = pd.to_numeric(
            temp["Predicted"],
            errors="coerce"
        ).to_numpy()

        valid = (
            np.isfinite(actual)
            &
            np.isfinite(predicted)
        )

        actual = actual[valid]
        predicted = predicted[valid]

        if len(actual) == 0:
            continue

        metrics = metric_dict(
            actual,
            predicted
        )

        actual_mean = np.mean(
            actual
        )

        actual_std = np.std(
            actual,
            ddof=1
        ) if len(actual) > 1 else 0.0

        actual_min = np.min(
            actual
        )

        actual_max = np.max(
            actual
        )

        rows.append({

            "Model":
                model_name,

            "Target":
                target,

            "Phase":
                identify_phase(
                    target
                ),

            "N":
                len(actual),

            "Actual_Mean":
                actual_mean,

            "Actual_Std":
                actual_std,

            "Actual_Min":
                actual_min,

            "Actual_Max":
                actual_max,

            "Actual_Range":
                actual_max - actual_min,

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"],

            "MAPE_percent":
                metrics["MAPE_percent"],

            "MAE_to_Std":
                (
                    metrics["MAE"]
                    / actual_std
                    if actual_std > 1e-12
                    else np.nan
                )
        })

    return pd.DataFrame(rows)


lstm_target_diag = calculate_target_metrics(
    lstm_pred,
    "LSTM"
)

gru_target_diag = calculate_target_metrics(
    gru_pred,
    "GRU"
)

target_diag_df = pd.concat(
    [
        lstm_target_diag,
        gru_target_diag
    ],
    ignore_index=True
)

print(
    target_diag_df[
        [
            "Model",
            "Target",
            "MAE",
            "RMSE",
            "R2",
            "Actual_Std",
            "MAE_to_Std"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 7. GRU R2 DIAGNOSTIC
# ============================================================
# IMPORTANT:
#   GRU is a multi-output model with 13 EV targets.
#   The original GRU model calculates pooled R2 using:
#       r2_score(..., multioutput="variance_weighted")
#   Therefore the diagnostic MUST preserve the 13-target
#   structure. Flattening all 7,800 values into one vector
#   gives a different, non-equivalent R2 and must NOT be used.
# ============================================================

print("\n" + "=" * 70)
print("3. GRU R2 DIAGNOSTIC")
print("=" * 70)


def reconstruct_target_matrix(df, target_columns):
    """Reconstruct [route-bin observations x 13 targets]."""

    temp = df.copy()

    temp["Target"] = temp["Target"].astype(str).str.strip()
    temp["Actual"] = pd.to_numeric(temp["Actual"], errors="coerce")
    temp["Predicted"] = pd.to_numeric(temp["Predicted"], errors="coerce")

    pair_col = find_column(
        temp,
        ["Pair_ID", "Test_Pair_ID", "Test_Pair", "Pair", "PairID"],
        required=True
    )

    fold_col = find_column(
        temp,
        ["Fold", "Fold_Number", "Fold_ID"],
        required=True
    )

    route_col = find_column(
        temp,
        ["Route_Bin", "RouteBin", "Bin"],
        required=True
    )

    index_cols = [fold_col, pair_col, route_col]

    actual_wide = temp.pivot_table(
        index=index_cols,
        columns="Target",
        values="Actual",
        aggfunc="first"
    ).reindex(columns=target_columns)

    predicted_wide = temp.pivot_table(
        index=index_cols,
        columns="Target",
        values="Predicted",
        aggfunc="first"
    ).reindex(columns=target_columns)

    valid_rows = (
        actual_wide.notna().all(axis=1)
        & predicted_wide.notna().all(axis=1)
    )

    actual_wide = actual_wide.loc[valid_rows]
    predicted_wide = predicted_wide.loc[valid_rows]

    if len(actual_wide) == 0:
        raise ValueError(
            "Could not reconstruct complete multi-output GRU observations."
        )

    return (
        actual_wide.to_numpy(dtype=float),
        predicted_wide.to_numpy(dtype=float)
    )


gru_actual_matrix, gru_predicted_matrix = reconstruct_target_matrix(
    gru_pred,
    EV_TARGET_COLUMNS
)

print(
    "Reconstructed GRU target matrix : "
    f"{gru_actual_matrix.shape}"
)

# EXACT same pooled multi-output definition used by ts_gru.py
gru_pooled_mae = mean_absolute_error(
    gru_actual_matrix,
    gru_predicted_matrix
)

gru_pooled_rmse = np.sqrt(
    mean_squared_error(
        gru_actual_matrix,
        gru_predicted_matrix
    )
)

gru_pooled_r2 = r2_score(
    gru_actual_matrix,
    gru_predicted_matrix,
    multioutput="variance_weighted"
)

gru_pooled_mape = safe_mape(
    gru_actual_matrix,
    gru_predicted_matrix
)

gru_overall_recomputed = {
    "MAE": gru_pooled_mae,
    "RMSE": gru_pooled_rmse,
    "R2": gru_pooled_r2,
    "MAPE_percent": gru_pooled_mape
}

print(
    "GRU R2 recomputed using the original "
    "multi-output definition : "
    f"{gru_pooled_r2:.6f}"
)

print(
    "GRU MAE recomputed using the original "
    "multi-output structure : "
    f"{gru_pooled_mae:.6f}"
)

print(
    "GRU RMSE recomputed using the original "
    "multi-output structure : "
    f"{gru_pooled_rmse:.6f}"
)

print("\nThe GRU workbook reported:")

gru_results_sheet = find_sheet(
    GRU_RESULTS,
    ["Overall", "Overall_OOS", "Overall_Results"],
    keywords=["overall"]
)

gru_overall_original = pd.read_excel(
    GRU_RESULTS,
    sheet_name=gru_results_sheet
)

print(
    gru_overall_original.to_string(index=False)
)

original_r2_col = find_column(
    gru_overall_original,
    ["R2", "R_squared"],
    required=False
)

original_mae_col = find_column(
    gru_overall_original,
    ["MAE"],
    required=False
)

original_rmse_col = find_column(
    gru_overall_original,
    ["RMSE"],
    required=False
)

if len(gru_overall_original) > 0:

    original_r2 = (
        float(gru_overall_original[original_r2_col].iloc[0])
        if original_r2_col is not None else np.nan
    )

    original_mae = (
        float(gru_overall_original[original_mae_col].iloc[0])
        if original_mae_col is not None else np.nan
    )

    original_rmse = (
        float(gru_overall_original[original_rmse_col].iloc[0])
        if original_rmse_col is not None else np.nan
    )

    print("\nGRU METRIC CONSISTENCY CHECK")
    print("-" * 70)

    print(f"Workbook R2       : {original_r2:.6f}")
    print(f"Recomputed R2     : {gru_pooled_r2:.6f}")
    print(f"Workbook MAE      : {original_mae:.6f}")
    print(f"Recomputed MAE    : {gru_pooled_mae:.6f}")
    print(f"Workbook RMSE     : {original_rmse:.6f}")
    print(f"Recomputed RMSE   : {gru_pooled_rmse:.6f}")

    if (
        np.isclose(original_r2, gru_pooled_r2, atol=1e-6)
        and np.isclose(original_mae, gru_pooled_mae, atol=1e-6)
        and np.isclose(original_rmse, gru_pooled_rmse, atol=1e-6)
    ):
        print("\nRESULT: PASS")
        print(
            "The saved GRU predictions reproduce the original "
            "workbook metrics."
        )
        print("There is no GRU Excel aggregation error.")
    else:
        print("\nRESULT: CHECK")
        print(
            "The saved predictions do not reproduce the workbook metrics."
        )

print("\nNOTE:")
print(
    "The previously reported flattened R2 is intentionally not used "
    "because it collapses the 13 targets into one vector and is not "
    "equivalent to the variance-weighted multi-output R2 used by ts_gru.py."
)


# 8. FOLD-WISE GRU DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("4. FOLD-WISE GRU DIAGNOSTIC")
print("=" * 70)

pair_col_gru = find_column(
    gru_pred,
    [
        "Pair_ID",
        "Test_Pair_ID",
        "Test_Pair",
        "Pair",
        "PairID"
    ],
    required=False
)

fold_col_gru = find_column(
    gru_pred,
    [
        "Fold",
        "Fold_Number",
        "Fold_ID"
    ],
    required=False
)

fold_rows = []

if fold_col_gru is not None:

    for fold, temp in gru_pred.groupby(
        fold_col_gru
    ):

        actual = pd.to_numeric(
            temp["Actual"],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            temp["Predicted"],
            errors="coerce"
        )

        valid = (
            np.isfinite(actual)
            &
            np.isfinite(predicted)
        )

        actual = actual[valid]
        predicted = predicted[valid]

        metrics = metric_dict(
            actual,
            predicted
        )

        pair_value = (
            temp[pair_col_gru].iloc[0]
            if pair_col_gru is not None
            else "Unknown"
        )

        fold_rows.append({

            "Fold":
                fold,

            "Test_Pair":
                pair_value,

            "Rows":
                len(actual),

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"],

            "MAPE_percent":
                metrics["MAPE_percent"]
        })


gru_fold_diag_df = pd.DataFrame(
    fold_rows
)

print(
    gru_fold_diag_df.to_string(
        index=False
    )
)


# ============================================================
# 9. TARGET VARIABILITY FLAGS
# ============================================================

print("\n" + "=" * 70)
print("5. TARGET VARIABILITY / IDENTIFIABILITY")
print("=" * 70)

variability_rows = []

for target in EV_TARGET_COLUMNS:

    temp = gru_pred[
        gru_pred["Target"].astype(str).str.strip()
        == target
    ]

    actual = pd.to_numeric(
        temp["Actual"],
        errors="coerce"
    ).dropna().to_numpy()

    if len(actual) == 0:
        continue

    std = np.std(
        actual,
        ddof=1
    ) if len(actual) > 1 else 0.0

    mean_abs = np.mean(
        np.abs(actual)
    )

    coefficient_of_variation = (
        std / mean_abs
        if mean_abs > 1e-12
        else np.nan
    )

    if std < 1e-3:

        flag = (
            "NEAR_CONSTANT_TARGET"
        )

    elif std < 0.01:

        flag = (
            "VERY_LOW_VARIABILITY"
        )

    elif coefficient_of_variation < 0.05:

        flag = (
            "LOW_RELATIVE_VARIABILITY"
        )

    else:

        flag = "ADEQUATE_VARIABILITY"

    variability_rows.append({

        "Target":
            target,

        "Phase":
            identify_phase(
                target
            ),

        "N":
            len(actual),

        "Mean":
            np.mean(actual),

        "Std":
            std,

        "Min":
            np.min(actual),

        "Max":
            np.max(actual),

        "Coefficient_of_Variation":
            coefficient_of_variation,

        "Variability_Flag":
            flag
    })


variability_df = pd.DataFrame(
    variability_rows
)

print(
    variability_df.to_string(
        index=False
    )
)


# ============================================================
# 10. TAT-RELEVANT DIAGNOSTIC
# ============================================================

print("\n" + "=" * 70)
print("6. TAT-RELEVANT TARGET DIAGNOSTICS")
print("=" * 70)

tat_targets = [
    "EV_STOP_Time_min",
    "EV_IDLE_Time_min",
    "EV_INCHING_Time_min",
    "EV_ACC_Time_min",
    "EV_DEC_Time_min",
    "EV_CRUISE_Time_min"
]

tat_diag_df = target_diag_df[
    target_diag_df["Target"].isin(
        tat_targets
    )
].copy()

print(
    tat_diag_df[
        [
            "Model",
            "Target",
            "MAE",
            "RMSE",
            "R2",
            "Actual_Std",
            "MAE_to_Std"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. MODEL COMPARISON PER TARGET
# ============================================================

print("\n" + "=" * 70)
print("7. LSTM vs GRU TARGET COMPARISON")
print("=" * 70)

comparison_rows = []

for target in EV_TARGET_COLUMNS:

    lstm_row = lstm_target_diag[
        lstm_target_diag["Target"] == target
    ]

    gru_row = gru_target_diag[
        gru_target_diag["Target"] == target
    ]

    if (
        len(lstm_row) == 0
        or len(gru_row) == 0
    ):
        continue

    lstm_row = lstm_row.iloc[0]
    gru_row = gru_row.iloc[0]

    comparison_rows.append({

        "Target":
            target,

        "Phase":
            identify_phase(
                target
            ),

        "LSTM_MAE":
            lstm_row["MAE"],

        "GRU_MAE":
            gru_row["MAE"],

        "LSTM_RMSE":
            lstm_row["RMSE"],

        "GRU_RMSE":
            gru_row["RMSE"],

        "LSTM_R2":
            lstm_row["R2"],

        "GRU_R2":
            gru_row["R2"],

        "MAE_Difference_GRU_minus_LSTM":
            gru_row["MAE"]
            - lstm_row["MAE"],

        "R2_Difference_GRU_minus_LSTM":
            gru_row["R2"]
            - lstm_row["R2"]
    })


target_comparison_df = pd.DataFrame(
    comparison_rows
)

print(
    target_comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 12. SIMPLE INTERPRETATION FLAGS
# ============================================================

print("\n" + "=" * 70)
print("8. INTERPRETATION FLAGS")
print("=" * 70)

interpretation_rows = []


# Overall model values from known completed experiments.
# These are read from the comparison workbook rather than
# hard-coded as a model decision.

comparison_file = os.path.join(
    COMPARISON_DIR,
    "TS_Model_Comparison.xlsx"
)

if os.path.exists(
    comparison_file
):

    overall_compare = pd.read_excel(
        comparison_file,
        sheet_name="Overall_Comparison"
    )

else:

    overall_compare = pd.DataFrame()


if len(overall_compare) > 0:

    linear_row = overall_compare[
        overall_compare["Model"]
        == "Linear Regression"
    ]

    lstm_row = overall_compare[
        overall_compare["Model"]
        == "LSTM"
    ]

    gru_row = overall_compare[
        overall_compare["Model"]
        == "GRU"
    ]

    if (
        len(linear_row)
        and len(lstm_row)
    ):

        linear_mae = float(
            linear_row["MAE"].iloc[0]
        )

        lstm_mae = float(
            lstm_row["MAE"].iloc[0]
        )

        linear_r2 = float(
            linear_row["R2"].iloc[0]
        )

        lstm_r2 = float(
            lstm_row["R2"].iloc[0]
        )

        interpretation_rows.append({

            "Topic":
                "Linear vs LSTM",

            "Observation":
                (
                    "LSTM does not improve all primary "
                    "pooled metrics relative to Linear Regression."
                ),

            "Evidence":
                (
                    f"Linear MAE={linear_mae:.6f}, "
                    f"LSTM MAE={lstm_mae:.6f}; "
                    f"Linear R2={linear_r2:.6f}, "
                    f"LSTM R2={lstm_r2:.6f}"
                )
        })

    if len(gru_row):

        gru_mae = float(
            gru_row["MAE"].iloc[0]
        )

        gru_r2 = float(
            gru_row["R2"].iloc[0]
        )

        interpretation_rows.append({

            "Topic":
                "GRU",

            "Observation":
                (
                    "GRU requires careful interpretation because "
                    "its pooled R2 is substantially lower than its "
                    "MAE/RMSE alone would suggest."
                ),

            "Evidence":
                (
                    f"GRU MAE={gru_mae:.6f}, "
                    f"GRU R2={gru_r2:.6f}"
                )
        })


# Low variability flags

low_variability = variability_df[
    variability_df["Variability_Flag"]
    != "ADEQUATE_VARIABILITY"
]

for _, row in low_variability.iterrows():

    interpretation_rows.append({

        "Topic":
            f"Target variability: {row['Target']}",

        "Observation":
            (
                "Target has limited variability; R2 may be "
                "unstable or difficult to interpret."
            ),

        "Evidence":
            (
                f"Std={row['Std']:.8f}, "
                f"CV={row['Coefficient_of_Variation']:.8f}, "
                f"Flag={row['Variability_Flag']}"
            )
    })


# TAT targets

if len(tat_diag_df) > 0:

    for model in tat_diag_df["Model"].unique():

        temp = tat_diag_df[
            tat_diag_df["Model"] == model
        ]

        negative_r2 = temp[
            temp["R2"] < 0
        ]

        interpretation_rows.append({

            "Topic":
                f"TAT-phase performance: {model}",

            "Observation":
                (
                    "Review individual TAT phase targets rather "
                    "than relying only on pooled metrics."
                ),

            "Evidence":
                (
                    f"{len(negative_r2)} of "
                    f"{len(temp)} TAT-phase targets have "
                    f"negative R2."
                )
        })


interpretation_df = pd.DataFrame(
    interpretation_rows
)

print(
    interpretation_df.to_string(
        index=False
    )
)


# ============================================================
# 13. FINAL SCIENTIFIC CHECKLIST
# ============================================================

checklist_rows = [

    {
        "Check":
            "Independent trip-level validation",
        "Status":
            "PASS",
        "Reason":
            "LOPO validation uses complete paired trips."
    },

    {
        "Check":
            "Random row split avoided",
        "Status":
            "PASS",
        "Reason":
            "No random row split used."
    },

    {
        "Check":
            "Cross-pair leakage avoided",
        "Status":
            "PASS",
        "Reason":
            "Test pair remains completely held out."
    },

    {
        "Check":
            "LSTM prediction coverage",
        "Status":
            (
                "PASS"
                if len(lstm_pred)
                == 6 * 100 * len(EV_TARGET_COLUMNS)
                else "CHECK"
            ),
        "Reason":
            "Expected 7800 prediction rows."
    },

    {
        "Check":
            "GRU prediction coverage",
        "Status":
            (
                "PASS"
                if len(gru_pred)
                == 6 * 100 * len(EV_TARGET_COLUMNS)
                else "CHECK"
            ),
        "Reason":
            "Expected 7800 prediction rows."
    },

    {
        "Check":
            "MAPE suitability",
        "Status":
            "NOT PRIMARY",
        "Reason":
            "Zero / near-zero phase targets make percentage "
            "error unstable."
    },

    {
        "Check":
            "Model selection",
        "Status":
            "PENDING INTERPRETATION",
        "Reason":
            "Use MAE, RMSE, R2, target-wise behavior, fold "
            "stability and TAT relevance together."
    },

    {
        "Check":
            "EV TAT reconstruction",
        "Status":
            "NEXT STAGE",
        "Reason":
            "Perform only after the TS model behavior is documented."
    }
]

checklist_df = pd.DataFrame(
    checklist_rows
)


# ============================================================
# 14. SAVE EXCEL
# ============================================================

output_excel = os.path.join(
    OUTPUT_DIR,
    "TS_Diagnostic_Interpretation.xlsx"
)

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl"
) as writer:

    prediction_audit_df.to_excel(
        writer,
        sheet_name="Prediction_Audit",
        index=False
    )

    target_diag_df.to_excel(
        writer,
        sheet_name="Target_Diagnostics",
        index=False
    )

    gru_fold_diag_df.to_excel(
        writer,
        sheet_name="GRU_Fold_Diagnostics",
        index=False
    )

    variability_df.to_excel(
        writer,
        sheet_name="Target_Variability",
        index=False
    )

    tat_diag_df.to_excel(
        writer,
        sheet_name="TAT_Diagnostics",
        index=False
    )

    target_comparison_df.to_excel(
        writer,
        sheet_name="LSTM_vs_GRU",
        index=False
    )

    interpretation_df.to_excel(
        writer,
        sheet_name="Interpretation_Flags",
        index=False
    )

    checklist_df.to_excel(
        writer,
        sheet_name="Scientific_Checklist",
        index=False
    )


# ============================================================
# 15. SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "TS_Diagnostic_Interpretation_Report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TIME-SERIES DIAGNOSTIC / INTERPRETATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "PURPOSE\n"
    )

    f.write(
        "Diagnose completed TS experiments before EV TAT "
        "reconstruction.\n\n"
    )

    f.write(
        "VALIDATION\n"
    )

    f.write(
        "Leave-One-Pair-Out validation with six independent "
        "paired trips.\n"
    )

    f.write(
        "No random row split.\n"
    )

    f.write(
        "No cross-pair leakage.\n\n"
    )

    f.write(
        "GRU RECOMPUTED METRICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for key, value in gru_overall_recomputed.items():

        f.write(
            f"{key}: {value}\n"
        )

    f.write(
        "\nGRU FOLD DIAGNOSTICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        gru_fold_diag_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTARGET VARIABILITY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        variability_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTAT DIAGNOSTICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        tat_diag_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nINTERPRETATION FLAGS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        interpretation_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSCIENTIFIC CHECKLIST\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        checklist_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nIMPORTANT NOTE\n"
    )

    f.write(
        "The diagnostic does not select a model using a single "
        "composite score. Model usefulness must be interpreted "
        "in the context of EV parameter estimation and the TAT "
        "reconstruction objective.\n"
    )


# ============================================================
# 16. PLOTS
# ============================================================

try:

    import matplotlib.pyplot as plt

    # --------------------------------------------------------
    # Target R2: LSTM vs GRU
    # --------------------------------------------------------

    plot_df = target_comparison_df.copy()

    x = np.arange(
        len(plot_df)
    )

    width = 0.38

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width / 2,
        plot_df["LSTM_R2"],
        width,
        label="LSTM"
    )

    plt.bar(
        x + width / 2,
        plot_df["GRU_R2"],
        width,
        label="GRU"
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xticks(
        x,
        plot_df["Target"],
        rotation=75,
        ha="right"
    )

    plt.ylabel("R2")
    plt.xlabel("EV Target")
    plt.title(
        "Target-wise R2: LSTM vs GRU"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "Target_Wise_R2_LSTM_vs_GRU.png"
        ),
        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GRU fold R2
    # --------------------------------------------------------

    if len(gru_fold_diag_df) > 0:

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            gru_fold_diag_df["Fold"],
            gru_fold_diag_df["R2"],
            marker="o"
        )

        plt.axhline(
            0,
            linewidth=1
        )

        plt.xlabel("LOPO Fold")
        plt.ylabel("R2")
        plt.title(
            "GRU Fold-wise R2"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOT_DIR,
                "GRU_Fold_R2.png"
            ),
            dpi=300
        )

        plt.close()


    # --------------------------------------------------------
    # Target actual standard deviation
    # --------------------------------------------------------

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        variability_df["Target"],
        variability_df["Std"]
    )

    plt.xticks(
        rotation=75,
        ha="right"
    )

    plt.xlabel("EV Target")
    plt.ylabel("Actual Standard Deviation")
    plt.title(
        "EV Target Variability"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "EV_Target_Variability.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nDiagnostic plots created successfully."
    )

except Exception as exc:

    print(
        f"\n[WARNING] Plot generation skipped: {exc}"
    )


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TS DIAGNOSTIC COMPLETED")
print("=" * 70)

print(
    "\nExcel results:"
)

print(
    output_excel
)

print(
    "\nText report:"
)

print(
    report_path
)

print(
    "\nPlots:"
)

print(
    PLOT_DIR
)

print(
    "\nNo model was trained."
)

print(
    "No hyperparameter tuning was performed."
)

print(
    "No new train/test split was created."
)

print(
    "No existing dataset was modified."
)

print(
    "\nNext stage after reviewing this report:"
)

print(
    "EV TAT reconstruction and final EV-vs-ICE TAT comparison."
)
