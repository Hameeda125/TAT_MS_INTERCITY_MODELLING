# ============================================================
# TS MODEL COMPARISON - ROBUST VERSION
# ============================================================
# Compares completed Time-Series experiments:
#   - Training Mean
#   - Route-Bin Mean
#   - Linear Regression
#   - LSTM
#   - GRU
#
# Validation:
#   Leave-One-Pair-Out (LOPO)
#
# Primary metrics:
#   MAE, RMSE, R2
#
# MAPE:
#   Reference only because zero / near-zero phase quantities
#   make percentage error unstable.
#
# IMPORTANT:
#   - Automatically detects workbook sheet names.
#   - Handles different column names such as Test_Pair,
#     Pair_ID, Test_Pair_ID, etc.
#   - Recalculates LSTM/GRU target-wise metrics directly from
#     saved predictions when possible, avoiding inconsistent
#     aggregation across workbook formats.
#   - Does NOT train any model.
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

PLOT_DIR = os.path.join(
    COMPARISON_DIR,
    "plots"
)

os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 2. AUTHORITATIVE TARGET NAMES
# ============================================================

TIMESERIES_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if TIMESERIES_DIR not in sys.path:
    sys.path.insert(0, TIMESERIES_DIR)

from ts_data_loader import EV_TARGET_COLUMNS


# ============================================================
# 3. INPUT FILES
# ============================================================

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


# ============================================================
# 4. FILE CHECK
# ============================================================

print("=" * 70)
print("TS MODEL COMPARISON")
print("=" * 70)

required_files = {
    "Baseline Results": BASELINE_RESULTS,
    "LSTM Results": LSTM_RESULTS,
    "LSTM Predictions": LSTM_PREDICTIONS,
    "GRU Results": GRU_RESULTS,
    "GRU Predictions": GRU_PREDICTIONS,
}

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )

    print(f"[FOUND] {name}")


# ============================================================
# 5. GENERAL HELPERS
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


def find_sheet(
    excel_file,
    candidates,
    keywords=None
):

    xls = pd.ExcelFile(excel_file)
    sheets = xls.sheet_names

    normalized = {
        normalize_text(sheet): sheet
        for sheet in sheets
    }

    # Exact normalized match
    for candidate in candidates:

        key = normalize_text(candidate)

        if key in normalized:
            return normalized[key]

    # Keyword fallback
    if keywords:

        for sheet in sheets:

            sheet_norm = normalize_text(sheet)

            if all(
                normalize_text(keyword)
                in sheet_norm
                for keyword in keywords
            ):
                return sheet

    raise ValueError(
        f"\nRequired sheet not found in:\n"
        f"{excel_file}\n\n"
        f"Available sheets:\n{sheets}"
    )


def load_model_workbook(
    path,
    model_name
):

    overall_sheet = find_sheet(
        path,
        [
            "Overall",
            "Overall_OOS",
            "Overall_Results",
            "Overall_OOS_Results"
        ],
        keywords=["overall"]
    )

    fold_sheet = find_sheet(
        path,
        [
            "Fold_Results",
            "Fold_Wise",
            "Fold_Wise_Results"
        ],
        keywords=["fold"]
    )

    target_sheet = find_sheet(
        path,
        [
            "Target_Wise",
            "Target_Wise_OOS",
            "Target_Results"
        ],
        keywords=["target"]
    )

    overall = pd.read_excel(
        path,
        sheet_name=overall_sheet
    )

    folds = pd.read_excel(
        path,
        sheet_name=fold_sheet
    )

    targets = pd.read_excel(
        path,
        sheet_name=target_sheet
    )

    print(f"{model_name} sheets:")
    print(f"  Overall     : {overall_sheet}")
    print(f"  Fold        : {fold_sheet}")
    print(f"  Target-wise : {target_sheet}")

    return overall, folds, targets


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

        key = normalize_text(candidate)

        if key in normalized:
            return normalized[key]

    if required:

        raise KeyError(
            f"\nCould not find any of {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def standardize_model_name(name):

    key = normalize_text(name)

    mapping = {
        "trainingmean": "Training Mean",
        "routebinmean": "Route-Bin Mean",
        "linearregression": "Linear Regression",
        "ridge": "Ridge",
        "elasticnet": "ElasticNet",
        "pls": "PLS",
        "randomforest": "RandomForest",
        "extratrees": "ExtraTrees",
        "svr": "SVR",
        "lstm": "LSTM",
        "gru": "GRU",
    }

    return mapping.get(
        key,
        str(name).strip()
    )


def safe_mape(
    y_true,
    y_pred,
    epsilon=1e-6
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    denominator = np.maximum(
        np.abs(y_true),
        epsilon
    )

    return (
        np.mean(
            np.abs(
                (y_true - y_pred)
                / denominator
            )
        )
        * 100
    )


def metric_dict(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),
        "R2": r2_score(
            y_true,
            y_pred,
            multioutput="variance_weighted"
        ),
        "MAPE_percent": safe_mape(
            y_true,
            y_pred
        )
    }


def prepare_target_table(
    df,
    default_model=None
):

    temp = df.copy()

    target_col = find_column(
        temp,
        [
            "Target",
            "EV_Target",
            "Target_Name"
        ]
    )

    temp["Target"] = (
        temp[target_col]
        .astype(str)
        .str.strip()
    )

    # Find metrics
    for metric in [
        "MAE",
        "RMSE",
        "R2",
        "MAPE_percent"
    ]:

        col = find_column(
            temp,
            [metric],
            required=False
        )

        if col is not None:

            temp[metric] = pd.to_numeric(
                temp[col],
                errors="coerce"
            )

        else:

            temp[metric] = np.nan

    model_col = find_column(
        temp,
        ["Model"],
        required=False
    )

    if default_model is not None:

        temp["Model"] = default_model

    elif model_col is not None:

        temp["Model"] = (
            temp[model_col]
            .apply(
                standardize_model_name
            )
        )

    else:

        temp["Model"] = "Unknown"

    return temp[
        [
            "Target",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent",
            "Model"
        ]
    ]


def identify_phase(target):

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
# 6. LOAD WORKBOOKS
# ============================================================

print("\n" + "=" * 70)
print("LOADING BASELINE RESULTS")
print("=" * 70)

(
    baseline_overall,
    baseline_folds,
    baseline_target
) = load_model_workbook(
    BASELINE_RESULTS,
    "Baseline"
)

print(
    f"Baseline overall rows : "
    f"{len(baseline_overall)}"
)

print(
    f"Baseline fold rows    : "
    f"{len(baseline_folds)}"
)

print(
    f"Baseline target rows  : "
    f"{len(baseline_target)}"
)


print("\n" + "=" * 70)
print("LOADING LSTM RESULTS")
print("=" * 70)

(
    lstm_overall,
    lstm_folds,
    lstm_target
) = load_model_workbook(
    LSTM_RESULTS,
    "LSTM"
)

print(
    f"LSTM overall rows : "
    f"{len(lstm_overall)}"
)

print(
    f"LSTM fold rows    : "
    f"{len(lstm_folds)}"
)

print(
    f"LSTM target rows  : "
    f"{len(lstm_target)}"
)


print("\n" + "=" * 70)
print("LOADING GRU RESULTS")
print("=" * 70)

(
    gru_overall,
    gru_folds,
    gru_target
) = load_model_workbook(
    GRU_RESULTS,
    "GRU"
)

print(
    f"GRU overall rows : "
    f"{len(gru_overall)}"
)

print(
    f"GRU fold rows    : "
    f"{len(gru_folds)}"
)

print(
    f"GRU target rows  : "
    f"{len(gru_target)}"
)


# ============================================================
# 7. LOAD PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING SAVED PREDICTIONS")
print("=" * 70)

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
# 8. PREDICTION STRUCTURE
# ============================================================

required_prediction_columns = {
    "Target",
    "Actual",
    "Predicted"
}

for model_name, prediction_df in [
    ("LSTM", lstm_pred),
    ("GRU", gru_pred)
]:

    missing = (
        required_prediction_columns
        - set(prediction_df.columns)
    )

    if missing:

        raise KeyError(
            f"{model_name} prediction file is missing "
            f"required columns: {sorted(missing)}"
        )

    pair_col = find_column(
        prediction_df,
        [
            "Pair_ID",
            "Test_Pair",
            "Test_Pair_ID",
            "Pair",
            "PairID"
        ],
        required=False
    )

    fold_col = find_column(
        prediction_df,
        [
            "Fold",
            "Fold_Number",
            "Fold_ID"
        ],
        required=False
    )

    print(
        f"[PASS] {model_name} prediction structure"
    )

    print(
        f"       Pair column : "
        f"{pair_col if pair_col else 'not present'}"
    )

    print(
        f"       Fold column : "
        f"{fold_col if fold_col else 'not present'}"
    )


# ============================================================
# 9. OVERALL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("OVERALL MODEL COMPARISON")
print("=" * 70)

overall_rows = []

# Baseline models
for _, row in baseline_overall.iterrows():

    overall_rows.append({

        "Model":
            standardize_model_name(
                row["Model"]
            ),

        "Validation":
            "Leave-One-Pair-Out",

        "Number_of_Pairs":
            row.get(
                "Number_of_Pairs",
                6
            ),

        "MAE":
            pd.to_numeric(
                row["MAE"],
                errors="coerce"
            ),

        "RMSE":
            pd.to_numeric(
                row["RMSE"],
                errors="coerce"
            ),

        "R2":
            pd.to_numeric(
                row["R2"],
                errors="coerce"
            ),

        "MAPE_percent":
            pd.to_numeric(
                row.get(
                    "MAPE_percent",
                    np.nan
                ),
                errors="coerce"
            )
    })


# LSTM
lstm_row = lstm_overall.iloc[0]

overall_rows.append({

    "Model":
        "LSTM",

    "Validation":
        "Leave-One-Pair-Out",

    "Number_of_Pairs":
        lstm_row.get(
            "Number_of_Pairs",
            6
        ),

    "MAE":
        pd.to_numeric(
            lstm_row["MAE"],
            errors="coerce"
        ),

    "RMSE":
        pd.to_numeric(
            lstm_row["RMSE"],
            errors="coerce"
        ),

    "R2":
        pd.to_numeric(
            lstm_row["R2"],
            errors="coerce"
        ),

    "MAPE_percent":
        pd.to_numeric(
            lstm_row.get(
                "MAPE_percent",
                np.nan
            ),
            errors="coerce"
        )
})


# GRU
gru_row = gru_overall.iloc[0]

overall_rows.append({

    "Model":
        "GRU",

    "Validation":
        "Leave-One-Pair-Out",

    "Number_of_Pairs":
        gru_row.get(
            "Number_of_Pairs",
            6
        ),

    "MAE":
        pd.to_numeric(
            gru_row["MAE"],
            errors="coerce"
        ),

    "RMSE":
        pd.to_numeric(
            gru_row["RMSE"],
            errors="coerce"
        ),

    "R2":
        pd.to_numeric(
            gru_row["R2"],
            errors="coerce"
        ),

    "MAPE_percent":
        pd.to_numeric(
            gru_row.get(
                "MAPE_percent",
                np.nan
            ),
            errors="coerce"
        )
})


overall_df = pd.DataFrame(
    overall_rows
)


# Metric-specific ranks only.
overall_df["MAE_Rank"] = (
    overall_df["MAE"]
    .rank(
        method="min",
        ascending=True
    )
)

overall_df["RMSE_Rank"] = (
    overall_df["RMSE"]
    .rank(
        method="min",
        ascending=True
    )
)

overall_df["R2_Rank"] = (
    overall_df["R2"]
    .rank(
        method="min",
        ascending=False
    )
)


print(
    overall_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2",
            "MAPE_percent"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. FOLD COLUMN RESOLUTION
# ============================================================

def get_fold_pair_columns(df):

    fold_col = find_column(
        df,
        [
            "Fold",
            "Fold_Number",
            "Fold_ID"
        ],
        required=False
    )

    pair_col = find_column(
        df,
        [
            "Test_Pair",
            "Test_Pair_ID",
            "Pair_ID",
            "Pair",
            "PairID"
        ],
        required=False
    )

    return fold_col, pair_col


# ============================================================
# 11. FOLD-WISE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FOLD-WISE COMPARISON")
print("=" * 70)

fold_rows = []


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

base_fold_col, base_pair_col = get_fold_pair_columns(
    baseline_folds
)

print(
    f"Baseline fold columns -> "
    f"Fold: {base_fold_col}, Pair: {base_pair_col}"
)

for _, row in baseline_folds.iterrows():

    pair_value = (
        row[base_pair_col]
        if base_pair_col is not None
        else row[base_fold_col]
        if base_fold_col is not None
        else "Unknown"
    )

    fold_value = (
        row[base_fold_col]
        if base_fold_col is not None
        else np.nan
    )

    fold_rows.append({

        "Model":
            standardize_model_name(
                row["Model"]
            ),

        "Fold":
            fold_value,

        "Test_Pair":
            pair_value,

        "MAE":
            pd.to_numeric(
                row["MAE"],
                errors="coerce"
            ),

        "RMSE":
            pd.to_numeric(
                row["RMSE"],
                errors="coerce"
            ),

        "R2":
            pd.to_numeric(
                row["R2"],
                errors="coerce"
            ),

        "MAPE_percent":
            pd.to_numeric(
                row.get(
                    "MAPE_percent",
                    np.nan
                ),
                errors="coerce"
            )
    })


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

lstm_fold_col, lstm_pair_col = get_fold_pair_columns(
    lstm_folds
)

print(
    f"LSTM fold columns -> "
    f"Fold: {lstm_fold_col}, Pair: {lstm_pair_col}"
)

for _, row in lstm_folds.iterrows():

    pair_value = (
        row[lstm_pair_col]
        if lstm_pair_col is not None
        else row[lstm_fold_col]
        if lstm_fold_col is not None
        else "Unknown"
    )

    fold_value = (
        row[lstm_fold_col]
        if lstm_fold_col is not None
        else np.nan
    )

    fold_rows.append({

        "Model":
            "LSTM",

        "Fold":
            fold_value,

        "Test_Pair":
            pair_value,

        "MAE":
            pd.to_numeric(
                row["MAE"],
                errors="coerce"
            ),

        "RMSE":
            pd.to_numeric(
                row["RMSE"],
                errors="coerce"
            ),

        "R2":
            pd.to_numeric(
                row["R2"],
                errors="coerce"
            ),

        "MAPE_percent":
            pd.to_numeric(
                row.get(
                    "MAPE_percent",
                    np.nan
                ),
                errors="coerce"
            )
    })


# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------

gru_fold_col, gru_pair_col = get_fold_pair_columns(
    gru_folds
)

print(
    f"GRU fold columns -> "
    f"Fold: {gru_fold_col}, Pair: {gru_pair_col}"
)

for _, row in gru_folds.iterrows():

    pair_value = (
        row[gru_pair_col]
        if gru_pair_col is not None
        else row[gru_fold_col]
        if gru_fold_col is not None
        else "Unknown"
    )

    fold_value = (
        row[gru_fold_col]
        if gru_fold_col is not None
        else np.nan
    )

    fold_rows.append({

        "Model":
            "GRU",

        "Fold":
            fold_value,

        "Test_Pair":
            pair_value,

        "MAE":
            pd.to_numeric(
                row["MAE"],
                errors="coerce"
            ),

        "RMSE":
            pd.to_numeric(
                row["RMSE"],
                errors="coerce"
            ),

        "R2":
            pd.to_numeric(
                row["R2"],
                errors="coerce"
            ),

        "MAPE_percent":
            pd.to_numeric(
                row.get(
                    "MAPE_percent",
                    np.nan
                ),
                errors="coerce"
            )
    })


fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# 12. MODEL STABILITY
# ============================================================

stability_rows = []

for model in fold_df["Model"].unique():

    temp = fold_df[
        fold_df["Model"] == model
    ]

    stability_rows.append({

        "Model":
            model,

        "Folds":
            len(temp),

        "Mean_MAE":
            temp["MAE"].mean(),

        "Std_MAE":
            temp["MAE"].std(
                ddof=1
            ),

        "Mean_RMSE":
            temp["RMSE"].mean(),

        "Std_RMSE":
            temp["RMSE"].std(
                ddof=1
            ),

        "Mean_R2":
            temp["R2"].mean(),

        "Std_R2":
            temp["R2"].std(
                ddof=1
            ),

        "Minimum_R2":
            temp["R2"].min(),

        "Maximum_R2":
            temp["R2"].max()
    })


stability_df = pd.DataFrame(
    stability_rows
)

stability_df["R2_Range"] = (
    stability_df["Maximum_R2"]
    - stability_df["Minimum_R2"]
)


# ============================================================
# 13. TARGET-WISE METRICS FROM SAVED PREDICTIONS
# ============================================================
#
# This is deliberately done for LSTM and GRU from their actual
# saved out-of-sample predictions. It avoids the inconsistency
# where one workbook stores 78 target rows (6 folds x 13)
# while another stores 13 pooled target rows.
# ============================================================

def target_metrics_from_predictions(
    prediction_df,
    model_name
):

    rows = []

    for target_name in EV_TARGET_COLUMNS:

        temp = prediction_df[
            prediction_df["Target"].astype(str).str.strip()
            == target_name
        ].copy()

        if len(temp) == 0:
            continue

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

        try:

            r2 = r2_score(
                actual,
                predicted
            )

        except Exception:

            r2 = np.nan

        rows.append({

            "Target":
                target_name,

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
                r2,

            "MAPE_percent":
                safe_mape(
                    actual,
                    predicted
                ),

            "Model":
                model_name
        })

    return pd.DataFrame(rows)


lstm_target_from_predictions = (
    target_metrics_from_predictions(
        lstm_pred,
        "LSTM"
    )
)

gru_target_from_predictions = (
    target_metrics_from_predictions(
        gru_pred,
        "GRU"
    )
)


# ------------------------------------------------------------
# Baseline target-wise
# ------------------------------------------------------------

baseline_target_df = prepare_target_table(
    baseline_target
)


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

target_df = pd.concat(
    [
        baseline_target_df,
        lstm_target_from_predictions,
        gru_target_from_predictions
    ],
    ignore_index=True
)

target_df["Phase"] = (
    target_df["Target"]
    .apply(identify_phase)
)


# ============================================================
# 14. TARGET SUMMARY
# ============================================================

target_summary_rows = []

for model in target_df["Model"].unique():

    temp = target_df[
        target_df["Model"] == model
    ]

    target_summary_rows.append({

        "Model":
            model,

        "Targets_Evaluated":
            len(temp),

        "Mean_Target_MAE":
            temp["MAE"].mean(),

        "Mean_Target_RMSE":
            temp["RMSE"].mean(),

        "Mean_Target_R2":
            temp["R2"].mean(),

        "Median_Target_R2":
            temp["R2"].median(),

        "Targets_R2_Positive":
            int(
                (
                    temp["R2"] > 0
                ).sum()
            ),

        "Targets_R2_Negative":
            int(
                (
                    temp["R2"] < 0
                ).sum()
            )
    })


target_summary_df = pd.DataFrame(
    target_summary_rows
)


# ============================================================
# 15. PHASE-WISE SUMMARY
# ============================================================

phase_order = [
    "OVERALL",
    "STOP",
    "IDLE",
    "INCHING",
    "ACCELERATION",
    "DECELERATION",
    "CRUISE"
]

phase_rows = []

for model in target_df["Model"].unique():

    model_data = target_df[
        target_df["Model"] == model
    ]

    for phase in phase_order:

        temp = model_data[
            model_data["Phase"] == phase
        ]

        if len(temp) == 0:
            continue

        phase_rows.append({

            "Model":
                model,

            "Phase":
                phase,

            "Number_of_Targets":
                len(temp),

            "Mean_MAE":
                temp["MAE"].mean(),

            "Mean_RMSE":
                temp["RMSE"].mean(),

            "Mean_R2":
                temp["R2"].mean(),

            "Minimum_R2":
                temp["R2"].min(),

            "Maximum_R2":
                temp["R2"].max()
        })


phase_summary_df = pd.DataFrame(
    phase_rows
)


# ============================================================
# 16. TAT-RELEVANT TARGETS
# ============================================================

tat_targets = [
    "EV_STOP_Time_min",
    "EV_IDLE_Time_min",
    "EV_INCHING_Time_min",
    "EV_ACC_Time_min",
    "EV_DEC_Time_min",
    "EV_CRUISE_Time_min"
]

tat_target_df = target_df[
    target_df["Target"].isin(
        tat_targets
    )
].copy()


tat_summary_rows = []

for model in tat_target_df["Model"].unique():

    temp = tat_target_df[
        tat_target_df["Model"] == model
    ]

    tat_summary_rows.append({

        "Model":
            model,

        "TAT_Phase_Targets":
            len(temp),

        "Mean_MAE":
            temp["MAE"].mean(),

        "Mean_RMSE":
            temp["RMSE"].mean(),

        "Mean_R2":
            temp["R2"].mean(),

        "Minimum_R2":
            temp["R2"].min(),

        "Maximum_R2":
            temp["R2"].max()
    })


tat_summary_df = pd.DataFrame(
    tat_summary_rows
)


# ============================================================
# 17. PREDICTION COVERAGE AUDIT
# ============================================================

prediction_audit_rows = []

for model_name, prediction_df in [
    ("LSTM", lstm_pred),
    ("GRU", gru_pred)
]:

    pair_col = find_column(
        prediction_df,
        [
            "Pair_ID",
            "Test_Pair",
            "Test_Pair_ID",
            "Pair",
            "PairID"
        ],
        required=False
    )

    fold_col = find_column(
        prediction_df,
        [
            "Fold",
            "Fold_Number",
            "Fold_ID"
        ],
        required=False
    )

    target_count = (
        prediction_df["Target"]
        .nunique()
    )

    pair_count = (
        prediction_df[pair_col].nunique()
        if pair_col is not None
        else np.nan
    )

    fold_count = (
        prediction_df[fold_col].nunique()
        if fold_col is not None
        else np.nan
    )

    prediction_audit_rows.append({

        "Model":
            model_name,

        "Prediction_Rows":
            len(prediction_df),

        "Unique_Targets":
            target_count,

        "Unique_Pairs":
            pair_count,

        "Unique_Folds":
            fold_count,

        "Expected_Targets":
            len(EV_TARGET_COLUMNS),

        "Expected_Pairs":
            6,

        "Expected_Rows":
            6 * 100 * len(EV_TARGET_COLUMNS)
    })


prediction_audit_df = pd.DataFrame(
    prediction_audit_rows
)


# ============================================================
# 18. SAVE EXCEL
# ============================================================

output_excel = os.path.join(
    COMPARISON_DIR,
    "TS_Model_Comparison.xlsx"
)

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl"
) as writer:

    overall_df.to_excel(
        writer,
        sheet_name="Overall_Comparison",
        index=False
    )

    fold_df.to_excel(
        writer,
        sheet_name="Fold_Wise",
        index=False
    )

    stability_df.to_excel(
        writer,
        sheet_name="Model_Stability",
        index=False
    )

    target_df.to_excel(
        writer,
        sheet_name="Target_Wise",
        index=False
    )

    target_summary_df.to_excel(
        writer,
        sheet_name="Target_Summary",
        index=False
    )

    phase_summary_df.to_excel(
        writer,
        sheet_name="Phase_Wise",
        index=False
    )

    tat_target_df.to_excel(
        writer,
        sheet_name="TAT_Relevant_Targets",
        index=False
    )

    tat_summary_df.to_excel(
        writer,
        sheet_name="TAT_Summary",
        index=False
    )

    prediction_audit_df.to_excel(
        writer,
        sheet_name="Prediction_Audit",
        index=False
    )


# ============================================================
# 19. SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    COMPARISON_DIR,
    "TS_Model_Comparison_Report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TIME-SERIES MODEL COMPARISON REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "MODELS COMPARED\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for model in overall_df["Model"]:
        f.write(
            f"- {model}\n"
        )

    f.write(
        "\nVALIDATION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Validation method : Leave-One-Pair-Out\n"
    )

    f.write(
        "Independent pairs : 6\n"
    )

    f.write(
        "Random row split  : No\n"
    )

    f.write(
        "Cross-pair mixing : No\n"
    )

    f.write(
        "Primary metrics   : MAE, RMSE, R2\n"
    )

    f.write(
        "MAPE              : Reference only\n"
    )

    f.write(
        "\nOVERALL RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        overall_df[
            [
                "Model",
                "MAE",
                "RMSE",
                "R2",
                "MAPE_percent"
            ]
        ].to_string(
            index=False
        )
    )

    f.write(
        "\n\nMODEL STABILITY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        stability_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTARGET SUMMARY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        target_summary_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nTAT-RELEVANT TARGET SUMMARY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    if len(tat_summary_df) > 0:

        f.write(
            tat_summary_df.to_string(
                index=False
            )
        )

    else:

        f.write(
            "No TAT-relevant targets found.\n"
        )

    f.write(
        "\n\nPREDICTION AUDIT\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        prediction_audit_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nINTERPRETATION NOTES\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "1. MAE and RMSE are the primary prediction-error metrics.\n"
    )

    f.write(
        "2. R2 should be interpreted together with MAE and RMSE.\n"
    )

    f.write(
        "3. MAPE is unstable for zero / near-zero phase quantities.\n"
    )

    f.write(
        "4. Fold-to-fold variation is important because only six "
        "independent paired trips are available.\n"
    )

    f.write(
        "5. No combined composite model score is created.\n"
    )

    f.write(
        "6. LSTM and GRU target-wise metrics are calculated from "
        "their saved out-of-sample predictions for consistency.\n"
    )

    f.write(
        "7. This script only compares completed experiments; "
        "no model is retrained.\n"
    )


# ============================================================
# 20. PLOTS
# ============================================================

try:

    import matplotlib.pyplot as plt

    # Overall MAE
    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        overall_df["Model"],
        overall_df["MAE"]
    )

    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title(
        "Time-Series Model Comparison - MAE"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "TS_Model_MAE.png"
        ),
        dpi=300
    )

    plt.close()


    # Overall RMSE
    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        overall_df["Model"],
        overall_df["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title(
        "Time-Series Model Comparison - RMSE"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "TS_Model_RMSE.png"
        ),
        dpi=300
    )

    plt.close()


    # Overall R2
    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        overall_df["Model"],
        overall_df["R2"]
    )

    plt.xlabel("Model")
    plt.ylabel("R2")
    plt.title(
        "Time-Series Model Comparison - R2"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "TS_Model_R2.png"
        ),
        dpi=300
    )

    plt.close()


    # Fold-wise MAE
    plt.figure(
        figsize=(10, 5)
    )

    for model in fold_df["Model"].unique():

        temp = fold_df[
            fold_df["Model"] == model
        ].sort_values(
            "Fold"
        )

        plt.plot(
            temp["Fold"],
            temp["MAE"],
            marker="o",
            label=model
        )

    plt.xlabel("LOPO Fold")
    plt.ylabel("MAE")
    plt.title(
        "Fold-wise MAE Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "TS_Fold_Wise_MAE.png"
        ),
        dpi=300
    )

    plt.close()


    # Fold-wise R2
    plt.figure(
        figsize=(10, 5)
    )

    for model in fold_df["Model"].unique():

        temp = fold_df[
            fold_df["Model"] == model
        ].sort_values(
            "Fold"
        )

        plt.plot(
            temp["Fold"],
            temp["R2"],
            marker="o",
            label=model
        )

    plt.xlabel("LOPO Fold")
    plt.ylabel("R2")
    plt.title(
        "Fold-wise R2 Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOT_DIR,
            "TS_Fold_Wise_R2.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nPlots created successfully."
    )

except Exception as exc:

    print(
        f"\n[WARNING] Plot generation skipped: {exc}"
    )


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("TS MODEL COMPARISON COMPLETED")
print("=" * 70)

print(
    "\nOVERALL COMPARISON"
)

print(
    overall_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nMODEL STABILITY"
)

print(
    stability_df[
        [
            "Model",
            "Mean_MAE",
            "Std_MAE",
            "Mean_RMSE",
            "Std_RMSE",
            "Mean_R2",
            "Std_R2",
            "Minimum_R2",
            "Maximum_R2"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nPREDICTION AUDIT"
)

print(
    prediction_audit_df.to_string(
        index=False
    )
)

print(
    "\nResults saved to:"
)

print(
    output_excel
)

print(
    "\nReport saved to:"
)

print(
    report_path
)

print(
    "\nPlots saved to:"
)

print(
    PLOT_DIR
)

print(
    "\nNo model was trained."
)

print(
    "No random row split was used."
)

print(
    "No cross-pair sequence mixing was introduced."
)

print(
    "\nNext step: interpret the consolidated TS results "
    "before EV TAT reconstruction."
)
