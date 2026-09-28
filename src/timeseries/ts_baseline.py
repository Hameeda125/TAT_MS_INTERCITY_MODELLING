"""
TS BASELINE MODELS
==================

Purpose
-------
Establish leakage-safe baseline performance for the
ICE -> EV route-progress time-series prediction problem.

Dataset
-------
X : (n_pairs, sequence_length, n_ice_features)
Y : (n_pairs, sequence_length, n_ev_targets)

Current dataset
---------------
X : (6, 100, 17)
Y : (6, 100, 13)

Validation
----------
Leave-One-Pair-Out (LOPO)

Each complete paired trip is kept entirely in either
training or testing.

No random row-level splitting.
No cross-pair leakage.
No sliding windows.

Models
------
1. Training Mean Baseline
2. Route-Bin Mean Baseline
3. Point-wise Multi-output Linear Regression

Important
---------
The 6 paired trips are the independent experimental units.
The 100 route bins within each trip are ordered observations,
not 100 independent trips.

Outputs
-------
Results/TimeSeries/Baseline/
    TS_Baseline_Results.xlsx
    TS_Baseline_Report.txt
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# IMPORTANT:
# ts_data_loader.py is in the same folder as this file.
from ts_data_loader import (
    ICE_INPUT_COLUMNS,
    EV_TARGET_COLUMNS,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_FILE = (
    PROJECT_ROOT
    / "Results"
    / "TimeSeries"
    / "Dataset"
    / "TS_Sequence_Dataset.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Results"
    / "TimeSeries"
    / "Baseline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_XLSX = (
    OUTPUT_DIR
    / "TS_Baseline_Results.xlsx"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "TS_Baseline_Report.txt"
)


# ============================================================
# EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_PAIRS = 6
EXPECTED_BINS = 100
EXPECTED_ICE_FEATURES = 17
EXPECTED_EV_TARGETS = 13


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def safe_mape(
    y_true,
    y_pred,
    epsilon=1e-6,
):
    """
    Calculate MAPE while excluding zero / near-zero
    true values.

    This is important because phase-level variables can
    legitimately contain zero values.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.abs(y_true) > epsilon

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[mask]
                    - y_pred[mask]
                )
                / y_true[mask]
            )
        )
        * 100.0
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate pooled out-of-sample metrics.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(
        true_flat,
        pred_flat,
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_flat,
            pred_flat,
        )
    )

    try:
        r2 = r2_score(
            true_flat,
            pred_flat,
        )
    except Exception:
        r2 = np.nan

    mape = safe_mape(
        true_flat,
        pred_flat,
    )

    return (
        mae,
        rmse,
        r2,
        mape,
    )


# ============================================================
# TARGET-WISE METRICS
# ============================================================

def calculate_target_metrics(
    y_true,
    y_pred,
    target_names,
    model_name,
    fold=None,
    test_pair=None,
):
    """
    Calculate metrics independently for every EV target.
    """

    records = []

    for target_index, target_name in enumerate(
        target_names
    ):

        true_values = y_true[:, target_index]
        pred_values = y_pred[:, target_index]

        mae = mean_absolute_error(
            true_values,
            pred_values,
        )

        rmse = np.sqrt(
            mean_squared_error(
                true_values,
                pred_values,
            )
        )

        try:
            r2 = r2_score(
                true_values,
                pred_values,
            )
        except Exception:
            r2 = np.nan

        mape = safe_mape(
            true_values,
            pred_values,
        )

        records.append(
            {
                "Fold": fold,
                "Test_Pair_ID": test_pair,
                "Model": model_name,
                "Target": target_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_percent": mape,
            }
        )

    return records


# ============================================================
# LOAD SEQUENCE DATASET
# ============================================================

def load_sequence_dataset():

    print("=" * 70)
    print("LOADING TS SEQUENCE DATASET")
    print("=" * 70)

    if not SEQUENCE_FILE.exists():

        raise FileNotFoundError(
            f"\nSequence dataset not found:\n"
            f"{SEQUENCE_FILE}\n"
        )

    data = np.load(
        SEQUENCE_FILE,
        allow_pickle=True,
    )

    # --------------------------------------------------------
    # Load only the arrays that are actually required
    # --------------------------------------------------------

    X = data["X"].astype(float)

    Y = data["Y"].astype(float)

    pair_ids = data["Pair_ID"].astype(str)

    route_bins = data["Route_Bin"]

    route_progress = data["Route_Progress"]

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Do NOT read:
    #
    # data["ICE_Feature_Names"]
    # data["EV_Target_Names"]
    #
    # Instead use the authoritative definitions from
    # ts_data_loader.py.
    # --------------------------------------------------------

    ice_features = np.array(
        ICE_INPUT_COLUMNS,
        dtype=str,
    )

    ev_targets = np.array(
        EV_TARGET_COLUMNS,
        dtype=str,
    )

    print()
    print("Sequence file:")
    print(SEQUENCE_FILE)

    print()
    print(f"X shape       : {X.shape}")
    print(f"Y shape       : {Y.shape}")
    print(f"Pair IDs      : {list(pair_ids)}")
    print(f"Route bins    : {route_bins.shape}")
    print(
        f"Route progress: {route_progress.shape}"
    )

    # ========================================================
    # STRUCTURAL VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING DATASET STRUCTURE")
    print("=" * 70)

    expected_x_shape = (
        EXPECTED_PAIRS,
        EXPECTED_BINS,
        EXPECTED_ICE_FEATURES,
    )

    expected_y_shape = (
        EXPECTED_PAIRS,
        EXPECTED_BINS,
        EXPECTED_EV_TARGETS,
    )

    if X.shape != expected_x_shape:

        raise ValueError(
            "\nUnexpected X shape.\n"
            f"Expected : {expected_x_shape}\n"
            f"Found    : {X.shape}\n"
        )

    if Y.shape != expected_y_shape:

        raise ValueError(
            "\nUnexpected Y shape.\n"
            f"Expected : {expected_y_shape}\n"
            f"Found    : {Y.shape}\n"
        )

    if len(ice_features) != EXPECTED_ICE_FEATURES:

        raise ValueError(
            "ICE feature count does not match X."
        )

    if len(ev_targets) != EXPECTED_EV_TARGETS:

        raise ValueError(
            "EV target count does not match Y."
        )

    print()
    print(
        f"Expected X shape : {expected_x_shape}"
    )
    print(
        f"Found X shape    : {X.shape}"
    )
    print(
        "X shape validation : PASS"
    )

    print()
    print(
        f"Expected Y shape : {expected_y_shape}"
    )
    print(
        f"Found Y shape    : {Y.shape}"
    )
    print(
        "Y shape validation : PASS"
    )

    # ========================================================
    # FINITE VALUE CHECK
    # ========================================================

    if not np.isfinite(X).all():

        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(Y).all():

        raise ValueError(
            "Y contains NaN or infinite values."
        )

    print()
    print(
        "X finite values : PASS"
    )

    print(
        "Y finite values : PASS"
    )

    # ========================================================
    # PAIR VALIDATION
    # ========================================================

    if len(pair_ids) != EXPECTED_PAIRS:

        raise ValueError(
            "Unexpected number of paired trips."
        )

    if len(np.unique(pair_ids)) != EXPECTED_PAIRS:

        raise ValueError(
            "Duplicate Pair_ID values detected."
        )

    print()
    print(
        f"Independent paired trips : {len(pair_ids)}"
    )

    print(
        "Pair structure : PASS"
    )

    return (
        X,
        Y,
        pair_ids,
        route_bins,
        route_progress,
        ice_features,
        ev_targets,
    )


# ============================================================
# TRAINING MEAN BASELINE
# ============================================================

def predict_training_mean(
    Y_train,
    n_test_bins,
):
    """
    Predict one overall EV target mean calculated
    exclusively from the training trips.
    """

    target_mean = np.mean(
        Y_train,
        axis=(0, 1),
    )

    predictions = np.tile(
        target_mean,
        (
            n_test_bins,
            1,
        ),
    )

    return predictions


# ============================================================
# ROUTE-BIN MEAN BASELINE
# ============================================================

def predict_route_bin_mean(
    Y_train,
):
    """
    For each route-progress bin, calculate the mean EV
    target across the training trips.

    Input:
        Y_train = (n_train_pairs, 100, 13)

    Output:
        prediction = (100, 13)
    """

    route_bin_mean = np.mean(
        Y_train,
        axis=0,
    )

    return route_bin_mean


# ============================================================
# LINEAR REGRESSION BASELINE
# ============================================================

def train_linear_baseline(
    X_train,
    Y_train,
    X_test,
):
    """
    Point-wise multi-output linear regression.

    Training:
        all route bins from the training trips

    Testing:
        all route bins from the held-out trip

    Scaling:
        fitted only on training data.
    """

    n_train_pairs = X_train.shape[0]

    # --------------------------------------------------------
    # Flatten only after separating train/test by Pair_ID.
    # --------------------------------------------------------

    X_train_flat = X_train.reshape(
        n_train_pairs * EXPECTED_BINS,
        EXPECTED_ICE_FEATURES,
    )

    Y_train_flat = Y_train.reshape(
        n_train_pairs * EXPECTED_BINS,
        EXPECTED_EV_TARGETS,
    )

    X_test_flat = X_test.reshape(
        EXPECTED_BINS,
        EXPECTED_ICE_FEATURES,
    )

    # --------------------------------------------------------
    # Scaling is part of the training pipeline.
    # It is fitted only using training data.
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "linear_regression",
                LinearRegression(),
            ),
        ]
    )

    model.fit(
        X_train_flat,
        Y_train_flat,
    )

    Y_pred_flat = model.predict(
        X_test_flat,
    )

    return Y_pred_flat


# ============================================================
# MAIN BASELINE EXPERIMENT
# ============================================================

def run_baselines():

    (
        X,
        Y,
        pair_ids,
        route_bins,
        route_progress,
        ice_features,
        ev_targets,
    ) = load_sequence_dataset()

    print()
    print("=" * 70)
    print("LEAVE-ONE-PAIR-OUT BASELINE VALIDATION")
    print("=" * 70)

    print()
    print("Validation strategy:")
    print(
        "  One complete paired trip = one test fold"
    )
    print(
        "  No random row-level splitting"
    )
    print(
        "  No cross-pair sequences"
    )
    print(
        "  No sliding windows"
    )
    print()

    n_pairs = X.shape[0]

    # --------------------------------------------------------
    # Containers for pooled out-of-sample predictions
    # --------------------------------------------------------

    pooled_predictions = {
        "Training Mean": [],
        "Route-Bin Mean": [],
        "Linear Regression": [],
    }

    pooled_true = []

    fold_records = []

    target_records = []

    # ========================================================
    # LEAVE-ONE-PAIR-OUT
    # ========================================================

    for test_index in range(n_pairs):

        train_indices = [
            i
            for i in range(n_pairs)
            if i != test_index
        ]

        test_indices = [
            test_index
        ]

        X_train = X[train_indices]

        Y_train = Y[train_indices]

        X_test = X[test_indices]

        Y_test = Y[test_indices]

        test_pair = pair_ids[test_index]

        # Remove the pair dimension from the single
        # held-out sequence.

        X_test_sequence = X_test[0]

        Y_test_sequence = Y_test[0]

        print("-" * 70)

        print(
            f"Fold {test_index + 1}/{n_pairs}"
        )

        print(
            f"Test pair : {test_pair}"
        )

        print(
            "Training pairs : "
            f"{[pair_ids[i] for i in train_indices]}"
        )

        print()

        # ====================================================
        # MODEL 1
        # ====================================================

        pred_training_mean = (
            predict_training_mean(
                Y_train,
                EXPECTED_BINS,
            )
        )

        print(
            "  Training Mean     : PASS"
        )

        # ====================================================
        # MODEL 2
        # ====================================================

        pred_route_bin_mean = (
            predict_route_bin_mean(
                Y_train,
            )
        )

        print(
            "  Route-Bin Mean    : PASS"
        )

        # ====================================================
        # MODEL 3
        # ====================================================

        pred_linear = (
            train_linear_baseline(
                X_train,
                Y_train,
                X_test,
            )
        )

        print(
            "  Linear Regression : PASS"
        )

        # ====================================================
        # STORE TRUE VALUES
        # ====================================================

        pooled_true.append(
            Y_test_sequence
        )

        # ====================================================
        # STORE PREDICTIONS
        # ====================================================

        pooled_predictions[
            "Training Mean"
        ].append(
            pred_training_mean
        )

        pooled_predictions[
            "Route-Bin Mean"
        ].append(
            pred_route_bin_mean
        )

        pooled_predictions[
            "Linear Regression"
        ].append(
            pred_linear
        )

        # ====================================================
        # FOLD METRICS
        # ====================================================

        fold_models = [
            (
                "Training Mean",
                pred_training_mean,
            ),
            (
                "Route-Bin Mean",
                pred_route_bin_mean,
            ),
            (
                "Linear Regression",
                pred_linear,
            ),
        ]

        for model_name, prediction in fold_models:

            mae, rmse, r2, mape = (
                calculate_metrics(
                    Y_test_sequence,
                    prediction,
                )
            )

            fold_records.append(
                {
                    "Fold": test_index + 1,
                    "Test_Pair_ID": test_pair,
                    "Model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "MAPE_percent": mape,
                }
            )

            # ------------------------------------------------
            # Target-wise metrics
            # ------------------------------------------------

            target_records.extend(
                calculate_target_metrics(
                    Y_test_sequence,
                    prediction,
                    ev_targets,
                    model_name,
                    fold=test_index + 1,
                    test_pair=test_pair,
                )
            )

    # ========================================================
    # POOLED OUT-OF-SAMPLE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("POOLED OUT-OF-SAMPLE BASELINE RESULTS")
    print("=" * 70)

    pooled_true = np.vstack(
        pooled_true
    )

    overall_records = []

    for model_name in pooled_predictions:

        pooled_pred = np.vstack(
            pooled_predictions[model_name]
        )

        mae, rmse, r2, mape = (
            calculate_metrics(
                pooled_true,
                pooled_pred,
            )
        )

        overall_records.append(
            {
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_percent": mape,
                "Test_Pairs": n_pairs,
                "Test_Bins": (
                    n_pairs
                    * EXPECTED_BINS
                ),
            }
        )

        print()
        print(model_name)

        print(
            f"  MAE  : {mae:.6f}"
        )

        print(
            f"  RMSE : {rmse:.6f}"
        )

        print(
            f"  R2   : {r2:.6f}"
        )

        if np.isnan(mape):

            print(
                "  MAPE : NaN"
            )

        else:

            print(
                f"  MAPE : {mape:.6f}%"
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    overall_df = pd.DataFrame(
        overall_records
    )

    fold_df = pd.DataFrame(
        fold_records
    )

    target_df = pd.DataFrame(
        target_records
    )

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    with pd.ExcelWriter(
        RESULT_XLSX,
        engine="openpyxl",
    ) as writer:

        overall_df.to_excel(
            writer,
            sheet_name="Overall_OOS",
            index=False,
        )

        fold_df.to_excel(
            writer,
            sheet_name="Fold_Results",
            index=False,
        )

        target_df.to_excel(
            writer,
            sheet_name="Target_Wise_OOS",
            index=False,
        )

        # ----------------------------------------------------
        # ICE feature list
        # ----------------------------------------------------

        pd.DataFrame(
            {
                "ICE_Feature_Index": range(
                    1,
                    len(ice_features) + 1,
                ),
                "ICE_Feature": ice_features,
            }
        ).to_excel(
            writer,
            sheet_name="ICE_Features",
            index=False,
        )

        # ----------------------------------------------------
        # EV target list
        # ----------------------------------------------------

        pd.DataFrame(
            {
                "EV_Target_Index": range(
                    1,
                    len(ev_targets) + 1,
                ),
                "EV_Target": ev_targets,
            }
        ).to_excel(
            writer,
            sheet_name="EV_Targets",
            index=False,
        )

    # ========================================================
    # SAVE TEXT REPORT
    # ========================================================

    with open(
        REPORT_TXT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "TS BASELINE MODEL REPORT\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        f.write(
            "DATASET STRUCTURE\n"
        )

        f.write(
            f"Number of paired trips : "
            f"{n_pairs}\n"
        )

        f.write(
            f"Sequence length        : "
            f"{EXPECTED_BINS}\n"
        )

        f.write(
            f"ICE input features     : "
            f"{EXPECTED_ICE_FEATURES}\n"
        )

        f.write(
            f"EV target features     : "
            f"{EXPECTED_EV_TARGETS}\n\n"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        f.write(
            "VALIDATION STRATEGY\n"
        )

        f.write(
            "Leave-One-Pair-Out (LOPO)\n"
        )

        f.write(
            "Each complete paired trip is "
            "held out as the test set.\n"
        )

        f.write(
            "No random row-level splitting "
            "was performed.\n"
        )

        f.write(
            "No sliding windows were created.\n"
        )

        f.write(
            "No information from the held-out "
            "pair was used for training.\n\n"
        )

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        f.write(
            "MODELS EVALUATED\n"
        )

        f.write(
            "1. Training Mean\n"
        )

        f.write(
            "2. Route-Bin Mean\n"
        )

        f.write(
            "3. Point-wise Multi-output "
            "Linear Regression\n\n"
        )

        # ----------------------------------------------------
        # Overall results
        # ----------------------------------------------------

        f.write(
            "OVERALL POOLED "
            "OUT-OF-SAMPLE RESULTS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for _, row in overall_df.iterrows():

            f.write(
                f"\n{row['Model']}\n"
            )

            f.write(
                f"  MAE  : "
                f"{row['MAE']:.6f}\n"
            )

            f.write(
                f"  RMSE : "
                f"{row['RMSE']:.6f}\n"
            )

            f.write(
                f"  R2   : "
                f"{row['R2']:.6f}\n"
            )

            if pd.isna(
                row["MAPE_percent"]
            ):

                f.write(
                    "  MAPE : NaN\n"
                )

            else:

                f.write(
                    f"  MAPE : "
                    f"{row['MAPE_percent']:.6f}%\n"
                )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        f.write(
            "\n\nIMPORTANT INTERPRETATION\n"
        )

        f.write(
            "The six paired trips are the "
            "independent experimental units.\n"
        )

        f.write(
            "The 100 route bins within each "
            "trip are ordered observations,\n"
        )

        f.write(
            "not 100 independent trips.\n"
        )

        f.write(
            "Performance is therefore evaluated "
            "using complete-trip LOPO validation.\n"
        )

        f.write(
            "The Linear Regression scaler is "
            "fitted independently within each "
            "training fold.\n"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("TS BASELINE MODELING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Number of independent pairs : "
        f"{n_pairs}"
    )

    print(
        f"Sequence length             : "
        f"{EXPECTED_BINS}"
    )

    print(
        f"ICE features                : "
        f"{EXPECTED_ICE_FEATURES}"
    )

    print(
        f"EV targets                  : "
        f"{EXPECTED_EV_TARGETS}"
    )

    print()
    print(
        "Validation : Leave-One-Pair-Out"
    )

    print(
        "Models     : 3"
    )

    print(
        "Test unit  : Complete paired trip"
    )

    print()
    print(
        "Results saved:"
    )

    print(
        RESULT_XLSX
    )

    print()
    print(
        "Report saved:"
    )

    print(
        REPORT_TXT
    )

    print()
    print(
        "No LSTM trained."
    )

    print(
        "No GRU trained."
    )

    print(
        "No random row-level split performed."
    )

    print(
        "No test-pair information used during training."
    )

    return (
        overall_df,
        fold_df,
        target_df,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_baselines()