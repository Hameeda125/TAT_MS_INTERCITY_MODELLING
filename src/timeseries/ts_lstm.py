"""
TS LSTM MODEL
=============

Purpose
-------
Predict the EV route-progress sequence from the ICE
route-progress sequence using an LSTM.

Dataset
-------
X : (6, 100, 17)
Y : (6, 100, 13)

Validation
----------
Leave-One-Pair-Out (LOPO)

Each complete paired trip is treated as one independent
experimental unit.

For every fold:

    5 complete trips -> training
    1 complete trip  -> testing

No random row-level split.
No sliding windows.
No cross-pair leakage.

Scaling
-------
X and Y are standardized separately.

The scalers are fitted ONLY on the five training trips
inside each LOPO fold.

The held-out test trip is transformed using the training
scalers and then inverse-transformed after prediction.

Model
-----
Small sequence-to-sequence LSTM:

    17 ICE features
          |
          v
       LSTM
    hidden = 16
          |
          v
      Linear
          |
          v
    13 EV targets

The model is intentionally small because there are only
6 independent paired trips.

Outputs
-------
Results/TimeSeries/LSTM/
    TS_LSTM_Results.xlsx
    TS_LSTM_Report.txt
    TS_LSTM_Fold_Predictions.xlsx
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
    / "LSTM"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_XLSX = (
    OUTPUT_DIR
    / "TS_LSTM_Results.xlsx"
)

REPORT_TXT = (
    OUTPUT_DIR
    / "TS_LSTM_Report.txt"
)

FOLD_PREDICTIONS_XLSX = (
    OUTPUT_DIR
    / "TS_LSTM_Fold_Predictions.xlsx"
)


# ============================================================
# EXPECTED DATASET STRUCTURE
# ============================================================

EXPECTED_PAIRS = 6
EXPECTED_BINS = 100
EXPECTED_ICE_FEATURES = 17
EXPECTED_EV_TARGETS = 13


# ============================================================
# LSTM HYPERPARAMETERS
# ============================================================

HIDDEN_SIZE = 16
NUM_LAYERS = 1

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

EPOCHS = 500

PRINT_EVERY = 50


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# METRICS
# ============================================================

def safe_mape(
    y_true,
    y_pred,
    epsilon=1e-6,
):
    """
    MAPE excluding zero / near-zero true values.
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
    Calculate pooled metrics across all targets
    and route bins.
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

    return mae, rmse, r2, mape


# ============================================================
# TARGET-WISE METRICS
# ============================================================

def calculate_target_metrics(
    y_true,
    y_pred,
    target_names,
    fold,
    test_pair,
):
    """
    Calculate metrics separately for every EV target.
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
                "Target": target_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_percent": mape,
            }
        )

    return records


# ============================================================
# LOAD DATA
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

    X = data["X"].astype(np.float32)

    Y = data["Y"].astype(np.float32)

    pair_ids = data["Pair_ID"].astype(str)

    route_bins = data["Route_Bin"]

    route_progress = data["Route_Progress"]

    # --------------------------------------------------------
    # Feature names come from the authoritative loader
    # definitions rather than relying on optional NPZ keys.
    # --------------------------------------------------------

    from ts_data_loader import (
        ICE_INPUT_COLUMNS,
        EV_TARGET_COLUMNS,
    )

    ice_features = np.array(
        ICE_INPUT_COLUMNS,
        dtype=str,
    )

    ev_targets = np.array(
        EV_TARGET_COLUMNS,
        dtype=str,
    )

    # ========================================================
    # STRUCTURAL CHECKS
    # ========================================================

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

    if not np.isfinite(X).all():

        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.isfinite(Y).all():

        raise ValueError(
            "Y contains NaN or infinite values."
        )

    if len(pair_ids) != EXPECTED_PAIRS:

        raise ValueError(
            "Unexpected number of paired trips."
        )

    print()
    print("Sequence file:")
    print(SEQUENCE_FILE)

    print()
    print(
        f"X shape       : {X.shape}"
    )

    print(
        f"Y shape       : {Y.shape}"
    )

    print(
        f"Pair IDs      : {list(pair_ids)}"
    )

    print()
    print(
        "Dataset structure : PASS"
    )

    print(
        "Finite values     : PASS"
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
# LSTM MODEL
# ============================================================

class TSLSTM(nn.Module):
    """
    Sequence-to-sequence LSTM.

    Input:
        (batch, sequence_length, 17)

    Output:
        (batch, sequence_length, 13)
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x):

        lstm_output, _ = self.lstm(x)

        output = self.output_layer(
            lstm_output
        )

        return output


# ============================================================
# SCALE DATA
# ============================================================

def fit_training_scalers(
    X_train,
    Y_train,
):
    """
    Fit scalers ONLY using training trips.

    Flattening here does NOT destroy sequence order.
    It is only used to estimate feature-wise scaling
    parameters.
    """

    n_train_pairs = X_train.shape[0]

    # --------------------------------------------------------
    # X scaler
    # --------------------------------------------------------

    X_train_flat = X_train.reshape(
        n_train_pairs * EXPECTED_BINS,
        EXPECTED_ICE_FEATURES,
    )

    X_scaler = StandardScaler()

    X_scaler.fit(
        X_train_flat
    )

    # --------------------------------------------------------
    # Y scaler
    # --------------------------------------------------------

    Y_train_flat = Y_train.reshape(
        n_train_pairs * EXPECTED_BINS,
        EXPECTED_EV_TARGETS,
    )

    Y_scaler = StandardScaler()

    Y_scaler.fit(
        Y_train_flat
    )

    return (
        X_scaler,
        Y_scaler,
    )


def transform_sequences(
    X,
    Y,
    X_scaler,
    Y_scaler,
):
    """
    Transform sequence data using already-fitted scalers.
    """

    n_pairs = X.shape[0]

    X_flat = X.reshape(
        n_pairs * EXPECTED_BINS,
        EXPECTED_ICE_FEATURES,
    )

    Y_flat = Y.reshape(
        n_pairs * EXPECTED_BINS,
        EXPECTED_EV_TARGETS,
    )

    X_scaled_flat = X_scaler.transform(
        X_flat
    )

    Y_scaled_flat = Y_scaler.transform(
        Y_flat
    )

    X_scaled = X_scaled_flat.reshape(
        n_pairs,
        EXPECTED_BINS,
        EXPECTED_ICE_FEATURES,
    )

    Y_scaled = Y_scaled_flat.reshape(
        n_pairs,
        EXPECTED_BINS,
        EXPECTED_EV_TARGETS,
    )

    return (
        X_scaled.astype(np.float32),
        Y_scaled.astype(np.float32),
    )


# ============================================================
# TRAIN LSTM
# ============================================================

def train_lstm(
    X_train_scaled,
    Y_train_scaled,
):
    """
    Train the LSTM using only the training pairs.

    The complete training sequence is presented as one
    batch. This preserves each complete trip as a sequence.
    """

    model = TSLSTM(
        input_size=EXPECTED_ICE_FEATURES,
        hidden_size=HIDDEN_SIZE,
        output_size=EXPECTED_EV_TARGETS,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    X_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    Y_tensor = torch.tensor(
        Y_train_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    model.train()

    training_losses = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        optimizer.zero_grad()

        predictions = model(
            X_tensor
        )

        loss = criterion(
            predictions,
            Y_tensor,
        )

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping for stable recurrent training
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        training_losses.append(
            loss.item()
        )

        if (
            epoch == 1
            or epoch % PRINT_EVERY == 0
            or epoch == EPOCHS
        ):

            print(
                f"    Epoch "
                f"{epoch:4d}/{EPOCHS} "
                f"Loss: "
                f"{loss.item():.6f}"
            )

    return (
        model,
        training_losses,
    )


# ============================================================
# PREDICT
# ============================================================

def predict_lstm(
    model,
    X_test_scaled,
    Y_scaler,
):
    """
    Predict the held-out trip and convert predictions
    back to the original physical units.
    """

    X_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    model.eval()

    with torch.no_grad():

        prediction_scaled = model(
            X_tensor
        ).cpu().numpy()

    # --------------------------------------------------------
    # Convert back to original units
    # --------------------------------------------------------

    n_test_pairs = prediction_scaled.shape[0]

    prediction_flat = prediction_scaled.reshape(
        n_test_pairs * EXPECTED_BINS,
        EXPECTED_EV_TARGETS,
    )

    prediction_original_flat = (
        Y_scaler.inverse_transform(
            prediction_flat
        )
    )

    prediction_original = (
        prediction_original_flat.reshape(
            n_test_pairs,
            EXPECTED_BINS,
            EXPECTED_EV_TARGETS,
        )
    )

    return prediction_original


# ============================================================
# MAIN LSTM EXPERIMENT
# ============================================================

def run_lstm():

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
    print("TS LSTM MODELING")
    print("=" * 70)

    print()
    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Hidden size   : {HIDDEN_SIZE}"
    )

    print(
        f"Layers        : {NUM_LAYERS}"
    )

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    print(
        f"Epochs        : {EPOCHS}"
    )

    print(
        f"Weight decay  : {WEIGHT_DECAY}"
    )

    print()
    print(
        "Validation : Leave-One-Pair-Out"
    )

    print(
        "Independent unit : Complete paired trip"
    )

    print(
        "Scaling : Fold-specific"
    )

    # ========================================================
    # STORAGE
    # ========================================================

    pooled_true = []

    pooled_predictions = []

    fold_records = []

    target_records = []

    prediction_records = []

    # ========================================================
    # LOPO
    # ========================================================

    n_pairs = X.shape[0]

    for test_index in range(
        n_pairs
    ):

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

        test_pair = pair_ids[
            test_index
        ]

        print()
        print("=" * 70)
        print(
            f"FOLD {test_index + 1}/{n_pairs}"
        )
        print("=" * 70)

        print(
            f"Test pair : {test_pair}"
        )

        print(
            "Training pairs : "
            f"{[pair_ids[i] for i in train_indices]}"
        )

        # ====================================================
        # FIT SCALERS ON TRAINING DATA ONLY
        # ====================================================

        (
            X_scaler,
            Y_scaler,
        ) = fit_training_scalers(
            X_train,
            Y_train,
        )

        print()
        print(
            "Training scalers fitted : PASS"
        )

        # ====================================================
        # TRANSFORM
        # ====================================================

        (
            X_train_scaled,
            Y_train_scaled,
        ) = transform_sequences(
            X_train,
            Y_train,
            X_scaler,
            Y_scaler,
        )

        # Test Y is not needed for scaling during prediction,
        # but we retain it in original units for evaluation.

        X_test_scaled, _ = (
            transform_sequences(
                X_test,
                Y_test,
                X_scaler,
                Y_scaler,
            )
        )

        print(
            "Training data scaled : PASS"
        )

        print(
            "Test data transformed "
            "using training scalers : PASS"
        )

        # ====================================================
        # TRAIN
        # ====================================================

        print()
        print(
            "Training LSTM..."
        )

        model, training_losses = (
            train_lstm(
                X_train_scaled,
                Y_train_scaled,
            )
        )

        print(
            "LSTM training : PASS"
        )

        # ====================================================
        # PREDICT
        # ====================================================

        prediction = predict_lstm(
            model,
            X_test_scaled,
            Y_scaler,
        )

        # Remove test-pair dimension

        prediction = prediction[0]

        actual = Y_test[0]

        # ====================================================
        # METRICS
        # ====================================================

        mae, rmse, r2, mape = (
            calculate_metrics(
                actual,
                prediction,
            )
        )

        print()
        print(
            "Fold results:"
        )

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

        # ====================================================
        # STORE
        # ====================================================

        pooled_true.append(
            actual
        )

        pooled_predictions.append(
            prediction
        )

        fold_records.append(
            {
                "Fold": test_index + 1,
                "Test_Pair_ID": test_pair,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_percent": mape,
                "Epochs": EPOCHS,
                "Hidden_Size": HIDDEN_SIZE,
                "Learning_Rate": LEARNING_RATE,
                "Weight_Decay": WEIGHT_DECAY,
            }
        )

        # ====================================================
        # TARGET-WISE
        # ====================================================

        target_records.extend(
            calculate_target_metrics(
                actual,
                prediction,
                ev_targets,
                test_index + 1,
                test_pair,
            )
        )

        # ====================================================
        # SAVE PREDICTIONS FOR THIS FOLD
        # ====================================================

        for bin_index in range(
            EXPECTED_BINS
        ):

            for target_index, target_name in enumerate(
                ev_targets
            ):

                prediction_records.append(
                    {
                        "Fold": test_index + 1,
                        "Test_Pair_ID": test_pair,
                        "Route_Bin": int(
                            route_bins[bin_index]
                        ),
                        "Route_Progress": float(
                            route_progress[bin_index]
                        ),
                        "Target": target_name,
                        "Actual": float(
                            actual[
                                bin_index,
                                target_index,
                            ]
                        ),
                        "Predicted": float(
                            prediction[
                                bin_index,
                                target_index,
                            ]
                        ),
                    }
                )

    # ========================================================
    # POOLED RESULTS
    # ========================================================

    pooled_true = np.vstack(
        pooled_true
    )

    pooled_predictions = np.vstack(
        pooled_predictions
    )

    (
        pooled_mae,
        pooled_rmse,
        pooled_r2,
        pooled_mape,
    ) = calculate_metrics(
        pooled_true,
        pooled_predictions,
    )

    print()
    print("=" * 70)
    print(
        "POOLED OUT-OF-SAMPLE LSTM RESULTS"
    )
    print("=" * 70)

    print()
    print(
        f"MAE  : {pooled_mae:.6f}"
    )

    print(
        f"RMSE : {pooled_rmse:.6f}"
    )

    print(
        f"R2   : {pooled_r2:.6f}"
    )

    if np.isnan(pooled_mape):

        print(
            "MAPE : NaN"
        )

    else:

        print(
            f"MAPE : {pooled_mape:.6f}%"
        )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    overall_df = pd.DataFrame(
        [
            {
                "Model": "LSTM",
                "MAE": pooled_mae,
                "RMSE": pooled_rmse,
                "R2": pooled_r2,
                "MAPE_percent": pooled_mape,
                "Test_Pairs": n_pairs,
                "Test_Bins": (
                    n_pairs
                    * EXPECTED_BINS
                ),
                "Sequence_Length": EXPECTED_BINS,
                "ICE_Features": EXPECTED_ICE_FEATURES,
                "EV_Targets": EXPECTED_EV_TARGETS,
                "Hidden_Size": HIDDEN_SIZE,
                "Num_Layers": NUM_LAYERS,
                "Epochs": EPOCHS,
                "Learning_Rate": LEARNING_RATE,
                "Weight_Decay": WEIGHT_DECAY,
            }
        ]
    )

    fold_df = pd.DataFrame(
        fold_records
    )

    target_df = pd.DataFrame(
        target_records
    )

    prediction_df = pd.DataFrame(
        prediction_records
    )

    # ========================================================
    # SAVE MAIN EXCEL
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
    # SAVE PREDICTIONS
    # ========================================================

    prediction_df.to_excel(
        FOLD_PREDICTIONS_XLSX,
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
            "TS LSTM MODEL REPORT\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "DATASET\n"
        )

        f.write(
            f"Independent paired trips : "
            f"{n_pairs}\n"
        )

        f.write(
            f"Sequence length : "
            f"{EXPECTED_BINS}\n"
        )

        f.write(
            f"ICE input features : "
            f"{EXPECTED_ICE_FEATURES}\n"
        )

        f.write(
            f"EV target features : "
            f"{EXPECTED_EV_TARGETS}\n\n"
        )

        f.write(
            "VALIDATION\n"
        )

        f.write(
            "Leave-One-Pair-Out (LOPO)\n"
        )

        f.write(
            "Each complete paired trip was "
            "held out as the test set.\n"
        )

        f.write(
            "No random row-level splitting.\n"
        )

        f.write(
            "No sliding windows.\n"
        )

        f.write(
            "No cross-pair sequences.\n\n"
        )

        f.write(
            "SCALING\n"
        )

        f.write(
            "X and Y were scaled independently "
            "within every training fold.\n"
        )

        f.write(
            "Test data used only the scalers "
            "fitted on the corresponding training fold.\n\n"
        )

        f.write(
            "MODEL\n"
        )

        f.write(
            f"Hidden size : {HIDDEN_SIZE}\n"
        )

        f.write(
            f"Layers : {NUM_LAYERS}\n"
        )

        f.write(
            f"Epochs : {EPOCHS}\n"
        )

        f.write(
            f"Learning rate : {LEARNING_RATE}\n"
        )

        f.write(
            f"Weight decay : {WEIGHT_DECAY}\n"
        )

        f.write(
            f"Device : {DEVICE}\n\n"
        )

        f.write(
            "POOLED OUT-OF-SAMPLE RESULTS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"MAE  : {pooled_mae:.6f}\n"
        )

        f.write(
            f"RMSE : {pooled_rmse:.6f}\n"
        )

        f.write(
            f"R2   : {pooled_r2:.6f}\n"
        )

        if np.isnan(pooled_mape):

            f.write(
                "MAPE : NaN\n"
            )

        else:

            f.write(
                f"MAPE : "
                f"{pooled_mape:.6f}%\n"
            )

        f.write(
            "\nIMPORTANT INTERPRETATION\n"
        )

        f.write(
            "The six paired trips are the "
            "independent experimental units.\n"
        )

        f.write(
            "The 100 route-progress bins within "
            "each trip are ordered observations.\n"
        )

        f.write(
            "They must not be interpreted as "
            "600 independent samples.\n"
        )

        f.write(
            "LSTM performance should therefore "
            "be interpreted as exploratory evidence "
            "given the very small number of "
            "independent paired trips.\n"
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TS LSTM MODELING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Independent pairs : {n_pairs}"
    )

    print(
        f"Sequence length   : {EXPECTED_BINS}"
    )

    print(
        f"ICE features      : {EXPECTED_ICE_FEATURES}"
    )

    print(
        f"EV targets        : {EXPECTED_EV_TARGETS}"
    )

    print()
    print(
        "Validation : Leave-One-Pair-Out"
    )

    print(
        "Scaling    : Fold-specific"
    )

    print(
        "Test unit  : Complete paired trip"
    )

    print()
    print(
        "Pooled OOS results:"
    )

    print(
        f"MAE  : {pooled_mae:.6f}"
    )

    print(
        f"RMSE : {pooled_rmse:.6f}"
    )

    print(
        f"R2   : {pooled_r2:.6f}"
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
        "Fold predictions saved:"
    )

    print(
        FOLD_PREDICTIONS_XLSX
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
        "No random row-level split performed."
    )

    print(
        "No cross-pair leakage."
    )

    print(
        "No sliding windows created."
    )

    return (
        overall_df,
        fold_df,
        target_df,
        prediction_df,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_lstm()