# ============================================================
# TS GRU MODEL
# EV Parameter Estimation from ICE Route-Progress Sequences
#
# Input  : 17 ICE features
# Output : 13 EV targets
# Sequence length : 100 route bins
# Validation : Leave-One-Pair-Out (LOPO)
# Scaling : Fold-specific StandardScaler
# ============================================================

import os
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATH SETUP
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Results",
    "TimeSeries",
    "Dataset",
    "TS_Sequence_Dataset.npz"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "Results",
    "TimeSeries",
    "GRU"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 2. IMPORT AUTHORITATIVE FEATURE NAMES
# ============================================================

TIMESERIES_DIR = os.path.dirname(os.path.abspath(__file__))

if TIMESERIES_DIR not in sys.path:
    sys.path.insert(0, TIMESERIES_DIR)

from ts_data_loader import ICE_INPUT_COLUMNS, EV_TARGET_COLUMNS


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. MODEL SETTINGS
# ============================================================

HIDDEN_SIZE = 16
NUM_LAYERS = 1

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

EPOCHS = 500

GRAD_CLIP = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("TS GRU MODEL")
print("=" * 70)

print(f"Device        : {DEVICE}")
print(f"Hidden size   : {HIDDEN_SIZE}")
print(f"GRU layers    : {NUM_LAYERS}")
print(f"Learning rate : {LEARNING_RATE}")
print(f"Weight decay  : {WEIGHT_DECAY}")
print(f"Epochs        : {EPOCHS}")


# ============================================================
# 5. LOAD SEQUENCE DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Sequence dataset not found:\n{DATA_PATH}"
    )

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"].astype(np.float32)
Y = data["Y"].astype(np.float32)
pair_ids = data["Pair_ID"].astype(str)

print("\nDATA")
print("-" * 70)
print(f"X shape : {X.shape}")
print(f"Y shape : {Y.shape}")

print(f"Number of pairs : {len(pair_ids)}")


# ============================================================
# 6. VALIDATION
# ============================================================

if X.ndim != 3:
    raise ValueError(f"Expected X to be 3D, got {X.ndim}D")

if Y.ndim != 3:
    raise ValueError(f"Expected Y to be 3D, got {Y.ndim}D")

if X.shape[0] != Y.shape[0]:
    raise ValueError("X and Y pair counts do not match")

if X.shape[1] != Y.shape[1]:
    raise ValueError("X and Y sequence lengths do not match")

if X.shape[2] != len(ICE_INPUT_COLUMNS):
    raise ValueError(
        f"Expected {len(ICE_INPUT_COLUMNS)} ICE features, "
        f"found {X.shape[2]}"
    )

if Y.shape[2] != len(EV_TARGET_COLUMNS):
    raise ValueError(
        f"Expected {len(EV_TARGET_COLUMNS)} EV targets, "
        f"found {Y.shape[2]}"
    )

if np.isnan(X).any():
    raise ValueError("NaN detected in X")

if np.isnan(Y).any():
    raise ValueError("NaN detected in Y")

if np.isinf(X).any():
    raise ValueError("Infinite value detected in X")

if np.isinf(Y).any():
    raise ValueError("Infinite value detected in Y")


# ============================================================
# 7. GRU MODEL
# ============================================================

class GRUModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        gru_output, _ = self.gru(x)

        output = self.output_layer(gru_output)

        return output


# ============================================================
# 8. METRIC FUNCTION
# ============================================================

def safe_mape(y_true, y_pred, epsilon=1e-6):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.maximum(
        np.abs(y_true),
        epsilon
    )

    return np.mean(
        np.abs((y_true - y_pred) / denominator)
    ) * 100


# ============================================================
# 9. STORAGE
# ============================================================

fold_results = []

all_y_true = []
all_y_pred = []

fold_predictions = []


# ============================================================
# 10. LEAVE-ONE-PAIR-OUT CROSS VALIDATION
# ============================================================

unique_pairs = np.unique(pair_ids)

print("\n" + "=" * 70)
print("LEAVE-ONE-PAIR-OUT CROSS VALIDATION")
print("=" * 70)

print(f"Unique pairs : {len(unique_pairs)}")


for fold_number, test_pair in enumerate(unique_pairs, start=1):

    print("\n" + "-" * 70)
    print(
        f"FOLD {fold_number}/{len(unique_pairs)} "
        f"| Test pair: {test_pair}"
    )
    print("-" * 70)

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    train_mask = pair_ids != test_pair
    test_mask = pair_ids == test_pair

    X_train = X[train_mask]
    Y_train = Y[train_mask]

    X_test = X[test_mask]
    Y_test = Y[test_mask]

    print(f"Training pairs : {np.sum(train_mask)}")
    print(f"Testing pairs  : {np.sum(test_mask)}")


    # --------------------------------------------------------
    # SCALING
    #
    # IMPORTANT:
    # Scalers are fitted ONLY on training pairs.
    # --------------------------------------------------------

    n_train, seq_len, n_features = X_train.shape
    n_test = X_test.shape[0]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()


    # Flatten sequence dimensions temporarily
    X_train_flat = X_train.reshape(
        -1,
        n_features
    )

    Y_train_flat = Y_train.reshape(
        -1,
        Y_train.shape[2]
    )

    X_test_flat = X_test.reshape(
        -1,
        n_features
    )

    Y_test_flat = Y_test.reshape(
        -1,
        Y_test.shape[2]
    )


    # Fit only using training data
    X_train_scaled = x_scaler.fit_transform(
        X_train_flat
    )

    Y_train_scaled = y_scaler.fit_transform(
        Y_train_flat
    )

    # Transform test using training scalers
    X_test_scaled = x_scaler.transform(
        X_test_flat
    )


    # Reshape back into sequences
    X_train_scaled = X_train_scaled.reshape(
        n_train,
        seq_len,
        n_features
    )

    Y_train_scaled = Y_train_scaled.reshape(
        n_train,
        seq_len,
        Y_train.shape[2]
    )

    X_test_scaled = X_test_scaled.reshape(
        n_test,
        seq_len,
        n_features
    )


    # --------------------------------------------------------
    # PYTORCH TENSORS
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    Y_train_tensor = torch.tensor(
        Y_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = GRUModel(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        output_size=Y_train.shape[2],
        num_layers=NUM_LAYERS
    ).to(DEVICE)


    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    training_losses = []

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        predictions = model(
            X_train_tensor
        )

        loss = criterion(
            predictions,
            Y_train_tensor
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP
        )

        optimizer.step()

        training_losses.append(
            loss.item()
        )


    print(
        f"Final training loss : "
        f"{training_losses[-1]:.6f}"
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        Y_pred_scaled = model(
            X_test_tensor
        ).cpu().numpy()


    # --------------------------------------------------------
    # INVERSE TRANSFORM
    # --------------------------------------------------------

    Y_pred_flat = Y_pred_scaled.reshape(
        -1,
        Y_test.shape[2]
    )

    Y_pred = y_scaler.inverse_transform(
        Y_pred_flat
    )

    Y_pred = Y_pred.reshape(
        Y_test.shape
    )


    # --------------------------------------------------------
    # FLATTEN FOR POOLED METRICS
    # --------------------------------------------------------

    y_true_flat = Y_test.reshape(
        -1,
        Y_test.shape[2]
    )

    y_pred_flat = Y_pred.reshape(
        -1,
        Y_pred.shape[2]
    )


    # --------------------------------------------------------
    # OVERALL FOLD METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true_flat,
        y_pred_flat
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true_flat,
            y_pred_flat
        )
    )

    r2 = r2_score(
        y_true_flat,
        y_pred_flat,
        multioutput="variance_weighted"
    )

    mape = safe_mape(
        y_true_flat,
        y_pred_flat
    )


    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R2   : {r2:.6f}")
    print(f"MAPE : {mape:.6f}%")


    fold_results.append({
        "Fold": fold_number,
        "Test_Pair": test_pair,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_percent": mape,
        "Final_Training_Loss": training_losses[-1]
    })


    all_y_true.append(y_true_flat)
    all_y_pred.append(y_pred_flat)


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    for bin_index in range(seq_len):

        for target_index, target_name in enumerate(
            EV_TARGET_COLUMNS
        ):

            fold_predictions.append({
                "Fold": fold_number,
                "Pair_ID": test_pair,
                "Route_Bin": bin_index,
                "Target": target_name,
                "Actual": Y_test[0, bin_index, target_index],
                "Predicted": Y_pred[0, bin_index, target_index],
                "Absolute_Error": abs(
                    Y_test[0, bin_index, target_index]
                    - Y_pred[0, bin_index, target_index]
                )
            })


# ============================================================
# 11. POOLED OUT-OF-SAMPLE METRICS
# ============================================================

all_y_true = np.vstack(all_y_true)
all_y_pred = np.vstack(all_y_pred)

pooled_mae = mean_absolute_error(
    all_y_true,
    all_y_pred
)

pooled_rmse = np.sqrt(
    mean_squared_error(
        all_y_true,
        all_y_pred
    )
)

pooled_r2 = r2_score(
    all_y_true,
    all_y_pred,
    multioutput="variance_weighted"
)

pooled_mape = safe_mape(
    all_y_true,
    all_y_pred
)


# ============================================================
# 12. TARGET-WISE METRICS
# ============================================================

target_results = []

for i, target_name in enumerate(
    EV_TARGET_COLUMNS
):

    actual = all_y_true[:, i]
    predicted = all_y_pred[:, i]

    target_mae = mean_absolute_error(
        actual,
        predicted
    )

    target_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    try:
        target_r2 = r2_score(
            actual,
            predicted
        )
    except Exception:
        target_r2 = np.nan

    target_mape = safe_mape(
        actual,
        predicted
    )

    target_results.append({
        "Target": target_name,
        "MAE": target_mae,
        "RMSE": target_rmse,
        "R2": target_r2,
        "MAPE_percent": target_mape
    })


# ============================================================
# 13. SUMMARY TABLE
# ============================================================

overall_results = pd.DataFrame([
    {
        "Model": "GRU",
        "Validation": "Leave-One-Pair-Out",
        "Number_of_Pairs": len(unique_pairs),
        "Sequence_Length": X.shape[1],
        "Input_Features": X.shape[2],
        "Output_Targets": Y.shape[2],
        "MAE": pooled_mae,
        "RMSE": pooled_rmse,
        "R2": pooled_r2,
        "MAPE_percent": pooled_mape
    }
])


fold_results_df = pd.DataFrame(
    fold_results
)

target_results_df = pd.DataFrame(
    target_results
)

predictions_df = pd.DataFrame(
    fold_predictions
)


# ============================================================
# 14. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("GRU OVERALL OUT-OF-SAMPLE RESULTS")
print("=" * 70)

print(
    f"MAE  : {pooled_mae:.6f}"
)

print(
    f"RMSE : {pooled_rmse:.6f}"
)

print(
    f"R2   : {pooled_r2:.6f}"
)

print(
    f"MAPE : {pooled_mape:.6f}%"
)


print("\nTARGET-WISE RESULTS")
print("-" * 70)

print(
    target_results_df.to_string(
        index=False
    )
)


# ============================================================
# 15. SAVE EXCEL RESULTS
# ============================================================

excel_path = os.path.join(
    RESULT_DIR,
    "TS_GRU_Results.xlsx"
)

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    overall_results.to_excel(
        writer,
        sheet_name="Overall",
        index=False
    )

    fold_results_df.to_excel(
        writer,
        sheet_name="Fold_Results",
        index=False
    )

    target_results_df.to_excel(
        writer,
        sheet_name="Target_Wise",
        index=False
    )


# ============================================================
# 16. SAVE FOLD PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    RESULT_DIR,
    "TS_GRU_Fold_Predictions.xlsx"
)

predictions_df.to_excel(
    prediction_path,
    index=False
)


# ============================================================
# 17. SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULT_DIR,
    "TS_GRU_Report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TS GRU MODEL REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "MODEL CONFIGURATION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Input features       : {X.shape[2]}\n"
    )

    f.write(
        f"Output targets       : {Y.shape[2]}\n"
    )

    f.write(
        f"Sequence length      : {X.shape[1]}\n"
    )

    f.write(
        f"Number of pairs      : {len(unique_pairs)}\n"
    )

    f.write(
        f"Hidden size          : {HIDDEN_SIZE}\n"
    )

    f.write(
        f"GRU layers           : {NUM_LAYERS}\n"
    )

    f.write(
        f"Learning rate        : {LEARNING_RATE}\n"
    )

    f.write(
        f"Weight decay         : {WEIGHT_DECAY}\n"
    )

    f.write(
        f"Epochs               : {EPOCHS}\n"
    )

    f.write(
        f"Gradient clipping    : {GRAD_CLIP}\n"
    )

    f.write(
        f"Random seed          : {SEED}\n"
    )

    f.write(
        f"Validation           : Leave-One-Pair-Out\n"
    )

    f.write(
        "Scaling              : Fold-specific StandardScaler\n"
    )

    f.write(
        "Random row split     : No\n"
    )

    f.write(
        "Sliding windows      : No\n"
    )

    f.write(
        "Cross-pair sequence  : No\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "POOLED OUT-OF-SAMPLE RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"MAE                  : {pooled_mae:.6f}\n"
    )

    f.write(
        f"RMSE                 : {pooled_rmse:.6f}\n"
    )

    f.write(
        f"R2                   : {pooled_r2:.6f}\n"
    )

    f.write(
        f"MAPE (%)             : {pooled_mape:.6f}\n"
    )

    f.write(
        "\n"
    )

    f.write(
        "NOTE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "MAPE is sensitive to zero and near-zero phase quantities "
        "and should not be treated as the primary metric.\n"
    )

    f.write(
        "Only six independent paired trips are available; therefore "
        "the results are preliminary and should be interpreted with "
        "caution.\n"
    )


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("GRU RUN COMPLETED")
print("=" * 70)

print(
    f"Results saved to:\n{excel_path}"
)

print(
    f"Predictions saved to:\n{prediction_path}"
)

print(
    f"Report saved to:\n{report_path}"
)

print("\nNo hyperparameter tuning performed.")
print("No random row split performed.")
print("No cross-pair leakage introduced.")