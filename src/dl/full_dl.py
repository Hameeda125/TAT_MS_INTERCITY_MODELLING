from __future__ import annotations

from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.preprocessing.data_loader import load_excel


# ============================================================
# FULL DL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = Path(
    r"D:\TAT_MS_INTERCITY\Data\Results\Master_Summary\Training_Data\EV_ICE_Training_STRICT.xlsx"
)

RESULTS_DIR = PROJECT_ROOT / "Results" / "DL" / "Full_Model"
MODEL_DIR = PROJECT_ROOT / "models" / "DL" / "Full_Model"

RANDOM_SEED = 42

EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_1 = 32
HIDDEN_2 = 16


# ============================================================
# 15 ICE INPUT FEATURES
# ============================================================

ICE_FEATURES = [
    "ICE Total Distance (km)",
    "ICE Total TAT (min)",
    "ICE Average Moving Speed (km/h)",

    "ICE STOP Time (min)",
    "ICE IDLE Time (min)",
    "ICE INCHING Time (min)",
    "ICE ACC Time (min)",
    "ICE DEC Time (min)",
    "ICE CRUISE Time (min)",

    "ICE STOP Distance (km)",
    "ICE IDLE Distance (km)",
    "ICE INCHING Distance (km)",
    "ICE ACC Distance (km)",
    "ICE DEC Distance (km)",
    "ICE CRUISE Distance (km)",
]


# ============================================================
# 15 EV TARGET FEATURES
# ============================================================

EV_TARGETS = [
    "EV Total Distance (km)",
    "EV Total TAT (min)",
    "EV Average Moving Speed (km/h)",

    "EV STOP Time (min)",
    "EV IDLE Time (min)",
    "EV INCHING Time (min)",
    "EV ACC Time (min)",
    "EV DEC Time (min)",
    "EV CRUISE Time (min)",

    "EV STOP Distance (km)",
    "EV IDLE Distance (km)",
    "EV INCHING Distance (km)",
    "EV ACC Distance (km)",
    "EV DEC Distance (km)",
    "EV CRUISE Distance (km)",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DNN MODEL
# ============================================================

class FullDLRegressor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_1,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2,
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_2,
                output_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


# ============================================================
# SAFE MAPE
# ============================================================

def safe_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    mask = np.abs(y_true) > 1e-8

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


# ============================================================
# METRICS
# ============================================================

def pooled_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {

        "MAE":
            float(
                mean_absolute_error(
                    y_true.ravel(),
                    y_pred.ravel(),
                )
            ),

        "RMSE":
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true.ravel(),
                        y_pred.ravel(),
                    )
                )
            ),

        "R2":
            float(
                r2_score(
                    y_true.ravel(),
                    y_pred.ravel(),
                )
            ),

        "MAPE":
            safe_mape(
                y_true.ravel(),
                y_pred.ravel(),
            ),
    }


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "STRICT training workbook was not found:\n"
            f"{INPUT_FILE}"
        )

    dataframe = load_excel(
        INPUT_FILE,
        sheet_name=0,
        header=3,
    )

    if "Pair ID" not in dataframe.columns:

        raise ValueError(
            "Pair ID column is required "
            "for LOPO validation."
        )

    required_columns = (
        ICE_FEATURES
        + EV_TARGETS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required modelling columns are missing:\n"
            f"{missing_columns}"
        )

    modelling_data = dataframe[
        required_columns
    ]

    if modelling_data.isna().any().any():

        raise ValueError(
            "Missing values found in modelling columns."
        )

    non_numeric = [

        column

        for column in modelling_data.columns

        if not pd.api.types.is_numeric_dtype(
            modelling_data[column]
        )
    ]

    if non_numeric:

        raise ValueError(
            "Non-numeric modelling columns found:\n"
            f"{non_numeric}"
        )

    return dataframe


# ============================================================
# TRAIN ONE LOPO FOLD
# ============================================================

def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:

    # --------------------------------------------------------
    # Scaling is fitted ONLY on training data
    # --------------------------------------------------------

    x_scaler = StandardScaler()

    y_scaler = StandardScaler()

    X_train_scaled = (
        x_scaler.fit_transform(
            X_train
        )
    )

    y_train_scaled = (
        y_scaler.fit_transform(
            y_train
        )
    )

    X_test_scaled = (
        x_scaler.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FullDLRegressor(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.MSELoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        prediction = model(
            X_tensor
        )

        loss = loss_function(
            prediction,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        scaled_prediction = (
            model(
                X_test_tensor
            )
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Convert back to original units
    # --------------------------------------------------------

    prediction = (
        y_scaler.inverse_transform(
            scaled_prediction
        )
    )

    return prediction


# ============================================================
# LEAVE-ONE-PAIR-OUT VALIDATION
# ============================================================

def run_lopo(
    dataframe: pd.DataFrame,
):

    X = dataframe[
        ICE_FEATURES
    ].to_numpy(
        dtype=float
    )

    y = dataframe[
        EV_TARGETS
    ].to_numpy(
        dtype=float
    )

    pair_ids = (
        dataframe["Pair ID"]
        .astype(str)
        .to_numpy()
    )

    unique_pairs = list(
        dict.fromkeys(
            pair_ids
        )
    )

    all_true = []

    all_pred = []

    fold_rows = []

    for pair_id in unique_pairs:

        print(
            f"\nProcessing LOPO fold: {pair_id}"
        )

        test_mask = (
            pair_ids == pair_id
        )

        train_mask = (
            ~test_mask
        )

        X_train = X[
            train_mask
        ]

        y_train = y[
            train_mask
        ]

        X_test = X[
            test_mask
        ]

        y_test = y[
            test_mask
        ]

        try:

            prediction = train_fold(
                X_train,
                y_train,
                X_test,
            )

            metrics = pooled_metrics(
                y_test,
                prediction,
            )

            all_true.append(
                y_test
            )

            all_pred.append(
                prediction
            )

            fold_rows.append({

                "Pair ID":
                    pair_id,

                "Status":
                    "SUCCESS",

                "MAE":
                    metrics["MAE"],

                "RMSE":
                    metrics["RMSE"],

                "R2":
                    metrics["R2"],

                "MAPE":
                    metrics["MAPE"],
            })

            print(
                f"  MAE  : {metrics['MAE']:.6f}"
            )

            print(
                f"  RMSE : {metrics['RMSE']:.6f}"
            )

            print(
                f"  R2   : {metrics['R2']:.6f}"
            )

            print(
                f"  MAPE : {metrics['MAPE']:.6f}"
            )

        except Exception as error:

            fold_rows.append({

                "Pair ID":
                    pair_id,

                "Status":
                    "FAILED",

                "Reason":
                    str(error),
            })

            print(
                f"  FAILED: {error}"
            )

    if not all_true:

        raise RuntimeError(
            "All LOPO folds failed."
        )

    y_true = np.vstack(
        all_true
    )

    y_pred = np.vstack(
        all_pred
    )

    overall_metrics = pooled_metrics(
        y_true,
        y_pred,
    )

    summary = pd.DataFrame([{

        "Model":
            "Full_DNN",

        "Samples":
            len(dataframe),

        "Input_Features":
            len(ICE_FEATURES),

        "Output_Targets":
            len(EV_TARGETS),

        "LOPO_Folds_Successful":
            len(all_true),

        "LOPO_Folds_Total":
            len(unique_pairs),

        "MAE":
            overall_metrics["MAE"],

        "RMSE":
            overall_metrics["RMSE"],

        "R2":
            overall_metrics["R2"],

        "MAPE":
            overall_metrics["MAPE"],
    }])

    fold_dataframe = pd.DataFrame(
        fold_rows
    )

    return (
        summary,
        fold_dataframe,
        y_true,
        y_pred,
    )


# ============================================================
# TARGET-WISE OOS METRICS
# ============================================================

def target_wise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for index, target in enumerate(
        EV_TARGETS
    ):

        true = y_true[
            :,
            index
        ]

        pred = y_pred[
            :,
            index
        ]

        if np.allclose(
            true,
            true[0]
        ):

            r2 = float("nan")

        else:

            r2 = float(
                r2_score(
                    true,
                    pred,
                )
            )

        rows.append({

            "Target":
                target,

            "MAE":
                float(
                    mean_absolute_error(
                        true,
                        pred,
                    )
                ),

            "RMSE":
                float(
                    np.sqrt(
                        mean_squared_error(
                            true,
                            pred,
                        )
                    )
                ),

            "R2":
                r2,

            "MAPE":
                safe_mape(
                    true,
                    pred,
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FINAL MODEL
# ============================================================

def fit_final_model(
    dataframe: pd.DataFrame,
):

    X = dataframe[
        ICE_FEATURES
    ].to_numpy(
        dtype=float
    )

    y = dataframe[
        EV_TARGETS
    ].to_numpy(
        dtype=float
    )

    x_scaler = StandardScaler()

    y_scaler = StandardScaler()

    X_scaled = (
        x_scaler.fit_transform(
            X
        )
    )

    y_scaled = (
        y_scaler.fit_transform(
            y
        )
    )

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_scaled,
        dtype=torch.float32,
    )

    model = FullDLRegressor(
        input_dim=len(
            ICE_FEATURES
        ),
        output_dim=len(
            EV_TARGETS
        ),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.MSELoss()

    model.train()

    for epoch in range(
        EPOCHS
    ):

        optimizer.zero_grad()

        prediction = model(
            X_tensor
        )

        loss = loss_function(
            prediction,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

    return {

        "model":
            model,

        "x_scaler":
            x_scaler,

        "y_scaler":
            y_scaler,

        "input_features":
            ICE_FEATURES,

        "target_features":
            EV_TARGETS,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "hidden_layers":
            [
                HIDDEN_1,
                HIDDEN_2
            ],

        "random_seed":
            RANDOM_SEED,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    set_seed()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FULL DL — EV PREDICTION FROM ICE"
    )

    print(
        "=" * 70
    )

    print(
        f"Input workbook : {INPUT_FILE}"
    )

    print(
        "Device         : "
        f"{'cuda' if torch.cuda.is_available() else 'cpu'}"
    )

    print(
        "Data scope     : STRICT pairs only"
    )

    print(
        "Validation     : LOPO by Pair ID"
    )

    print(
        f"Input/Output   : "
        f"{len(ICE_FEATURES)} -> "
        f"{len(EV_TARGETS)}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dataframe = (
        load_training_data()
    )

    print(
        f"Rows loaded    : "
        f"{len(dataframe)}"
    )

    print(
        f"Unique Pair IDs: "
        f"{dataframe['Pair ID'].nunique()}"
    )

    # --------------------------------------------------------
    # LOPO
    # --------------------------------------------------------

    (
        summary,
        folds,
        y_true,
        y_pred,
    ) = run_lopo(
        dataframe
    )

    # --------------------------------------------------------
    # Target-wise results
    # --------------------------------------------------------

    target_dataframe = (
        target_wise_metrics(
            y_true,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save LOPO summary
    # --------------------------------------------------------

    summary.to_excel(
        RESULTS_DIR
        / "Full_DL_LOPO_Summary.xlsx",
        index=False,
    )

    # --------------------------------------------------------
    # Save fold results
    # --------------------------------------------------------

    folds.to_excel(
        RESULTS_DIR
        / "Full_DL_LOPO_Folds.xlsx",
        index=False,
    )

    # --------------------------------------------------------
    # Save target-wise results
    # --------------------------------------------------------

    target_dataframe.to_excel(
        RESULTS_DIR
        / "Full_DL_Target_Wise_OOS.xlsx",
        index=False,
    )

    # --------------------------------------------------------
    # Save OOS predictions
    # --------------------------------------------------------

    prediction_dataframe = pd.DataFrame()

    for index, target in enumerate(
        EV_TARGETS
    ):

        prediction_dataframe[
            f"Actual {target}"
        ] = y_true[
            :,
            index
        ]

        prediction_dataframe[
            f"Predicted {target}"
        ] = y_pred[
            :,
            index
        ]

    prediction_dataframe.to_excel(
        RESULTS_DIR
        / "Full_DL_OOS_Predictions.xlsx",
        index=False,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    configuration = pd.DataFrame([

        [
            "Architecture",
            f"15 -> {HIDDEN_1} -> "
            f"{HIDDEN_2} -> 15",
        ],

        [
            "Epochs",
            EPOCHS,
        ],

        [
            "Learning rate",
            LEARNING_RATE,
        ],

        [
            "Weight decay",
            WEIGHT_DECAY,
        ],

        [
            "Scaling",
            "StandardScaler fitted "
            "inside each LOPO fold",
        ],

        [
            "Validation",
            "Leave-One-Pair-Out "
            "by Pair ID",
        ],

        [
            "Charging/refuelling",
            "Separate; not a DL target",
        ],

        [
            "Samples",
            len(dataframe),
        ],

    ], columns=[
        "Parameter",
        "Value",
    ])

    configuration.to_excel(
        RESULTS_DIR
        / "Full_DL_Configuration.xlsx",
        index=False,
    )

    # --------------------------------------------------------
    # Train final model on all STRICT pairs
    # --------------------------------------------------------

    final_model = fit_final_model(
        dataframe
    )

    final_model_path = (
        MODEL_DIR
        / "Full_DNN_Final.pkl"
    )

    with open(
        final_model_path,
        "wb",
    ) as file:

        pickle.dump(
            final_model,
            file
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nFULL DL LOPO RESULT"
    )

    print(
        "-" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nTARGET-WISE OOS RESULT"
    )

    print(
        "-" * 70
    )

    print(
        target_dataframe.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        final_model_path
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FULL DL COMPLETED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()