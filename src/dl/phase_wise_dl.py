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
# PHASE-WISE DL CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    r"D:\TAT_MS_INTERCITY\Data\Results\Master_Summary\Training_Data\EV_ICE_Training_STRICT.xlsx"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "Results" / "DL" / "Phase_Wise_Model"
MODEL_DIR = PROJECT_ROOT / "models" / "DL" / "Phase_Wise_Model"

RANDOM_SEED = 42

EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_1 = 8
HIDDEN_2 = 4


# ============================================================
# PHASES
# Charging/refuelling is intentionally excluded.
# ============================================================

PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
]


# ============================================================
# PHASE INPUTS AND TARGETS
# ============================================================

PHASE_FEATURES = {
    phase: [
        f"ICE {phase} Time (min)",
        f"ICE {phase} Distance (km)",
    ]
    for phase in PHASES
}

PHASE_TARGETS = {
    phase: [
        f"EV {phase} Time (min)",
        f"EV {phase} Distance (km)",
    ]
    for phase in PHASES
}


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
# PHASE DNN
# ============================================================

class PhaseDLRegressor(nn.Module):

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
    ) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, output_dim),
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
        ) * 100.0
    )


# ============================================================
# METRICS
# ============================================================

def pooled_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {
        "MAE": float(
            mean_absolute_error(
                y_true.ravel(),
                y_pred.ravel(),
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true.ravel(),
                    y_pred.ravel(),
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true.ravel(),
                y_pred.ravel(),
            )
        ),
        "MAPE": safe_mape(
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
            "Pair ID column is required for LOPO validation."
        )

    required_columns = []

    for phase in PHASES:
        required_columns.extend(
            PHASE_FEATURES[phase]
        )
        required_columns.extend(
            PHASE_TARGETS[phase]
        )

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Required phase-wise columns are missing:\n"
            f"{missing}"
        )

    modelling = dataframe[required_columns]

    if modelling.isna().any().any():

        raise ValueError(
            "Missing values found in phase-wise modelling columns."
        )

    non_numeric = [
        column
        for column in modelling.columns
        if not pd.api.types.is_numeric_dtype(
            modelling[column]
        )
    ]

    if non_numeric:

        raise ValueError(
            "Non-numeric phase-wise modelling columns found:\n"
            f"{non_numeric}"
        )

    return dataframe


# ============================================================
# TRAIN ONE PHASE / ONE LOPO FOLD
# ============================================================

def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_test_scaled = x_scaler.transform(X_test)

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

    model = PhaseDLRegressor(
        input_dim=2,
        output_dim=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.MSELoss()

    model.train()

    for _ in range(EPOCHS):

        optimizer.zero_grad()

        prediction = model(X_tensor)

        loss = loss_function(
            prediction,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        scaled_prediction = (
            model(X_test_tensor)
            .cpu()
            .numpy()
        )

    return y_scaler.inverse_transform(
        scaled_prediction
    )


# ============================================================
# RUN LOPO FOR ONE PHASE
# ============================================================

def run_phase_lopo(
    dataframe: pd.DataFrame,
    phase: str,
):

    features = PHASE_FEATURES[phase]
    targets = PHASE_TARGETS[phase]

    X = dataframe[features].to_numpy(dtype=float)
    y = dataframe[targets].to_numpy(dtype=float)

    pair_ids = (
        dataframe["Pair ID"]
        .astype(str)
        .to_numpy()
    )

    unique_pairs = list(
        dict.fromkeys(pair_ids)
    )

    all_true = []
    all_pred = []
    successful_pair_ids = []
    fold_rows = []

    for pair_id in unique_pairs:

        print(
            f"    LOPO fold: {pair_id}"
        )

        test_mask = pair_ids == pair_id
        train_mask = ~test_mask

        try:

            prediction = train_fold(
                X[train_mask],
                y[train_mask],
                X[test_mask],
            )

            true = y[test_mask]

            metrics = pooled_metrics(
                true,
                prediction,
            )

            all_true.append(true)
            all_pred.append(prediction)
            successful_pair_ids.append(pair_id)

            fold_rows.append({
                "Phase": phase,
                "Pair ID": pair_id,
                "Status": "SUCCESS",
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE": metrics["MAPE"],
            })

            print(
                f"      MAE={metrics['MAE']:.6f}, "
                f"RMSE={metrics['RMSE']:.6f}, "
                f"R2={metrics['R2']:.6f}, "
                f"MAPE={metrics['MAPE']:.6f}"
            )

        except Exception as error:

            fold_rows.append({
                "Phase": phase,
                "Pair ID": pair_id,
                "Status": "FAILED",
                "Reason": str(error),
            })

            print(
                f"      FAILED: {error}"
            )

    if not all_true:

        raise RuntimeError(
            f"All LOPO folds failed for phase {phase}."
        )

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    metrics = pooled_metrics(
        y_true,
        y_pred,
    )

    summary = {
        "Phase": phase,
        "Model": "Phase_DNN",
        "Samples": len(dataframe),
        "Input_Features": 2,
        "Output_Targets": 2,
        "LOPO_Folds_Successful": len(all_true),
        "LOPO_Folds_Total": len(unique_pairs),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "MAPE": metrics["MAPE"],
    }

    return (
        summary,
        pd.DataFrame(fold_rows),
        y_true,
        y_pred,
        successful_pair_ids,
    )


# ============================================================
# TARGET-WISE METRICS
# ============================================================

def target_wise_metrics(
    phase: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for index, target in enumerate(
        PHASE_TARGETS[phase]
    ):

        true = y_true[:, index]
        pred = y_pred[:, index]

        if np.allclose(
            true,
            true[0],
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
            "Phase": phase,
            "Target": target,
            "MAE": float(
                mean_absolute_error(
                    true,
                    pred,
                )
            ),
            "RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        true,
                        pred,
                    )
                )
            ),
            "R2": r2,
            "MAPE": safe_mape(
                true,
                pred,
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# FINAL MODEL FOR ONE PHASE
# ============================================================

def fit_final_model(
    dataframe: pd.DataFrame,
    phase: str,
):

    features = PHASE_FEATURES[phase]
    targets = PHASE_TARGETS[phase]

    X = dataframe[features].to_numpy(dtype=float)
    y = dataframe[targets].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_scaled,
        dtype=torch.float32,
    )

    model = PhaseDLRegressor(
        input_dim=2,
        output_dim=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.MSELoss()

    model.train()

    for _ in range(EPOCHS):

        optimizer.zero_grad()

        prediction = model(X_tensor)

        loss = loss_function(
            prediction,
            y_tensor,
        )

        loss.backward()

        optimizer.step()

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "phase": phase,
        "input_features": features,
        "target_features": targets,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_layers": [
            HIDDEN_1,
            HIDDEN_2,
        ],
        "random_seed": RANDOM_SEED,
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
        "PHASE-WISE DL — EV PREDICTION FROM ICE"
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
        "Phases         : "
        + ", ".join(PHASES)
    )

    print(
        "Architecture   : "
        f"2 -> {HIDDEN_1} -> "
        f"{HIDDEN_2} -> 2"
    )

    print(
        "=" * 70
    )

    dataframe = load_training_data()

    print(
        f"Rows loaded    : {len(dataframe)}"
    )

    print(
        f"Unique Pair IDs: "
        f"{dataframe['Pair ID'].nunique()}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []
    all_folds = []
    all_targets = []

    for phase in PHASES:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"PHASE: {phase}"
        )

        print(
            "=" * 70
        )

        (
            summary,
            folds,
            y_true,
            y_pred,
            pair_ids,
        ) = run_phase_lopo(
            dataframe,
            phase,
        )

        target_df = target_wise_metrics(
            phase,
            y_true,
            y_pred,
        )

        all_summaries.append(
            summary
        )

        all_folds.append(
            folds
        )

        all_targets.append(
            target_df
        )

        phase_dir = (
            RESULTS_DIR / phase
        )

        phase_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame([summary]).to_excel(
            phase_dir
            / f"{phase}_DL_LOPO_Summary.xlsx",
            index=False,
        )

        folds.to_excel(
            phase_dir
            / f"{phase}_DL_LOPO_Folds.xlsx",
            index=False,
        )

        target_df.to_excel(
            phase_dir
            / f"{phase}_DL_Target_Wise_OOS.xlsx",
            index=False,
        )

        prediction_df = pd.DataFrame({
            "Pair ID": pair_ids,
        })

        for index, target in enumerate(
            PHASE_TARGETS[phase]
        ):

            prediction_df[
                f"Actual {target}"
            ] = y_true[:, index]

            prediction_df[
                f"Predicted {target}"
            ] = y_pred[:, index]

        prediction_df.to_excel(
            phase_dir
            / f"{phase}_DL_OOS_Predictions.xlsx",
            index=False,
        )

        # Final model trained on all STRICT pairs.
        final_model = fit_final_model(
            dataframe,
            phase,
        )

        phase_model_dir = (
            MODEL_DIR / phase
        )

        phase_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            phase_model_dir
            / f"{phase}_DNN_Final.pkl"
        )

        with open(
            model_path,
            "wb",
        ) as file:

            pickle.dump(
                final_model,
                file,
            )

        print(
            f"\n{phase} DL RESULT"
        )

        print(
            pd.DataFrame([summary])
            .to_string(index=False)
        )

        print(
            f"Model saved: {model_path}"
        )

    # --------------------------------------------------------
    # Combined outputs
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        all_summaries
    )

    folds_df = pd.concat(
        all_folds,
        ignore_index=True,
    )

    targets_df = pd.concat(
        all_targets,
        ignore_index=True,
    )

    summary_df.to_excel(
        RESULTS_DIR
        / "Phase_Wise_DL_LOPO_Summary.xlsx",
        index=False,
    )

    folds_df.to_excel(
        RESULTS_DIR
        / "Phase_Wise_DL_LOPO_Folds.xlsx",
        index=False,
    )

    targets_df.to_excel(
        RESULTS_DIR
        / "Phase_Wise_DL_Target_Wise_OOS.xlsx",
        index=False,
    )

    configuration = pd.DataFrame([

        [
            "Architecture",
            f"2 -> {HIDDEN_1} -> "
            f"{HIDDEN_2} -> 2",
        ],

        [
            "Phases",
            ", ".join(PHASES),
        ],

        [
            "Samples",
            len(dataframe),
        ],

        [
            "Validation",
            "Leave-One-Pair-Out by Pair ID",
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
            "StandardScaler fitted separately "
            "inside each LOPO fold",
        ],

        [
            "Charging/refuelling",
            "Excluded; handled separately",
        ],

    ], columns=[
        "Parameter",
        "Value",
    ])

    configuration.to_excel(
        RESULTS_DIR
        / "Phase_Wise_DL_Configuration.xlsx",
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE-WISE DL COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nCombined results:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_DIR}"
    )

    print(
        f"\nModels saved to:\n"
        f"{MODEL_DIR}"
    )

    print(
        "\n"
        + "=" * 70
    )


if __name__ == "__main__":
    main()
