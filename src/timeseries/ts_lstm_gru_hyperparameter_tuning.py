# ============================================================
# TS LSTM + GRU CONTROLLED HYPERPARAMETER TUNING - GPU VERSION
# ============================================================
# Purpose:
#   Leakage-safe hyperparameter tuning for the existing TS LSTM
#   and GRU models, followed by completely held-out outer LOPO
#   evaluation.
#
# IMPORTANT:
#   - Existing model/result files are NOT modified.
#   - No random row split.
#   - No cross-pair leakage.
#   - Outer test pair is NEVER used during tuning.
#   - Charging time is NOT included here.
#
# Validation design:
#
#   OUTER LOPO
#       1 complete pair = final test pair
#       remaining 5 pairs = development data
#
#   INNER LOPO
#       performed only inside the 5 development pairs
#       to select hyperparameters.
#
# After selection:
#       retrain on all 5 development pairs
#       evaluate once on the untouched outer test pair
#
# Current data:
#   6 independent paired trips
#   100 route bins / pair
#   17 ICE inputs
#   13 EV targets
#
# Tuning objective:
#   Mean absolute error of the SIX-PHASE EV DRIVING TAT
#   on inner validation pairs.
#
# Six phase times:
#   STOP + IDLE + INCHING + ACC + DEC + CRUISE
#
# This is deliberately more comprehensive than the earlier
# 8-12-combination version because the user has GPU available.
#
# Search:
#   24 controlled hyperparameter combinations
#   per model per outer fold.
#
# ============================================================

import os
import sys
import copy
import itertools
import random
import time

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
# 1. REPRODUCIBILITY + DEVICE
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("TS LSTM + GRU CONTROLLED HYPERPARAMETER TUNING")
print("=" * 70)

print(
    f"\nDevice: {DEVICE}"
)

if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print(
        "WARNING: CUDA GPU was not detected."
    )


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "Results",
    "TimeSeries",
)

DATASET_PATH = os.path.join(
    RESULTS_ROOT,
    "Dataset",
    "TS_Sequence_Dataset.npz",
)

OUTPUT_DIR = os.path.join(
    RESULTS_ROOT,
    "Tuning",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 3. TUNING SETTINGS
# ============================================================
#
# 24 deliberately selected combinations.
#
# We use combinations that cover:
#   - small / medium / larger hidden representations
#   - 1 / 2 recurrent layers
#   - low / medium learning rate
#   - no / small weight decay
#   - no / small dropout
#
# We avoid a huge Cartesian grid because six independent trips
# cannot support an enormous hyperparameter search reliably.
#
# 24 combinations x 5 inner folds x 6 outer folds x 2 models
# = 1440 training runs.
#
# GPU is recommended.
# ============================================================

SEARCH_SPACE = [

    # hidden=16
    {
        "hidden_size": 16,
        "num_layers": 1,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "dropout": 0.0,
    },

    {
        "hidden_size": 16,
        "num_layers": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
    },

    {
        "hidden_size": 16,
        "num_layers": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.0,
    },

    {
        "hidden_size": 16,
        "num_layers": 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "dropout": 0.1,
    },

    {
        "hidden_size": 16,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.1,
    },

    {
        "hidden_size": 16,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.1,
    },

    # hidden=32
    {
        "hidden_size": 32,
        "num_layers": 1,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "dropout": 0.0,
    },

    {
        "hidden_size": 32,
        "num_layers": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
    },

    {
        "hidden_size": 32,
        "num_layers": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.0,
    },

    {
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "dropout": 0.1,
    },

    {
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.1,
    },

    {
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.1,
    },

    # hidden=64
    {
        "hidden_size": 64,
        "num_layers": 1,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "dropout": 0.0,
    },

    {
        "hidden_size": 64,
        "num_layers": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
    },

    {
        "hidden_size": 64,
        "num_layers": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.0,
    },

    {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "dropout": 0.1,
    },

    {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.1,
    },

    {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.1,
    },

    # additional regularization variants
    {
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "dropout": 0.2,
    },

    {
        "hidden_size": 32,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.2,
    },

    {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "dropout": 0.2,
    },

    {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.2,
    },

    {
        "hidden_size": 16,
        "num_layers": 1,
        "learning_rate": 0.00025,
        "weight_decay": 0.0001,
        "dropout": 0.0,
    },

    {
        "hidden_size": 32,
        "num_layers": 1,
        "learning_rate": 0.00025,
        "weight_decay": 0.0001,
        "dropout": 0.0,
    },
]

MAX_EPOCHS = 500

PATIENCE = 50

MIN_DELTA = 1e-5

GRADIENT_CLIP = 1.0


# ============================================================
# 4. AUTHORITATIVE TARGETS
# ============================================================

EV_TARGET_COLUMNS = [
    "EV_Moving_Average_Speed_kmh",
    "EV_STOP_Time_min",
    "EV_IDLE_Time_min",
    "EV_INCHING_Time_min",
    "EV_ACC_Time_min",
    "EV_DEC_Time_min",
    "EV_CRUISE_Time_min",
    "EV_STOP_Distance_km",
    "EV_IDLE_Distance_km",
    "EV_INCHING_Distance_km",
    "EV_ACC_Distance_km",
    "EV_DEC_Distance_km",
    "EV_CRUISE_Distance_km",
]

TAT_TARGETS = [
    "EV_STOP_Time_min",
    "EV_IDLE_Time_min",
    "EV_INCHING_Time_min",
    "EV_ACC_Time_min",
    "EV_DEC_Time_min",
    "EV_CRUISE_Time_min",
]

TAT_INDICES = [
    EV_TARGET_COLUMNS.index(
        target
    )
    for target in TAT_TARGETS
]


# ============================================================
# 5. LOAD DATA
# ============================================================

print(
    "\nLoading sequence dataset..."
)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

data = np.load(
    DATASET_PATH,
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
    f"X shape: {X.shape}"
)

print(
    f"Y shape: {Y.shape}"
)

print(
    f"Pairs: {len(pair_ids)}"
)

print(
    f"Search combinations: "
    f"{len(SEARCH_SPACE)}"
)


# ============================================================
# 6. VALIDATE DATA
# ============================================================

if X.ndim != 3 or Y.ndim != 3:
    raise ValueError(
        "X and Y must both be 3D."
    )

if X.shape[:2] != Y.shape[:2]:
    raise ValueError(
        "X and Y sequence dimensions do not match."
    )

if len(pair_ids) != X.shape[0]:
    raise ValueError(
        "Pair_ID count does not match X."
    )

if len(set(pair_ids)) != len(pair_ids):
    raise ValueError(
        "Duplicate Pair_ID values found."
    )

if not np.isfinite(X).all():
    raise ValueError(
        "Non-finite X values found."
    )

if not np.isfinite(Y).all():
    raise ValueError(
        "Non-finite Y values found."
    )

if Y.shape[-1] != len(
    EV_TARGET_COLUMNS
):
    raise ValueError(
        "Y target count does not match "
        "authoritative target list."
    )


# ============================================================
# 7. MODEL
# ============================================================

class SequenceModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        model_type,
    ):

        super().__init__()

        effective_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        if model_type == "LSTM":

            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=effective_dropout,
            )

        elif model_type == "GRU":

            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=effective_dropout,
            )

        else:

            raise ValueError(
                "model_type must be LSTM or GRU."
            )

        self.output = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(
        self,
        x,
    ):

        sequence_output, _ = self.rnn(
            x
        )

        return self.output(
            sequence_output
        )


# ============================================================
# 8. SCALING
# ============================================================

def fit_scalers(
    X_train,
    Y_train,
):

    x_scaler = StandardScaler()

    y_scaler = StandardScaler()

    X_train_2d = X_train.reshape(
        -1,
        X_train.shape[-1],
    )

    Y_train_2d = Y_train.reshape(
        -1,
        Y_train.shape[-1],
    )

    x_scaler.fit(
        X_train_2d
    )

    y_scaler.fit(
        Y_train_2d
    )

    return (
        x_scaler,
        y_scaler,
    )


def transform_X(
    X_input,
    scaler,
):

    X_2d = X_input.reshape(
        -1,
        X_input.shape[-1],
    )

    return scaler.transform(
        X_2d
    ).reshape(
        X_input.shape
    )


def transform_Y(
    Y_input,
    scaler,
):

    Y_2d = Y_input.reshape(
        -1,
        Y_input.shape[-1],
    )

    return scaler.transform(
        Y_2d
    ).reshape(
        Y_input.shape
    )


# ============================================================
# 9. TRAIN
# ============================================================

def train_model(
    X_train,
    Y_train,
    params,
    model_type,
):

    x_scaler, y_scaler = fit_scalers(
        X_train,
        Y_train,
    )

    X_scaled = transform_X(
        X_train,
        x_scaler,
    )

    Y_scaled = transform_Y(
        Y_train,
        y_scaler,
    )

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    Y_tensor = torch.tensor(
        Y_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    model = SequenceModel(
        input_size=X.shape[-1],
        hidden_size=params[
            "hidden_size"
        ],
        output_size=Y.shape[-1],
        num_layers=params[
            "num_layers"
        ],
        dropout=params[
            "dropout"
        ],
        model_type=model_type,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params[
            "learning_rate"
        ],
        weight_decay=params[
            "weight_decay"
        ],
    )

    criterion = nn.MSELoss()

    best_loss = np.inf

    best_state = None

    patience_counter = 0

    epochs_used = 0

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            X_tensor
        )

        loss = criterion(
            prediction,
            Y_tensor,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        current_loss = float(
            loss.detach()
            .cpu()
            .item()
        )

        epochs_used = epoch + 1

        if (
            best_loss
            -
            current_loss
            >
            MIN_DELTA
        ):

            best_loss = current_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >=
            PATIENCE
        ):
            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        model,
        x_scaler,
        y_scaler,
        epochs_used,
    )


# ============================================================
# 10. PREDICTION
# ============================================================

def predict_model(
    model,
    x_scaler,
    y_scaler,
    X_input,
):

    X_scaled = transform_X(
        X_input,
        x_scaler,
    )

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    model.eval()

    with torch.no_grad():

        prediction_scaled = (
            model(
                X_tensor
            )
            .cpu()
            .numpy()
        )

    prediction_2d = (
        prediction_scaled.reshape(
            -1,
            prediction_scaled.shape[-1],
        )
    )

    prediction = (
        y_scaler.inverse_transform(
            prediction_2d
        )
        .reshape(
            prediction_scaled.shape
        )
    )

    return prediction


# ============================================================
# 11. TAT AGGREGATE SCORE
# ============================================================

def aggregate_six_phase_tat(
    Y_values,
):

    return np.sum(
        Y_values[
            :,
            :,
            TAT_INDICES,
        ],
        axis=(1, 2),
    )


def six_phase_tat_mae(
    Y_actual,
    Y_predicted,
):

    actual_tat = aggregate_six_phase_tat(
        Y_actual
    )

    predicted_tat = aggregate_six_phase_tat(
        Y_predicted
    )

    return float(
        mean_absolute_error(
            actual_tat,
            predicted_tat,
        )
    )


# ============================================================
# 12. INNER-LOPO TUNING
# ============================================================

def select_hyperparameters(
    X_development,
    Y_development,
    development_pair_ids,
    model_type,
    outer_fold,
    outer_test_pair,
):

    candidates = []

    for combination_number, params in enumerate(
        SEARCH_SPACE,
        start=1,
    ):

        inner_scores = []

        for inner_test_pair in development_pair_ids:

            inner_train_mask = (
                development_pair_ids
                !=
                inner_test_pair
            )

            inner_test_mask = (
                development_pair_ids
                ==
                inner_test_pair
            )

            X_inner_train = (
                X_development[
                    inner_train_mask
                ]
            )

            Y_inner_train = (
                Y_development[
                    inner_train_mask
                ]
            )

            X_inner_test = (
                X_development[
                    inner_test_mask
                ]
            )

            Y_inner_test = (
                Y_development[
                    inner_test_mask
                ]
            )

            model, x_scaler, y_scaler, epochs_used = (
                train_model(
                    X_inner_train,
                    Y_inner_train,
                    params,
                    model_type,
                )
            )

            prediction = predict_model(
                model,
                x_scaler,
                y_scaler,
                X_inner_test,
            )

            score = six_phase_tat_mae(
                Y_inner_test,
                prediction,
            )

            inner_scores.append(
                score
            )

        mean_inner_score = float(
            np.mean(
                inner_scores
            )
        )

        candidates.append(
            {
                "Model":
                    model_type,

                "Outer_Fold":
                    outer_fold,

                "Outer_Test_Pair":
                    outer_test_pair,

                "Combination":
                    combination_number,

                "Hidden_Size":
                    params[
                        "hidden_size"
                    ],

                "Num_Layers":
                    params[
                        "num_layers"
                    ],

                "Learning_Rate":
                    params[
                        "learning_rate"
                    ],

                "Weight_Decay":
                    params[
                        "weight_decay"
                    ],

                "Dropout":
                    params[
                        "dropout"
                    ],

                "Inner_LOPO_TAT_MAE_min":
                    mean_inner_score,

                "Inner_Fold_Count":
                    len(inner_scores),
            }
        )

    candidate_df = pd.DataFrame(
        candidates
    )

    best_index = candidate_df[
        "Inner_LOPO_TAT_MAE_min"
    ].idxmin()

    best_row = candidate_df.loc[
        best_index
    ]

    best_params = {
        "hidden_size":
            int(
                best_row[
                    "Hidden_Size"
                ]
            ),

        "num_layers":
            int(
                best_row[
                    "Num_Layers"
                ]
            ),

        "learning_rate":
            float(
                best_row[
                    "Learning_Rate"
                ]
            ),

        "weight_decay":
            float(
                best_row[
                    "Weight_Decay"
                ]
            ),

        "dropout":
            float(
                best_row[
                    "Dropout"
                ]
            ),
    }

    return (
        candidate_df,
        best_params,
        float(
            best_row[
                "Inner_LOPO_TAT_MAE_min"
            ]
        ),
    )


# ============================================================
# 13. OUTER NESTED LOPO
# ============================================================

all_tuning_rows = []

selected_parameter_rows = []

outer_prediction_rows = []

outer_summary_rows = []

start_time = time.time()

for model_type in [
    "LSTM",
    "GRU",
]:

    print("\n" + "=" * 70)
    print(
        f"OUTER LOPO TUNING: {model_type}"
    )
    print("=" * 70)

    for outer_fold, test_pair in enumerate(
        pair_ids,
        start=1,
    ):

        outer_train_mask = (
            pair_ids
            !=
            test_pair
        )

        outer_test_mask = (
            pair_ids
            ==
            test_pair
        )

        X_development = X[
            outer_train_mask
        ]

        Y_development = Y[
            outer_train_mask
        ]

        development_pair_ids = pair_ids[
            outer_train_mask
        ]

        X_outer_test = X[
            outer_test_mask
        ]

        Y_outer_test = Y[
            outer_test_mask
        ]

        print(
            f"\n{model_type} | "
            f"Outer fold {outer_fold}/"
            f"{len(pair_ids)} | "
            f"Test pair: {test_pair}"
        )

        tuning_df, best_params, inner_score = (
            select_hyperparameters(
                X_development,
                Y_development,
                development_pair_ids,
                model_type,
                outer_fold,
                test_pair,
            )
        )

        all_tuning_rows.extend(
            tuning_df.to_dict(
                orient="records"
            )
        )

        print(
            "  Selected parameters:"
        )

        print(
            f"    hidden_size  = "
            f"{best_params['hidden_size']}"
        )

        print(
            f"    num_layers   = "
            f"{best_params['num_layers']}"
        )

        print(
            f"    learning_rate= "
            f"{best_params['learning_rate']}"
        )

        print(
            f"    weight_decay = "
            f"{best_params['weight_decay']}"
        )

        print(
            f"    dropout      = "
            f"{best_params['dropout']}"
        )

        print(
            f"  Inner LOPO TAT MAE = "
            f"{inner_score:.6f} min"
        )

        selected_parameter_rows.append(
            {
                "Model":
                    model_type,

                "Outer_Fold":
                    outer_fold,

                "Test_Pair_ID":
                    test_pair,

                **{
                    key: value
                    for key, value
                    in best_params.items()
                },

                "Inner_LOPO_TAT_MAE_min":
                    inner_score,
            }
        )

        # ----------------------------------------------------
        # REFIT USING ALL DEVELOPMENT PAIRS
        # ----------------------------------------------------

        model, x_scaler, y_scaler, epochs_used = (
            train_model(
                X_development,
                Y_development,
                best_params,
                model_type,
            )
        )

        # ----------------------------------------------------
        # FINAL OUTER TEST
        # ----------------------------------------------------

        prediction = predict_model(
            model,
            x_scaler,
            y_scaler,
            X_outer_test,
        )

        test_tat_mae = six_phase_tat_mae(
            Y_outer_test,
            prediction,
        )

        actual_tat = aggregate_six_phase_tat(
            Y_outer_test
        )

        predicted_tat = aggregate_six_phase_tat(
            prediction
        )

        print(
            f"  Outer test six-phase TAT MAE = "
            f"{test_tat_mae:.6f} min"
        )

        for sample_index in range(
            X_outer_test.shape[0]
        ):

            pair_id = test_pair

            for bin_index in range(
                X_outer_test.shape[1]
            ):

                for target_index, target in enumerate(
                    EV_TARGET_COLUMNS
                ):

                    if target not in TAT_TARGETS:
                        continue

                    outer_prediction_rows.append(
                        {
                            "Model":
                                model_type,

                            "Outer_Fold":
                                outer_fold,

                            "Test_Pair_ID":
                                pair_id,

                            "Route_Bin":
                                bin_index,

                            "Target":
                                target,

                            "Actual":
                                float(
                                    Y_outer_test[
                                        sample_index,
                                        bin_index,
                                        target_index,
                                    ]
                                ),

                            "Predicted":
                                float(
                                    prediction[
                                        sample_index,
                                        bin_index,
                                        target_index,
                                    ]
                                ),
                        }
                    )

        outer_summary_rows.append(
            {
                "Model":
                    model_type,

                "Outer_Fold":
                    outer_fold,

                "Test_Pair_ID":
                    test_pair,

                "Inner_LOPO_TAT_MAE_min":
                    inner_score,

                "Outer_TAT_MAE_min":
                    test_tat_mae,

                "Actual_Six_Phase_TAT_min":
                    float(
                        actual_tat[0]
                    ),

                "Predicted_Six_Phase_TAT_min":
                    float(
                        predicted_tat[0]
                    ),

                "Absolute_TAT_Error_min":
                    float(
                        abs(
                            actual_tat[0]
                            -
                            predicted_tat[0]
                        )
                    ),

                "Epochs_Used":
                    epochs_used,

                **{
                    key: value
                    for key, value
                    in best_params.items()
                },
            }
        )


# ============================================================
# 14. FINAL POOLED PHASE METRICS
# ============================================================

predictions_df = pd.DataFrame(
    outer_prediction_rows
)

selected_df = pd.DataFrame(
    selected_parameter_rows
)

tuning_df = pd.DataFrame(
    all_tuning_rows
)

outer_summary_df = pd.DataFrame(
    outer_summary_rows
)

pooled_phase_rows = []

for model_type in [
    "LSTM",
    "GRU",
]:

    model_df = predictions_df[
        predictions_df["Model"]
        ==
        model_type
    ]

    for target in TAT_TARGETS:

        target_df = model_df[
            model_df["Target"]
            ==
            target
        ]

        actual = target_df[
            "Actual"
        ].to_numpy()

        predicted = target_df[
            "Predicted"
        ].to_numpy()

        pooled_phase_rows.append(
            {
                "Model":
                    model_type,

                "Target":
                    target,

                "N":
                    len(actual),

                "MAE_min":
                    mean_absolute_error(
                        actual,
                        predicted,
                    ),

                "RMSE_min":
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            predicted,
                        )
                    ),

                "R2":
                    (
                        r2_score(
                            actual,
                            predicted,
                        )
                        if not np.allclose(
                            actual,
                            actual[0],
                        )
                        else np.nan
                    ),
            }
        )

pooled_phase_df = pd.DataFrame(
    pooled_phase_rows
)


# ============================================================
# 15. POOLED SIX-PHASE TAT RESULTS
# ============================================================

aggregate_rows = []

for model_type in [
    "LSTM",
    "GRU",
]:

    temp = outer_summary_df[
        outer_summary_df["Model"]
        ==
        model_type
    ]

    actual = temp[
        "Actual_Six_Phase_TAT_min"
    ].to_numpy()

    predicted = temp[
        "Predicted_Six_Phase_TAT_min"
    ].to_numpy()

    if len(actual) > 1:
        r2 = r2_score(
            actual,
            predicted,
        )
    else:
        r2 = np.nan

    aggregate_rows.append(
        {
            "Model":
                model_type,

            "N_Pairs":
                len(actual),

            "MAE_min":
                mean_absolute_error(
                    actual,
                    predicted,
                ),

            "RMSE_min":
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted,
                    )
                ),

            "R2":
                r2,
        }
    )

aggregate_df = pd.DataFrame(
    aggregate_rows
)


# ============================================================
# 16. SANITY CHECKS
# ============================================================

expected_prediction_rows = (
    len(pair_ids)
    *
    X.shape[1]
    *
    len(TAT_TARGETS)
    *
    2
)

if len(predictions_df) != expected_prediction_rows:
    raise ValueError(
        "Unexpected prediction coverage. "
        f"Expected {expected_prediction_rows}, "
        f"found {len(predictions_df)}."
    )

if len(selected_df) != (
    len(pair_ids) * 2
):
    raise ValueError(
        "Expected exactly one selected "
        "parameter set per model per outer fold."
    )

if len(
    outer_summary_df
) != (
    len(pair_ids) * 2
):
    raise ValueError(
        "Unexpected outer-fold summary count."
    )


# ============================================================
# 17. SAVE RESULTS
# ============================================================

output_excel = os.path.join(
    OUTPUT_DIR,
    "TS_Tuned_LSTM_GRU_Nested_LOPO.xlsx",
)

with pd.ExcelWriter(
    output_excel,
    engine="openpyxl",
) as writer:

    tuning_df.to_excel(
        writer,
        sheet_name="Tuning_Search",
        index=False,
    )

    selected_df.to_excel(
        writer,
        sheet_name="Selected_Params",
        index=False,
    )

    outer_summary_df.to_excel(
        writer,
        sheet_name="Outer_LOPO_TAT",
        index=False,
    )

    pooled_phase_df.to_excel(
        writer,
        sheet_name="TAT_Phase_OOS",
        index=False,
    )

    aggregate_df.to_excel(
        writer,
        sheet_name="Six_Phase_TAT",
        index=False,
    )

    predictions_df.to_excel(
        writer,
        sheet_name="OOS_Predictions",
        index=False,
    )


# ============================================================
# 18. REPORT
# ============================================================

elapsed_minutes = (
    time.time() - start_time
) / 60.0

report_path = os.path.join(
    OUTPUT_DIR,
    "TS_Tuned_LSTM_GRU_Nested_LOPO_Report.txt",
)

report_lines = []

report_lines.append(
    "TS LSTM + GRU CONTROLLED HYPERPARAMETER "
    "TUNING REPORT"
)

report_lines.append(
    "=" * 80
)

report_lines.append(
    ""
)

report_lines.append(
    f"Device: {DEVICE}"
)

if DEVICE.type == "cuda":
    report_lines.append(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

report_lines.append(
    f"Pairs: {len(pair_ids)}"
)

report_lines.append(
    f"Route bins per pair: {X.shape[1]}"
)

report_lines.append(
    f"Hyperparameter combinations: "
    f"{len(SEARCH_SPACE)}"
)

report_lines.append(
    f"Maximum epochs: {MAX_EPOCHS}"
)

report_lines.append(
    ""
)

report_lines.append(
    "VALIDATION DESIGN"
)

report_lines.append(
    "-" * 80
)

report_lines.append(
    "Nested Leave-One-Pair-Out."
)

report_lines.append(
    "Outer test pair was completely excluded from "
    "hyperparameter selection."
)

report_lines.append(
    "Inner LOPO was performed only on the five "
    "outer-development pairs."
)

report_lines.append(
    "After selection, the selected model was refit "
    "on all five development pairs and evaluated "
    "once on the untouched outer test pair."
)

report_lines.append(
    ""
)

report_lines.append(
    "TUNING OBJECTIVE"
)

report_lines.append(
    "-" * 80
)

report_lines.append(
    "Mean absolute error of aggregate six-phase "
    "EV driving TAT on inner LOPO validation."
)

report_lines.append(
    "Charging time is excluded and remains a "
    "separate component of final EV TAT."
)

report_lines.append(
    ""
)

report_lines.append(
    "SELECTED PARAMETERS"
)

report_lines.append(
    "-" * 80
)

report_lines.append(
    selected_df.to_string(
        index=False
    )
)

report_lines.append(
    ""
)

report_lines.append(
    "POOLED TAT PHASE OOS RESULTS"
)

report_lines.append(
    "-" * 80
)

report_lines.append(
    pooled_phase_df.to_string(
        index=False
    )
)

report_lines.append(
    ""
)

report_lines.append(
    "POOLED SIX-PHASE TAT OOS RESULTS"
)

report_lines.append(
    "-" * 80
)

report_lines.append(
    aggregate_df.to_string(
        index=False
    )
)

report_lines.append(
    ""
)

report_lines.append(
    f"Runtime: {elapsed_minutes:.2f} minutes"
)

report_lines.append(
    ""
)

report_lines.append(
    "IMPORTANT LIMITATION"
)

report_lines.append(
    "-" * 80
)

report_lines.append(
    "Only six independent paired trips are currently "
    "available. Therefore, tuning and performance "
    "results should be interpreted as preliminary."
)

report_lines.append(
    "The purpose of nested LOPO is to obtain a "
    "leakage-controlled estimate despite the small "
    "number of independent trips."
)

report_lines.append(
    "When additional paired trips become available, "
    "the same pipeline can be rerun with the expanded "
    "dataset."
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


print("\n" + "=" * 70)
print("NESTED LOPO TUNING COMPLETED")
print("=" * 70)

print(
    f"\nResults workbook:\n{output_excel}"
)

print(
    f"\nReport:\n{report_path}"
)

print(
    f"\nRuntime: {elapsed_minutes:.2f} minutes"
)

print(
    "\nExisting LSTM/GRU result files were not modified."
)

print(
    "No charging-time calculation was changed."
)

print(
    "No dataset was modified."
)
