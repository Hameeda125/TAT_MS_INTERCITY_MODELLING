"""
TAT MS INTERCITY
PHASE-WISE ML MODELLING

Purpose
-------
Train separate multi-output regression models for each of the six
non-charging phases:

    STOP
    IDLE
    INCHING
    ACC
    DEC
    CRUISE

For every phase:

    ICE phase time + ICE phase distance
                    ↓
              ML model
                    ↓
    EV phase time + EV phase distance

Charging/refuelling is intentionally excluded from this module.
It is calculated separately by src.prediction.charging_time.

Historical data:
    EV_ICE_Training_STRICT.xlsx

Validation:
    Leave-One-Pair-Out Cross-Validation (LOPO-CV)

Models:
    LinearRegression
    Ridge
    ElasticNet
    PLSRegression
    RandomForestRegressor
    ExtraTreesRegressor
    SVR

The best model is selected independently for each phase using
out-of-sample MAE, with RMSE used as the secondary criterion.

IMPORTANT
---------
The historical dataset currently contains only six STRICT pairs.
Therefore, the evaluation is exploratory and should not be interpreted
as strong generalization evidence.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================================
# PATHS
# ============================================================================

TRAINING_FILE = Path(
    r"D:\TAT_MS_INTERCITY\Data\Results\Master_Summary"
    r"\Training_Data\EV_ICE_Training_STRICT.xlsx"
)

RESULTS_ROOT = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING\Results\ML\Phase_Wise_Model"
)

MODEL_ROOT = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING\models\ML\Phase_Wise_Model"
)

RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PHASE DEFINITIONS
# ============================================================================

PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
]


def phase_column_prefix(phase: str) -> str:
    """Map phase name to the naming used in the training workbook."""
    mapping = {
        "STOP": "STOP",
        "IDLE": "IDLE",
        "INCHING": "INCHING",
        "ACC": "ACC",
        "DEC": "DEC",
        "CRUISE": "CRUISE",
    }
    return mapping[phase]


# ============================================================================
# MODELS
# ============================================================================

def build_models() -> dict[str, object]:
    """
    Build the same candidate model family used in Full ML.

    Scaling is performed inside pipelines so that each LOPO fold fits
    preprocessing only on the training pairs.
    """
    return {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                (
                    "model",
                    MultiOutputRegressor(
                        Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                (
                                    "elasticnet",
                                    ElasticNet(
                                        alpha=0.01,
                                        l1_ratio=0.5,
                                        max_iter=10000,
                                    ),
                                ),
                            ]
                        )
                    ),
                )
            ]
        ),
        "PLS": PLSRegression(
            n_components=1,
            scale=True,
            max_iter=500,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features=1.0,
            min_samples_leaf=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            max_features=1.0,
            min_samples_leaf=1,
        ),
        "SVR": MultiOutputRegressor(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(C=10.0, epsilon=0.1, kernel="rbf")),
                ]
            )
        ),
    }


# ============================================================================
# DATA VALIDATION
# ============================================================================

def expected_columns_for_phase(phase: str) -> tuple[list[str], list[str]]:
    prefix = phase_column_prefix(phase)

    x_columns = [
        f"ICE {prefix} Time (min)",
        f"ICE {prefix} Distance (km)",
    ]

    y_columns = [
        f"EV {prefix} Time (min)",
        f"EV {prefix} Distance (km)",
    ]

    return x_columns, y_columns


def load_training_data() -> pd.DataFrame:
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training workbook does not exist:\n{TRAINING_FILE}"
        )

    dataframe = pd.read_excel(
        TRAINING_FILE,
        header=3,
        engine="openpyxl",
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    if dataframe.empty:
        raise ValueError("Training workbook is empty.")

    return dataframe


def validate_data(dataframe: pd.DataFrame) -> None:
    if "Pair ID" not in dataframe.columns:
        raise ValueError("Required column 'Pair ID' is missing.")

    if dataframe["Pair ID"].isna().any():
        raise ValueError("Pair ID contains missing values.")

    if dataframe["Pair ID"].duplicated().any():
        raise ValueError("Duplicate Pair IDs detected.")

    if len(dataframe) < 2:
        raise ValueError("At least two paired observations are required.")

    missing = []

    for phase in PHASES:
        x_columns, y_columns = expected_columns_for_phase(phase)
        for column in x_columns + y_columns:
            if column not in dataframe.columns:
                missing.append(column)

    if missing:
        raise ValueError(
            "Missing required phase-wise modelling columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    modelling_columns = []

    for phase in PHASES:
        x_columns, y_columns = expected_columns_for_phase(phase)
        modelling_columns.extend(x_columns)
        modelling_columns.extend(y_columns)

    numeric_check = dataframe[modelling_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if numeric_check.isna().any().any():
        bad_columns = numeric_check.columns[
            numeric_check.isna().any()
        ].tolist()
        raise ValueError(
            "Non-numeric or missing values found in modelling columns:\n"
            + "\n".join(f"  - {column}" for column in bad_columns)
        )

    print("\nPhase-wise structure validation : PASS")


# ============================================================================
# METRICS
# ============================================================================

def safe_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100.0
    )


def safe_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) < 2:
        return np.nan

    if np.isclose(np.var(actual), 0.0):
        return np.nan

    return float(r2_score(actual, predicted))


def pooled_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "R2": safe_r2(actual, predicted),
        "MAPE (%)": safe_mape(actual, predicted),
    }


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_model(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> object:
    """
    Fit one candidate model.

    PLS requires n_components <= min(n_samples, n_features).
    With two inputs and LOPO training on five pairs, n_components=1 is safe.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)

    return model


# ============================================================================
# PHASE-WISE LOPO-CV
# ============================================================================

def evaluate_phase(
    dataframe: pd.DataFrame,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all candidate models for one phase using LOPO-CV.

    Returns:
        overall_metrics
        target_metrics
        held_out_predictions
        fold_status
    """
    x_columns, y_columns = expected_columns_for_phase(phase)

    x = dataframe[x_columns].astype(float)
    y = dataframe[y_columns].astype(float)
    pair_ids = dataframe["Pair ID"].astype(str).to_numpy()

    unique_pairs = list(dict.fromkeys(pair_ids))

    all_predictions: list[dict] = []
    fold_status: list[dict] = []

    for fold_number, test_pair in enumerate(unique_pairs, start=1):
        test_mask = pair_ids == test_pair
        train_mask = ~test_mask

        x_train = x.loc[train_mask]
        y_train = y.loc[train_mask]
        x_test = x.loc[test_mask]
        y_test = y.loc[test_mask]

        print(
            f"\n  Fold {fold_number}/{len(unique_pairs)}"
            f" | Test Pair: {test_pair}"
        )

        for model_name, model_template in build_models().items():
            # Rebuild each model so every fold is independent.
            model = build_models()[model_name]

            status = "SUCCESS"
            error_message = ""

            try:
                fitted = fit_model(
                    model_name,
                    model,
                    x_train,
                    y_train,
                )

                prediction = np.asarray(
                    fitted.predict(x_test),
                    dtype=float,
                )

                if prediction.ndim == 1:
                    prediction = prediction.reshape(1, -1)

                for target_index, target_name in enumerate(y_columns):
                    all_predictions.append(
                        {
                            "Phase": phase,
                            "Pair ID": test_pair,
                            "Model": model_name,
                            "Target": target_name,
                            "Actual": float(
                                y_test.iloc[0, target_index]
                            ),
                            "Predicted": float(
                                prediction[0, target_index]
                            ),
                        }
                    )

                print(f"    {model_name:<18} SUCCESS")

            except Exception as error:
                status = "FAILED"
                error_message = str(error)
                print(
                    f"    {model_name:<18} FAILED: "
                    f"{error_message}"
                )

            fold_status.append(
                {
                    "Phase": phase,
                    "Fold": fold_number,
                    "Test Pair": test_pair,
                    "Model": model_name,
                    "Status": status,
                    "Error": error_message,
                }
            )

    prediction_df = pd.DataFrame(all_predictions)
    fold_status_df = pd.DataFrame(fold_status)

    overall_rows = []
    target_rows = []

    for model_name in build_models().keys():
        model_predictions = prediction_df[
            prediction_df["Model"] == model_name
        ]

        if model_predictions.empty:
            continue

        actual = model_predictions["Actual"].to_numpy()
        predicted = model_predictions["Predicted"].to_numpy()

        metrics = pooled_metrics(actual, predicted)

        overall_rows.append(
            {
                "Phase": phase,
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE (%)": metrics["MAPE (%)"],
                "N Predictions": len(model_predictions),
            }
        )

        for target_name in y_columns:
            target_predictions = model_predictions[
                model_predictions["Target"] == target_name
            ]

            target_actual = target_predictions["Actual"].to_numpy()
            target_predicted = target_predictions["Predicted"].to_numpy()

            target_metric = pooled_metrics(
                target_actual,
                target_predicted,
            )

            target_rows.append(
                {
                    "Phase": phase,
                    "Model": model_name,
                    "Target": target_name,
                    "MAE": target_metric["MAE"],
                    "RMSE": target_metric["RMSE"],
                    "R2": target_metric["R2"],
                    "MAPE (%)": target_metric["MAPE (%)"],
                    "N": len(target_predictions),
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    target_df = pd.DataFrame(target_rows)

    return (
        overall_df,
        target_df,
        prediction_df,
        fold_status_df,
    )


# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_best_models(
    overall_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the best model independently for each phase.

    Primary criterion:
        lowest pooled OOS MAE

    Secondary criterion:
        lowest pooled OOS RMSE
    """
    selections = []

    for phase in PHASES:
        phase_results = overall_df[
            overall_df["Phase"] == phase
        ].copy()

        if phase_results.empty:
            continue

        phase_results = phase_results.sort_values(
            by=["MAE", "RMSE"],
            ascending=[True, True],
        ).reset_index(drop=True)

        best = phase_results.iloc[0]

        selections.append(
            {
                "Phase": phase,
                "Best Model": best["Model"],
                "MAE": best["MAE"],
                "RMSE": best["RMSE"],
                "R2": best["R2"],
                "MAPE (%)": best["MAPE (%)"],
                "Selection Basis": "Lowest pooled LOPO-CV MAE; RMSE as tie-breaker",
            }
        )

    return pd.DataFrame(selections)


# ============================================================================
# FINAL MODEL FIT
# ============================================================================

def fit_final_models(
    dataframe: pd.DataFrame,
    selections_df: pd.DataFrame,
) -> None:
    """
    Fit the selected model for each phase using all STRICT pairs.

    These models are saved for later prediction of new ICE trips.
    """
    print("\n" + "=" * 68)
    print("FINAL PHASE-WISE MODEL FIT ON ALL STRICT PAIRS")
    print("=" * 68)

    for _, selection in selections_df.iterrows():
        phase = str(selection["Phase"])
        model_name = str(selection["Best Model"])

        x_columns, y_columns = expected_columns_for_phase(phase)

        x = dataframe[x_columns].astype(float)
        y = dataframe[y_columns].astype(float)

        model = build_models()[model_name]

        try:
            fit_model(
                model_name,
                model,
                x,
                y,
            )

            phase_dir = MODEL_ROOT / phase
            phase_dir.mkdir(parents=True, exist_ok=True)

            model_path = (
                phase_dir
                / f"{model_name}_{phase}_Phase.pkl"
            )

            joblib.dump(model, model_path)

            metadata = {
                "phase": phase,
                "model": model_name,
                "input_columns": x_columns,
                "target_columns": y_columns,
                "training_pairs": dataframe["Pair ID"]
                .astype(str)
                .tolist(),
            }

            joblib.dump(
                metadata,
                phase_dir / f"{phase}_metadata.pkl",
            )

            print(
                f"{phase:<10} {model_name:<18} SUCCESS"
            )
            print(f"           Saved: {model_path}")

        except Exception as error:
            print(
                f"{phase:<10} {model_name:<18} FAILED: "
                f"{error}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 68)
    print("TAT MS INTERCITY")
    print("PHASE-WISE ML MODELLING")
    print("=" * 68)

    dataframe = load_training_data()

    print("\nHistorical STRICT summary data loaded.")
    print(f"Rows    : {len(dataframe)}")
    print(f"Columns : {len(dataframe.columns)}")

    validate_data(dataframe)

    print("\n" + "=" * 68)
    print("PHASE-WISE ML CONFIGURATION")
    print("=" * 68)

    print("Training pairs : STRICT only")
    print("Phases         : STOP, IDLE, INCHING, ACC, DEC, CRUISE")
    print("Inputs / phase : ICE time + ICE distance")
    print("Targets / phase: EV time + EV distance")
    print("Charging       : Separate pipeline")
    print("Validation     : Leave-One-Pair-Out CV")
    print("Models         : 7 candidate ML models")

    all_overall = []
    all_target = []
    all_predictions = []
    all_fold_status = []

    for phase in PHASES:
        print("\n" + "=" * 68)
        print(f"PHASE: {phase}")
        print("=" * 68)

        overall_df, target_df, prediction_df, fold_status_df = (
            evaluate_phase(
                dataframe,
                phase,
            )
        )

        all_overall.append(overall_df)
        all_target.append(target_df)
        all_predictions.append(prediction_df)
        all_fold_status.append(fold_status_df)

        print("\nPhase overall OOS comparison:")
        print(
            overall_df.sort_values(
                by=["MAE", "RMSE"]
            ).to_string(index=False)
        )

    overall_results = pd.concat(
        all_overall,
        ignore_index=True,
    )

    target_results = pd.concat(
        all_target,
        ignore_index=True,
    )

    held_out_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    fold_status = pd.concat(
        all_fold_status,
        ignore_index=True,
    )

    selections = select_best_models(overall_results)

    # ------------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------------

    comparison_path = (
        RESULTS_ROOT
        / "Phase_Wise_ML_Model_Comparison.xlsx"
    )

    target_path = (
        RESULTS_ROOT
        / "Phase_Wise_ML_Per_Target_Metrics.xlsx"
    )

    predictions_path = (
        RESULTS_ROOT
        / "Phase_Wise_ML_Held_Out_Predictions.xlsx"
    )

    detailed_path = (
        RESULTS_ROOT
        / "Phase_Wise_ML_Detailed_Evaluation.xlsx"
    )

    selection_path = (
        RESULTS_ROOT
        / "Phase_Wise_Best_Model_By_Phase.xlsx"
    )

    overall_results.to_excel(
        comparison_path,
        index=False,
    )

    target_results.to_excel(
        target_path,
        index=False,
    )

    held_out_predictions.to_excel(
        predictions_path,
        index=False,
    )

    selections.to_excel(
        selection_path,
        index=False,
    )

    with pd.ExcelWriter(
        detailed_path,
        engine="openpyxl",
    ) as writer:
        overall_results.to_excel(
            writer,
            sheet_name="Overall_OOS_Metrics",
            index=False,
        )
        target_results.to_excel(
            writer,
            sheet_name="Target_Wise_OOS",
            index=False,
        )
        selections.to_excel(
            writer,
            sheet_name="Best_Model_By_Phase",
            index=False,
        )
        held_out_predictions.to_excel(
            writer,
            sheet_name="Held_Out_Predictions",
            index=False,
        )
        fold_status.to_excel(
            writer,
            sheet_name="Fold_Model_Status",
            index=False,
        )

    print("\n" + "=" * 68)
    print("BEST MODEL BY PHASE")
    print("=" * 68)
    print(
        selections.to_string(index=False)
    )

    print("\nSaved model comparison:")
    print(comparison_path)

    print("\nSaved target-wise metrics:")
    print(target_path)

    print("\nSaved held-out predictions:")
    print(predictions_path)

    print("\nSaved detailed evaluation:")
    print(detailed_path)

    print("\nSaved best-model selection:")
    print(selection_path)

    # ------------------------------------------------------------------------
    # Fit final selected models on all STRICT pairs.
    # ------------------------------------------------------------------------

    fit_final_models(
        dataframe,
        selections,
    )

    print("\n" + "=" * 68)
    print("PHASE-WISE ML PIPELINE COMPLETED")
    print("=" * 68)

    print(f"Phases evaluated       : {len(PHASES)}")
    print("Candidate ML models    : 7")
    print(f"STRICT training pairs  : {dataframe['Pair ID'].nunique()}")
    print("\nCharging/refuelling is NOT part of phase-wise ML.")
    print("Use src.prediction.charging_time for EV charging time.")
    print("\nBest model selection is based on held-out LOPO-CV MAE.")
    print(
        "Because only six STRICT pairs are available, "
        "results are exploratory."
    )


if __name__ == "__main__":
    main()
