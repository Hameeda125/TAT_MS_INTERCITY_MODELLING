from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    r"D:\TAT_MS_INTERCITY\Data\Results\Master_Summary"
    r"\Training_Data\EV_ICE_Training_STRICT.xlsx"
)

RESULTS_DIR = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING\Results\ML\Full_Model"
)

MODEL_DIR = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING\models\ML\Full_Model"
)


# ============================================================
# MODEL INPUTS
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
# MODEL TARGETS
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

PAIR_COLUMN = "Pair ID"


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data() -> pd.DataFrame:
    """
    Load the existing STRICT historical summary workbook.

    This function only loads data.
    It does not clean, impute, engineer, normalize, or scale data.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training file was not found:\n{INPUT_FILE}"
        )

    dataframe = pd.read_excel(
        INPUT_FILE,
        sheet_name="01_Training_Data",
        header=3,
        engine="openpyxl",
    )

    if dataframe.empty:
        raise ValueError("The training worksheet is empty.")

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_full_model_data(dataframe: pd.DataFrame) -> None:
    """
    Validate the exact Full Model structure.

    No values are removed or modified.
    """

    required_columns = ICE_FEATURES + EV_TARGETS + [PAIR_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required Full Model columns are missing:\n"
            + "\n".join(f" - {column}" for column in missing_columns)
        )

    if len(ICE_FEATURES) != 15:
        raise ValueError("The Full Model must contain exactly 15 ICE inputs.")

    if len(EV_TARGETS) != 15:
        raise ValueError("The Full Model must contain exactly 15 EV targets.")

    if dataframe[PAIR_COLUMN].isna().any():
        raise ValueError("Pair ID contains missing values.")

    if dataframe[PAIR_COLUMN].duplicated().any():
        duplicate_pairs = (
            dataframe.loc[
                dataframe[PAIR_COLUMN].duplicated(),
                PAIR_COLUMN,
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate Pair IDs were found:\n"
            + "\n".join(f" - {pair}" for pair in duplicate_pairs)
        )

    for column in ICE_FEATURES + EV_TARGETS:
        converted = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if converted.isna().any():
            raise ValueError(
                f"Non-numeric or missing values found in modelling column:\n"
                f"{column}"
            )


# ============================================================
# PREPARE X / Y / GROUPS
# ============================================================

def prepare_full_model_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    X = dataframe[ICE_FEATURES].copy()
    Y = dataframe[EV_TARGETS].copy()
    groups = dataframe[PAIR_COLUMN].copy()

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="raise")

    for column in Y.columns:
        Y[column] = pd.to_numeric(Y[column], errors="raise")

    return X, Y, groups


# ============================================================
# MODEL REGISTRY
# ============================================================

def build_candidate_models(
    n_training_samples: int,
) -> dict[str, object]:
    """
    Build conservative candidate models.

    Scaling is inside the Pipeline for models that require it,
    so scaling is fitted independently inside every LOPO fold.
    """

    models: dict[str, object] = {}

    models["LinearRegression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    models["Ridge"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    models["ElasticNet"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    ElasticNet(
                        alpha=0.1,
                        l1_ratio=0.5,
                        max_iter=10000,
                        random_state=42,
                    )
                ),
            ),
        ]
    )

    # With 5 training observations in each LOPO fold,
    # 2 PLS components is the maximum used here.
    pls_components = min(
        2,
        max(1, n_training_samples - 1),
        len(ICE_FEATURES),
    )

    models["PLS"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                PLSRegression(
                    n_components=pls_components
                ),
            ),
        ]
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    models["SVR"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    SVR(
                        kernel="rbf",
                        C=1.0,
                        epsilon=0.1,
                    )
                ),
            ),
        ]
    )

    return models


# ============================================================
# METRIC HELPERS
# ============================================================

def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    MAPE excluding zero actual values.

    MAPE is supplementary because very small actual values
    can produce very large percentages.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (actual != 0)
    )

    if not np.any(mask):
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def calculate_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Safe R² calculation.

    R² requires at least two observations and non-zero
    variance in the actual values.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) < 2:
        return np.nan

    if np.isclose(np.var(actual), 0.0):
        return np.nan

    return float(r2_score(actual, predicted))


def calculate_vector_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE (%)": np.nan,
        }

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "R2": calculate_r2(actual, predicted),
        "MAPE (%)": calculate_mape(actual, predicted),
    }


# ============================================================
# LOPO-CV
# ============================================================

def run_lopo_cv(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    groups: pd.Series,
) -> tuple[
    dict[str, list[dict]],
    dict[str, list[dict]],
    dict[str, list[dict]],
]:
    """
    Leave-One-Pair-Out Cross-Validation.

    For each fold:
        - one complete Pair ID is held out
        - remaining pairs are training data
        - baseline uses only the training-pair mean
        - model is fitted only on the training pairs

    Returns:
        model predictions
        baseline predictions
        fold status records
    """

    unique_pairs = groups.unique()

    if len(unique_pairs) < 3:
        raise ValueError(
            "At least 3 unique Pair IDs are required."
        )

    n_training_samples = len(unique_pairs) - 1
    models = build_candidate_models(n_training_samples)

    prediction_records = {
        model_name: []
        for model_name in models
    }

    baseline_records = []
    status_records = []

    print("\n" + "=" * 75)
    print("LEAVE-ONE-PAIR-OUT CROSS-VALIDATION")
    print("=" * 75)
    print(f"Unique pairs : {len(unique_pairs)}")

    for fold_number, test_pair in enumerate(
        unique_pairs,
        start=1,
    ):

        train_mask = groups != test_pair
        test_mask = groups == test_pair

        X_train = X.loc[train_mask]
        Y_train = Y.loc[train_mask]
        X_test = X.loc[test_mask]
        Y_test = Y.loc[test_mask]

        print(
            f"\nFold {fold_number}/{len(unique_pairs)}"
            f" | Test Pair: {test_pair}"
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------
        # Mean is calculated ONLY from the training pairs.
        # ----------------------------------------------------

        baseline_prediction = (
            Y_train.mean(axis=0)
            .to_numpy(dtype=float)
        )

        for row_number in range(len(Y_test)):

            record = {
                "Fold": fold_number,
                "Pair ID": test_pair,
            }

            for target_index, target in enumerate(EV_TARGETS):
                record[f"Actual {target}"] = Y_test.iloc[
                    row_number,
                    target_index,
                ]
                record[f"Predicted {target}"] = baseline_prediction[
                    target_index
                ]

            baseline_records.append(record)

        # ----------------------------------------------------
        # Models
        # ----------------------------------------------------

        for model_name, model in models.items():

            status = {
                "Fold": fold_number,
                "Pair ID": test_pair,
                "Model": model_name,
                "Status": "FAILED",
                "Error": "",
            }

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, Y_train)

                prediction = np.asarray(
                    model.predict(X_test),
                    dtype=float,
                )

                if prediction.shape != Y_test.shape:
                    raise ValueError(
                        f"Unexpected prediction shape "
                        f"{prediction.shape}; expected "
                        f"{Y_test.shape}."
                    )

                for row_number in range(len(X_test)):

                    record = {
                        "Fold": fold_number,
                        "Pair ID": test_pair,
                    }

                    for target_index, target in enumerate(EV_TARGETS):
                        record[f"Actual {target}"] = Y_test.iloc[
                            row_number,
                            target_index,
                        ]
                        record[f"Predicted {target}"] = prediction[
                            row_number,
                            target_index
                        ]

                    prediction_records[model_name].append(record)

                status["Status"] = "SUCCESS"

                print(
                    f"  {model_name:<20} SUCCESS"
                )

            except Exception as error:
                status["Error"] = str(error)

                print(
                    f"  {model_name:<20} FAILED"
                )
                print(
                    f"      Reason: {error}"
                )

            status_records.append(status)

    return (
        prediction_records,
        baseline_records,
        status_records,
    )


# ============================================================
# COMBINE PREDICTIONS
# ============================================================

def combine_prediction_records(
    prediction_records: dict[str, list[dict]],
) -> dict[str, pd.DataFrame]:

    combined = {}

    for model_name, records in prediction_records.items():
        if records:
            combined[model_name] = pd.DataFrame(records)

    return combined


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_prediction_table(
    dataframe: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:

    target_records = []
    pair_records = []

    # --------------------------------------------------------
    # Target-wise OOS metrics
    # --------------------------------------------------------

    for target in EV_TARGETS:

        actual = dataframe[
            f"Actual {target}"
        ].to_numpy(dtype=float)

        predicted = dataframe[
            f"Predicted {target}"
        ].to_numpy(dtype=float)

        metrics = calculate_vector_metrics(
            actual,
            predicted,
        )

        target_records.append(
            {
                "Model": model_name,
                "Target": target,
                **metrics,
                "N": int(
                    np.isfinite(actual)
                    .astype(int)
                    .sum()
                ),
            }
        )

    target_metrics = pd.DataFrame(target_records)

    # --------------------------------------------------------
    # Overall pooled metrics
    # --------------------------------------------------------

    all_actual = []
    all_predicted = []

    for target in EV_TARGETS:
        all_actual.extend(
            dataframe[
                f"Actual {target}"
            ].to_numpy(dtype=float)
        )
        all_predicted.extend(
            dataframe[
                f"Predicted {target}"
            ].to_numpy(dtype=float)
        )

    overall_metrics = calculate_vector_metrics(
        np.asarray(all_actual),
        np.asarray(all_predicted),
    )

    # --------------------------------------------------------
    # Pair-wise metrics
    # --------------------------------------------------------

    for pair_id, pair_df in dataframe.groupby(
        "Pair ID",
        sort=False,
    ):

        pair_actual = []
        pair_predicted = []

        for target in EV_TARGETS:
            pair_actual.extend(
                pair_df[
                    f"Actual {target}"
                ].to_numpy(dtype=float)
            )
            pair_predicted.extend(
                pair_df[
                    f"Predicted {target}"
                ].to_numpy(dtype=float)
            )

        metrics = calculate_vector_metrics(
            np.asarray(pair_actual),
            np.asarray(pair_predicted),
        )

        pair_records.append(
            {
                "Model": model_name,
                "Pair ID": pair_id,
                **metrics,
            }
        )

    pair_metrics = pd.DataFrame(pair_records)

    return (
        target_metrics,
        overall_metrics,
        pair_metrics,
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def evaluate_all_models(
    prediction_data: dict[str, pd.DataFrame],
    baseline_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    overall_records = []
    target_frames = []
    pair_frames = []

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_target, baseline_overall, baseline_pair = (
        evaluate_prediction_table(
            baseline_data,
            "TrainingMeanBaseline",
        )
    )

    overall_records.append(
        {
            "Model": "TrainingMeanBaseline",
            **baseline_overall,
        }
    )

    target_frames.append(baseline_target)
    pair_frames.append(baseline_pair)

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    for model_name, dataframe in prediction_data.items():

        target_metrics, overall_metrics, pair_metrics = (
            evaluate_prediction_table(
                dataframe,
                model_name,
            )
        )

        overall_records.append(
            {
                "Model": model_name,
                **overall_metrics,
            }
        )

        target_frames.append(target_metrics)
        pair_frames.append(pair_metrics)

    overall_results = pd.DataFrame(
        overall_records
    )

    # --------------------------------------------------------
    # Rank models.
    # Baseline remains visible but is not ranked as an ML model.
    # --------------------------------------------------------

    ml_only = overall_results[
        overall_results["Model"]
        != "TrainingMeanBaseline"
    ].copy()

    ml_only = ml_only.sort_values(
        by=["RMSE", "MAE"],
        ascending=[True, True],
    ).reset_index(drop=True)

    ml_only["Overall Rank"] = np.arange(
        1,
        len(ml_only) + 1,
    )

    overall_results = overall_results.merge(
        ml_only[
            ["Model", "Overall Rank"]
        ],
        on="Model",
        how="left",
    )

    overall_results["Better Than Baseline"] = (
        overall_results["RMSE"]
        < float(
            baseline_overall["RMSE"]
        )
    )

    overall_results = overall_results.sort_values(
        by="Overall Rank",
        na_position="last",
    ).reset_index(drop=True)

    target_results = pd.concat(
        target_frames,
        ignore_index=True,
    )

    pair_results = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    return (
        overall_results,
        target_results,
        pair_results,
    )


# ============================================================
# TARGET-WISE MODEL RANKING
# ============================================================

def create_target_ranking(
    target_results: pd.DataFrame,
) -> pd.DataFrame:

    ml_targets = target_results[
        target_results["Model"]
        != "TrainingMeanBaseline"
    ].copy()

    ml_targets["MAE Rank"] = (
        ml_targets.groupby("Target")["MAE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    ml_targets["RMSE Rank"] = (
        ml_targets.groupby("Target")["RMSE"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    best_mae = (
        ml_targets[
            ml_targets["MAE Rank"] == 1
        ]
        .groupby("Model")
        .size()
        .reset_index(
            name="Targets with Best MAE"
        )
        .sort_values(
            "Targets with Best MAE",
            ascending=False,
        )
    )

    return best_mae


# ============================================================
# ERROR ANALYSIS
# ============================================================

def create_error_analysis(
    prediction_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    records = []

    for model_name, dataframe in prediction_data.items():

        for target in EV_TARGETS:

            actual = dataframe[
                f"Actual {target}"
            ].to_numpy(dtype=float)

            predicted = dataframe[
                f"Predicted {target}"
            ].to_numpy(dtype=float)

            error = predicted - actual

            records.append(
                {
                    "Model": model_name,
                    "Target": target,
                    "Mean Signed Error": float(
                        np.mean(error)
                    ),
                    "Mean Absolute Error": float(
                        np.mean(np.abs(error))
                    ),
                    "Maximum Absolute Error": float(
                        np.max(np.abs(error))
                    ),
                    "Overprediction Count": int(
                        np.sum(error > 0)
                    ),
                    "Underprediction Count": int(
                        np.sum(error < 0)
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    prediction_data: dict[str, pd.DataFrame],
    baseline_data: pd.DataFrame,
    status_data: pd.DataFrame,
    overall_results: pd.DataFrame,
    target_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    target_ranking: pd.DataFrame,
    error_analysis: pd.DataFrame,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Overall comparison
    # --------------------------------------------------------

    overall_path = (
        RESULTS_DIR
        / "Full_ML_Model_Comparison.xlsx"
    )

    overall_results.to_excel(
        overall_path,
        index=False,
    )

    # --------------------------------------------------------
    # Detailed analysis workbook
    # --------------------------------------------------------

    analysis_path = (
        RESULTS_DIR
        / "Full_ML_Detailed_Evaluation.xlsx"
    )

    with pd.ExcelWriter(
        analysis_path,
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

        pair_results.to_excel(
            writer,
            sheet_name="Pair_Wise_OOS",
            index=False,
        )

        target_ranking.to_excel(
            writer,
            sheet_name="Best_Model_By_Target",
            index=False,
        )

        error_analysis.to_excel(
            writer,
            sheet_name="Signed_Error_Analysis",
            index=False,
        )

        status_data.to_excel(
            writer,
            sheet_name="Fold_Model_Status",
            index=False,
        )

        baseline_data.to_excel(
            writer,
            sheet_name="Baseline_Predictions",
            index=False,
        )

    # --------------------------------------------------------
    # Per-target compatibility output
    # --------------------------------------------------------

    per_target_path = (
        RESULTS_DIR
        / "Full_ML_Per_Target_Metrics.xlsx"
    )

    target_results.to_excel(
        per_target_path,
        index=False,
    )

    # --------------------------------------------------------
    # Held-out predictions
    # --------------------------------------------------------

    prediction_path = (
        RESULTS_DIR
        / "Full_ML_Held_Out_Predictions.xlsx"
    )

    with pd.ExcelWriter(
        prediction_path,
        engine="openpyxl",
    ) as writer:

        for model_name, dataframe in prediction_data.items():

            dataframe.to_excel(
                writer,
                sheet_name=model_name[:31],
                index=False,
            )

    print(
        f"\nSaved model comparison:\n{overall_path}"
    )

    print(
        f"\nSaved detailed evaluation:\n{analysis_path}"
    )

    print(
        f"\nSaved per-target metrics:\n{per_target_path}"
    )

    print(
        f"\nSaved held-out predictions:\n{prediction_path}"
    )


# ============================================================
# FINAL MODEL FIT
# ============================================================

def fit_final_models(
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> dict[str, object]:
    """
    Fit final candidate models on all available STRICT pairs.

    These are NOT used for CV evaluation.
    They are saved for later new-ICE inference after model
    selection.
    """

    models = build_candidate_models(
        n_training_samples=len(X)
    )

    final_models = {}

    print("\n" + "=" * 75)
    print("FINAL MODEL FIT ON ALL STRICT PAIRS")
    print("=" * 75)

    for model_name, model in models.items():

        try:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, Y)

            final_models[model_name] = model

            print(
                f"{model_name:<20} SUCCESS"
            )

        except Exception as error:

            print(
                f"{model_name:<20} FAILED"
            )
            print(
                f"    Reason: {error}"
            )

    return final_models


# ============================================================
# SAVE FINAL MODELS
# ============================================================

def save_final_models(
    final_models: dict[str, object],
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, model in final_models.items():

        model_path = (
            MODEL_DIR
            / f"{model_name}_Full.pkl"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"Saved final model: {model_path.name}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n" + "=" * 75)
    print("TAT MS INTERCITY")
    print("FULL ML MODELLING")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Load existing STRICT summary data
    # --------------------------------------------------------

    dataframe = load_training_data()

    print("\nHistorical summary data loaded.")
    print(f"Rows    : {len(dataframe)}")
    print(f"Columns : {len(dataframe.columns)}")

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    validate_full_model_data(dataframe)

    print(
        "\nFull Model structure validation : PASS"
    )

    # --------------------------------------------------------
    # 3. Prepare X / Y / groups
    # --------------------------------------------------------

    X, Y, groups = prepare_full_model_data(
        dataframe
    )

    print("\n" + "=" * 75)
    print("FULL MODEL DATA")
    print("=" * 75)

    print(f"Unique Pair IDs : {groups.nunique()}")
    print(f"ICE input shape : {X.shape}")
    print(f"EV target shape : {Y.shape}")

    # --------------------------------------------------------
    # 4. LOPO-CV
    # --------------------------------------------------------

    (
        prediction_records,
        baseline_records,
        status_records,
    ) = run_lopo_cv(
        X,
        Y,
        groups,
    )

    prediction_data = combine_prediction_records(
        prediction_records
    )

    if not prediction_data:
        raise RuntimeError(
            "No model produced valid held-out predictions."
        )

    baseline_data = pd.DataFrame(
        baseline_records
    )

    status_data = pd.DataFrame(
        status_records
    )

    # --------------------------------------------------------
    # 5. Evaluate held-out predictions
    # --------------------------------------------------------

    (
        overall_results,
        target_results,
        pair_results,
    ) = evaluate_all_models(
        prediction_data,
        baseline_data,
    )

    target_ranking = create_target_ranking(
        target_results
    )

    error_analysis = create_error_analysis(
        prediction_data
    )

    # --------------------------------------------------------
    # 6. Display overall comparison
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FULL ML OVERALL OUT-OF-SAMPLE COMPARISON")
    print("=" * 75)

    print(
        overall_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. Display target-wise ExtraTrees result
    # --------------------------------------------------------

    if "ExtraTrees" in prediction_data:

        extra_target = target_results[
            target_results["Model"] == "ExtraTrees"
        ].copy()

        print("\n" + "=" * 75)
        print("EXTRATREES TARGET-WISE OUT-OF-SAMPLE RESULTS")
        print("=" * 75)

        print(
            extra_target[
                [
                    "Target",
                    "MAE",
                    "RMSE",
                    "R2",
                    "MAPE (%)",
                    "N",
                ]
            ]
            .sort_values("MAE")
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 8. Save CV results and diagnostics
    # --------------------------------------------------------

    save_results(
        prediction_data=prediction_data,
        baseline_data=baseline_data,
        status_data=status_data,
        overall_results=overall_results,
        target_results=target_results,
        pair_results=pair_results,
        target_ranking=target_ranking,
        error_analysis=error_analysis,
    )

    # --------------------------------------------------------
    # 9. Fit final models on all STRICT pairs
    # --------------------------------------------------------

    final_models = fit_final_models(
        X,
        Y,
    )

    save_final_models(
        final_models
    )

    # --------------------------------------------------------
    # 10. Final status
    # --------------------------------------------------------

    successful_models = (
        status_data[
            status_data["Status"] == "SUCCESS"
        ]["Model"]
        .nunique()
        if not status_data.empty
        else 0
    )

    total_models = len(
        build_candidate_models(
            len(unique_pairs := groups.unique())
        )
    )

    print("\n" + "=" * 75)
    print("FULL ML PIPELINE COMPLETED")
    print("=" * 75)

    print(
        f"\nLOPO-CV pairs : {groups.nunique()}"
    )

    print(
        f"Candidate ML models : {total_models}"
    )

    print(
        f"Models with successful folds : "
        f"{successful_models}"
    )

    print(
        "\nEvaluation is based on held-out paired trips."
    )

    print(
        "\nThe TrainingMeanBaseline is included as a "
        "reference and is not treated as an ML model."
    )

    print(
        "\nFinal candidate models were fitted on all "
        "available STRICT pairs for later inference."
    )

    print(
        "\nIMPORTANT:"
        "\nDo not interpret the pooled overall R² as "
        "17 individual target accuracies."
        "\nUse the Target_Wise_OOS sheet for target-level "
        "research interpretation."
    )


if __name__ == "__main__":
    main()
