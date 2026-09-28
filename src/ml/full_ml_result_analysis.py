from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "Results" / "ML" / "Full_Model"
OUTPUT_DIR = INPUT_DIR / "Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = INPUT_DIR / "Full_ML_Model_Comparison.xlsx"
TARGET_METRICS_FILE = INPUT_DIR / "Full_ML_Per_Target_Metrics.xlsx"
PREDICTIONS_FILE = INPUT_DIR / "Full_ML_Held_Out_Predictions.xlsx"


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(ss_tot, 0):
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1 - ss_res / ss_tot)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE (%)": np.nan}

    error = y_true - y_pred

    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": r2_score_safe(y_true, y_pred),
        "MAPE (%)": safe_mape(y_true, y_pred),
    }


def main():
    print("=" * 70)
    print("FULL ML RESULT ANALYSIS")
    print("=" * 70)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(PREDICTIONS_FILE)

    predictions = pd.read_excel(PREDICTIONS_FILE, sheet_name=None)

    all_target_rows = []
    pair_rows = []

    for model, df in predictions.items():
        if df.empty:
            continue

        print(f"Processing: {model}")

        pair_ids = df["Pair ID"].astype(str)

        actual_cols = [c for c in df.columns if c.startswith("Actual EV ")]
        for actual_col in actual_cols:
            target = actual_col.replace("Actual ", "")
            pred_col = f"Predicted {target}"

            if pred_col not in df.columns:
                continue

            metrics = calculate_metrics(df[actual_col], df[pred_col])

            all_target_rows.append({
                "Model": model,
                "Target": target,
                **metrics,
                "N": int(df[[actual_col, pred_col]].dropna().shape[0]),
            })

        # Pair-level aggregate errors.
        for i, row in df.iterrows():
            pair_id = pair_ids.iloc[i]
            errors = []

            for actual_col in actual_cols:
                target = actual_col.replace("Actual ", "")
                pred_col = f"Predicted {target}"

                if pred_col not in df.columns:
                    continue

                actual = row[actual_col]
                pred = row[pred_col]

                if pd.notna(actual) and pd.notna(pred):
                    errors.append(abs(float(actual) - float(pred)))

            if errors:
                pair_rows.append({
                    "Model": model,
                    "Pair ID": pair_id,
                    "Mean Absolute Error Across 17 Targets": float(np.mean(errors)),
                    "Maximum Absolute Error Across 17 Targets": float(np.max(errors)),
                })

    target_metrics = pd.DataFrame(all_target_rows)
    pair_metrics = pd.DataFrame(pair_rows)

    # Rank models within each target using MAE.
    target_metrics["Target_MAE_Rank"] = (
        target_metrics.groupby("Target")["MAE"]
        .rank(method="min", ascending=True)
    )

    target_metrics = target_metrics.sort_values(
        ["Target", "MAE", "RMSE"]
    )

    # Best model count across targets.
    best_model_counts = (
        target_metrics[target_metrics["Target_MAE_Rank"] == 1]
        .groupby("Model")
        .size()
        .reset_index(name="Number of Targets with Best MAE")
        .sort_values("Number of Targets with Best MAE", ascending=False)
    )

    # Recalculate overall metrics from all out-of-sample predictions.
    overall_rows = []

    for model, df in predictions.items():
        actual_cols = [c for c in df.columns if c.startswith("Actual EV ")]
        y_true = []
        y_pred = []

        for actual_col in actual_cols:
            target = actual_col.replace("Actual ", "")
            pred_col = f"Predicted {target}"
            if pred_col not in df.columns:
                continue

            y_true.extend(df[actual_col].tolist())
            y_pred.extend(df[pred_col].tolist())

        metrics = calculate_metrics(y_true, y_pred)
        overall_rows.append({"Model": model, **metrics})

    overall_metrics = pd.DataFrame(overall_rows).sort_values("RMSE")

    # Save analysis workbook.
    output_file = OUTPUT_DIR / "Full_ML_Result_Analysis.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        overall_metrics.to_excel(writer, sheet_name="Overall_OOS_Metrics", index=False)
        target_metrics.to_excel(writer, sheet_name="Target_Wise_OOS_Metrics", index=False)
        pair_metrics.to_excel(writer, sheet_name="Pair_Wise_Errors", index=False)
        best_model_counts.to_excel(writer, sheet_name="Best_Model_By_Target", index=False)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    # 1. Overall model comparison.
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = overall_metrics.sort_values("RMSE")
    ax.bar(plot_df["Model"], plot_df["RMSE"])
    ax.set_title("Full ML: Out-of-Sample RMSE by Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_Overall_RMSE_Comparison.png", dpi=300)
    plt.close(fig)

    # 2. Target-wise MAE for each model.
    pivot_mae = target_metrics.pivot(
        index="Target", columns="Model", values="MAE"
    )

    ax = pivot_mae.plot(kind="bar", figsize=(15, 8))
    ax.set_title("Full ML: Target-Wise Out-of-Sample MAE")
    ax.set_ylabel("MAE")
    ax.set_xlabel("EV Target")
    ax.tick_params(axis="x", rotation=75)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_Target_Wise_MAE.png", dpi=300)
    plt.close(fig)

    # 3. ExtraTrees target-wise metrics.
    extra = target_metrics[target_metrics["Model"] == "ExtraTrees"].copy()

    ax = extra.sort_values("MAE").plot(
        x="Target", y="MAE", kind="bar", figsize=(14, 7), legend=False
    )
    ax.set_title("ExtraTrees: Target-Wise Out-of-Sample MAE")
    ax.set_ylabel("MAE")
    ax.set_xlabel("EV Target")
    ax.tick_params(axis="x", rotation=75)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_ExtraTrees_Target_MAE.png", dpi=300)
    plt.close(fig)

    # 4. ExtraTrees actual vs predicted for every target.
    et_df = predictions["ExtraTrees"]

    actual_cols = [c for c in et_df.columns if c.startswith("Actual EV ")]

    for actual_col in actual_cols:
        target = actual_col.replace("Actual ", "")
        pred_col = f"Predicted {target}"

        if pred_col not in et_df.columns:
            continue

        y_true = et_df[actual_col].astype(float).to_numpy()
        y_pred = et_df[pred_col].astype(float).to_numpy()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred)

        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(finite):
            lo = min(y_true[finite].min(), y_pred[finite].min())
            hi = max(y_true[finite].max(), y_pred[finite].max())
            if np.isclose(lo, hi):
                lo -= 1
                hi += 1
            ax.plot([lo, hi], [lo, hi], linestyle="--")

        metrics = calculate_metrics(y_true, y_pred)
        ax.set_title(f"ExtraTrees: Actual vs Predicted\n{target}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.text(
            0.05, 0.95,
            f"MAE = {metrics['MAE']:.3f}\n"
            f"RMSE = {metrics['RMSE']:.3f}\n"
            f"R² = {metrics['R2']:.3f}" if pd.notna(metrics["R2"])
            else f"MAE = {metrics['MAE']:.3f}\nRMSE = {metrics['RMSE']:.3f}\nR² = N/A",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", alpha=0.8),
        )
        fig.tight_layout()

        safe_name = (
            target.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        fig.savefig(
            OUTPUT_DIR / f"Actual_vs_Predicted_ExtraTrees_{safe_name}.png",
            dpi=300,
        )
        plt.close(fig)

    print("\nAnalysis completed.")
    print(f"Results folder: {OUTPUT_DIR}")
    print(f"Workbook: {output_file}")


if __name__ == "__main__":
    main()
