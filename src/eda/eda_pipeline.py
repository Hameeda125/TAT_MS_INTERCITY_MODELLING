from __future__ import annotations



import argparse
import sys
import warnings

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS / IMPORT BOOTSTRAP
# ============================================================

SCRIPT_VERSION = "2026-09-29_INTERCITY_EDA_V4_SINGULAR_MATRIX_SAFE"

# This file is expected at:
#   TAT_MS_INTERCITY_MODELLING/src/eda/eda_pipeline.py
#
# Derive the modelling-project root from the script location.
# Add that root to sys.path so BOTH of these commands work:
#   python .\src\eda\eda_pipeline.py
#   python -m src.eda.eda_pipeline
#
# This changes Python's import search path only. It does not
# modify any scientific data or STRICT/NONSTRICT classification.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_TEXT = str(PROJECT_ROOT)

if PROJECT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEXT)

from src.preprocessing.data_loader import load_excel

PART1_PROJECT_ROOT = (
    PROJECT_ROOT.parent
    / "TAT_MS_INTERCITY"
)

TRAINING_DATA_ROOT = (
    PART1_PROJECT_ROOT
    / "Data"
    / "Results"
    / "Master_Summary"
    / "Training_Data"
)

STRICT_INPUT = (
    TRAINING_DATA_ROOT
    / "EV_ICE_Training_STRICT.xlsx"
)

NONSTRICT_INPUT = (
    TRAINING_DATA_ROOT
    / "EV_ICE_Training_NONSTRICT.xlsx"
)

BASE_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "EDA"
)

# The active class-specific output root is assigned by
# run_eda() before any EDA stage writes a file.
OUTPUT_ROOT = BASE_OUTPUT_ROOT


# ============================================================

# FEATURE DEFINITIONS

# ============================================================



ICE_FEATURES = [

    "ICE Total Distance (km)",

    "ICE Total TAT (min)",

    "ICE Average Moving Speed (km/h)",

    "ICE CHARGING_REFUELING Time (min)",

    "ICE STOP Time (min)",

    "ICE IDLE Time (min)",

    "ICE INCHING Time (min)",

    "ICE ACC Time (min)",

    "ICE DEC Time (min)",

    "ICE CRUISE Time (min)",

    "ICE CHARGING_REFUELING Distance (km)",

    "ICE STOP Distance (km)",

    "ICE IDLE Distance (km)",

    "ICE INCHING Distance (km)",

    "ICE ACC Distance (km)",

    "ICE DEC Distance (km)",

    "ICE CRUISE Distance (km)",

]



EV_TARGETS = [

    "EV Total Distance (km)",

    "EV Total TAT (min)",

    "EV Average Moving Speed (km/h)",

    "EV CHARGING_REFUELING Time (min)",

    "EV STOP Time (min)",

    "EV IDLE Time (min)",

    "EV INCHING Time (min)",

    "EV ACC Time (min)",

    "EV DEC Time (min)",

    "EV CRUISE Time (min)",

    "EV CHARGING_REFUELING Distance (km)",

    "EV STOP Distance (km)",

    "EV IDLE Distance (km)",

    "EV INCHING Distance (km)",

    "EV ACC Distance (km)",

    "EV DEC Distance (km)",

    "EV CRUISE Distance (km)",

]



PHASES = [

    "CHARGING_REFUELING",

    "STOP",

    "IDLE",

    "INCHING",

    "ACC",

    "DEC",

    "CRUISE",

]



REQUIRED_METADATA = [

    "Pair ID",

    "Pair Class",

]





# ============================================================

# OUTPUT DIRECTORIES

# ============================================================



def create_output_directories() -> None:

    """

    Create the complete EDA output structure.

    """



    folders = [

        OUTPUT_ROOT / "01_Data_Audit",

        OUTPUT_ROOT / "02_Descriptive_Statistics",

        OUTPUT_ROOT / "03_Variability",

        OUTPUT_ROOT / "04_Correlation",

        OUTPUT_ROOT / "05_Physical_Consistency",

        OUTPUT_ROOT / "06_Outlier_Analysis",

        OUTPUT_ROOT / "07_ICE_EV_Relationships",

        OUTPUT_ROOT / "08_TAT_Comparison",

        OUTPUT_ROOT / "09_Baseline_Learnability",

        OUTPUT_ROOT / "10_Multicollinearity",

        OUTPUT_ROOT / "11_Research_Summary",

        OUTPUT_ROOT / "Figures" / "distributions",

        OUTPUT_ROOT / "Figures" / "correlations",

        OUTPUT_ROOT / "Figures" / "relationships",

        OUTPUT_ROOT / "Figures" / "variability",

        OUTPUT_ROOT / "Figures" / "physical_consistency",

        OUTPUT_ROOT / "Figures" / "tat",

    ]



    for folder in folders:

        folder.mkdir(

            parents=True,

            exist_ok=True,

        )





# ============================================================

# GENERAL HELPERS

# ============================================================



def safe_numeric(series: pd.Series) -> pd.Series:

    """

    Convert a series to numeric values.



    Non-numeric values become NaN.

    Original dataframe is never modified.

    """



    return pd.to_numeric(

        series,

        errors="coerce",

    )





def save_excel(

    dataframe: pd.DataFrame,

    path: Path,

    index: bool = False,

) -> None:

    """

    Save a dataframe to Excel.

    """



    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )



    dataframe.to_excel(

        path,

        index=index,

    )





def save_figure(path: Path) -> None:

    """

    Save and close the current matplotlib figure.

    """



    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )



    plt.tight_layout()



    plt.savefig(

        path,

        dpi=300,

        bbox_inches="tight",

    )



    plt.close()





def plot_correlation_heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: Path,
    *,
    annotate: bool,
    figsize: tuple[float, float],
) -> None:
    """
    Plot a correlation heatmap using matplotlib only.

    NaN cells are masked. Correlation values are displayed on the
    plot only when annotate=True.
    """

    values = matrix.to_numpy(
        dtype=float
    )

    masked_values = np.ma.masked_invalid(
        values
    )

    figure, axis = plt.subplots(
        figsize=figsize
    )

    image = axis.imshow(
        masked_values,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axis.set_xticks(
        np.arange(
            len(matrix.columns)
        )
    )

    axis.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    axis.set_xticklabels(
        matrix.columns,
        rotation=90,
    )

    axis.set_yticklabels(
        matrix.index,
    )

    axis.set_title(
        title
    )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    if annotate:

        for row_index in range(
            values.shape[0]
        ):

            for column_index in range(
                values.shape[1]
            ):

                value = values[
                    row_index,
                    column_index,
                ]

                if np.isfinite(value):

                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

    save_figure(
        path
    )



def check_required_columns(

    dataframe: pd.DataFrame,

    required_columns: list[str],

) -> list[str]:

    """

    Return required columns that are missing.

    """



    return [

        column

        for column in required_columns

        if column not in dataframe.columns

    ]





# ============================================================

# STRICT / NONSTRICT CLASS VALIDATION

# ============================================================


def validate_pair_class_for_eda(
    dataframe: pd.DataFrame,
    expected_pair_class: str,
) -> dict:
    """
    Validate the upstream Pair Class before EDA.

    IMPORTANT:
    - Pair Class is READ from Part 1.
    - Pair Class is VALIDATED against the selected workbook.
    - Pair Class is PRESERVED.
    - STRICT/NONSTRICT is never recalculated or overwritten.
    """

    normalized_expected = (
        str(expected_pair_class)
        .strip()
        .upper()
    )

    if normalized_expected not in {
        "STRICT",
        "NONSTRICT",
    }:

        raise ValueError(
            "expected_pair_class must be STRICT "
            "or NONSTRICT."
        )

    if "Pair Class" not in dataframe.columns:

        raise ValueError(
            "Required column 'Pair Class' "
            "is missing from the training data."
        )

    pair_class = (
        dataframe["Pair Class"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    missing_mask = (
        pair_class.isna()
        | pair_class.eq("")
    )

    valid_values = {
        "STRICT",
        "NONSTRICT",
    }

    observed_nonblank = pair_class[
        ~missing_mask
    ]

    invalid_values = sorted(
        set(observed_nonblank.dropna())
        - valid_values
    )

    mismatch_mask = (
        (~missing_mask)
        & pair_class.ne(
            normalized_expected
        )
    )

    counts = {
        str(key): int(value)
        for key, value
        in observed_nonblank.value_counts().items()
    }

    report = {
        "expected": normalized_expected,
        "counts": counts,
        "missing": int(
            missing_mask.sum()
        ),
        "invalid_values": invalid_values,
        "mismatch_count": int(
            mismatch_mask.sum()
        ),
    }

    if report["missing"] > 0:

        raise ValueError(
            "Missing Pair Class values detected: "
            f"{report['missing']}"
        )

    if report["invalid_values"]:

        raise ValueError(
            "Invalid Pair Class values detected: "
            f"{report['invalid_values']}"
        )

    if report["mismatch_count"] > 0:

        raise ValueError(
            "Pair Class mismatch detected. "
            f"Expected {normalized_expected}, "
            f"but {report['mismatch_count']} row(s) "
            "contain another class."
        )

    return report




# ============================================================

# 1. DATA AUDIT

# ============================================================



def run_data_audit(

    dataframe: pd.DataFrame,

) -> None:

    """

    Perform structural and data-quality audit.



    No data is removed or modified.

    """



    output_dir = OUTPUT_ROOT / "01_Data_Audit"



    modelling_columns = (

        ICE_FEATURES + EV_TARGETS

    )



    required_columns = (

        modelling_columns + REQUIRED_METADATA

    )



    missing_required = check_required_columns(

        dataframe,

        required_columns,

    )



    # --------------------------------------------------------

    # Overall dataset overview

    # --------------------------------------------------------



    audit_summary = pd.DataFrame(

        [

            {

                "Rows": int(len(dataframe)),

                "Columns": int(len(dataframe.columns)),

                "Unique Pair IDs": (

                    dataframe["Pair ID"].nunique()

                    if "Pair ID" in dataframe.columns

                    else np.nan

                ),

                "Duplicate Rows": int(

                    dataframe.duplicated().sum()

                ),

                "Duplicate Columns": int(

                    dataframe.columns.duplicated().sum()

                ),

                "Total Missing Cells": int(

                    dataframe.isna().sum().sum()

                ),

                "ICE Features": len(ICE_FEATURES),

                "EV Targets": len(EV_TARGETS),

                "Total Modelling Variables": len(

                    modelling_columns

                ),

                "Missing Required Columns": len(

                    missing_required

                ),

            }

        ]

    )



    save_excel(

        audit_summary,

        output_dir / "Dataset_Overview.xlsx",

    )



    # --------------------------------------------------------

    # Variable-level quality report

    # --------------------------------------------------------



    quality_records = []



    for column in modelling_columns:



        numeric = safe_numeric(

            dataframe[column]

        )



        numeric_array = numeric.to_numpy(

            dtype=float,

            na_value=np.nan,

        )



        infinite_mask = np.isinf(

            numeric_array

        )



        conversion_failure_mask = (

            dataframe[column].notna()

            & numeric.isna()

        )



        quality_records.append(

            {

                "Variable": column,

                "Group": (

                    "ICE"

                    if column.startswith("ICE ")

                    else "EV"

                ),

                "Original Data Type": str(

                    dataframe[column].dtype

                ),

                "Numeric Interpretable": bool(

                    not conversion_failure_mask.any()

                ),

                "Missing Values": int(

                    dataframe[column].isna().sum()

                ),

                "Numeric Conversion Failures": int(

                    conversion_failure_mask.sum()

                ),

                "Infinite Values": int(

                    infinite_mask.sum()

                ),

                "Unique Values": int(

                    dataframe[column].nunique(

                        dropna=True

                    )

                ),

                "Constant": (

                    dataframe[column].nunique(

                        dropna=True

                    ) <= 1

                ),

            }

        )



    quality_report = pd.DataFrame(

        quality_records

    )



    save_excel(

        quality_report,

        output_dir / "Data_Quality_Report.xlsx",

    )



    # --------------------------------------------------------

    # Pair ID report

    # --------------------------------------------------------



    if "Pair ID" in dataframe.columns:



        pair_report = (

            dataframe

            .groupby(

                "Pair ID",

                dropna=False,

            )

            .size()

            .reset_index(

                name="Rows"

            )

        )



        pair_report["One Row Per Pair"] = (

            pair_report["Rows"] == 1

        )



        save_excel(

            pair_report,

            output_dir / "Pair_ID_Report.xlsx",

        )



    # --------------------------------------------------------

    # Missing required columns

    # --------------------------------------------------------



    if missing_required:



        save_excel(

            pd.DataFrame(

                {

                    "Missing Required Column":

                        missing_required

                }

            ),

            output_dir / "Missing_Columns.xlsx",

        )





# ============================================================

# 2. DESCRIPTIVE STATISTICS

# ============================================================



def descriptive_statistics(

    dataframe: pd.DataFrame,

    columns: list[str],

) -> pd.DataFrame:

    """

    Calculate trip-level descriptive statistics.

    """



    records = []



    for column in columns:



        values = safe_numeric(

            dataframe[column]

        ).dropna()



        if len(values) == 0:



            records.append(

                {

                    "Variable": column,

                    "N": 0,

                    "Mean": np.nan,

                    "Median": np.nan,

                    "Std": np.nan,

                    "Min": np.nan,

                    "Q1": np.nan,

                    "Q3": np.nan,

                    "Max": np.nan,

                    "Range": np.nan,

                    "IQR": np.nan,

                    "Variance": np.nan,

                    "CV (%)": np.nan,

                    "Unique": 0,

                    "Zero Count": 0,

                    "Missing": int(

                        dataframe[column].isna().sum()

                    ),

                }

            )



            continue



        mean_value = values.mean()

        std_value = values.std()



        if (

            pd.notna(mean_value)

            and mean_value != 0

        ):

            cv = (

                abs(std_value / mean_value)

                * 100

            )

        else:

            cv = np.nan



        records.append(

            {

                "Variable": column,

                "N": int(len(values)),

                "Mean": mean_value,

                "Median": values.median(),

                "Std": std_value,

                "Min": values.min(),

                "Q1": values.quantile(0.25),

                "Q3": values.quantile(0.75),

                "Max": values.max(),

                "Range": (

                    values.max()

                    - values.min()

                ),

                "IQR": (

                    values.quantile(0.75)

                    - values.quantile(0.25)

                ),

                "Variance": values.var(),

                "CV (%)": cv,

                "Unique": int(

                    values.nunique()

                ),

                "Zero Count": int(

                    (values == 0).sum()

                ),

                "Missing": int(

                    dataframe[column].isna().sum()

                ),

            }

        )



    return pd.DataFrame(records)





def run_descriptive_statistics(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "02_Descriptive_Statistics"

    )



    ice_stats = descriptive_statistics(

        dataframe,

        ICE_FEATURES,

    )



    ev_stats = descriptive_statistics(

        dataframe,

        EV_TARGETS,

    )



    combined_stats = pd.concat(

        [

            ice_stats.assign(

                Group="ICE"

            ),

            ev_stats.assign(

                Group="EV"

            ),

        ],

        ignore_index=True,

    )



    save_excel(

        ice_stats,

        output_dir / "ICE_Statistics.xlsx",

    )



    save_excel(

        ev_stats,

        output_dir / "EV_Statistics.xlsx",

    )



    save_excel(

        combined_stats,

        output_dir / "Combined_Statistics.xlsx",

    )





# ============================================================

# 3. VARIABILITY ANALYSIS

# ============================================================



def run_variability_analysis(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "03_Variability"

    )



    records = []



    for column in (

        ICE_FEATURES + EV_TARGETS

    ):



        values = safe_numeric(

            dataframe[column]

        )



        unique_count = values.nunique(

            dropna=True

        )



        records.append(

            {

                "Variable": column,

                "Group": (

                    "ICE"

                    if column.startswith("ICE ")

                    else "EV"

                ),

                "Unique Values": unique_count,

                "Variance": values.var(),

                "Std": values.std(),

                "Mean": values.mean(),

                "Constant": (

                    unique_count <= 1

                ),

                "Near Constant": (

                    unique_count <= 2

                ),

            }

        )



    report = pd.DataFrame(records)



    save_excel(

        report,

        output_dir / "Variability_Report.xlsx",

    )



    # --------------------------------------------------------

    # Standard deviation visualization

    # --------------------------------------------------------



    plot_data = report.sort_values(

        "Std",

        ascending=False,

    )



    plt.figure(

        figsize=(12, 10)

    )



    plt.barh(

        plot_data["Variable"],

        plot_data["Std"],

    )



    plt.xlabel(

        "Standard Deviation"

    )



    plt.ylabel(

        "Variable"

    )



    plt.title(

        "Variability of Modelling Variables"

    )



    plt.gca().invert_yaxis()



    save_figure(

        OUTPUT_ROOT

        / "Figures"

        / "variability"

        / "Variable_Standard_Deviation.png"

    )





# ============================================================

# 4. SAFE CORRELATION

# ============================================================



def safe_pairwise_correlation(

    dataframe: pd.DataFrame,

    x_column: str,

    y_column: str,

    method: str,

) -> float:

    """

    Calculate a pairwise correlation safely.



    Correlation is not calculated when:

    - fewer than 2 valid paired observations exist

    - either variable is constant



    Constant variables remain in the dataset.

    """



    pair = dataframe[

        [x_column, y_column]

    ].apply(

        pd.to_numeric,

        errors="coerce",

    ).dropna()



    if len(pair) < 2:

        return np.nan



    x = pair[x_column]

    y = pair[y_column]



    if x.nunique() <= 1:

        return np.nan



    if y.nunique() <= 1:

        return np.nan



    if method == "pearson":



        return x.corr(

            y,

            method="pearson",

        )



    if method == "spearman":



        return x.corr(

            y,

            method="spearman",

        )



    raise ValueError(

        f"Unsupported correlation method: {method}"

    )





def calculate_correlation(

    dataframe: pd.DataFrame,

    columns: list[str],

    method: str,

) -> pd.DataFrame:

    """

    Calculate a within-group correlation matrix

    without generating warnings for constant variables.

    """



    correlation_matrix = pd.DataFrame(

        np.nan,

        index=columns,

        columns=columns,

        dtype=float,

    )



    for i, column_x in enumerate(columns):



        values_x = safe_numeric(

            dataframe[column_x]

        ).dropna()



        # Diagonal is defined only when the

        # variable has variability.

        if (

            len(values_x) >= 2

            and values_x.nunique() > 1

        ):

            correlation_matrix.loc[

                column_x,

                column_x,

            ] = 1.0



        for column_y in columns[i + 1:]:



            correlation = (

                safe_pairwise_correlation(

                    dataframe,

                    column_x,

                    column_y,

                    method,

                )

            )



            correlation_matrix.loc[

                column_x,

                column_y,

            ] = correlation



            correlation_matrix.loc[

                column_y,

                column_x,

            ] = correlation



    return correlation_matrix





def run_correlation_analysis(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "04_Correlation"

    )



    # --------------------------------------------------------

    # ICE ↔ ICE

    # --------------------------------------------------------



    ice_pearson = calculate_correlation(

        dataframe,

        ICE_FEATURES,

        "pearson",

    )



    ice_spearman = calculate_correlation(

        dataframe,

        ICE_FEATURES,

        "spearman",

    )



    # --------------------------------------------------------

    # EV ↔ EV

    # --------------------------------------------------------



    ev_pearson = calculate_correlation(

        dataframe,

        EV_TARGETS,

        "pearson",

    )



    ev_spearman = calculate_correlation(

        dataframe,

        EV_TARGETS,

        "spearman",

    )



    # --------------------------------------------------------

    # ICE → EV

    # --------------------------------------------------------



    ice_ev_pearson = pd.DataFrame(

        np.nan,

        index=ICE_FEATURES,

        columns=EV_TARGETS,

        dtype=float,

    )



    ice_ev_spearman = pd.DataFrame(

        np.nan,

        index=ICE_FEATURES,

        columns=EV_TARGETS,

        dtype=float,

    )



    ice_ev_status = pd.DataFrame(

        "",

        index=ICE_FEATURES,

        columns=EV_TARGETS,

        dtype=object,

    )



    for ice_column in ICE_FEATURES:



        for ev_column in EV_TARGETS:



            pair = dataframe[

                [ice_column, ev_column]

            ].apply(

                pd.to_numeric,

                errors="coerce",

            ).dropna()



            if len(pair) < 2:



                ice_ev_status.loc[

                    ice_column,

                    ev_column,

                ] = "INSUFFICIENT_DATA"



                continue



            ice_values = pair[

                ice_column

            ]



            ev_values = pair[

                ev_column

            ]



            if (

                ice_values.nunique() <= 1

                or ev_values.nunique() <= 1

            ):



                ice_ev_status.loc[

                    ice_column,

                    ev_column,

                ] = "CONSTANT_INPUT"



                continue



            ice_ev_pearson.loc[

                ice_column,

                ev_column,

            ] = ice_values.corr(

                ev_values,

                method="pearson",

            )



            ice_ev_spearman.loc[

                ice_column,

                ev_column,

            ] = ice_values.corr(

                ev_values,

                method="spearman",

            )



            ice_ev_status.loc[

                ice_column,

                ev_column,

            ] = "CALCULATED"



    # --------------------------------------------------------

    # Save correlation matrices

    # --------------------------------------------------------



    save_excel(

        ice_pearson,

        output_dir

        / "ICE_Pearson_Correlation.xlsx",

        index=True,

    )



    save_excel(

        ice_spearman,

        output_dir

        / "ICE_Spearman_Correlation.xlsx",

        index=True,

    )



    save_excel(

        ev_pearson,

        output_dir

        / "EV_Pearson_Correlation.xlsx",

        index=True,

    )



    save_excel(

        ev_spearman,

        output_dir

        / "EV_Spearman_Correlation.xlsx",

        index=True,

    )



    save_excel(

        ice_ev_pearson,

        output_dir

        / "ICE_EV_Pearson_Correlation.xlsx",

        index=True,

    )



    save_excel(

        ice_ev_spearman,

        output_dir

        / "ICE_EV_Spearman_Correlation.xlsx",

        index=True,

    )



    save_excel(

        ice_ev_status,

        output_dir

        / "ICE_EV_Correlation_Status.xlsx",

        index=True,

    )



    # --------------------------------------------------------

    # Correlation summary

    # --------------------------------------------------------



    records = []



    for ice_column in ICE_FEATURES:



        for ev_column in EV_TARGETS:



            pearson = ice_ev_pearson.loc[

                ice_column,

                ev_column,

            ]



            spearman = ice_ev_spearman.loc[

                ice_column,

                ev_column,

            ]



            status = ice_ev_status.loc[

                ice_column,

                ev_column,

            ]



            records.append(

                {

                    "ICE Variable": ice_column,

                    "EV Variable": ev_column,

                    "Status": status,

                    "Pearson": pearson,

                    "Spearman": spearman,

                    "Absolute Pearson": (

                        abs(pearson)

                        if pd.notna(pearson)

                        else np.nan

                    ),

                    "Absolute Spearman": (

                        abs(spearman)

                        if pd.notna(spearman)

                        else np.nan

                    ),

                }

            )



    correlation_summary = pd.DataFrame(

        records

    )



    correlation_summary = (

        correlation_summary.sort_values(

            "Absolute Pearson",

            ascending=False,

            na_position="last",

        )

    )



    save_excel(

        correlation_summary,

        output_dir

        / "ICE_EV_Correlation_Summary.xlsx",

    )



    # --------------------------------------------------------

    # Heatmap: ICE

    # --------------------------------------------------------



    plot_correlation_heatmap(

        ice_pearson,

        "ICE Feature Pearson Correlation",

        OUTPUT_ROOT

        / "Figures"

        / "correlations"

        / "ICE_Pearson_Correlation_Heatmap.png",

        annotate=True,

        figsize=(14, 12),

    )



    # --------------------------------------------------------

    # Heatmap: EV

    # --------------------------------------------------------



    plot_correlation_heatmap(

        ev_pearson,

        "EV Target Pearson Correlation",

        OUTPUT_ROOT

        / "Figures"

        / "correlations"

        / "EV_Pearson_Correlation_Heatmap.png",

        annotate=True,

        figsize=(14, 12),

    )



    # --------------------------------------------------------

    # Heatmap: ICE → EV

    # --------------------------------------------------------



    plot_correlation_heatmap(

        ice_ev_pearson,

        "ICE → EV Pearson Correlation",

        OUTPUT_ROOT

        / "Figures"

        / "correlations"

        / "ICE_EV_Pearson_Correlation_Heatmap.png",

        annotate=False,

        figsize=(16, 12),

    )



    print(

        "   Correlation analysis completed "

        "with constant-variable protection."

    )





# ============================================================

# 5. DISTRIBUTION ANALYSIS

# ============================================================



def run_distribution_analysis(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "Figures"

        / "distributions"

    )



    all_columns = (

        ICE_FEATURES + EV_TARGETS

    )



    for column in all_columns:



        values = safe_numeric(

            dataframe[column]

        ).dropna()



        if len(values) == 0:

            continue



        plt.figure(

            figsize=(8, 5)

        )



        # KDE is deliberately not used as the

        # primary representation because the current

        # dataset contains very few trips.



        plt.hist(

            values,

            bins=min(

                10,

                max(

                    3,

                    len(values),

                ),

            ),

        )



        plt.xlabel(column)

        plt.ylabel("Number of Trips")



        plt.title(

            f"Distribution: {column}"

        )



        filename = (

            column

            .replace("/", "_")

            .replace("\\\\", "_")

            .replace(":", "_")

        )



        save_figure(

            output_dir

            / f"{filename}.png"

        )





# ============================================================

# 6. OUTLIER ANALYSIS

# ============================================================



def run_outlier_analysis(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "06_Outlier_Analysis"

    )



    records = []



    for column in (

        ICE_FEATURES + EV_TARGETS

    ):



        values = safe_numeric(

            dataframe[column]

        ).dropna()



        if len(values) == 0:

            continue



        q1 = values.quantile(0.25)

        q3 = values.quantile(0.75)



        iqr = q3 - q1



        lower_bound = (

            q1 - 1.5 * iqr

        )



        upper_bound = (

            q3 + 1.5 * iqr

        )



        outlier_mask = (

            (values < lower_bound)

            | (values > upper_bound)

        )



        records.append(

            {

                "Variable": column,

                "Q1": q1,

                "Q3": q3,

                "IQR": iqr,

                "Lower Bound": lower_bound,

                "Upper Bound": upper_bound,

                "Outlier Count": int(

                    outlier_mask.sum()

                ),

                "Outlier Percentage": float(

                    outlier_mask.mean()

                    * 100

                ),

            }

        )



        # ----------------------------------------------------

        # Boxplot

        # ----------------------------------------------------



        plt.figure(

            figsize=(8, 5)

        )



        plt.boxplot(
            values,
            orientation="horizontal",
        )



        plt.xlabel(column)



        plt.title(

            f"Outlier Screening: {column}"

        )



        filename = (

            column

            .replace("/", "_")

            .replace("\\\\", "_")

            .replace(":", "_")

        )



        save_figure(

            OUTPUT_ROOT

            / "Figures"

            / "variability"

            / f"Boxplot_{filename}.png"

        )



    report = pd.DataFrame(records)



    save_excel(

        report,

        output_dir / "Outlier_Report.xlsx",

    )





# ============================================================

# 7. ICE → EV RELATIONSHIPS

# ============================================================



def run_relationship_analysis(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "07_ICE_EV_Relationships"

    )



    records = []



    # --------------------------------------------------------

    # All ICE → EV relationships

    # --------------------------------------------------------



    for ice_column in ICE_FEATURES:



        for ev_column in EV_TARGETS:



            pair = dataframe[

                [ice_column, ev_column]

            ].apply(

                pd.to_numeric,

                errors="coerce",

            ).dropna()



            if len(pair) < 2:

                continue



            x = pair[ice_column]

            y = pair[ev_column]



            if (

                x.nunique() <= 1

                or y.nunique() <= 1

            ):

                continue



            pearson = x.corr(

                y,

                method="pearson",

            )



            spearman = x.corr(

                y,

                method="spearman",

            )



            records.append(

                {

                    "ICE Variable": ice_column,

                    "EV Variable": ev_column,

                    "N": int(len(pair)),

                    "Pearson": pearson,

                    "Spearman": spearman,

                    "Absolute Pearson": abs(

                        pearson

                    ),

                    "Absolute Spearman": abs(

                        spearman

                    ),

                }

            )



    relationship_report = pd.DataFrame(

        records

    )



    if not relationship_report.empty:



        relationship_report = (

            relationship_report.sort_values(

                "Absolute Pearson",

                ascending=False,

            )

        )



    save_excel(

        relationship_report,

        output_dir / "Relationship_Report.xlsx",

    )



    # --------------------------------------------------------

    # Corresponding ICE → EV relationships

    # --------------------------------------------------------



    corresponding_records = []



    for ice_column, ev_column in zip(

        ICE_FEATURES,

        EV_TARGETS,

    ):



        pair = dataframe[

            [ice_column, ev_column]

        ].apply(

            pd.to_numeric,

            errors="coerce",

        ).dropna()



        if len(pair) < 2:

            continue



        x = pair[ice_column]

        y = pair[ev_column]



        if (

            x.nunique() <= 1

            or y.nunique() <= 1

        ):



            corresponding_records.append(

                {

                    "ICE Variable": ice_column,

                    "EV Variable": ev_column,

                    "N": int(len(pair)),

                    "Pearson": np.nan,

                    "Spearman": np.nan,

                    "Status": "CONSTANT_INPUT",

                }

            )



            continue



        corresponding_records.append(

            {

                "ICE Variable": ice_column,

                "EV Variable": ev_column,

                "N": int(len(pair)),

                "Pearson": x.corr(

                    y,

                    method="pearson",

                ),

                "Spearman": x.corr(

                    y,

                    method="spearman",

                ),

                "Status": "CALCULATED",

            }

        )



    corresponding_report = pd.DataFrame(

        corresponding_records

    )



    save_excel(

        corresponding_report,

        output_dir

        / "Corresponding_ICE_EV_Relationships.xlsx",

    )



    # --------------------------------------------------------

    # Top ICE → EV relationships

    # --------------------------------------------------------



    if relationship_report.empty:

        return



    top_relationships = (

        relationship_report.head(15)

    )



    for number, row in enumerate(

        top_relationships.itertuples(

            index=False

        ),

        start=1,

    ):



        ice_column = row[0]

        ev_column = row[1]

        pearson_value = row[3]



        x = safe_numeric(

            dataframe[ice_column]

        )



        y = safe_numeric(

            dataframe[ev_column]

        )



        valid = pd.concat(

            [x, y],

            axis=1,

        ).dropna()



        if len(valid) < 2:

            continue



        if (

            valid.iloc[:, 0].nunique()

            <= 1

            or valid.iloc[:, 1].nunique()

            <= 1

        ):

            continue



        plt.figure(

            figsize=(7, 6)

        )



        x_values = valid.iloc[:, 0]

        y_values = valid.iloc[:, 1]



        plt.scatter(

            x_values,

            y_values,

            s=60,

        )



        # Fit the simple diagnostic trend line explicitly instead of
        # np.polyfit().  This avoids poorly-conditioned RankWarning
        # messages on very small EDA samples while preserving the same
        # ordinary least-squares straight-line interpretation.
        x_array = x_values.to_numpy(dtype=float)
        y_array = y_values.to_numpy(dtype=float)

        x_mean = float(np.mean(x_array))
        y_mean = float(np.mean(y_array))
        x_centered = x_array - x_mean

        denominator = float(
            np.dot(x_centered, x_centered)
        )

        if denominator <= 0:
            plt.close()
            continue

        slope = float(
            np.dot(
                x_centered,
                y_array - y_mean,
            )
            / denominator
        )

        intercept = float(
            y_mean - slope * x_mean
        )



        regression_x = np.linspace(

            x_values.min(),

            x_values.max(),

            100,

        )



        regression_y = (

            slope * regression_x

            + intercept

        )



        plt.plot(

            regression_x,

            regression_y,

        )



        plt.xlabel(

            ice_column

        )



        plt.ylabel(

            ev_column

        )



        plt.title(

            f"ICE → EV Relationship #{number}\n"

            f"Pearson r = {pearson_value:.3f}"

        )



        save_figure(

            OUTPUT_ROOT

            / "Figures"

            / "relationships"

            / f"{number:02d}_ICE_EV_relationship.png"

        )





# ============================================================

# 8. PHYSICAL CONSISTENCY

# ============================================================



def run_physical_consistency_analysis(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "05_Physical_Consistency"

    )



    records = []



    for vehicle in ["ICE", "EV"]:



        total_distance_column = (

            f"{vehicle} Total Distance (km)"

        )



        total_tat_column = (

            f"{vehicle} Total TAT (min)"

        )



        distance_columns = [

            f"{vehicle} {phase} Distance (km)"

            for phase in PHASES

        ]



        time_columns = [

            f"{vehicle} {phase} Time (min)"

            for phase in PHASES

        ]



        for index, row in dataframe.iterrows():



            phase_distance_values = [

                pd.to_numeric(

                    row[column],

                    errors="coerce",

                )

                for column in distance_columns

            ]



            phase_time_values = [

                pd.to_numeric(

                    row[column],

                    errors="coerce",

                )

                for column in time_columns

            ]



            phase_distance_sum = np.nansum(

                phase_distance_values

            )



            phase_time_sum = np.nansum(

                phase_time_values

            )



            total_distance = pd.to_numeric(

                row[total_distance_column],

                errors="coerce",

            )



            total_tat = pd.to_numeric(

                row[total_tat_column],

                errors="coerce",

            )



            distance_difference = (

                total_distance

                - phase_distance_sum

            )



            tat_difference = (

                total_tat

                - phase_time_sum

            )



            records.append(

                {

                    "Pair ID": (

                        row["Pair ID"]

                        if "Pair ID"

                        in dataframe.columns

                        else index

                    ),

                    "Vehicle": vehicle,

                    "Total Distance (km)": (

                        total_distance

                    ),

                    "Phase Distance Sum (km)": (

                        phase_distance_sum

                    ),

                    "Distance Difference (km)": (

                        distance_difference

                    ),

                    "Total TAT (min)": total_tat,

                    "Phase Time Sum (min)": (

                        phase_time_sum

                    ),

                    "TAT Difference (min)": (

                        tat_difference

                    ),

                }

            )



    report = pd.DataFrame(

        records

    )



    save_excel(

        report,

        output_dir

        / "Physical_Consistency_Report.xlsx",

    )



    # --------------------------------------------------------

    # Distance consistency visualization

    # --------------------------------------------------------



    for vehicle in ["ICE", "EV"]:



        subset = report[

            report["Vehicle"] == vehicle

        ]



        if subset.empty:

            continue



        plt.figure(

            figsize=(9, 6)

        )



        plt.scatter(

            subset["Total Distance (km)"],

            subset["Phase Distance Sum (km)"],

            s=70,

        )



        minimum = min(

            subset["Total Distance (km)"].min(),

            subset["Phase Distance Sum (km)"].min(),

        )



        maximum = max(

            subset["Total Distance (km)"].max(),

            subset["Phase Distance Sum (km)"].max(),

        )



        plt.plot(

            [minimum, maximum],

            [minimum, maximum],

            linestyle="--",

        )



        plt.xlabel(

            "Total Distance (km)"

        )



        plt.ylabel(

            "Sum of Phase Distances (km)"

        )



        plt.title(

            f"{vehicle}: Total Distance vs "

            "Phase Distance Sum"

        )



        save_figure(

            OUTPUT_ROOT

            / "Figures"

            / "physical_consistency"

            / f"{vehicle}_Distance_Consistency.png"

        )





# ============================================================

# 9. TAT COMPARISON

# ============================================================



def run_tat_comparison(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "08_TAT_Comparison"

    )



    records = []



    # --------------------------------------------------------

    # Overall paired TAT comparison

    # --------------------------------------------------------



    for index, row in dataframe.iterrows():



        pair_id = (

            row["Pair ID"]

            if "Pair ID"

            in dataframe.columns

            else index

        )



        ice_tat = pd.to_numeric(

            row["ICE Total TAT (min)"],

            errors="coerce",

        )



        ev_tat = pd.to_numeric(

            row["EV Total TAT (min)"],

            errors="coerce",

        )



        if (

            pd.notna(ice_tat)

            and pd.notna(ev_tat)

        ):



            difference = (

                ev_tat - ice_tat

            )



            absolute_difference = abs(

                difference

            )



            if ice_tat != 0:



                percentage_difference = (

                    difference

                    / ice_tat

                    * 100

                )



            else:



                percentage_difference = np.nan



            if difference > 0:

                higher_vehicle = "EV"



            elif difference < 0:

                higher_vehicle = "ICE"



            else:

                higher_vehicle = "Equal"



        else:



            difference = np.nan

            absolute_difference = np.nan

            percentage_difference = np.nan

            higher_vehicle = "Not available"



        records.append(

            {

                "Pair ID": pair_id,

                "ICE Total TAT (min)": ice_tat,

                "EV Total TAT (min)": ev_tat,

                "EV - ICE TAT Difference (min)": (

                    difference

                ),

                "Absolute TAT Difference (min)": (

                    absolute_difference

                ),

                "TAT Difference (%)": (

                    percentage_difference

                ),

                "Higher TAT": higher_vehicle,

            }

        )



    tat_report = pd.DataFrame(

        records

    )



    save_excel(

        tat_report,

        output_dir / "TAT_Comparison.xlsx",

    )



    # --------------------------------------------------------

    # Overall TAT summary

    # --------------------------------------------------------



    summary = pd.DataFrame(

        [

            {

                "Number of Pairs": len(

                    tat_report

                ),

                "Mean ICE TAT (min)": (

                    tat_report[

                        "ICE Total TAT (min)"

                    ].mean()

                ),

                "Mean EV TAT (min)": (

                    tat_report[

                        "EV Total TAT (min)"

                    ].mean()

                ),

                "Mean EV - ICE Difference (min)": (

                    tat_report[

                        "EV - ICE TAT Difference (min)"

                    ].mean()

                ),

                "Median EV - ICE Difference (min)": (

                    tat_report[

                        "EV - ICE TAT Difference (min)"

                    ].median()

                ),

                "EV Higher TAT Count": int(

                    (

                        tat_report["Higher TAT"]

                        == "EV"

                    ).sum()

                ),

                "ICE Higher TAT Count": int(

                    (

                        tat_report["Higher TAT"]

                        == "ICE"

                    ).sum()

                ),

                "Equal TAT Count": int(

                    (

                        tat_report["Higher TAT"]

                        == "Equal"

                    ).sum()

                ),

            }

        ]

    )



    save_excel(

        summary,

        output_dir

        / "TAT_Comparison_Summary.xlsx",

    )



    # --------------------------------------------------------

    # Paired TAT bar chart

    # --------------------------------------------------------



    if not tat_report.empty:



        x = np.arange(

            len(tat_report)

        )



        width = 0.35



        plt.figure(

            figsize=(12, 6)

        )



        plt.bar(

            x - width / 2,

            tat_report[

                "ICE Total TAT (min)"

            ],

            width,

            label="ICE",

        )



        plt.bar(

            x + width / 2,

            tat_report[

                "EV Total TAT (min)"

            ],

            width,

            label="EV",

        )



        plt.xlabel(

            "Paired Trip"

        )



        plt.ylabel(

            "Total TAT (min)"

        )



        plt.title(

            "Paired ICE vs EV Total TAT"

        )



        plt.xticks(

            x,

            tat_report[

                "Pair ID"

            ].astype(str),

            rotation=45,

            ha="right",

        )



        plt.legend()



        save_figure(

            OUTPUT_ROOT

            / "Figures"

            / "tat"

            / "Paired_ICE_EV_Total_TAT.png"

        )



    # --------------------------------------------------------

    # Phase-wise TAT comparison

    # --------------------------------------------------------



    phase_records = []



    for index, row in dataframe.iterrows():



        pair_id = (

            row["Pair ID"]

            if "Pair ID"

            in dataframe.columns

            else index

        )



        for phase in PHASES:



            ice_column = (

                f"ICE {phase} Time (min)"

            )



            ev_column = (

                f"EV {phase} Time (min)"

            )



            ice_time = pd.to_numeric(

                row[ice_column],

                errors="coerce",

            )



            ev_time = pd.to_numeric(

                row[ev_column],

                errors="coerce",

            )



            phase_records.append(

                {

                    "Pair ID": pair_id,

                    "Phase": phase,

                    "ICE Time (min)": ice_time,

                    "EV Time (min)": ev_time,

                    "EV - ICE Difference (min)": (

                        ev_time - ice_time

                    ),

                }

            )



    phase_report = pd.DataFrame(

        phase_records

    )



    save_excel(

        phase_report,

        output_dir

        / "Phase_Wise_TAT_Comparison.xlsx",

    )



    phase_summary = (

        phase_report

        .groupby("Phase")

        .agg(

            {

                "ICE Time (min)": "mean",

                "EV Time (min)": "mean",

                "EV - ICE Difference (min)": "mean",

            }

        )

        .reset_index()

    )



    save_excel(

        phase_summary,

        output_dir

        / "Phase_Wise_TAT_Summary.xlsx",

    )



    # --------------------------------------------------------

    # Phase-wise TAT difference plot

    # --------------------------------------------------------



    plt.figure(

        figsize=(12, 6)

    )



    plt.bar(

        phase_summary["Phase"],

        phase_summary[

            "EV - ICE Difference (min)"

        ],

    )



    plt.axhline(

        0,

        linestyle="--",

    )



    plt.xlabel(

        "Driving Phase"

    )



    plt.ylabel(

        "Mean EV - ICE Time Difference (min)"

    )



    plt.title(

        "Phase-wise EV vs ICE Time Difference"

    )



    plt.xticks(

        rotation=30

    )



    save_figure(

        OUTPUT_ROOT

        / "Figures"

        / "tat"

        / "Phase_Wise_EV_minus_ICE_Time.png"

    )





# ============================================================

# 10. BASELINE LEARNABILITY

# ============================================================



def run_baseline_learnability(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "09_Baseline_Learnability"

    )



    records = []



    for target in EV_TARGETS:



        values = safe_numeric(

            dataframe[target]

        ).dropna()



        if len(values) == 0:

            continue



        mean_target = values.mean()



        baseline_mae = np.mean(

            np.abs(

                values

                - mean_target

            )

        )



        baseline_rmse = np.sqrt(

            np.mean(

                (

                    values

                    - mean_target

                ) ** 2

            )

        )



        records.append(

            {

                "Target": target,

                "N": int(len(values)),

                "Target Mean": mean_target,

                "Target Std": values.std(),

                "Target Variance": values.var(),

                "Mean Baseline MAE": (

                    baseline_mae

                ),

                "Mean Baseline RMSE": (

                    baseline_rmse

                ),

                "R2 Baseline": np.nan,

                "Interpretation": (

                    "Reference baseline only; "

                    "not a model-validation result."

                ),

            }

        )



    report = pd.DataFrame(

        records

    )



    save_excel(

        report,

        output_dir

        / "Target_Learnability_Baseline.xlsx",

    )





# ============================================================

# 11. MULTICOLLINEARITY / VIF

# ============================================================



def run_multicollinearity_analysis(

    dataframe: pd.DataFrame,

) -> None:

    """
    Run a diagnostic VIF analysis on ICE predictors.

    This implementation deliberately avoids direct matrix inversion and
    statsmodels' VIF helper.  Instead, each auxiliary regression is solved
    with NumPy SVD-based least squares (np.linalg.lstsq), which remains
    stable when the predictor matrix is singular or rank deficient.

    Constant predictors are retained in the source dataset but excluded
    from VIF estimation because VIF is undefined for a constant predictor.

    Perfect multicollinearity is reported as infinite VIF rather than
    raising a singular-matrix exception.  No modelling column is deleted,
    altered, imputed, or reclassified.
    """

    output_dir = (

        OUTPUT_ROOT

        / "10_Multicollinearity"

    )

    numeric_data = dataframe[

        ICE_FEATURES

    ].apply(

        pd.to_numeric,

        errors="coerce",

    )

    # Work on an EDA-only numeric copy.  The source dataframe is untouched.
    # Non-finite values cannot participate in a least-squares VIF estimate.
    numeric_data = (

        numeric_data

        .replace([np.inf, -np.inf], np.nan)

        .dropna()

    )

    constant_columns = [

        column

        for column in numeric_data.columns

        if numeric_data[column].nunique(

            dropna=True

        ) <= 1

    ]

    active_columns = [

        column

        for column in numeric_data.columns

        if column not in constant_columns

    ]

    active_data = numeric_data[

        active_columns

    ].copy()

    n_rows = int(

        active_data.shape[0]

    )

    n_columns = int(

        active_data.shape[1]

    )

    records = []

    for column in constant_columns:

        records.append(

            {

                "Variable": column,

                "VIF": np.nan,

                "R2 Auxiliary": np.nan,

                "Status": "CONSTANT_INPUT",

                "Observations": n_rows,

                "Predictors Used": n_columns,

                "Auxiliary Rank": np.nan,

                "Auxiliary Parameters": np.nan,

                "Reason": (

                    "Constant predictor retained in the "

                    "dataset but excluded from VIF "

                    "estimation because VIF is undefined."

                ),

            }

        )

    if n_columns == 0:

        records.append(

            {

                "Variable": np.nan,

                "VIF": np.nan,

                "R2 Auxiliary": np.nan,

                "Status": "NOT MEANINGFUL",

                "Observations": n_rows,

                "Predictors Used": n_columns,

                "Auxiliary Rank": np.nan,

                "Auxiliary Parameters": np.nan,

                "Reason": (

                    "No non-constant ICE predictors are "

                    "available for VIF estimation."

                ),

            }

        )

        save_excel(

            pd.DataFrame(records),

            output_dir

            / "VIF_Report.xlsx",

        )

        return

    # With p predictors, each auxiliary regression uses p-1 predictors
    # plus an intercept.  Requiring n > p avoids zero residual degrees
    # of freedom and prevents a misleading VIF calculation.
    if n_rows <= n_columns:

        for column in active_columns:

            records.append(

                {

                    "Variable": column,

                    "VIF": np.nan,

                    "R2 Auxiliary": np.nan,

                    "Status": "NOT MEANINGFUL",

                    "Observations": n_rows,

                    "Predictors Used": n_columns,

                    "Auxiliary Rank": np.nan,

                    "Auxiliary Parameters": n_columns,

                    "Reason": (

                        "Observations are not greater than "

                        "the number of non-constant predictors. "

                        "VIF is deferred until more paired trips "

                        "are available."

                    ),

                }

            )

        save_excel(

            pd.DataFrame(records),

            output_dir

            / "VIF_Report.xlsx",

        )

        return

    # Standardization improves numerical conditioning without changing R2
    # or VIF.  It is performed only on this temporary diagnostic copy.
    means = active_data.mean(axis=0)

    standard_deviations = active_data.std(

        axis=0,

        ddof=0,

    )

    standardized = (

        active_data - means

    ) / standard_deviations

    standardized_matrix = standardized.to_numpy(

        dtype=float

    )

    # Global matrix diagnostics.  Rank deficiency is informative but is
    # not treated as an exception: individual VIFs below are obtained by
    # SVD-based least squares and perfect dependence is reported as INF.
    global_rank = int(

        np.linalg.matrix_rank(

            standardized_matrix

        )

    )

    singular_values = np.linalg.svd(

        standardized_matrix,

        compute_uv=False,

    )

    if singular_values.size == 0:

        condition_number = np.nan

    else:

        largest_sv = float(

            singular_values[0]

        )

        smallest_sv = float(

            singular_values[-1]

        )

        sv_tolerance = (

            np.finfo(float).eps

            * max(

                standardized_matrix.shape

            )

            * max(largest_sv, 1.0)

        )

        if smallest_sv <= sv_tolerance:

            condition_number = np.inf

        else:

            condition_number = (

                largest_sv / smallest_sv

            )

    diagnostics = pd.DataFrame(

        [

            {

                "Observations": n_rows,

                "Non-Constant Predictors": n_columns,

                "Matrix Rank": global_rank,

                "Rank Deficient": (

                    global_rank < n_columns

                ),

                "Condition Number": condition_number,

                "Method": (

                    "Standardized SVD-based least squares; "

                    "no direct matrix inversion"

                ),

            }

        ]

    )

    save_excel(

        diagnostics,

        output_dir

        / "VIF_Matrix_Diagnostics.xlsx",

    )

    perfect_tolerance = 1e-10

    for index, column in enumerate(

        active_columns

    ):

        y = standardized_matrix[:, index]

        x_without_target = np.delete(

            standardized_matrix,

            index,

            axis=1,

        )

        design = np.column_stack(

            [

                np.ones(

                    n_rows,

                    dtype=float,

                ),

                x_without_target,

            ]

        )

        auxiliary_parameters = int(

            design.shape[1]

        )

        try:

            coefficients, _, auxiliary_rank, _ = (

                np.linalg.lstsq(

                    design,

                    y,

                    rcond=None,

                )

            )

            fitted = design @ coefficients

            residual = y - fitted

            sse = float(

                np.dot(

                    residual,

                    residual,

                )

            )

            centered_y = y - float(

                np.mean(y)

            )

            sst = float(

                np.dot(

                    centered_y,

                    centered_y,

                )

            )

            if sst <= np.finfo(float).eps:

                vif_value = np.nan

                r_squared = np.nan

                status = "UNDEFINED"

                reason = (

                    "Auxiliary target has effectively zero "

                    "variance after standardization."

                )

            else:

                r_squared = float(

                    1.0 - (sse / sst)

                )

                # Clamp tiny floating-point excursions only.
                if r_squared < 0 and abs(

                    r_squared

                ) <= perfect_tolerance:

                    r_squared = 0.0

                if r_squared > 1 and abs(

                    r_squared - 1.0

                ) <= perfect_tolerance:

                    r_squared = 1.0

                one_minus_r2 = (

                    1.0 - r_squared

                )

                if one_minus_r2 <= perfect_tolerance:

                    vif_value = np.inf

                    status = "INFINITE"

                    reason = (

                        "Perfect or numerically perfect "

                        "multicollinearity detected. VIF is "

                        "reported as infinite; no matrix "

                        "inversion was attempted."

                    )

                else:

                    vif_value = float(

                        1.0 / one_minus_r2

                    )

                    if (

                        int(auxiliary_rank)

                        < auxiliary_parameters

                    ):

                        status = (

                            "CALCULATED_RANK_DEFICIENT"

                        )

                        reason = (

                            "Auxiliary design is rank deficient, "

                            "but SVD least squares produced a "

                            "stable fitted-value R2 without direct "

                            "matrix inversion."

                        )

                    else:

                        status = "CALCULATED"

                        reason = ""

        except np.linalg.LinAlgError as error:

            vif_value = np.nan

            r_squared = np.nan

            auxiliary_rank = np.nan

            status = "NUMERICAL_FAILURE"

            reason = (

                "SVD least-squares estimation failed: "

                f"{error}"

            )

        records.append(

            {

                "Variable": column,

                "VIF": vif_value,

                "R2 Auxiliary": r_squared,

                "Status": status,

                "Observations": n_rows,

                "Predictors Used": n_columns,

                "Auxiliary Rank": auxiliary_rank,

                "Auxiliary Parameters": (

                    auxiliary_parameters

                ),

                "Reason": reason,

            }

        )

    vif_report = pd.DataFrame(

        records

    )

    save_excel(

        vif_report,

        output_dir

        / "VIF_Report.xlsx",

    )



# ============================================================

# 12. RESEARCH SUMMARY

# ============================================================



def create_research_summary(

    dataframe: pd.DataFrame,

) -> None:



    output_dir = (

        OUTPUT_ROOT

        / "11_Research_Summary"

    )



    ev_tat = safe_numeric(

        dataframe[

            "EV Total TAT (min)"

        ]

    )



    ice_tat = safe_numeric(

        dataframe[

            "ICE Total TAT (min)"

        ]

    )



    tat_difference = (

        ev_tat - ice_tat

    )



    summary_records = [

        {

            "Metric": (

                "Number of paired trips"

            ),

            "Value": int(

                len(dataframe)

            ),

            "Interpretation": (

                "Number of independent paired "

                "trips available for summary-level "

                "analysis."

            ),

        },

        {

            "Metric": (

                "Number of ICE predictors"

            ),

            "Value": len(

                ICE_FEATURES

            ),

            "Interpretation": (

                "Full-model ICE input variables."

            ),

        },

        {

            "Metric": (

                "Number of EV targets"

            ),

            "Value": len(

                EV_TARGETS

            ),

            "Interpretation": (

                "Full-model EV output variables."

            ),

        },

        {

            "Metric": (

                "Mean EV - ICE Total TAT Difference (min)"

            ),

            "Value": tat_difference.mean(),

            "Interpretation": (

                "Positive values indicate higher "

                "EV TAT on average; negative values "

                "indicate higher ICE TAT."

            ),

        },

        {

            "Metric": (

                "Median EV - ICE Total TAT Difference (min)"

            ),

            "Value": tat_difference.median(),

            "Interpretation": (

                "Median paired TAT difference."

            ),

        },

    ]



    summary = pd.DataFrame(

        summary_records

    )



    save_excel(

        summary,

        output_dir

        / "EDA_Research_Summary.xlsx",

    )





# ============================================================

# MAIN EDA PIPELINE

# ============================================================


def run_eda(

    input_path: Path,

    expected_pair_class: str,

) -> None:

    """
    Run the complete diagnostic EDA for one pair class.

    STRICT and NONSTRICT are analysed separately and written to
    separate output folders. Pair Class is validated only; it is
    never recalculated or overwritten.
    """



    global OUTPUT_ROOT



    normalized_class = (

        str(expected_pair_class)

        .strip()

        .upper()

    )



    if normalized_class not in {

        "STRICT",

        "NONSTRICT",

    }:



        raise ValueError(

            "expected_pair_class must be "

            "STRICT or NONSTRICT."

        )



    OUTPUT_ROOT = (

        BASE_OUTPUT_ROOT

        / normalized_class

    )



    print()

    print("=" * 70)

    print(

        "INTERCITY EV–ICE EDA PIPELINE"

    )

    print("=" * 70)



    print(

        f"Version   : {SCRIPT_VERSION}"

    )



    print(

        f"Pair Class: {normalized_class}"

    )



    print()

    print("Input file:")

    print(input_path)



    print()

    print("Output directory:")

    print(OUTPUT_ROOT)



    if not input_path.exists():



        raise FileNotFoundError(

            "EDA input workbook was not found:\n"

            f"{input_path}"

        )



    if not input_path.is_file():



        raise ValueError(

            "EDA input path is not a file:\n"

            f"{input_path}"

        )



    print()

    print("Loading dataset...")



    dataframe = load_excel(

        input_path,

        sheet_name="01_Training_Data",

        header=3,

    )



    print(

        f"Loaded {len(dataframe)} rows "

        f"and {len(dataframe.columns)} columns."

    )



    # --------------------------------------------------------

    # Structural validation before EDA

    # --------------------------------------------------------



    required_columns = (

        ICE_FEATURES

        + EV_TARGETS

        + REQUIRED_METADATA

    )



    missing_columns = check_required_columns(

        dataframe,

        required_columns,

    )



    if missing_columns:



        raise ValueError(

            "Required columns are missing "

            "from the training data:\n"

            + "\n".join(

                f"  - {column}"

                for column in missing_columns

            )

        )



    pair_class_report = (

        validate_pair_class_for_eda(

            dataframe,

            normalized_class,

        )

    )



    print()

    print("STRICT / NONSTRICT validation:")

    print(

        f"  Expected         : "

        f"{pair_class_report['expected']}"

    )

    print(

        f"  Observed counts  : "

        f"{pair_class_report['counts']}"

    )

    print(

        f"  Missing          : "

        f"{pair_class_report['missing']}"

    )

    print(

        f"  Invalid values   : "

        f"{pair_class_report['invalid_values']}"

    )

    print(

        f"  Class mismatches : "

        f"{pair_class_report['mismatch_count']}"

    )

    print(

        "  Handling         : "

        "READ / VALIDATE / PRESERVE ONLY"

    )

    print(

        "  Recalculation    : NONE"

    )

    print(

        "  Overwrite        : NONE"

    )



    # Output folders are created only after the input has passed

    # required-column and Pair Class validation.

    create_output_directories()



    if len(dataframe) < 30:

        print()
        print(
            "EDA NOTE: This dataset has fewer than "
            "30 paired observations."
        )
        print(
            "Correlation and regression-style EDA "
            "outputs are diagnostic/exploratory only."
        )



    print()

    print("Running EDA stages...")



    print(

        "1/11  Data audit"

    )

    run_data_audit(

        dataframe

    )



    print(

        "2/11  Descriptive statistics"

    )

    run_descriptive_statistics(

        dataframe

    )



    print(

        "3/11  Variability analysis"

    )

    run_variability_analysis(

        dataframe

    )



    print(

        "4/11  Correlation analysis"

    )

    run_correlation_analysis(

        dataframe

    )



    print(

        "5/11  Distribution analysis"

    )

    run_distribution_analysis(

        dataframe

    )



    print(

        "6/11  Outlier analysis"

    )

    run_outlier_analysis(

        dataframe

    )



    print(

        "7/11  ICE → EV relationship analysis"

    )

    run_relationship_analysis(

        dataframe

    )



    print(

        "8/11  Physical consistency analysis"

    )

    run_physical_consistency_analysis(

        dataframe

    )



    print(

        "9/11  TAT comparison"

    )

    run_tat_comparison(

        dataframe

    )



    print(

        "10/11 Baseline learnability"

    )

    run_baseline_learnability(

        dataframe

    )



    print(

        "11/11 Multicollinearity analysis"

    )

    run_multicollinearity_analysis(

        dataframe

    )



    create_research_summary(

        dataframe

    )



    print()

    print("=" * 70)

    print(

        f"EDA COMPLETED SUCCESSFULLY — "

        f"{normalized_class}"

    )

    print("=" * 70)



    print()

    print("Input:")

    print(input_path)



    print()

    print("Output:")

    print(OUTPUT_ROOT)



    print()

    print("IMPORTANT:")

    print(

        "No rows or columns were removed."

    )

    print(

        "No imputation was performed."

    )

    print(

        "No scaling or normalization was performed."

    )

    print(

        "No feature transformation was performed."

    )

    print(

        "No outliers were removed."

    )



    print()

    print(

        "Constant variables were retained."

    )



    print(
        "Constant-variable correlations are handled "
        "explicitly and reported as NaN where undefined."
    )
    print(
        "Constant predictors remain in the source data; "
        "VIF excludes them only from the diagnostic calculation."
    )



    print()

    print(

        "STRICT/NONSTRICT was read, validated, "

        "and preserved only."

    )



    print(

        "The EDA is diagnostic only."

    )



    print()




# ============================================================

# COMMAND-LINE INTERFACE

# ============================================================


def parse_arguments() -> argparse.Namespace:



    parser = argparse.ArgumentParser(

        description=(

            "EDA pipeline for paired Intercity "

            "EV–ICE summary data."

        )

    )



    parser.add_argument(

        "--dataset",

        type=str.upper,

        choices=[

            "STRICT",

            "NONSTRICT",

            "BOTH",

        ],

        default="BOTH",

        help=(

            "Dataset to analyse. Default: BOTH."

        ),

    )



    parser.add_argument(

        "--strict-input",

        type=Path,

        default=STRICT_INPUT,

        help=(

            "Path to EV_ICE_Training_STRICT.xlsx"

        ),

    )



    parser.add_argument(

        "--nonstrict-input",

        type=Path,

        default=NONSTRICT_INPUT,

        help=(

            "Path to EV_ICE_Training_NONSTRICT.xlsx"

        ),

    )



    return parser.parse_args()




# ============================================================

# PROGRAM ENTRY POINT

# ============================================================


if __name__ == "__main__":



    args = parse_arguments()



    jobs = []



    if args.dataset in {

        "STRICT",

        "BOTH",

    }:



        jobs.append(

            (

                "STRICT",

                args.strict_input.resolve(),

            )

        )



    if args.dataset in {

        "NONSTRICT",

        "BOTH",

    }:



        jobs.append(

            (

                "NONSTRICT",

                args.nonstrict_input.resolve(),

            )

        )



    for pair_class, input_path in jobs:



        run_eda(

            input_path=input_path,

            expected_pair_class=pair_class,

        )



    print()

    print("=" * 70)

    print("ALL REQUESTED EDA RUNS COMPLETED")

    print("=" * 70)

