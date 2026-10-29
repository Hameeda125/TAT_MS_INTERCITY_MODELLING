from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import load_excel


SCRIPT_VERSION = "2026-09-29_MODELLING_DATA_VALIDATOR_V2"


# ============================================================
# ICE INPUT FEATURES — 17
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


# ============================================================
# EV TARGETS — 17
# ============================================================

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


# ============================================================
# METADATA / CONTEXT
# ============================================================

REQUIRED_METADATA = [
    "Pair ID",
    "Pair Class",
]

OPTIONAL_CONTEXT_COLUMNS = [
    "Pair",
    "Route Group",
    "Route Direction",
    "EV Date",
    "ICE Date",
    "EV Start Time",
    "EV End Time",
    "ICE Start Time",
    "ICE End Time",
    "Same Calendar Day",
    "Start-Time Difference (min)",
]

MODELLING_COLUMNS = ICE_FEATURES + EV_TARGETS


# ============================================================
# AUTHORITATIVE PART 1 TRAINING PATH
# ============================================================

def get_training_data_directory() -> Path:
    """
    Return the authoritative Part 1 Training_Data directory.

    Expected sibling-project layout:

        .../TAT_MS_INTERCITY/
        .../TAT_MS_INTERCITY_MODELLING/

    No recursive search is used.
    """

    current_project = Path(__file__).resolve().parents[2]
    part1_project = current_project.parent / "TAT_MS_INTERCITY"

    if not part1_project.exists():
        raise FileNotFoundError(
            "Part 1 project was not found:\n"
            f"{part1_project}"
        )

    if not part1_project.is_dir():
        raise ValueError(
            "Part 1 path is not a directory:\n"
            f"{part1_project}"
        )

    training_directory = (
        part1_project
        / "Data"
        / "Results"
        / "Master_Summary"
        / "Training_Data"
    )

    if not training_directory.exists():
        raise FileNotFoundError(
            "Authoritative Part 1 training-data directory "
            "was not found:\n"
            f"{training_directory}"
        )

    if not training_directory.is_dir():
        raise ValueError(
            "Training-data path is not a directory:\n"
            f"{training_directory}"
        )

    return training_directory


def find_training_workbook(
    pair_class: str = "STRICT",
) -> Path:
    """
    Locate the authoritative STRICT or NONSTRICT training workbook.

    Pair class is used only to select the upstream workbook.
    It is never recalculated here.
    """

    normalized_class = str(pair_class).strip().upper()

    if normalized_class not in {"STRICT", "NONSTRICT"}:
        raise ValueError(
            "pair_class must be STRICT or NONSTRICT. "
            f"Received: {pair_class!r}"
        )

    training_file = (
        get_training_data_directory()
        / f"EV_ICE_Training_{normalized_class}.xlsx"
    )

    if not training_file.exists():
        raise FileNotFoundError(
            f"Could not find the authoritative {normalized_class} "
            "training workbook:\n"
            f"{training_file}"
        )

    if not training_file.is_file():
        raise ValueError(
            "Training workbook path is not a file:\n"
            f"{training_file}"
        )

    return training_file


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(dataframe: pd.DataFrame) -> dict:
    """
    Check the expected ICE features, EV targets, and metadata.

    No data is modified.
    """

    actual_columns = list(dataframe.columns)

    missing_ice = [
        column
        for column in ICE_FEATURES
        if column not in actual_columns
    ]

    missing_ev = [
        column
        for column in EV_TARGETS
        if column not in actual_columns
    ]

    missing_metadata = [
        column
        for column in REQUIRED_METADATA
        if column not in actual_columns
    ]

    missing_optional_context = [
        column
        for column in OPTIONAL_CONTEXT_COLUMNS
        if column not in actual_columns
    ]

    duplicate_columns = (
        dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()
    )

    return {
        "missing_ice": missing_ice,
        "missing_ev": missing_ev,
        "missing_metadata": missing_metadata,
        "missing_optional_context": missing_optional_context,
        "duplicate_columns": duplicate_columns,
        "ice_found": len(ICE_FEATURES) - len(missing_ice),
        "ev_found": len(EV_TARGETS) - len(missing_ev),
    }


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def validate_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict:
    """
    Check whether modelling columns are numeric and finite.

    No data is modified.
    """

    results = {}

    for column in columns:
        if column not in dataframe.columns:
            continue

        original = dataframe[column]

        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        nonblank_original = (
            original.notna()
            & original.astype("string").str.strip().ne("")
        )

        conversion_failures = (
            nonblank_original
            & numeric.isna()
        )

        numeric_values = numeric.dropna()

        if len(numeric_values) > 0:
            finite_mask = np.isfinite(
                numeric_values.to_numpy(dtype=float)
            )
            finite_count = int(finite_mask.sum())
            infinite_count = int((~finite_mask).sum())
        else:
            finite_count = 0
            infinite_count = 0

        results[column] = {
            "missing": int(numeric.isna().sum()),
            "conversion_failures": int(
                conversion_failures.sum()
            ),
            "finite_values": finite_count,
            "infinite_values": infinite_count,
        }

    return results


# ============================================================
# VARIANCE / INFORMATION REPORT
# ============================================================

def calculate_variance_report(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Calculate descriptive/variance information.

    Nothing is removed based on this report.
    """

    rows = []

    for column in columns:
        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        non_null = values.dropna()

        if len(non_null) > 0:
            mean_value = float(non_null.mean())
            std_value = float(non_null.std())
            min_value = float(non_null.min())
            max_value = float(non_null.max())
        else:
            mean_value = np.nan
            std_value = np.nan
            min_value = np.nan
            max_value = np.nan

        unique_count = int(
            values.nunique(dropna=True)
        )

        rows.append(
            {
                "Feature": column,
                "Count": int(values.notna().sum()),
                "Missing": int(values.isna().sum()),
                "Unique": unique_count,
                "Mean": mean_value,
                "Std": std_value,
                "Min": min_value,
                "Max": max_value,
                "Constant": unique_count <= 1,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Feature",
            "Count",
            "Missing",
            "Unique",
            "Mean",
            "Std",
            "Min",
            "Max",
            "Constant",
        ],
    )


# ============================================================
# NON-NEGATIVE VALIDATION
# ============================================================

def validate_non_negative(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict:
    """
    Check for negative physical values.

    The 34 modelling variables are time, distance, or speed
    quantities and are expected to be non-negative.
    """

    results = {}

    for column in columns:
        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        results[column] = int(
            (values < 0).sum()
        )

    return results


# ============================================================
# TARGET-SPECIFIC VALIDATION
# ============================================================

def validate_targets(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Validate the 17 EV targets.

    Constant targets are reported only.
    They are not removed.
    """

    report = {}

    for target in EV_TARGETS:
        if target not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[target],
            errors="coerce",
        )

        report[target] = {
            "missing": int(values.isna().sum()),
            "unique_values": int(
                values.nunique(dropna=True)
            ),
            "constant": (
                values.nunique(dropna=True) <= 1
            ),
            "negative_values": int(
                (values < 0).sum()
            ),
        }

    return report


# ============================================================
# STRICT / NONSTRICT VALIDATION
# ============================================================

def validate_pair_class(
    dataframe: pd.DataFrame,
    expected_pair_class: str | None = None,
) -> dict:
    """
    Validate upstream Pair Class.

    This function never derives, recalculates, or overwrites
    STRICT/NONSTRICT. It only reads and validates Part 1 output.
    """

    if "Pair Class" not in dataframe.columns:
        return {
            "exists": False,
            "expected": expected_pair_class,
            "missing": None,
            "invalid_values": [],
            "mismatch_count": None,
            "counts": {},
            "status": "FAIL",
        }

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

    counts = {
        str(key): int(value)
        for key, value
        in observed_nonblank.value_counts().items()
    }

    normalized_expected = None

    if expected_pair_class is not None:
        normalized_expected = (
            str(expected_pair_class)
            .strip()
            .upper()
        )

        if normalized_expected not in valid_values:
            raise ValueError(
                "expected_pair_class must be STRICT, "
                "NONSTRICT, or None."
            )

        mismatch_count = int(
            (
                (~missing_mask)
                & pair_class.ne(normalized_expected)
            ).sum()
        )
    else:
        mismatch_count = 0

    status = (
        "PASS"
        if (
            int(missing_mask.sum()) == 0
            and not invalid_values
            and mismatch_count == 0
        )
        else "FAIL"
    )

    return {
        "exists": True,
        "expected": normalized_expected,
        "missing": int(missing_mask.sum()),
        "invalid_values": invalid_values,
        "mismatch_count": mismatch_count,
        "counts": counts,
        "status": status,
    }


# ============================================================
# PAIR ID VALIDATION
# ============================================================

def validate_pair_ids(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Validate Pair ID information.

    Pair ID is identification metadata only.
    """

    if "Pair ID" not in dataframe.columns:
        return {
            "exists": False,
            "unique_pairs": None,
            "missing_pair_ids": None,
            "duplicate_pair_ids": None,
        }

    pair_ids = dataframe["Pair ID"]

    missing_mask = (
        pair_ids.isna()
        | pair_ids.astype("string").str.strip().eq("")
    )

    valid_pair_ids = pair_ids[
        ~missing_mask
    ]

    unique_pairs = int(
        valid_pair_ids.nunique(dropna=True)
    )

    duplicate_pair_ids = int(
        valid_pair_ids.duplicated(
            keep=False
        ).sum()
    )

    return {
        "exists": True,
        "unique_pairs": unique_pairs,
        "missing_pair_ids": int(
            missing_mask.sum()
        ),
        "duplicate_pair_ids": duplicate_pair_ids,
    }


# ============================================================
# DUPLICATE ROW VALIDATION
# ============================================================

def validate_duplicate_rows(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Check exact duplicate rows.

    Duplicate rows are not removed.
    """

    duplicate_mask = dataframe.duplicated(
        keep=False
    )

    return {
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "duplicate_group_rows": int(
            duplicate_mask.sum()
        ),
    }


# ============================================================
# MODELLING READINESS
# ============================================================

def determine_modelling_readiness(
    column_report: dict,
    numeric_report: dict,
    *,
    pair_class_report: dict | None = None,
    pair_report: dict | None = None,
    duplicate_report: dict | None = None,
    negative_report: dict | None = None,
) -> dict:
    """
    Determine whether data is structurally ready.

    Constant variables are reported only.
    """

    structural_errors = []

    if column_report["missing_ice"]:
        structural_errors.append(
            "Missing ICE input features."
        )

    if column_report["missing_ev"]:
        structural_errors.append(
            "Missing EV target variables."
        )

    if column_report["missing_metadata"]:
        structural_errors.append(
            "Missing required metadata."
        )

    if column_report["duplicate_columns"]:
        structural_errors.append(
            "Duplicate column names detected."
        )

    conversion_failures = {
        column: report["conversion_failures"]
        for column, report
        in numeric_report.items()
        if report["conversion_failures"] > 0
    }

    if conversion_failures:
        structural_errors.append(
            "Numeric conversion failures detected."
        )

    missing_values = {
        column: report["missing"]
        for column, report
        in numeric_report.items()
        if report["missing"] > 0
    }

    if missing_values:
        structural_errors.append(
            "Missing values detected in modelling variables."
        )

    infinite_values = {
        column: report["infinite_values"]
        for column, report
        in numeric_report.items()
        if report["infinite_values"] > 0
    }

    if infinite_values:
        structural_errors.append(
            "Infinite values detected."
        )

    if negative_report is not None:
        negative_values = {
            column: count
            for column, count
            in negative_report.items()
            if count > 0
        }

        if negative_values:
            structural_errors.append(
                "Negative physical values detected "
                "in modelling variables."
            )

    if pair_class_report is not None:
        if not pair_class_report.get(
            "exists",
            False,
        ):
            structural_errors.append(
                "Pair Class column is missing."
            )
        else:
            if pair_class_report.get(
                "missing",
                0,
            ) > 0:
                structural_errors.append(
                    "Missing Pair Class values detected."
                )

            if pair_class_report.get(
                "invalid_values"
            ):
                structural_errors.append(
                    "Invalid Pair Class values detected."
                )

            if pair_class_report.get(
                "mismatch_count",
                0,
            ) > 0:
                structural_errors.append(
                    "Pair Class values do not match "
                    "the expected training workbook class."
                )

    if pair_report is not None:
        if not pair_report.get(
            "exists",
            False,
        ):
            structural_errors.append(
                "Pair ID column is missing."
            )
        else:
            if pair_report.get(
                "missing_pair_ids",
                0,
            ) > 0:
                structural_errors.append(
                    "Missing Pair ID values detected."
                )

            if pair_report.get(
                "duplicate_pair_ids",
                0,
            ) > 0:
                structural_errors.append(
                    "Duplicate Pair IDs detected "
                    "in pair-level training data."
                )

    if duplicate_report is not None:
        if duplicate_report.get(
            "duplicate_rows",
            0,
        ) > 0:
            structural_errors.append(
                "Exact duplicate rows detected."
            )

    return {
        "status": (
            "FAIL"
            if structural_errors
            else "PASS"
        ),
        "errors": structural_errors,
        "warnings": [],
    }


# ============================================================
# MAIN VALIDATION FUNCTION
# ============================================================

def validate_training_data(
    dataframe: pd.DataFrame,
    expected_pair_class: str | None = None,
) -> dict:
    """
    Complete validation of Part 1 training output.

    IMPORTANT:
    This function only reads and reports.

    It does NOT clean, remove, modify, impute, scale,
    transform, rename phases, recalculate Pair Class,
    or overwrite STRICT/NONSTRICT.
    """

    print("\n" + "=" * 70)
    print(
        "TAT MS INTERCITY — "
        "TRAINING DATA VALIDATION"
    )
    print("=" * 70)

    print(
        f"Validator version  : "
        f"{SCRIPT_VERSION}"
    )

    if expected_pair_class is not None:
        print(
            f"Expected Pair Class: "
            f"{str(expected_pair_class).strip().upper()}"
        )

    print("\n" + "-" * 70)
    print("DATASET INFORMATION")
    print("-" * 70)
    print(f"Rows              : {len(dataframe)}")
    print(f"Columns           : {len(dataframe.columns)}")
    print(
        "Duplicate rows    : "
        f"{int(dataframe.duplicated().sum())}"
    )
    print(
        "Duplicate columns : "
        f"{int(dataframe.columns.duplicated().sum())}"
    )

    column_report = validate_columns(
        dataframe
    )

    print("\n" + "-" * 70)
    print("ICE INPUT FEATURES")
    print("-" * 70)
    print(f"Expected : {len(ICE_FEATURES)}")
    print(
        f"Found    : "
        f"{column_report['ice_found']}"
    )

    if column_report["missing_ice"]:
        print("Status   : FAIL")
        print("Missing:")
        for column in column_report["missing_ice"]:
            print(f"  - {column}")
    else:
        print("Status   : PASS")

    print("\n" + "-" * 70)
    print("EV TARGETS")
    print("-" * 70)
    print(f"Expected : {len(EV_TARGETS)}")
    print(
        f"Found    : "
        f"{column_report['ev_found']}"
    )

    if column_report["missing_ev"]:
        print("Status   : FAIL")
        print("Missing:")
        for column in column_report["missing_ev"]:
            print(f"  - {column}")
    else:
        print("Status   : PASS")

    print("\n" + "-" * 70)
    print("REQUIRED METADATA")
    print("-" * 70)

    if column_report["missing_metadata"]:
        print("Status : FAIL")
        print("Missing:")
        for column in column_report[
            "missing_metadata"
        ]:
            print(f"  - {column}")
    else:
        print("Pair ID    : PRESENT")
        print("Pair Class : PRESENT")

    print("\n" + "-" * 70)
    print("OPTIONAL CONTEXT COLUMNS")
    print("-" * 70)

    if column_report[
        "missing_optional_context"
    ]:
        print(
            "Missing optional context columns "
            "(does not fail readiness):"
        )
        for column in column_report[
            "missing_optional_context"
        ]:
            print(f"  - {column}")
    else:
        print(
            "All expected context columns are present."
        )

    pair_class_report = (
        validate_pair_class(
            dataframe,
            expected_pair_class=(
                expected_pair_class
            ),
        )
    )

    print("\n" + "-" * 70)
    print("STRICT / NONSTRICT CLASSIFICATION")
    print("-" * 70)

    if not pair_class_report["exists"]:
        print("Pair Class : NOT FOUND")
    else:
        print(
            "Observed counts : "
            f"{pair_class_report['counts']}"
        )
        print(
            "Missing values  : "
            f"{pair_class_report['missing']}"
        )
        print(
            "Invalid values  : "
            f"{pair_class_report['invalid_values']}"
        )

        if pair_class_report[
            "expected"
        ] is not None:
            print(
                "Expected class  : "
                f"{pair_class_report['expected']}"
            )
            print(
                "Class mismatches: "
                f"{pair_class_report['mismatch_count']}"
            )

        print(
            "Classification handling: "
            "READ / VALIDATE / PRESERVE ONLY"
        )
        print(
            "Recalculation           : NONE"
        )
        print(
            "Overwrite               : NONE"
        )

    numeric_report = (
        validate_numeric_columns(
            dataframe,
            MODELLING_COLUMNS,
        )
    )

    conversion_failures = {
        column: values[
            "conversion_failures"
        ]
        for column, values
        in numeric_report.items()
        if values[
            "conversion_failures"
        ] > 0
    }

    infinite_values = {
        column: values[
            "infinite_values"
        ]
        for column, values
        in numeric_report.items()
        if values[
            "infinite_values"
        ] > 0
    }

    print("\n" + "-" * 70)
    print("NUMERIC VALIDATION")
    print("-" * 70)

    if conversion_failures:
        print(
            "Conversion failures detected:"
        )
        for column, count in (
            conversion_failures.items()
        ):
            print(f"  - {column}: {count}")
    else:
        print(
            "All modelling columns are "
            "numerically interpretable : PASS"
        )

    if infinite_values:
        print("\nInfinite values detected:")
        for column, count in (
            infinite_values.items()
        ):
            print(f"  - {column}: {count}")
    else:
        print("Infinite values : NONE")

    missing_report = {
        column: values["missing"]
        for column, values
        in numeric_report.items()
        if values["missing"] > 0
    }

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    if missing_report:
        print(
            "Missing values detected "
            "(structural failure for modelling variables):"
        )
        for column, count in (
            missing_report.items()
        ):
            print(f"  - {column}: {count}")
    else:
        print("Missing values : NONE")

    variance_report = (
        calculate_variance_report(
            dataframe,
            MODELLING_COLUMNS,
        )
    )

    constant_features = []

    if not variance_report.empty:
        constant_features = (
            variance_report[
                variance_report["Constant"]
            ]["Feature"].tolist()
        )

    print("\n" + "-" * 70)
    print("VARIANCE / INFORMATION CONTENT")
    print("-" * 70)

    if constant_features:
        print(
            "Constant variables detected "
            "(reported only):"
        )
        for column in constant_features:
            print(f"  - {column}")
    else:
        print(
            "No constant modelling variables found."
        )

    negative_report = (
        validate_non_negative(
            dataframe,
            MODELLING_COLUMNS,
        )
    )

    negative_values = {
        column: count
        for column, count
        in negative_report.items()
        if count > 0
    }

    print("\n" + "-" * 70)
    print("NEGATIVE VALUES")
    print("-" * 70)

    if negative_values:
        print(
            "Negative physical values detected "
            "(structural failure):"
        )
        for column, count in (
            negative_values.items()
        ):
            print(f"  - {column}: {count}")
    else:
        print(
            "No negative physical values detected."
        )

    target_report = validate_targets(
        dataframe
    )

    constant_targets = [
        target
        for target, report
        in target_report.items()
        if report["constant"]
    ]

    negative_targets = {
        target: report[
            "negative_values"
        ]
        for target, report
        in target_report.items()
        if report[
            "negative_values"
        ] > 0
    }

    print("\n" + "-" * 70)
    print("TARGET-SPECIFIC CHECK")
    print("-" * 70)
    print(
        f"Total EV targets checked : "
        f"{len(EV_TARGETS)}"
    )
    print(
        f"Constant EV targets      : "
        f"{len(constant_targets)}"
    )
    print(
        f"Targets with negatives   : "
        f"{len(negative_targets)}"
    )

    if constant_targets:
        print(
            "\nConstant targets "
            "(reported only):"
        )
        for target in constant_targets:
            print(f"  - {target}")

    if negative_targets:
        print(
            "\nTargets containing negative values:"
        )
        for target, count in (
            negative_targets.items()
        ):
            print(f"  - {target}: {count}")

    pair_report = validate_pair_ids(
        dataframe
    )

    print("\n" + "-" * 70)
    print("PAIR / TRIP INFORMATION")
    print("-" * 70)

    if pair_report["exists"]:
        print(
            "Unique Pair IDs       : "
            f"{pair_report['unique_pairs']}"
        )
        print(
            "Missing Pair IDs      : "
            f"{pair_report['missing_pair_ids']}"
        )
        print(
            "Rows with repeated Pair IDs : "
            f"{pair_report['duplicate_pair_ids']}"
        )
    else:
        print("Pair ID : NOT FOUND")

    duplicate_report = (
        validate_duplicate_rows(
            dataframe
        )
    )

    print("\n" + "-" * 70)
    print("DUPLICATE ROW CHECK")
    print("-" * 70)
    print(
        "Exact duplicate rows : "
        f"{duplicate_report['duplicate_rows']}"
    )
    print(
        "Rows in duplicate groups : "
        f"{duplicate_report['duplicate_group_rows']}"
    )

    readiness = (
        determine_modelling_readiness(
            column_report,
            numeric_report,
            pair_class_report=(
                pair_class_report
            ),
            pair_report=pair_report,
            duplicate_report=(
                duplicate_report
            ),
            negative_report=(
                negative_report
            ),
        )
    )

    print("\n" + "-" * 70)
    print("MODELLING READINESS")
    print("-" * 70)

    if readiness["status"] == "PASS":
        print(
            "Structural validation : PASS"
        )

        if constant_features:
            print("\nWARNING:")
            print(
                "Constant variables are present."
            )
            print(
                "They will NOT be removed here."
            )
            print(
                "They should be investigated "
                "during EDA/model preparation."
            )
    else:
        print(
            "Structural validation : FAIL"
        )
        print("\nReasons:")
        for error in readiness["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 70)

    if readiness["status"] == "PASS":
        if constant_features:
            print(
                "OVERALL VALIDATION : "
                "PASS WITH WARNINGS"
            )
        else:
            print(
                "OVERALL VALIDATION : PASS"
            )
    else:
        print(
            "OVERALL VALIDATION : FAIL"
        )

    print("=" * 70)

    return {
        "column_report": column_report,
        "numeric_report": numeric_report,
        "missing_report": missing_report,
        "variance_report": variance_report,
        "negative_report": negative_report,
        "target_report": target_report,
        "pair_class_report": pair_class_report,
        "pair_report": pair_report,
        "duplicate_report": duplicate_report,
        "constant_features": constant_features,
        "constant_targets": constant_targets,
        "readiness": readiness,
        "structural_pass": (
            readiness["status"] == "PASS"
        ),
    }


# ============================================================
# LOAD + VALIDATE ONE WORKBOOK
# ============================================================

def validate_training_workbook(
    pair_class: str,
) -> dict:
    """
    Load and validate one authoritative Part 1 workbook.
    """

    normalized_class = (
        str(pair_class)
        .strip()
        .upper()
    )

    training_file = (
        find_training_workbook(
            normalized_class
        )
    )

    print("\nTraining workbook:")
    print(training_file)

    data = load_excel(
        training_file,
        sheet_name="01_Training_Data",
        header=3,
    )

    result = validate_training_data(
        data,
        expected_pair_class=(
            normalized_class
        ),
    )

    result["training_file"] = str(
        training_file
    )
    result["pair_class"] = (
        normalized_class
    )

    return result


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    try:
        all_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for pair_class in (
            "STRICT",
            "NONSTRICT",
        ):

            print("\n\n" + "#" * 70)
            print(
                f"VALIDATING {pair_class} "
                "TRAINING DATA"
            )
            print("#" * 70)

            result = (
                validate_training_workbook(
                    pair_class
                )
            )

            all_results[
                pair_class
            ] = result

            if not result[
                "structural_pass"
            ]:
                raise SystemExit(
                    f"\n{pair_class} training data "
                    "structural validation failed."
                )

        print(
            "\nBoth STRICT and NONSTRICT "
            "training workbooks passed "
            "structural validation."
        )

    except Exception as error:
        print(
            "\nVALIDATION STOPPED | "
            f"{error}"
        )
        raise SystemExit(1)
