from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# EXCEL LOADER
# ============================================================

def load_excel(
    file_path: str | Path,
    sheet_name: str | int = 0,
    header: int = 0,
) -> pd.DataFrame:
    """
    Load an Excel worksheet into a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Path to the Excel workbook.

    sheet_name:
        Excel worksheet name or worksheet index.

    header:
        Zero-based row containing the column headers.

        Example:
            header=0 -> Excel row 1
            header=3 -> Excel row 4

    Returns
    -------
    pd.DataFrame
        Loaded Excel worksheet.
    """

    path = Path(file_path).expanduser()

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file:\n{path}"
        )

    if path.suffix.lower() not in {
        ".xlsx",
        ".xlsm",
    }:
        raise ValueError(
            f"Unsupported Excel file type: {path.suffix}. "
            "Expected .xlsx or .xlsm."
        )

    # --------------------------------------------------------
    # Read Excel
    # --------------------------------------------------------

    try:
        dataframe = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header,
            engine="openpyxl",
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to read Excel file:\n"
            f"{path}\n"
            f"Sheet: {sheet_name}\n"
            f"Reason: {error}"
        ) from error

    # --------------------------------------------------------
    # Validate dataframe
    # --------------------------------------------------------

    if dataframe is None:
        raise RuntimeError(
            f"No dataframe was returned from:\n"
            f"{path}\n"
            f"Sheet: {sheet_name}"
        )

    if dataframe.empty:
        raise ValueError(
            f"The worksheet is empty:\n"
            f"{path}\n"
            f"Sheet: {sheet_name}"
        )

    # --------------------------------------------------------
    # Clean column names only
    # --------------------------------------------------------

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """

    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input path is not a file:\n{path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Expected .csv."
        )

    last_error: Exception | None = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
            )
            break

        except UnicodeDecodeError as error:
            last_error = error

        except Exception as error:
            raise RuntimeError(
                f"Failed to read CSV file:\n"
                f"{path}\n"
                f"Reason: {error}"
            ) from error

    else:
        raise RuntimeError(
            f"Could not decode CSV file:\n"
            f"{path}\n"
            f"Reason: {last_error}"
        )

    if dataframe.empty:
        raise ValueError(
            f"The CSV file is empty:\n{path}"
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


# ============================================================
# DATAFRAME INSPECTION
# ============================================================

def inspect_dataframe(
    dataframe: pd.DataFrame,
    name: str = "Data",
) -> dict:
    """
    Generate a basic structural summary of a dataframe.

    This function reports data quality information only.
    It does not modify the dataframe.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    return {
        "name": name,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "missing_cells": int(
            dataframe.isna().sum().sum()
        ),
        "numeric_columns": int(
            dataframe.select_dtypes(
                include="number"
            ).shape[1]
        ),
        "categorical_columns": int(
            dataframe.select_dtypes(
                exclude="number"
            ).shape[1]
        ),
        "duplicate_columns": int(
            dataframe.columns.duplicated().sum()
        ),
    }


# ============================================================
# PRINT DATAFRAME SUMMARY
# ============================================================

def print_dataframe_summary(
    dataframe: pd.DataFrame,
    name: str = "Data",
) -> None:
    """
    Print a readable dataframe summary.
    """

    summary = inspect_dataframe(
        dataframe,
        name=name,
    )

    print("\n" + "=" * 60)
    print(f"{name.upper()} SUMMARY")
    print("=" * 60)

    print(
        f"Rows              : "
        f"{summary['rows']}"
    )

    print(
        f"Columns           : "
        f"{summary['columns']}"
    )

    print(
        f"Duplicate rows    : "
        f"{summary['duplicate_rows']}"
    )

    print(
        f"Missing cells     : "
        f"{summary['missing_cells']}"
    )

    print(
        f"Numeric columns   : "
        f"{summary['numeric_columns']}"
    )

    print(
        f"Other columns     : "
        f"{summary['categorical_columns']}"
    )

    print(
        f"Duplicate columns : "
        f"{summary['duplicate_columns']}"
    )

    print("\nColumns:")

    for number, column in enumerate(
        dataframe.columns,
        start=1,
    ):
        print(
            f"  {number:02d}. {column}"
        )

    print("=" * 60)