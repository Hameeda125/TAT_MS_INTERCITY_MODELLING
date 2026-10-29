r"""
TAT MS INTERCITY MODELLING
RAW ICE -> PREDICTION-READY ICE PIPELINE

Version:
    2026-09-30_ICE_INFERENCE_PREPROCESSING_V4_PATH_FIXED

Place this file at:
    D:\Hameeda_Projects\TAT_MS_INTERCITY_MODELLING\src\prediction\
    ice_raw_preprocessing_pipeline.py

Raw inputs:
    D:\Hameeda_Projects\TAT_MS_INTERCITY_MODELLING\Raw_ICE_Input_Data

For EVERY raw ICE workbook this script:
    1. detects the raw telemetry sheet/schema
    2. reuses the authoritative Part-1 cleaning rules
    3. reuses the authoritative Part-1 feature engineering
    4. reuses the authoritative Part-1 ICE phase segmentation
    5. reuses the authoritative Part-1 TAT aggregation
    6. produces ALL 17 ICE trip/phase parameters:
       - 3 trip-level parameters
       - 7 phase-time parameters
       - 7 phase-distance parameters
    7. preserves ALL 7 ICE phases:
       REFUELING, STOP, IDLE, INCH, ACC, DEC, CRUISE
    8. preserves timestamp-level phase-segmented data for Time Series
    9. extracts start/end coordinates, timestamps and trip distance
       for downstream charging logic
   10. writes one prediction-ready workbook per ICE file
   11. writes a separate STRICT-style 17-parameter table with one row
       for every successfully processed ICE trip

Each *_ICE_READY.xlsx workbook contains:
    05_All_ICE_Parameters_17
with the complete calculated 17-parameter row for that ICE trip.

A separate cross-trip table is also created at:
    Results/Prediction/ICE_17_Parameters_Table.xlsx

The EV vehicle/charger specification sheet is NOT read by this script.
The separate charging-time code will read those specifications directly
from the original ICE workbook when charging is calculated.

NOT used for a future ICE-only trip:
    - EV-ICE pairing
    - pair cropping
    - STRICT/NONSTRICT classification
    - EV-side telemetry processing

Pair Class is therefore NOT recalculated or assigned.

IMPORTANT:
    Keep the raw telemetry worksheet FIRST in the workbook.
    Other worksheets, including any EV/charger specification sheet,
    are ignored by THIS preprocessing script and remain available for
    the separate charging-time code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-09-30_ICE_INFERENCE_PREPROCESSING_V4_PATH_FIXED"

ICE_PARAMETER_COLUMNS_17 = [
    "ICE Total Distance (km)",
    "ICE Total TAT (min)",
    "ICE Average Moving Speed (km/h)",

    "ICE REFUELING Time (min)",
    "ICE STOP Time (min)",
    "ICE IDLE Time (min)",
    "ICE INCHING Time (min)",
    "ICE ACC Time (min)",
    "ICE DEC Time (min)",
    "ICE CRUISE Time (min)",

    "ICE REFUELING Distance (km)",
    "ICE STOP Distance (km)",
    "ICE IDLE Distance (km)",
    "ICE INCHING Distance (km)",
    "ICE ACC Distance (km)",
    "ICE DEC Distance (km)",
    "ICE CRUISE Distance (km)",
]

PHASE_MAP = {
    "REFUELING": "REFUELING",
    "STOP": "STOP",
    "IDLE": "IDLE",
    "INCHING": "INCH",
    "ACC": "ACC",
    "DEC": "DEC",
    "CRUISE": "CRUISE",
}



def get_project_root() -> Path:
    """
    Locate the TAT_MS_INTERCITY_MODELLING project root robustly.

    The script currently lives at:
        <project>/src/ice_raw_data/ice_processing.py

    Earlier versions assumed src/prediction/, which incorrectly made
    <project>/src/ice_raw_data the project root.
    """
    here = Path(__file__).resolve()

    # First, use the known project-folder name.
    for candidate in (here.parent, *here.parents):
        if candidate.name.casefold() == "tat_ms_intercity_modelling":
            return candidate

    # Fallback for a future renamed project: locate a parent that has the
    # modelling-project structure, not merely the script's immediate folder.
    for candidate in (here.parent, *here.parents):
        if (
            (candidate / "src").is_dir()
            and (candidate / "models").exists()
            and (candidate / "Results").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the TAT_MS_INTERCITY_MODELLING project root "
        f"from script path: {here}"
    )


PROJECT_ROOT = get_project_root()

RAW_INPUT_DIR = PROJECT_ROOT / "Raw_ICE_Input_Data"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "Results"
    / "Prediction"
    / "ICE_Ready"
)

PART1_SRC = (
    PROJECT_ROOT.parent
    / "TAT_MS_INTERCITY"
    / "TAT_MS_INTERCITY_PIPELINE"
    / "src"
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()



def _trip_id(path: Path) -> str:
    text = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        path.stem,
    ).strip("_")

    return text[:120] or "ICE_TRIP"


def import_part1_modules() -> dict[str, Any]:
    """
    Import the exact completed Part-1 science instead of duplicating it here.
    """

    expected_part1_files = [
        PART1_SRC / "config_loader.py",
        PART1_SRC / "schema_detector.py",
        PART1_SRC / "cleaned_file_writer.py",
        PART1_SRC / "feature_engineering.py",
        PART1_SRC / "phase_segmentation.py",
        PART1_SRC / "tat_computation.py",
    ]

    if not PART1_SRC.is_dir():
        raise FileNotFoundError(
            "Authoritative Part-1 source folder was not found:\n"
            f"{PART1_SRC}"
        )

    missing_part1_files = [
        str(path)
        for path in expected_part1_files
        if not path.is_file()
    ]

    if missing_part1_files:
        raise FileNotFoundError(
            "Part-1 source folder exists, but required authoritative "
            "module(s) are missing:\n  - "
            + "\n  - ".join(missing_part1_files)
        )

    sys.path.insert(
        0,
        str(PART1_SRC),
    )

    from config_loader import load_config
    from schema_detector import detect_file_schema
    from cleaned_file_writer import clean_telemetry_data_with_audit
    import feature_engineering
    import phase_segmentation
    import tat_computation

    return {
        "load_config": load_config,
        "detect_file_schema": detect_file_schema,
        "clean": clean_telemetry_data_with_audit,
        "feature_module": feature_engineering,
        "phase_module": phase_segmentation,
        "tat_module": tat_computation,
    }


def discover_raw_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Raw ICE folder was not found:\n{input_dir}"
        )

    files = [
        path
        for path in sorted(input_dir.iterdir())
        if (
            path.is_file()
            and not path.name.startswith("~$")
            and path.suffix.lower() in {".xlsx", ".xlsm"}
        )
    ]

    if not files:
        raise FileNotFoundError(
            f"No Excel ICE files were found in:\n{input_dir}"
        )

    return files


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def first_address(frame: pd.DataFrame) -> str:
    if "Address" not in frame.columns:
        return ""

    values = (
        frame["Address"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return (
        str(values.iloc[0])
        if not values.empty
        else ""
    )


def last_address(frame: pd.DataFrame) -> str:
    if "Address" not in frame.columns:
        return ""

    values = (
        frame["Address"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return (
        str(values.iloc[-1])
        if not values.empty
        else ""
    )


def prediction_qc(
    featured: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Outlier detection is REPORT-ONLY here unless Part-1 already handled it.

    This is deliberate: applying a new clipping/deletion rule only at
    prediction time would make the inference data scientifically inconsistent
    with the historical data used to train the models.
    """

    cleaning = config.get(
        "cleaning",
        {},
    ) or {}

    feature = config.get(
        "feature_engineering",
        {},
    ) or {}

    threshold = float(
        feature.get(
            "maximum_gps_implied_speed_kmh",
            cleaning.get(
                "maximum_realistic_speed_kmh",
                160.0,
            ),
        )
    )

    dt = pd.to_numeric(
        featured["Time Difference (s)"],
        errors="coerce",
    )

    distance = pd.to_numeric(
        featured["Haversine Distance (m)"],
        errors="coerce",
    )

    speed = pd.to_numeric(
        featured["Speed (km/h)"],
        errors="coerce",
    )

    implied = pd.Series(
        np.nan,
        index=featured.index,
        dtype=float,
    )

    positive = dt > 0

    implied.loc[positive] = (
        distance.loc[positive]
        / dt.loc[positive]
        * 3.6
    )

    return pd.DataFrame(
        [
            {
                "QC Check": "GPS-implied speed above threshold",
                "Threshold": threshold,
                "Flagged Rows": int(
                    (implied > threshold).sum()
                ),
                "Handling": (
                    "REPORT ONLY; no new prediction-only clipping/deletion"
                ),
            },
            {
                "QC Check": "Recorded speed above threshold",
                "Threshold": threshold,
                "Flagged Rows": int(
                    (speed > threshold).sum()
                ),
                "Handling": (
                    "REPORT ONLY; no new prediction-only clipping/deletion"
                ),
            },
        ]
    )


def build_summary_row(
    *,
    trip_id: str,
    source_file: Path,
    segmented: pd.DataFrame,
    tat_metrics: dict[str, Any],
    phase_metrics: dict[str, Any],
    versions: dict[str, str],
) -> dict[str, Any]:

    phase_values = tat_metrics[
        "phase_values"
    ]

    row: dict[str, Any] = {
        "Trip ID": trip_id,
        "Source ICE File": source_file.name,
        "Start Timestamp": tat_metrics["trip_start"],
        "End Timestamp": tat_metrics["trip_end"],
        "Start Latitude": float(
            segmented["Latitude"].iloc[0]
        ),
        "Start Longitude": float(
            segmented["Longitude"].iloc[0]
        ),
        "End Latitude": float(
            segmented["Latitude"].iloc[-1]
        ),
        "End Longitude": float(
            segmented["Longitude"].iloc[-1]
        ),
        "Start Address": first_address(segmented),
        "End Address": last_address(segmented),
        "ICE Total Distance (km)": (
            tat_metrics["total_distance_km"]
        ),
        "ICE Total TAT (min)": (
            tat_metrics["total_time_min"]
        ),
        "ICE Average Moving Speed (km/h)": (
            tat_metrics["average_moving_speed_kmh"]
        ),
    }

    for display_phase, phase_name in PHASE_MAP.items():
        values = phase_values[phase_name]

        row[
            f"ICE {display_phase} Time (min)"
        ] = values["time_min"]

        row[
            f"ICE {display_phase} Distance (km)"
        ] = values["distance_km"]

    row.update(
        {
            "Phase Time Closure Error (s)": (
                tat_metrics["phase_time_error_s"]
            ),
            "Phase Distance Closure Error (m)": (
                tat_metrics["phase_distance_error_m"]
            ),
            "Low-Confidence Phase Rows": (
                phase_metrics.get(
                    "low_confidence_fallback_rows",
                    0,
                )
            ),
            "Pair Class": "NOT_APPLICABLE_NEW_ICE_TRIP",
            "Pair Class Handling": (
                "NOT READ / NOT RECALCULATED / NOT OVERWRITTEN"
            ),
            "Inference Preprocessing Version": SCRIPT_VERSION,
            "Part1 Feature Engineering Version": (
                versions["feature"]
            ),
            "Part1 Phase Segmentation Version": (
                versions["phase"]
            ),
            "Part1 TAT Computation Version": (
                versions["tat"]
            ),
        }
    )

    model_values = np.asarray(
        [
            row[column]
            for column in ICE_PARAMETER_COLUMNS_17
        ],
        dtype=float,
    )

    if not np.isfinite(model_values).all():
        raise ValueError(
            "The generated 17 ICE parameters contain non-finite values."
        )

    if (model_values < 0).any():
        raise ValueError(
            "The generated 17 ICE parameters contain negative values."
        )

    return row


def write_ready_workbook(
    *,
    path: Path,
    cleaned: pd.DataFrame,
    featured: pd.DataFrame,
    segmented: pd.DataFrame,
    summary_row: dict[str, Any],
    cleaning_summary: dict[str, Any],
    removed_rows: pd.DataFrame,
    speed_audit: pd.DataFrame,
    phase_audit: pd.DataFrame,
    phase_metrics: dict[str, Any],
    qc: pd.DataFrame,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        [summary_row]
    )

    ice_parameters = summary[
        ICE_PARAMETER_COLUMNS_17
    ].copy()

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
    ) as writer:

        cleaned.to_excel(
            writer,
            sheet_name="01_Cleaned",
            index=False,
        )

        featured.to_excel(
            writer,
            sheet_name="02_Featured",
            index=False,
        )

        segmented.to_excel(
            writer,
            sheet_name="03_Phase_Segmented",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="04_Trip_Summary",
            index=False,
        )

        ice_parameters.to_excel(
            writer,
            sheet_name="05_All_ICE_Parameters_17",
            index=False,
        )

        pd.DataFrame(
            [cleaning_summary]
        ).to_excel(
            writer,
            sheet_name="06_Cleaning_Summary",
            index=False,
        )

        removed_rows.to_excel(
            writer,
            sheet_name="07_Removed_Rows",
            index=False,
        )

        speed_audit.to_excel(
            writer,
            sheet_name="08_Speed_Audit",
            index=False,
        )

        phase_audit.to_excel(
            writer,
            sheet_name="09_Phase_Audit",
            index=False,
        )

        pd.DataFrame(
            [phase_metrics]
        ).to_excel(
            writer,
            sheet_name="10_Phase_Metrics",
            index=False,
        )

        qc.to_excel(
            writer,
            sheet_name="11_Prediction_QC",
            index=False,
        )


# ============================================================================
# PROCESS ONE ICE FILE
# ============================================================================

def process_one(
    *,
    input_path: Path,
    output_root: Path,
    part1: dict[str, Any],
    config: dict,
) -> dict[str, Any]:

    trip_id = _trip_id(
        input_path
    )

    detector = part1[
        "detect_file_schema"
    ]

    schema = detector(
        input_path
    )

    if schema["schema"] == "UNSUPPORTED":
        raise ValueError(
            schema["error"]
            or "Unsupported telemetry workbook."
        )

    telemetry_sheet = str(
        schema["sheet_name"]
    )

    with pd.ExcelFile(
        input_path,
        engine="openpyxl",
    ) as excel:
        sheet_names = list(
            excel.sheet_names
        )

    if not sheet_names:
        raise ValueError(
            "Workbook contains no worksheets."
        )

    if sheet_names[0] != telemetry_sheet:
        raise ValueError(
            "Keep the raw telemetry worksheet as the FIRST worksheet. "
            "Other sheets are ignored by preprocessing and may be used "
            "later by the charging-time code.\n"
            f"Detected telemetry sheet: {telemetry_sheet!r}\n"
            f"First sheet: {sheet_names[0]!r}"
        )

    # 1. Cleaning.
    (
        cleaned,
        cleaning_summary,
        removed_rows,
        speed_audit,
    ) = part1["clean"](
        input_path,
        config,
    )

    if cleaned.empty:
        raise ValueError(
            "No telemetry rows remain after cleaning."
        )

    # 2. Feature engineering.
    feature_module = part1[
        "feature_module"
    ]

    feature_rules = (
        feature_module._settings(
            config
        )
    )

    featured, feature_audit = (
        feature_module.engineer_features(
            cleaned,
            feature_rules,
        )
    )

    if featured.empty:
        raise ValueError(
            "No telemetry rows remain after feature engineering."
        )

    # 3. Phase segmentation.
    phase_module = part1[
        "phase_module"
    ]

    start_time = pd.Timestamp(
        featured["Timestamp"].iloc[0]
    )

    end_time = pd.Timestamp(
        featured["Timestamp"].iloc[-1]
    )

    context = (
        phase_module.PairVehicleContext(
            vehicle="ICE",
            source_date=start_time.normalize(),
            source_trip_start=start_time,
            source_trip_end=end_time,
            route_origin=first_address(featured),
            route_destination=last_address(featured),
            source_file_name=input_path.name,
            route_category="INTERCITY",
        )
    )

    (
        segmented,
        phase_audit,
        _manual_event_audit,
        _replenishment_details,
        phase_metrics,
    ) = phase_module.segment_vehicle(
        source=featured,
        vehicle="ICE",
        context=context,
        manual_record=None,
    )

    # 4. TAT computation.
    tat_module = part1[
        "tat_module"
    ]

    tat_metrics = (
        tat_module._compute_metrics(
            segmented,
            "ICE",
        )
    )

    versions = {
        "feature": _clean_text(
            getattr(
                feature_module,
                "SCRIPT_VERSION",
                "UNKNOWN",
            )
        ),
        "phase": _clean_text(
            getattr(
                phase_module,
                "SCRIPT_VERSION",
                "UNKNOWN",
            )
        ),
        "tat": _clean_text(
            getattr(
                tat_module,
                "SCRIPT_VERSION",
                "UNKNOWN",
            )
        ),
    }

    summary_row = build_summary_row(
        trip_id=trip_id,
        source_file=input_path,
        segmented=segmented,
        tat_metrics=tat_metrics,
        phase_metrics=phase_metrics,
        versions=versions,
    )

    cleaning_summary = dict(
        cleaning_summary
    )

    cleaning_summary[
        "inference_preprocessing_version"
    ] = SCRIPT_VERSION

    cleaning_summary[
        "feature_audit"
    ] = json.dumps(
        feature_audit,
        default=str,
        sort_keys=True,
    )

    qc = prediction_qc(
        featured,
        config,
    )

    trip_output_dir = (
        output_root
        / trip_id
    )

    output_path = (
        trip_output_dir
        / f"{trip_id}_ICE_READY.xlsx"
    )

    write_ready_workbook(
        path=output_path,
        cleaned=cleaned,
        featured=featured,
        segmented=segmented,
        summary_row=summary_row,
        cleaning_summary=cleaning_summary,
        removed_rows=removed_rows,
        speed_audit=speed_audit,
        phase_audit=phase_audit,
        phase_metrics=phase_metrics,
        qc=qc,
    )

    summary_row[
        "Ready Workbook"
    ] = str(
        output_path
    )

    print(
        f"SUCCESS | {input_path.name}"
    )

    print(
        f"  Start     : {summary_row['Start Timestamp']}"
    )

    print(
        f"  End       : {summary_row['End Timestamp']}"
    )

    print(
        f"  Distance  : {summary_row['ICE Total Distance (km)']:.3f} km"
    )

    print(
        f"  ICE TAT   : {summary_row['ICE Total TAT (min)']:.3f} min"
    )

    print(
        f"  Saved     : {output_path}"
    )

    return summary_row


# ============================================================================
# PROCESS ALL ICE FILES
# ============================================================================

def run_all(
    *,
    input_dir: Path,
    output_dir: Path,
) -> None:

    print(
        "=" * 92
    )
    print(
        "RAW ICE -> PREDICTION-READY ICE PIPELINE"
    )
    print(
        "=" * 92
    )

    print(
        f"Version      : {SCRIPT_VERSION}"
    )
    print(
        f"Input        : {input_dir}"
    )
    print(
        f"Output       : {output_dir}"
    )
    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Part-1 src   : {PART1_SRC}"
    )
    print(
        "Pair Class   : NOT APPLICABLE / NOT RECALCULATED"
    )

    print(
        "ICE parameters: 17 (3 trip-level + 7 phase times + 7 phase distances)"
    )

    print(
        "ICE phases    : REFUELING, STOP, IDLE, INCH, ACC, DEC, CRUISE"
    )

    print(
        "EV specs      : NOT READ HERE; charging code reads them separately"
    )

    part1 = import_part1_modules()

    config = part1[
        "load_config"
    ]()

    files = discover_raw_files(
        input_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful: list[
        dict[str, Any]
    ] = []

    status_rows: list[
        dict[str, Any]
    ] = []

    for index, input_path in enumerate(
        files,
        start=1,
    ):

        print(
            "\n"
            + "-" * 92
        )

        print(
            f"[{index}/{len(files)}] "
            f"{input_path.name}"
        )

        try:
            row = process_one(
                input_path=input_path,
                output_root=output_dir,
                part1=part1,
                config=config,
            )

            successful.append(
                row
            )

            status_rows.append(
                {
                    "File": input_path.name,
                    "Status": "SUCCESS",
                    "Error": "",
                }
            )

        except Exception as error:

            message = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"FAILED | {input_path.name} | {message}"
            )

            traceback.print_exc()

            status_rows.append(
                {
                    "File": input_path.name,
                    "Status": "FAILED",
                    "Error": message,
                }
            )

    master_path = (
        output_dir
        / "ICE_Prediction_Input_Master.xlsx"
    )

    # STRICT-style cross-trip table:
    # one successful raw ICE file = one row;
    # Trip ID + Source ICE File + exactly the 17 calculated ICE parameters.
    parameter_table_dir = (
        PROJECT_ROOT
        / "Results"
        / "Prediction"
    )

    parameter_table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_table_path = (
        parameter_table_dir
        / "ICE_17_Parameters_Table.xlsx"
    )

    status_path = (
        output_dir
        / "ICE_Preprocessing_Run_Status.xlsx"
    )

    if successful:
        master = pd.DataFrame(
            successful
        )

        first_columns = [
            "Trip ID",
            "Source ICE File",
            "Start Timestamp",
            "End Timestamp",
            "Start Latitude",
            "Start Longitude",
            "End Latitude",
            "End Longitude",
            "Start Address",
            "End Address",
            *ICE_PARAMETER_COLUMNS_17,
        ]

        remaining = [
            column
            for column in master.columns
            if column not in first_columns
        ]

        master = master[
            first_columns
            + remaining
        ]

        master.to_excel(
            master_path,
            index=False,
        )

        # Separate compact table analogous to the historical STRICT/NONSTRICT
        # modelling tables: one row per ICE trip and only identifiers + the
        # complete 17 calculated ICE parameters.
        parameter_table_columns = [
            "Trip ID",
            "Source ICE File",
            *ICE_PARAMETER_COLUMNS_17,
        ]

        parameter_table = master[
            parameter_table_columns
        ].copy()

        with pd.ExcelWriter(
            parameter_table_path,
            engine="openpyxl",
        ) as writer:
            parameter_table.to_excel(
                writer,
                sheet_name="ICE_17_Parameters",
                index=False,
            )

    pd.DataFrame(
        status_rows
    ).to_excel(
        status_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 92
    )

    print(
        "PIPELINE COMPLETED"
    )

    print(
        "=" * 92
    )

    print(
        f"Files found       : {len(files)}"
    )

    print(
        f"Successful        : {len(successful)}"
    )

    print(
        f"Failed            : {len(files) - len(successful)}"
    )

    if successful:
        print(
            f"Master model input: {master_path}"
        )
        print(
            f"17-param table    : {parameter_table_path}"
        )

    print(
        f"Run status        : {status_path}"
    )

    print(
        "\nThe NEXT prediction stage will use EACH successful ICE file:"
    )

    print(
        "17 ICE parameters / phase sequence -> trained models; the separate charging code reads EV/charger specs -> predicted EV TAT"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_all(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
