"""
time_series_analysis.py

MASTER TIME-SERIES ANALYSIS PIPELINE
====================================

Seven analysis stages
---------------------
1. Telemetry & Data Structure Analysis
2. Phase Distribution & Episode Analysis
3. Sampling / Time-Gap Analysis
4. ICE-EV Temporal Alignment Analysis
5. Phase-Aware Temporal Representation Analysis
6. Time-Series Dataset Design & Validation
7. Time-Series Modelling Readiness Analysis

Important principles
--------------------
- Uses existing Part-1 processed telemetry.
- Does NOT clean the data.
- Does NOT re-run feature engineering.
- Does NOT re-run phase segmentation.
- Does NOT delete long time gaps.
- Uses the existing `phase` column.
- Uses only STRICT paired trips.
- Preserves Pair ID and temporal order.
- Does NOT force row-wise ICE-EV matching.
- Charging/refuelling remains separate from predictive modelling.
- This file performs ANALYSIS only.
- Actual TS model training will be implemented separately after
  representation/design decisions are confirmed.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\TAT_MS_INTERCITY_MODELLING"
)

# Existing Part-1 paired telemetry
PAIRED_DATA_ROOT = Path(
    r"D:\TAT_MS_INTERCITY\Data\Results\Paired"
)

# Output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "Results"
    / "TimeSeries"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "TS_Complete_Analysis.xlsx"
)


# =============================================================================
# EXISTING PART-1 TELEMETRY COLUMNS
# =============================================================================

TIMESTAMP_COL = "Timestamp"

SPEED_MS_COL = "Speed (m/s)"

SPEED_KMH_COL = "Speed (km/h)"

TIME_DIFF_COL = "Time Difference (s)"

DISTANCE_COL = "Haversine Distance (m)"

ACCEL_COL = "Acceleration (m/s^2)"

PHASE_COL = "phase"


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

MODELLING_PHASES = [
    "STOP",
    "IDLE",
    "INCHING",
    "ACC",
    "DEC",
    "CRUISE",
]


CHARGING_PHASES = [
    "CHARGING",
    "REFUELING",
    "CHARGING_REFUELING",
]


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalize_phase(value) -> str:
    """
    Normalize phase text for analysis.

    This does NOT change the source dataframe.
    """

    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    replacements = {
        "INCH": "INCHING",
        "CHARGE": "CHARGING",
        "REFUEL": "REFUELING",
        "CHARGING / REFUELING": "CHARGING_REFUELING",
        "CHARGING-REFUELING": "CHARGING_REFUELING",
    }

    return replacements.get(
        text,
        text,
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric for analysis."""

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_divide(
    numerator,
    denominator,
):
    """Safe scalar division."""

    if denominator is None:
        return np.nan

    if pd.isna(denominator):
        return np.nan

    if denominator == 0:
        return np.nan

    return numerator / denominator


def percentile(
    series: pd.Series,
    q: float,
):
    """Calculate percentile safely."""

    values = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .dropna()
    )

    if values.empty:
        return np.nan

    return float(
        np.percentile(
            values,
            q,
        )
    )


def get_duration_seconds(
    df: pd.DataFrame,
) -> float:
    """
    Calculate duration using Part-1 interval semantics.

    `Time Difference (s)` represents the interval from the previous
    datapoint to the current datapoint.

    Therefore:

        Total duration = sum(Time Difference)

    rather than simply:

        last timestamp - first timestamp
    """

    if TIME_DIFF_COL not in df.columns:
        return np.nan

    values = (
        safe_numeric(
            df[TIME_DIFF_COL]
        )
        .fillna(0)
        .clip(lower=0)
    )

    return float(
        values.sum()
    )


def get_distance_km(
    df: pd.DataFrame,
) -> float:
    """Calculate distance using Part-1 Haversine distance."""

    if DISTANCE_COL not in df.columns:
        return np.nan

    values = (
        safe_numeric(
            df[DISTANCE_COL]
        )
        .fillna(0)
        .clip(lower=0)
    )

    return float(
        values.sum() / 1000.0
    )


def get_timestamp_range(
    df: pd.DataFrame,
):
    """Return first and last valid timestamps."""

    if TIMESTAMP_COL not in df.columns:
        return pd.NaT, pd.NaT

    timestamps = (
        pd.to_datetime(
            df[TIMESTAMP_COL],
            errors="coerce",
        )
        .dropna()
    )

    if timestamps.empty:
        return pd.NaT, pd.NaT

    return (
        timestamps.min(),
        timestamps.max(),
    )


def safe_excel_sheet_name(
    name: str,
) -> str:
    """Create a valid Excel worksheet name."""

    invalid_characters = [
        "\\",
        "/",
        "*",
        "?",
        ":",
        "[",
        "]",
    ]

    for character in invalid_characters:
        name = name.replace(
            character,
            "_",
        )

    return name[:31]


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def discover_strict_files() -> list[Path]:
    """
    Find STRICT paired telemetry workbooks recursively.

    Only the Part-1 Paired directory is searched.

    This avoids accidentally reading duplicate copies elsewhere
    under the Data directory.
    """

    if not PAIRED_DATA_ROOT.exists():

        raise FileNotFoundError(
            "Paired data directory does not exist:\n"
            f"{PAIRED_DATA_ROOT}"
        )

    files = sorted(
        path
        for path in PAIRED_DATA_ROOT.rglob(
            "*.xlsx"
        )
        if "_STRICT_" in path.stem.upper()
    )

    unique_files = {}

    for file in files:

        key = str(
            file.resolve()
        ).lower()

        unique_files[key] = file

    files = sorted(
        unique_files.values()
    )

    if not files:

        raise FileNotFoundError(
            "No STRICT paired telemetry workbooks found under:\n"
            f"{PAIRED_DATA_ROOT}"
        )

    return files


def identify_vehicle_sheet(
    workbook: Path,
    vehicle: str,
) -> str | int:
    """
    Identify the EV or ICE worksheet.
    """

    excel_file = pd.ExcelFile(
        workbook,
        engine="openpyxl",
    )

    sheet_names = (
        excel_file.sheet_names
    )

    vehicle_upper = vehicle.upper()

    # -------------------------------------------------------------------------
    # First: check sheet names
    # -------------------------------------------------------------------------

    for sheet in sheet_names:

        if vehicle_upper in str(
            sheet
        ).upper():

            return sheet

    # -------------------------------------------------------------------------
    # Second: inspect worksheet columns
    # -------------------------------------------------------------------------

    for sheet in sheet_names:

        try:

            preview = pd.read_excel(
                workbook,
                sheet_name=sheet,
                nrows=5,
                engine="openpyxl",
            )

            columns = " ".join(
                str(column).upper()
                for column in preview.columns
            )

            if (
                TIMESTAMP_COL.upper()
                in columns
                and
                PHASE_COL.upper()
                in columns
            ):

                return sheet

        except Exception:
            continue

    raise ValueError(
        f"Could not identify {vehicle} sheet in:\n"
        f"{workbook}\n"
        f"Available sheets: {sheet_names}"
    )


def load_vehicle_data(
    workbook: Path,
    vehicle: str,
) -> pd.DataFrame:
    """Load one EV or ICE telemetry worksheet."""

    sheet = identify_vehicle_sheet(
        workbook,
        vehicle,
    )

    df = pd.read_excel(
        workbook,
        sheet_name=sheet,
        engine="openpyxl",
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required_columns = [
        TIMESTAMP_COL,
        SPEED_MS_COL,
        SPEED_KMH_COL,
        TIME_DIFF_COL,
        DISTANCE_COL,
        ACCEL_COL,
        PHASE_COL,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{vehicle} telemetry is missing required columns:\n"
            f"{missing_columns}\n"
            f"Workbook: {workbook}\n"
            f"Sheet: {sheet}"
        )

    return df


def extract_pair_id(
    workbook: Path,
) -> str:
    """Extract Pair ID from workbook filename."""

    stem = workbook.stem

    parts = stem.split("_")

    if (
        len(parts) >= 2
        and parts[0].upper() == "PAIR"
    ):

        return "_".join(
            parts[:2]
        )

    return stem


def load_all_pairs():
    """Load all available STRICT EV-ICE pairs."""

    files = discover_strict_files()

    records = []

    for workbook in files:

        pair_id = extract_pair_id(
            workbook
        )

        try:

            ev = load_vehicle_data(
                workbook,
                "EV",
            )

            ice = load_vehicle_data(
                workbook,
                "ICE",
            )

            records.append(
                {
                    "pair_id": pair_id,
                    "workbook": workbook,
                    "EV": ev,
                    "ICE": ice,
                }
            )

        except Exception as error:

            print(
                "\n[WARNING]"
                "\nCould not load:"
                f"\n{workbook}"
                "\nReason:"
                f"\n{error}"
            )

    if not records:

        raise RuntimeError(
            "No valid STRICT EV-ICE telemetry pairs could be loaded."
        )

    # -------------------------------------------------------------------------
    # Deduplicate Pair IDs
    # -------------------------------------------------------------------------

    unique_pairs = {}

    for record in records:

        unique_pairs[
            record["pair_id"]
        ] = record

    records = list(
        unique_pairs.values()
    )

    records.sort(
        key=lambda item:
        item["pair_id"]
    )

    return records


# =============================================================================
# STAGE 1
# TELEMETRY & DATA STRUCTURE ANALYSIS
# =============================================================================

def analyze_telemetry_structure(
    records,
):
    """Analyse basic telemetry structure."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            start_time, end_time = (
                get_timestamp_range(df)
            )

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Start Timestamp": start_time,
                    "End Timestamp": end_time,
                    "Telemetry Duration (min)": (
                        get_duration_seconds(df)
                        / 60
                    ),
                    "Distance (km)": (
                        get_distance_km(df)
                    ),
                    "Missing Cells": int(
                        df.isna()
                        .sum()
                        .sum()
                    ),
                    "Duplicate Rows": int(
                        df.duplicated()
                        .sum()
                    ),
                    "Duplicate Timestamps": int(
                        df[TIMESTAMP_COL]
                        .duplicated()
                        .sum()
                    ),
                    "Missing Phase": int(
                        df[PHASE_COL]
                        .isna()
                        .sum()
                    ),
                    "Unique Phases": int(
                        df[PHASE_COL]
                        .apply(
                            normalize_phase
                        )
                        .nunique()
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# STAGE 2
# PHASE DISTRIBUTION & EPISODE ANALYSIS
# =============================================================================

def calculate_phase_distribution(
    records,
):
    """Calculate phase-wise rows, duration and distance."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle].copy()

            df["_phase"] = (
                df[PHASE_COL]
                .apply(normalize_phase)
            )

            grouped = (
                df.groupby(
                    "_phase",
                    dropna=False,
                )
                .agg(
                    Rows=(
                        "_phase",
                        "size",
                    ),
                    Duration_s=(
                        TIME_DIFF_COL,
                        "sum",
                    ),
                    Distance_m=(
                        DISTANCE_COL,
                        "sum",
                    ),
                )
                .reset_index()
            )

            for _, row in grouped.iterrows():

                rows.append(
                    {
                        "Pair ID": pair_id,
                        "Vehicle": vehicle,
                        "Phase": row["_phase"],
                        "Rows": int(
                            row["Rows"]
                        ),
                        "Duration (s)": float(
                            row["Duration_s"]
                        ),
                        "Duration (min)": float(
                            row["Duration_s"]
                            / 60
                        ),
                        "Distance (km)": float(
                            row["Distance_m"]
                            / 1000
                        ),
                    }
                )

    return pd.DataFrame(rows)


def extract_phase_episodes(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Extract consecutive row-level phase runs.

    These are analytical episodes only.

    They are NOT used as ICE-EV alignment units.
    """

    data = df.copy()

    data["_phase"] = (
        data[PHASE_COL]
        .apply(normalize_phase)
    )

    phases = data[
        "_phase"
    ].tolist()

    if not phases:
        return []

    episodes = []

    start_index = 0

    current_phase = phases[0]

    for i in range(
        1,
        len(phases),
    ):

        if phases[i] != current_phase:

            segment = data.iloc[
                start_index:i
            ]

            duration = (
                safe_numeric(
                    segment[
                        TIME_DIFF_COL
                    ]
                )
                .fillna(0)
                .clip(lower=0)
                .sum()
            )

            distance = (
                safe_numeric(
                    segment[
                        DISTANCE_COL
                    ]
                )
                .fillna(0)
                .clip(lower=0)
                .sum()
            )

            episodes.append(
                {
                    "Phase": current_phase,
                    "Start Row": start_index,
                    "End Row": i - 1,
                    "Rows": (
                        i - start_index
                    ),
                    "Duration (s)": float(
                        duration
                    ),
                    "Distance (m)": float(
                        distance
                    ),
                }
            )

            start_index = i

            current_phase = phases[i]

    # -------------------------------------------------------------------------
    # Last episode
    # -------------------------------------------------------------------------

    segment = data.iloc[
        start_index:
    ]

    duration = (
        safe_numeric(
            segment[TIME_DIFF_COL]
        )
        .fillna(0)
        .clip(lower=0)
        .sum()
    )

    distance = (
        safe_numeric(
            segment[DISTANCE_COL]
        )
        .fillna(0)
        .clip(lower=0)
        .sum()
    )

    episodes.append(
        {
            "Phase": current_phase,
            "Start Row": start_index,
            "End Row": len(data) - 1,
            "Rows": (
                len(data)
                - start_index
            ),
            "Duration (s)": float(
                duration
            ),
            "Distance (m)": float(
                distance
            ),
        }
    )

    return episodes


def calculate_episode_analysis(
    records,
):
    """Generate phase episode statistics."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            episodes = (
                extract_phase_episodes(
                    record[vehicle]
                )
            )

            for number, episode in enumerate(
                episodes,
                start=1,
            ):

                rows.append(
                    {
                        "Pair ID": pair_id,
                        "Vehicle": vehicle,
                        "Episode Number": number,
                        **episode,
                        "Duration (min)": (
                            episode[
                                "Duration (s)"
                            ]
                            / 60
                        ),
                        "Distance (km)": (
                            episode[
                                "Distance (m)"
                            ]
                            / 1000
                        ),
                    }
                )

    return pd.DataFrame(rows)


def calculate_phase_sequences(
    records,
):
    """Create chronological phase sequences."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            episodes = (
                extract_phase_episodes(
                    record[vehicle]
                )
            )

            sequence = [
                episode["Phase"]
                for episode in episodes
            ]

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Number of Episodes": len(
                        sequence
                    ),
                    "Phase Sequence": (
                        " -> ".join(
                            sequence
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_phase_fragmentation(
    records,
):
    """Quantify phase fragmentation."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            episodes = (
                extract_phase_episodes(
                    df
                )
            )

            total_rows = len(df)

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Telemetry Rows": total_rows,
                    "Phase Episodes": len(
                        episodes
                    ),
                    "Episodes per 100 Rows": (
                        safe_divide(
                            len(episodes) * 100,
                            total_rows,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# STAGE 3
# SAMPLING / TIME-GAP ANALYSIS
# =============================================================================

def calculate_sampling_analysis(
    records,
):
    """Analyse telemetry sampling intervals."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            time_diff = (
                safe_numeric(
                    df[TIME_DIFF_COL]
                )
                .dropna()
            )

            positive = time_diff[
                time_diff > 0
            ]

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Rows": len(df),
                    "Positive Intervals": len(
                        positive
                    ),
                    "Median Interval (s)": (
                        positive.median()
                        if not positive.empty
                        else np.nan
                    ),
                    "Mean Interval (s)": (
                        positive.mean()
                        if not positive.empty
                        else np.nan
                    ),
                    "P25 Interval (s)": (
                        percentile(
                            positive,
                            25,
                        )
                    ),
                    "P75 Interval (s)": (
                        percentile(
                            positive,
                            75,
                        )
                    ),
                    "P95 Interval (s)": (
                        percentile(
                            positive,
                            95,
                        )
                    ),
                    "Maximum Interval (s)": (
                        positive.max()
                        if not positive.empty
                        else np.nan
                    ),
                    "Intervals > 30s": int(
                        (
                            positive > 30
                        ).sum()
                    ),
                    "Intervals > 60s": int(
                        (
                            positive > 60
                        ).sum()
                    ),
                    "Intervals > 300s": int(
                        (
                            positive > 300
                        ).sum()
                    ),
                    "Intervals > 600s": int(
                        (
                            positive > 600
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_gap_phase_relationship(
    records,
):
    """
    Analyse the phase associated with long intervals.

    No gaps are removed or reclassified.
    """

    rows = []

    thresholds = [
        30,
        60,
        180,
        300,
        600,
    ]

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle].copy()

            df["_time_diff"] = (
                safe_numeric(
                    df[TIME_DIFF_COL]
                )
                .fillna(0)
            )

            df["_phase"] = (
                df[PHASE_COL]
                .apply(normalize_phase)
            )

            for threshold in thresholds:

                subset = df[
                    df["_time_diff"]
                    > threshold
                ]

                phase_counts = Counter(
                    subset["_phase"]
                )

                rows.append(
                    {
                        "Pair ID": pair_id,
                        "Vehicle": vehicle,
                        "Gap Threshold (s)": (
                            threshold
                        ),
                        "Number of Gaps": len(
                            subset
                        ),
                        "Gap Duration (s)": float(
                            subset[
                                "_time_diff"
                            ].sum()
                        ),
                        "Most Common Phase": (
                            phase_counts.most_common(
                                1
                            )[0][0]
                            if phase_counts
                            else "NONE"
                        ),
                        "CHARGING/REFUELING Gaps": (
                            int(
                                sum(
                                    count
                                    for phase, count
                                    in phase_counts.items()
                                    if phase
                                    in CHARGING_PHASES
                                )
                            )
                        ),
                        "STOP Gaps": int(
                            phase_counts.get(
                                "STOP",
                                0,
                            )
                        ),
                        "IDLE Gaps": int(
                            phase_counts.get(
                                "IDLE",
                                0,
                            )
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# STAGE 4
# ICE-EV TEMPORAL ALIGNMENT ANALYSIS
# =============================================================================

def calculate_trip_summary(
    records,
):
    """Create trip-level ICE-EV comparison."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        data = {
            "Pair ID": pair_id,
        }

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            start, end = (
                get_timestamp_range(df)
            )

            duration = (
                get_duration_seconds(df)
            )

            distance = (
                get_distance_km(df)
            )

            moving_speed = (
                safe_numeric(
                    df[SPEED_KMH_COL]
                )
            )

            moving_speed = moving_speed[
                moving_speed > 0
            ]

            data[
                f"{vehicle} Rows"
            ] = len(df)

            data[
                f"{vehicle} Duration (min)"
            ] = (
                duration / 60
            )

            data[
                f"{vehicle} Distance (km)"
            ] = distance

            data[
                f"{vehicle} Average Moving Speed (km/h)"
            ] = (
                moving_speed.mean()
                if not moving_speed.empty
                else np.nan
            )

            data[
                f"{vehicle} Start"
            ] = start

            data[
                f"{vehicle} End"
            ] = end

        data[
            "EV-ICE Duration Difference (min)"
        ] = (
            data["EV Duration (min)"]
            - data["ICE Duration (min)"]
        )

        data[
            "EV-ICE Distance Difference (km)"
        ] = (
            data["EV Distance (km)"]
            - data["ICE Distance (km)"]
        )

        rows.append(data)

    return pd.DataFrame(rows)


def interpolate_series(
    df: pd.DataFrame,
    timestamp_grid: pd.Series,
    column: str,
):
    """
    Interpolate a numeric signal onto a common time grid.

    This is exploratory analysis only.
    It does not modify source telemetry.
    """

    data = df.copy()

    timestamps = pd.to_datetime(
        data[TIMESTAMP_COL],
        errors="coerce",
    )

    values = safe_numeric(
        data[column]
    )

    valid = (
        timestamps.notna()
        & values.notna()
    )

    timestamps = timestamps[
        valid
    ]

    values = values[
        valid
    ]

    if len(timestamps) < 2:

        return np.full(
            len(timestamp_grid),
            np.nan,
        )

    temp = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
        }
    )

    temp = (
        temp
        .groupby(
            "timestamp",
            as_index=False,
        )["value"]
        .mean()
        .sort_values(
            "timestamp"
        )
    )

    x = (
        temp["timestamp"]
        .astype("int64")
        .to_numpy()
        .astype(float)
    )

    y = temp[
        "value"
    ].to_numpy(
        dtype=float
    )

    grid_x = (
        pd.to_datetime(
            timestamp_grid
        )
        .astype("int64")
        .to_numpy()
        .astype(float)
    )

    inside = (
        (grid_x >= x.min())
        &
        (grid_x <= x.max())
    )

    output = np.full(
        len(grid_x),
        np.nan,
    )

    output[inside] = np.interp(
        grid_x[inside],
        x,
        y,
    )

    return output


def calculate_temporal_alignment(
    records,
):
    """
    Exploratory ICE-EV temporal signal comparison.

    A common 5-second elapsed-time grid is used only to investigate
    temporal similarity.

    This is NOT a row-by-row prediction mapping.
    """

    rows = []

    signals = [
        SPEED_KMH_COL,
        SPEED_MS_COL,
        ACCEL_COL,
    ]

    for record in records:

        pair_id = record["pair_id"]

        ice = record["ICE"].copy()

        ev = record["EV"].copy()

        ice_start, ice_end = (
            get_timestamp_range(ice)
        )

        ev_start, ev_end = (
            get_timestamp_range(ev)
        )

        if (
            pd.isna(ice_start)
            or
            pd.isna(ev_start)
        ):
            continue

        start = max(
            ice_start,
            ev_start,
        )

        end = min(
            ice_end,
            ev_end,
        )

        if (
            pd.isna(end)
            or end <= start
        ):
            continue

        grid = pd.date_range(
            start=start,
            end=end,
            freq="5s",
        )

        if len(grid) < 3:
            continue

        for signal in signals:

            ice_values = (
                interpolate_series(
                    ice,
                    grid,
                    signal,
                )
            )

            ev_values = (
                interpolate_series(
                    ev,
                    grid,
                    signal,
                )
            )

            valid = (
                np.isfinite(
                    ice_values
                )
                &
                np.isfinite(
                    ev_values
                )
            )

            if valid.sum() < 3:

                correlation = np.nan
                mae = np.nan
                rmse = np.nan

            else:

                x = ice_values[
                    valid
                ]

                y = ev_values[
                    valid
                ]

                if (
                    np.std(x) > 0
                    and
                    np.std(y) > 0
                ):

                    correlation = float(
                        np.corrcoef(
                            x,
                            y,
                        )[0, 1]
                    )

                else:

                    correlation = np.nan

                mae = float(
                    np.mean(
                        np.abs(
                            x - y
                        )
                    )
                )

                rmse = float(
                    np.sqrt(
                        np.mean(
                            (x - y) ** 2
                        )
                    )
                )

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Signal": signal,
                    "Grid Interval (s)": 5,
                    "Common Grid Points": len(
                        grid
                    ),
                    "Valid Comparison Points": int(
                        valid.sum()
                    ),
                    "Pearson Correlation": (
                        correlation
                    ),
                    "MAE": mae,
                    "RMSE": rmse,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# STAGE 5
# PHASE-AWARE TEMPORAL REPRESENTATION ANALYSIS
# =============================================================================

def calculate_phase_time_distance_matrix(
    records,
):
    """
    Compare ICE and EV phase duration and distance.

    This is phase-level comparison.

    It does not attempt to align individual episodes.
    """

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for phase in MODELLING_PHASES:

            row = {
                "Pair ID": pair_id,
                "Phase": phase,
            }

            for vehicle in [
                "ICE",
                "EV",
            ]:

                df = record[vehicle].copy()

                df["_phase"] = (
                    df[PHASE_COL]
                    .apply(
                        normalize_phase
                    )
                )

                subset = df[
                    df["_phase"]
                    == phase
                ]

                duration = (
                    safe_numeric(
                        subset[
                            TIME_DIFF_COL
                        ]
                    )
                    .fillna(0)
                    .clip(lower=0)
                    .sum()
                )

                distance = (
                    safe_numeric(
                        subset[
                            DISTANCE_COL
                        ]
                    )
                    .fillna(0)
                    .clip(lower=0)
                    .sum()
                )

                row[
                    f"{vehicle} Duration (min)"
                ] = (
                    duration / 60
                )

                row[
                    f"{vehicle} Distance (km)"
                ] = (
                    distance / 1000
                )

                row[
                    f"{vehicle} Rows"
                ] = len(subset)

            row[
                "EV-ICE Duration Difference (min)"
            ] = (
                row["EV Duration (min)"]
                -
                row["ICE Duration (min)"]
            )

            row[
                "EV-ICE Distance Difference (km)"
            ] = (
                row["EV Distance (km)"]
                -
                row["ICE Distance (km)"]
            )

            rows.append(row)

    return pd.DataFrame(rows)


def calculate_phase_order_analysis(
    records,
):
    """
    Compare the first occurrence order of major phases.

    This is descriptive only.
    """

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            episodes = (
                extract_phase_episodes(
                    record[vehicle]
                )
            )

            sequence = [
                episode["Phase"]
                for episode in episodes
            ]

            first_positions = {}

            for index, phase in enumerate(
                sequence,
                start=1,
            ):

                if (
                    phase
                    not in first_positions
                ):

                    first_positions[
                        phase
                    ] = index

            for phase in MODELLING_PHASES:

                rows.append(
                    {
                        "Pair ID": pair_id,
                        "Vehicle": vehicle,
                        "Phase": phase,
                        "First Episode Position": (
                            first_positions.get(
                                phase,
                                np.nan,
                            )
                        ),
                        "Occurs": (
                            phase
                            in first_positions
                        ),
                    }
                )

    return pd.DataFrame(rows)


def calculate_phase_aware_time_grid(
    records,
):
    """
    Create an exploratory normalized elapsed-time representation.

    Features:
        - Pair ID
        - Vehicle
        - original row
        - normalized elapsed time
        - cumulative distance
        - speed
        - acceleration
        - time difference
        - phase

    This representation retains time spent in STOP/IDLE/charging phases.
    """

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle].copy()

            time_diff = (
                safe_numeric(
                    df[TIME_DIFF_COL]
                )
                .fillna(0)
                .clip(lower=0)
            )

            distance = (
                safe_numeric(
                    df[DISTANCE_COL]
                )
                .fillna(0)
                .clip(lower=0)
            )

            cumulative_time = (
                time_diff.cumsum()
            )

            cumulative_distance = (
                distance.cumsum()
                / 1000
            )

            if len(
                cumulative_time
            ):

                total_time = (
                    cumulative_time.iloc[-1]
                )

            else:

                total_time = 0

            if total_time > 0:

                normalized_time = (
                    cumulative_time
                    / total_time
                )

            else:

                normalized_time = (
                    pd.Series(
                        np.zeros(
                            len(df)
                        )
                    )
                )

            speed = safe_numeric(
                df[SPEED_KMH_COL]
            )

            acceleration = safe_numeric(
                df[ACCEL_COL]
            )

            for index in range(
                len(df)
            ):

                rows.append(
                    {
                        "Pair ID": pair_id,
                        "Vehicle": vehicle,
                        "Original Row": index,
                        "Normalized Time": float(
                            normalized_time.iloc[
                                index
                            ]
                        ),
                        "Cumulative Distance (km)": float(
                            cumulative_distance.iloc[
                                index
                            ]
                        ),
                        "Speed (km/h)": (
                            speed.iloc[index]
                        ),
                        "Acceleration (m/s^2)": (
                            acceleration.iloc[
                                index
                            ]
                        ),
                        "Time Difference (s)": float(
                            time_diff.iloc[
                                index
                            ]
                        ),
                        "Phase": normalize_phase(
                            df[
                                PHASE_COL
                            ].iloc[index]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def calculate_distance_progress_representation(
    records,
):
    """
    Create an exploratory normalized distance-progress representation.

    This is useful mainly for moving-phase analysis.

    Stops and charging have zero/very small distance progression,
    so this representation is kept separate from time progression.
    """

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle].copy()

            distance = (
                safe_numeric(
                    df[DISTANCE_COL]
                )
                .fillna(0)
                .clip(lower=0)
            )

            cumulative_distance = (
                distance.cumsum()
                / 1000
            )

            if len(
                cumulative_distance
            ):

                total_distance = (
                    cumulative_distance.iloc[
                        -1
                    ]
                )

            else:

                total_distance = 0

            if total_distance > 0:

                normalized_distance = (
                    cumulative_distance
                    / total_distance
                )

            else:

                normalized_distance = (
                    pd.Series(
                        np.zeros(
                            len(df)
                        )
                    )
                )

            speed = safe_numeric(
                df[SPEED_KMH_COL]
            )

            acceleration = safe_numeric(
                df[ACCEL_COL]
            )

            time_diff = safe_numeric(
                df[TIME_DIFF_COL]
            ).fillna(0)

            for index in range(
                len(df)
            ):

                rows.append(
                    {
                        "Pair ID": pair_id,
                        "Vehicle": vehicle,
                        "Original Row": index,
                        "Normalized Distance": float(
                            normalized_distance.iloc[
                                index
                            ]
                        ),
                        "Cumulative Distance (km)": float(
                            cumulative_distance.iloc[
                                index
                            ]
                        ),
                        "Speed (km/h)": (
                            speed.iloc[index]
                        ),
                        "Acceleration (m/s^2)": (
                            acceleration.iloc[
                                index
                            ]
                        ),
                        "Time Difference (s)": float(
                            time_diff.iloc[
                                index
                            ]
                        ),
                        "Phase": normalize_phase(
                            df[
                                PHASE_COL
                            ].iloc[index]
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# STAGE 6
# TIME-SERIES DATASET DESIGN & VALIDATION
# =============================================================================

def calculate_dataset_design(
    records,
):
    """
    Validate availability of temporal modelling information.
    """

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            phase_values = (
                df[PHASE_COL]
                .apply(
                    normalize_phase
                )
            )

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Telemetry Rows": len(df),
                    "Temporal Order Available": (
                        TIMESTAMP_COL
                        in df.columns
                    ),
                    "Pair Identity Available": True,
                    "Phase Available": (
                        PHASE_COL
                        in df.columns
                    ),
                    "Speed Available": (
                        SPEED_KMH_COL
                        in df.columns
                    ),
                    "Acceleration Available": (
                        ACCEL_COL
                        in df.columns
                    ),
                    "Time Difference Available": (
                        TIME_DIFF_COL
                        in df.columns
                    ),
                    "Distance Increment Available": (
                        DISTANCE_COL
                        in df.columns
                    ),
                    "Charging/Refuelling Present": bool(
                        phase_values.isin(
                            CHARGING_PHASES
                        ).any()
                    ),
                    "Modelling Phases Present": int(
                        phase_values.isin(
                            MODELLING_PHASES
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_sequence_length_analysis(
    records,
):
    """Analyse sequence lengths for future window design."""

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Rows": len(df),
                    "Minimum Rows Required for 10-step Window": (
                        len(df) >= 10
                    ),
                    "Minimum Rows Required for 20-step Window": (
                        len(df) >= 20
                    ),
                    "Minimum Rows Required for 30-step Window": (
                        len(df) >= 30
                    ),
                    "Minimum Rows Required for 60-step Window": (
                        len(df) >= 60
                    ),
                    "Minimum Rows Required for 100-step Window": (
                        len(df) >= 100
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# STAGE 7
# TIME-SERIES MODEL DESIGN
# =============================================================================

def calculate_model_input_output_design():
    """
    Document the planned TS model families.

    This does not train the models.
    """

    rows = [
        {
            "Model": "Persistence",
            "Model Family": "Baseline",
            "Input": "Previous EV temporal value",
            "Target": "Next EV temporal value",
            "ICE Used": "No",
            "Purpose": "Naive temporal benchmark",
        },
        {
            "Model": "VAR",
            "Model Family": "Classical Time-Series",
            "Input": "Historical multivariate temporal signals",
            "Target": "Future multivariate temporal signals",
            "ICE Used": "Yes, in multivariate temporal representation",
            "Purpose": "Classical temporal benchmark",
        },
        {
            "Model": "XGBoost + Lag Features",
            "Model Family": "ML-based Temporal",
            "Input": "ICE temporal signals + lagged temporal features",
            "Target": "EV temporal signal",
            "ICE Used": "Yes",
            "Purpose": "Nonlinear lag-based temporal model",
        },
        {
            "Model": "LSTM",
            "Model Family": "Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Sequence dependency modelling",
        },
        {
            "Model": "GRU",
            "Model Family": "Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Gated sequence modelling",
        },
        {
            "Model": "Stacked LSTM",
            "Model Family": "Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Deeper sequence modelling",
        },
        {
            "Model": "Transformer",
            "Model Family": "Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Attention-based temporal modelling",
        },
        {
            "Model": "1D CNN",
            "Model Family": "Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Local temporal pattern extraction",
        },
        {
            "Model": "CNN-LSTM",
            "Model Family": "Hybrid Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Local + sequential pattern modelling",
        },
        {
            "Model": "TCN",
            "Model Family": "Deep Temporal",
            "Input": "ICE temporal sequences",
            "Target": "EV temporal sequence",
            "ICE Used": "Yes",
            "Purpose": "Dilated causal temporal modelling",
        },
    ]

    return pd.DataFrame(rows)


def calculate_modelling_readiness(
    records,
):
    """
    Conservative structural modelling-readiness assessment.

    Because only a small number of STRICT trips are currently available,
    this does NOT claim that the models will perform well.
    """

    total_pairs = len(records)

    rows = []

    checks = [
        (
            "STRICT paired trips available",
            total_pairs >= 2,
            f"{total_pairs} STRICT pairs found",
        ),
        (
            "Pair identity available",
            True,
            "Pair ID available for grouped validation",
        ),
        (
            "Temporal ordering available",
            True,
            "Timestamp column available",
        ),
        (
            "Phase information available",
            True,
            "Existing Part-1 phase column available",
        ),
        (
            "ICE telemetry available",
            all(
                len(
                    record["ICE"]
                ) > 0
                for record in records
            ),
            "ICE telemetry loaded",
        ),
        (
            "EV telemetry available",
            all(
                len(
                    record["EV"]
                ) > 0
                for record in records
            ),
            "EV telemetry loaded",
        ),
        (
            "Variable sequence lengths handled",
            True,
            "Trips retain original sequence lengths",
        ),
        (
            "Row-wise ICE-EV matching required",
            False,
            "Not required because telemetry row counts differ",
        ),
        (
            "Long gaps automatically removed",
            False,
            "No automatic deletion of long intervals",
        ),
        (
            "Part-1 segmentation rerun",
            False,
            "Existing phase labels are used directly",
        ),
        (
            "Charging/refuelling used as predictive target",
            False,
            "Handled separately from predictive modelling",
        ),
        (
            "LOPO validation possible",
            total_pairs >= 2,
            "Pair ID can be used as validation group",
        ),
    ]

    for (
        check_name,
        status,
        comment,
    ) in checks:

        rows.append(
            {
                "Check": check_name,
                "Status": (
                    "PASS"
                    if status
                    else "FAIL"
                ),
                "Comment": comment,
            }
        )

    return pd.DataFrame(rows)


def calculate_representation_comparison(
    records,
):
    """
    Compare time-progress and distance-progress representations.
    """

    rows = []

    for record in records:

        pair_id = record["pair_id"]

        for vehicle in [
            "ICE",
            "EV",
        ]:

            df = record[vehicle]

            time_diff = (
                safe_numeric(
                    df[TIME_DIFF_COL]
                )
                .fillna(0)
                .clip(lower=0)
            )

            distance = (
                safe_numeric(
                    df[DISTANCE_COL]
                )
                .fillna(0)
                .clip(lower=0)
            )

            total_time = (
                time_diff.sum()
            )

            total_distance = (
                distance.sum()
            )

            zero_distance_duration = (
                time_diff[
                    distance <= 0
                ].sum()
            )

            moving_duration = (
                time_diff[
                    distance > 0
                ].sum()
            )

            zero_distance_share = (
                safe_divide(
                    zero_distance_duration,
                    total_time,
                )
            )

            if not pd.isna(
                zero_distance_share
            ):

                zero_distance_share *= 100

            rows.append(
                {
                    "Pair ID": pair_id,
                    "Vehicle": vehicle,
                    "Total Duration (min)": (
                        total_time / 60
                    ),
                    "Total Distance (km)": (
                        total_distance / 1000
                    ),
                    "Zero-Distance Duration (min)": (
                        zero_distance_duration
                        / 60
                    ),
                    "Moving Duration (min)": (
                        moving_duration
                        / 60
                    ),
                    "Zero-Distance Duration Share (%)": (
                        zero_distance_share
                    ),
                    "Distance-Progress Representation": (
                        "Suitable mainly for moving-phase analysis"
                    ),
                    "Time-Progress Representation": (
                        "Suitable for full-trip temporal analysis"
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# MASTER SUMMARY
# =============================================================================

def create_master_summary(
    records,
    structure,
    phase_distribution,
    sampling,
    trip_summary,
    temporal_alignment,
    readiness,
):
    """Create one-page summary of the complete analysis."""

    total_pairs = len(records)

    total_rows_ice = sum(
        len(record["ICE"])
        for record in records
    )

    total_rows_ev = sum(
        len(record["EV"])
        for record in records
    )

    successful_checks = int(
        (
            readiness["Status"]
            == "PASS"
        ).sum()
    )

    total_checks = len(
        readiness
    )

    return pd.DataFrame(
        [
            {
                "Analysis": "STRICT paired trips",
                "Result": total_pairs,
                "Interpretation": (
                    "Number of paired trips included"
                ),
            },
            {
                "Analysis": "Total ICE telemetry rows",
                "Result": total_rows_ice,
                "Interpretation": (
                    "Existing Part-1 ICE telemetry"
                ),
            },
            {
                "Analysis": "Total EV telemetry rows",
                "Result": total_rows_ev,
                "Interpretation": (
                    "Existing Part-1 EV telemetry"
                ),
            },
            {
                "Analysis": "Phase information",
                "Result": "Available",
                "Interpretation": (
                    "Existing Part-1 phase labels used"
                ),
            },
            {
                "Analysis": "ICE-EV row-wise matching",
                "Result": "Not used",
                "Interpretation": (
                    "Telemetry row counts and timestamps differ"
                ),
            },
            {
                "Analysis": "Long time gaps",
                "Result": "Retained",
                "Interpretation": (
                    "Investigated using existing phase information"
                ),
            },
            {
                "Analysis": "Charging/refuelling",
                "Result": "Separate",
                "Interpretation": (
                    "Not treated as predictive target"
                ),
            },
            {
                "Analysis": "LOPO validation",
                "Result": (
                    "Supported"
                    if total_pairs >= 2
                    else "Not supported"
                ),
                "Interpretation": (
                    "Pair ID is the grouping unit"
                ),
            },
            {
                "Analysis": "Modelling readiness checks",
                "Result": (
                    f"{successful_checks}/{total_checks} PASS"
                ),
                "Interpretation": (
                    "Structural readiness only; "
                    "performance must be evaluated experimentally"
                ),
            },
        ]
    )


# =============================================================================
# WRITE EXCEL RESULTS
# =============================================================================

def write_results(
    summary,
    structure,
    phase_distribution,
    episodes,
    sequences,
    fragmentation,
    sampling,
    gap_phase,
    trip_summary,
    temporal_alignment,
    phase_summary,
    phase_order,
    time_grid,
    distance_grid,
    dataset_design,
    sequence_lengths,
    model_design,
    readiness,
    representation_comparison,
):
    """
    Write all analysis outputs into one Excel workbook.

    IMPORTANT:
    `writer.book[...]` is used to access worksheets.
    `writer[...]` is NOT valid for OpenpyxlWriter.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheets = {
        "Master_Summary": summary,

        "1_Telemetry": structure,

        "2_Phase_Distribution": (
            phase_distribution
        ),

        "2_Phase_Episodes": episodes,

        "2_Phase_Sequences": sequences,

        "2_Phase_Fragmentation": (
            fragmentation
        ),

        "3_Sampling": sampling,

        "3_Gap_Phase": gap_phase,

        "4_Trip_Alignment": trip_summary,

        "4_Temporal_Alignment": (
            temporal_alignment
        ),

        "5_Phase_Summary": phase_summary,

        "5_Phase_Order": phase_order,

        "5_Time_Progress": time_grid,

        "5_Distance_Progress": (
            distance_grid
        ),

        "6_Dataset_Design": dataset_design,

        "6_Sequence_Length": (
            sequence_lengths
        ),

        "7_Model_Design": model_design,

        "7_Readiness": readiness,

        "7_Representation_Comparison": (
            representation_comparison
        ),
    }

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl",
    ) as writer:

        for (
            sheet_name,
            dataframe,
        ) in sheets.items():

            final_sheet_name = (
                safe_excel_sheet_name(
                    sheet_name
                )
            )

            # -----------------------------------------------------------------
            # Write dataframe
            # -----------------------------------------------------------------

            dataframe.to_excel(
                writer,
                sheet_name=final_sheet_name,
                index=False,
            )

            # -----------------------------------------------------------------
            # Correct way to access worksheet
            # -----------------------------------------------------------------

            worksheet = writer.book[
                final_sheet_name
            ]

            # -----------------------------------------------------------------
            # Freeze header row
            # -----------------------------------------------------------------

            worksheet.freeze_panes = "A2"

            # -----------------------------------------------------------------
            # Auto-size columns
            # -----------------------------------------------------------------

            for column_cells in (
                worksheet.columns
            ):

                max_length = 0

                for cell in column_cells:

                    value = (
                        ""
                        if cell.value is None
                        else str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        len(value),
                    )

                worksheet.column_dimensions[
                    column_cells[
                        0
                    ].column_letter
                ].width = min(
                    max(
                        max_length + 2,
                        10,
                    ),
                    45,
                )

    print(
        "\nResults written to:"
        f"\n{OUTPUT_FILE}"
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TIME-SERIES MASTER ANALYSIS PIPELINE"
    )

    print(
        "=" * 80
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print(
        "\nLoading STRICT paired telemetry..."
    )

    records = load_all_pairs()

    print(
        f"STRICT pairs loaded: "
        f"{len(records)}"
    )

    for record in records:

        print(
            f"  {record['pair_id']}: "
            f"ICE={len(record['ICE'])} rows, "
            f"EV={len(record['EV'])} rows"
        )

    # =========================================================================
    # STAGE 1
    # =========================================================================

    print(
        "\n[1/7] "
        "Telemetry & Data Structure Analysis..."
    )

    structure = (
        analyze_telemetry_structure(
            records
        )
    )

    # =========================================================================
    # STAGE 2
    # =========================================================================

    print(
        "\n[2/7] "
        "Phase Distribution & Episode Analysis..."
    )

    phase_distribution = (
        calculate_phase_distribution(
            records
        )
    )

    episodes = (
        calculate_episode_analysis(
            records
        )
    )

    sequences = (
        calculate_phase_sequences(
            records
        )
    )

    fragmentation = (
        calculate_phase_fragmentation(
            records
        )
    )

    # =========================================================================
    # STAGE 3
    # =========================================================================

    print(
        "\n[3/7] "
        "Sampling / Time-Gap Analysis..."
    )

    sampling = (
        calculate_sampling_analysis(
            records
        )
    )

    gap_phase = (
        calculate_gap_phase_relationship(
            records
        )
    )

    # =========================================================================
    # STAGE 4
    # =========================================================================

    print(
        "\n[4/7] "
        "ICE-EV Temporal Alignment Analysis..."
    )

    trip_summary = (
        calculate_trip_summary(
            records
        )
    )

    temporal_alignment = (
        calculate_temporal_alignment(
            records
        )
    )

    # =========================================================================
    # STAGE 5
    # =========================================================================

    print(
        "\n[5/7] "
        "Phase-Aware Temporal Representation Analysis..."
    )

    phase_summary = (
        calculate_phase_time_distance_matrix(
            records
        )
    )

    phase_order = (
        calculate_phase_order_analysis(
            records
        )
    )

    time_grid = (
        calculate_phase_aware_time_grid(
            records
        )
    )

    distance_grid = (
        calculate_distance_progress_representation(
            records
        )
    )

    # =========================================================================
    # STAGE 6
    # =========================================================================

    print(
        "\n[6/7] "
        "Time-Series Dataset Design & Validation..."
    )

    dataset_design = (
        calculate_dataset_design(
            records
        )
    )

    sequence_lengths = (
        calculate_sequence_length_analysis(
            records
        )
    )

    # =========================================================================
    # STAGE 7
    # =========================================================================

    print(
        "\n[7/7] "
        "Time-Series Modelling Readiness Analysis..."
    )

    model_design = (
        calculate_model_input_output_design()
    )

    readiness = (
        calculate_modelling_readiness(
            records
        )
    )

    representation_comparison = (
        calculate_representation_comparison(
            records
        )
    )

    # =========================================================================
    # MASTER SUMMARY
    # =========================================================================

    summary = create_master_summary(
        records=records,
        structure=structure,
        phase_distribution=phase_distribution,
        sampling=sampling,
        trip_summary=trip_summary,
        temporal_alignment=temporal_alignment,
        readiness=readiness,
    )

    # =========================================================================
    # WRITE OUTPUT
    # =========================================================================

    print(
        "\nWriting consolidated analysis workbook..."
    )

    write_results(
        summary=summary,
        structure=structure,
        phase_distribution=phase_distribution,
        episodes=episodes,
        sequences=sequences,
        fragmentation=fragmentation,
        sampling=sampling,
        gap_phase=gap_phase,
        trip_summary=trip_summary,
        temporal_alignment=temporal_alignment,
        phase_summary=phase_summary,
        phase_order=phase_order,
        time_grid=time_grid,
        distance_grid=distance_grid,
        dataset_design=dataset_design,
        sequence_lengths=sequence_lengths,
        model_design=model_design,
        readiness=readiness,
        representation_comparison=(
            representation_comparison
        ),
    )

    # =========================================================================
    # TERMINAL SUMMARY
    # =========================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TIME-SERIES ANALYSIS COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSTRICT pairs       : "
        f"{len(records)}"
    )

    print(
        f"ICE telemetry rows: "
        f"{sum(len(r['ICE']) for r in records)}"
    )

    print(
        f"EV telemetry rows : "
        f"{sum(len(r['EV']) for r in records)}"
    )

    print(
        "\nImportant modelling decisions:"
    )

    print(
        "  - Existing Part-1 phase labels are retained."
    )

    print(
        "  - Part-1 cleaning is NOT rerun."
    )

    print(
        "  - Part-1 feature engineering is NOT rerun."
    )

    print(
        "  - Part-1 segmentation is NOT rerun."
    )

    print(
        "  - Long time gaps are investigated, not automatically removed."
    )

    print(
        "  - ICE and EV rows are NOT forced into one-to-one correspondence."
    )

    print(
        "  - Phase episodes are analysed but are NOT used as alignment units."
    )

    print(
        "  - Time-progress and distance-progress representations are examined."
    )

    print(
        "  - Charging/refuelling remains separate from predictive modelling."
    )

    print(
        "  - Pair ID is retained for LOPO validation."
    )

    print(
        "  - Final TS models are NOT trained by this analysis script."
    )

    print(
        "\nOutput:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        "\n"
        + "=" * 80
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()