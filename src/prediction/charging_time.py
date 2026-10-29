"""
TAT MS INTERCITY MODELLING
COMMON EV CHARGING-TIME PIPELINE

Version:
    2026-09-30_CHARGING_V6_SINGLE_SOC_RAW_SHEET2

PURPOSE
-------
Calculate ONE model-independent EN-ROUTE EV charging time for each processed
raw ICE trip.

The operational charging assumption is intentionally simple and fixed:

    Assumed departure SOC = 90%
    Minimum SOC           = 20%
    En-route target SOC   = 80%

The batch prediction pipeline does NOT create an 80% sensitivity scenario.

INPUTS
------
1. Results/Prediction/ICE_Ready/ICE_Prediction_Input_Master.xlsx
2. The original raw ICE workbook named in "Source ICE File"
3. EV/charger specifications from Sheet 2 (or a later non-telemetry sheet)

Required specification names:
    Battery Capacity
    Energy Consumption
    Charging Efficiency
    Charger Power

PHYSICAL MODEL
--------------
For trip distance D:

    E_trip = D * e

    E_initial_usable =
        C * (SOC_departure - SOC_min)

    E_enroute_battery =
        max(0, E_trip - E_initial_usable)

    E_recharge_window =
        C * (SOC_target - SOC_min)

For each required en-route charging stop:

    E_charger =
        E_battery_added / charging_efficiency

    T_charge =
        E_charger / charger_power

Only charging performed DURING the trip contributes to charging time.
Pre-trip/departure charging is not added to TAT.

OUTPUT
------
One row per ICE trip with one:

    Charging Time (min)

This value is consumed by final_tat_assembly.py.

PAIR CLASS / STRICT / NONSTRICT
-------------------------------
Not applicable to a new raw ICE trip.

This module:
    - does NOT read Pair Class,
    - does NOT recalculate Pair Class,
    - does NOT overwrite Pair Class,
    - does NOT infer STRICT/NONSTRICT.

COMPATIBILITY
-------------
The public calculate_charging_time() function retains the optional
departure_soc_override argument because the current historical Full-ML module
calls that API. The normal batch prediction pipeline does NOT use an override
and therefore always uses the single 90% operational assumption.

When the historical Full-ML code is updated for the final master pipeline, its
old sensitivity reporting can also be removed without changing this batch
charging workflow.
"""

from __future__ import annotations

import argparse
import math
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCRIPT_VERSION = "2026-09-30_CHARGING_V6_SINGLE_SOC_RAW_SHEET2"

ASSUMED_DEPARTURE_SOC = 0.90
MINIMUM_SOC = 0.20
CHARGE_TARGET_SOC = 0.80

DEFAULT_VEHICLE_ID = "INTERCITY_EBUS_PROTOTYPE_01"

# Temporary development fallback used by the callable API for historical-model
# compatibility. The raw-ICE batch workflow reads Sheet 2 and does not use this
# fallback when valid raw-workbook specifications are available.
PROTOTYPE_VEHICLE_SPECS: dict[str, dict[str, Any]] = {
    DEFAULT_VEHICLE_ID: {
        "battery_capacity_kwh": 423.0,
        "energy_consumption_kwh_per_km": 1.30,
        "charging_efficiency": 0.90,
        "charger_power_kw": 180.0,
        "start_soc": ASSUMED_DEPARTURE_SOC,
        "minimum_soc": MINIMUM_SOC,
        "charge_target_soc": CHARGE_TARGET_SOC,
    }
}


REQUIRED_MASTER_COLUMNS = [
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
    "ICE Total Distance (km)",
]


@dataclass(frozen=True)
class VehicleChargerSpecification:
    vehicle_id: str
    battery_capacity_kwh: float
    energy_consumption_kwh_per_km: float
    charging_efficiency: float
    charger_power_kw: float
    source_sheet: str
    start_soc: float = ASSUMED_DEPARTURE_SOC
    minimum_soc: float = MINIMUM_SOC
    charge_target_soc: float = CHARGE_TARGET_SOC


@dataclass(frozen=True)
class ChargingResult:
    vehicle_id: str
    trip_distance_km: float

    start_soc: float
    minimum_soc: float
    charge_target_soc: float

    trip_energy_required_kwh: float
    initial_usable_battery_energy_kwh: float
    initial_driving_range_km: float

    enroute_battery_energy_required_kwh: float
    enroute_energy_from_charger_kwh: float
    recharge_window_battery_energy_kwh: float

    number_of_charging_stops: int
    charging_required: bool

    battery_energy_per_stop_kwh: tuple[float, ...]
    charger_energy_per_stop_kwh: tuple[float, ...]
    charger_powers_kw: tuple[float, ...]
    charging_times_per_stop_h: tuple[float, ...]

    total_charging_time_h: float
    total_charging_time_min: float

    stop_rows: tuple[dict[str, Any], ...]


def detect_project_root(script_path: Path) -> Path:
    script_path = script_path.resolve()

    for candidate in (
        script_path.parent,
        *script_path.parents,
    ):
        if (
            candidate.name.casefold()
            == "tat_ms_intercity_modelling"
        ):
            return candidate

    for candidate in (
        script_path.parent,
        *script_path.parents,
    ):
        if (
            (candidate / "src").is_dir()
            and (candidate / "Results").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate TAT_MS_INTERCITY_MODELLING "
        f"from script path:\n{script_path}"
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(
        value
    ).strip()


def _normalise_key(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        _clean_text(
            value
        ).casefold(),
    )


def _canonical_spec_key(
    value: Any,
) -> str | None:
    key = _normalise_key(
        value
    )

    if key.startswith(
        "batterycapacity"
    ):
        return "battery_capacity_kwh"

    if key.startswith(
        "energyconsumption"
    ):
        return "energy_consumption_kwh_per_km"

    if key.startswith(
        "chargingefficiency"
    ):
        return "charging_efficiency"

    if key.startswith(
        "chargerpower"
    ):
        return "charger_power_kw"

    if (
        key.startswith(
            "vehicleid"
        )
        or key.startswith(
            "evvehicleid"
        )
    ):
        return "vehicle_id"

    return None


def _parse_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(
            value,
            (
                int,
                float,
            ),
        )
        and not isinstance(
            value,
            bool,
        )
    ):
        number = float(
            value
        )

    else:
        text = (
            _clean_text(
                value
            )
            .replace(
                ",",
                "",
            )
        )

        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
            text,
        )

        if not match:
            raise ValueError(
                f"Could not read numeric {name}: {value!r}"
            )

        number = float(
            match.group(
                0
            )
        )

    if not math.isfinite(
        number
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return number


def _validate_soc(
    value: float,
    name: str,
) -> None:
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def _validate_spec(
    spec: VehicleChargerSpecification,
) -> None:
    if (
        not math.isfinite(
            spec.battery_capacity_kwh
        )
        or spec.battery_capacity_kwh
        <= 0
    ):
        raise ValueError(
            "Battery Capacity must be finite and > 0 kWh."
        )

    if (
        not math.isfinite(
            spec.energy_consumption_kwh_per_km
        )
        or spec.energy_consumption_kwh_per_km
        <= 0
    ):
        raise ValueError(
            "Energy Consumption must be finite and > 0 kWh/km."
        )

    if (
        not math.isfinite(
            spec.charging_efficiency
        )
        or not (
            0
            < spec.charging_efficiency
            <= 1
        )
    ):
        raise ValueError(
            "Charging Efficiency must be in (0, 1]."
        )

    if (
        not math.isfinite(
            spec.charger_power_kw
        )
        or spec.charger_power_kw
        <= 0
    ):
        raise ValueError(
            "Charger Power must be finite and > 0 kW."
        )

    _validate_soc(
        spec.start_soc,
        "Departure SOC",
    )

    _validate_soc(
        spec.minimum_soc,
        "Minimum SOC",
    )

    _validate_soc(
        spec.charge_target_soc,
        "Charge Target SOC",
    )

    if (
        spec.start_soc
        <= spec.minimum_soc
    ):
        raise ValueError(
            "Departure SOC must be greater than minimum SOC."
        )

    if (
        spec.charge_target_soc
        <= spec.minimum_soc
    ):
        raise ValueError(
            "Charge Target SOC must be greater than minimum SOC."
        )


def _extract_specs_from_frame(
    frame: pd.DataFrame,
    sheet_name: str,
) -> VehicleChargerSpecification | None:
    found: dict[
        str,
        Any,
    ] = {}

    rows, cols = (
        frame.shape
    )

    for row_index in range(
        rows
    ):
        for column_index in range(
            cols
        ):
            canonical = (
                _canonical_spec_key(
                    frame.iat[
                        row_index,
                        column_index,
                    ]
                )
            )

            if canonical is None:
                continue

            value = None

            for next_column in range(
                column_index + 1,
                cols,
            ):
                candidate = (
                    frame.iat[
                        row_index,
                        next_column,
                    ]
                )

                if _clean_text(
                    candidate
                ):
                    value = candidate
                    break

            if (
                value is None
                and row_index + 1
                < rows
            ):
                candidate = (
                    frame.iat[
                        row_index + 1,
                        column_index,
                    ]
                )

                if _clean_text(
                    candidate
                ):
                    value = candidate

            if value is not None:
                found[
                    canonical
                ] = value

    required = {
        "battery_capacity_kwh",
        "energy_consumption_kwh_per_km",
        "charging_efficiency",
        "charger_power_kw",
    }

    if not required.issubset(
        found
    ):
        return None

    efficiency = _parse_number(
        found[
            "charging_efficiency"
        ],
        "Charging Efficiency",
    )

    if (
        1
        < efficiency
        <= 100
    ):
        efficiency /= 100.0

    spec = VehicleChargerSpecification(
        vehicle_id=(
            _clean_text(
                found.get(
                    "vehicle_id"
                )
            )
            or DEFAULT_VEHICLE_ID
        ),
        battery_capacity_kwh=(
            _parse_number(
                found[
                    "battery_capacity_kwh"
                ],
                "Battery Capacity",
            )
        ),
        energy_consumption_kwh_per_km=(
            _parse_number(
                found[
                    "energy_consumption_kwh_per_km"
                ],
                "Energy Consumption",
            )
        ),
        charging_efficiency=(
            efficiency
        ),
        charger_power_kw=(
            _parse_number(
                found[
                    "charger_power_kw"
                ],
                "Charger Power",
            )
        ),
        source_sheet=(
            sheet_name
        ),
        start_soc=(
            ASSUMED_DEPARTURE_SOC
        ),
        minimum_soc=(
            MINIMUM_SOC
        ),
        charge_target_soc=(
            CHARGE_TARGET_SOC
        ),
    )

    _validate_spec(
        spec
    )

    return spec


def read_vehicle_charger_specs(
    raw_workbook: Path,
) -> VehicleChargerSpecification:
    excel = pd.ExcelFile(
        raw_workbook
    )

    if len(
        excel.sheet_names
    ) < 2:
        raise ValueError(
            f"{raw_workbook.name} has only one worksheet. "
            "Put EV/charger specifications in Sheet 2."
        )

    checked: list[
        str
    ] = []

    for sheet_name in (
        excel.sheet_names[
            1:
        ]
    ):
        checked.append(
            sheet_name
        )

        frame = pd.read_excel(
            raw_workbook,
            sheet_name=(
                sheet_name
            ),
            header=None,
        )

        spec = (
            _extract_specs_from_frame(
                frame,
                sheet_name,
            )
        )

        if spec is not None:
            return spec

    raise ValueError(
        "Could not find all required EV/charger specifications "
        "in Sheet 2 or later. Required: Battery Capacity, "
        "Energy Consumption, Charging Efficiency, Charger Power. "
        f"Checked sheets: {checked}"
    )


def _mapping_to_specification(
    *,
    vehicle_id: str,
    vehicle_specs: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ]
    | None,
    departure_soc: float,
) -> VehicleChargerSpecification:
    source = (
        PROTOTYPE_VEHICLE_SPECS
        if vehicle_specs
        is None
        else vehicle_specs
    )

    if vehicle_id not in source:
        raise KeyError(
            "No charging specification found for vehicle: "
            f"{vehicle_id}"
        )

    data = source[
        vehicle_id
    ]

    spec = VehicleChargerSpecification(
        vehicle_id=(
            vehicle_id
        ),
        battery_capacity_kwh=float(
            data[
                "battery_capacity_kwh"
            ]
        ),
        energy_consumption_kwh_per_km=float(
            data[
                "energy_consumption_kwh_per_km"
            ]
        ),
        charging_efficiency=float(
            data[
                "charging_efficiency"
            ]
        ),
        charger_power_kw=float(
            data[
                "charger_power_kw"
            ]
        ),
        source_sheet=(
            "IN_CODE_COMPATIBILITY_SPEC"
        ),
        start_soc=float(
            departure_soc
        ),
        minimum_soc=float(
            data.get(
                "minimum_soc",
                MINIMUM_SOC,
            )
        ),
        charge_target_soc=float(
            data.get(
                "charge_target_soc",
                CHARGE_TARGET_SOC,
            )
        ),
    )

    _validate_spec(
        spec
    )

    return spec


def _calculate_from_spec(
    *,
    trip_distance_km: float,
    spec: VehicleChargerSpecification,
) -> ChargingResult:
    distance = float(
        trip_distance_km
    )

    if (
        not math.isfinite(
            distance
        )
        or distance
        <= 0
    ):
        raise ValueError(
            "Trip distance must be finite and > 0 km."
        )

    _validate_spec(
        spec
    )

    trip_energy = (
        distance
        * spec.energy_consumption_kwh_per_km
    )

    initial_usable = (
        spec.battery_capacity_kwh
        * (
            spec.start_soc
            - spec.minimum_soc
        )
    )

    initial_range = (
        initial_usable
        / spec.energy_consumption_kwh_per_km
    )

    enroute_battery = max(
        0.0,
        (
            trip_energy
            - initial_usable
        ),
    )

    recharge_window = (
        spec.battery_capacity_kwh
        * (
            spec.charge_target_soc
            - spec.minimum_soc
        )
    )

    if (
        not math.isfinite(
            recharge_window
        )
        or recharge_window
        <= 0
    ):
        raise ValueError(
            "Recharge-window battery energy must be > 0."
        )

    if math.isclose(
        enroute_battery,
        0.0,
        abs_tol=1e-12,
    ):
        stops = 0

    else:
        stops = math.ceil(
            enroute_battery
            / recharge_window
        )

    remaining = (
        enroute_battery
    )

    battery_per_stop: list[
        float
    ] = []

    charger_per_stop: list[
        float
    ] = []

    charging_times_h: list[
        float
    ] = []

    stop_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for stop_number in range(
        1,
        stops + 1,
    ):
        battery_energy = min(
            recharge_window,
            remaining,
        )

        charger_energy = (
            battery_energy
            / spec.charging_efficiency
        )

        charging_time_h = (
            charger_energy
            / spec.charger_power_kw
        )

        battery_per_stop.append(
            float(
                battery_energy
            )
        )

        charger_per_stop.append(
            float(
                charger_energy
            )
        )

        charging_times_h.append(
            float(
                charging_time_h
            )
        )

        stop_rows.append(
            {
                "Stop Number": (
                    stop_number
                ),
                "Assumed Departure SOC": (
                    spec.start_soc
                ),
                "Battery Energy Added (kWh)": (
                    float(
                        battery_energy
                    )
                ),
                "Energy Drawn From Charger (kWh)": (
                    float(
                        charger_energy
                    )
                ),
                "Charger Power (kW)": (
                    spec.charger_power_kw
                ),
                "Charging Time (min)": (
                    float(
                        charging_time_h
                        * 60.0
                    )
                ),
                "Is Partial Final Stop": bool(
                    battery_energy
                    < recharge_window
                    - 1e-12
                ),
            }
        )

        remaining -= (
            battery_energy
        )

    total_charger_energy = float(
        sum(
            charger_per_stop
        )
    )

    total_time_h = float(
        sum(
            charging_times_h
        )
    )

    total_time_min = (
        total_time_h
        * 60.0
    )

    powers = tuple(
        spec.charger_power_kw
        for _ in range(
            stops
        )
    )

    return ChargingResult(
        vehicle_id=(
            spec.vehicle_id
        ),
        trip_distance_km=(
            distance
        ),
        start_soc=float(
            spec.start_soc
        ),
        minimum_soc=float(
            spec.minimum_soc
        ),
        charge_target_soc=float(
            spec.charge_target_soc
        ),
        trip_energy_required_kwh=float(
            trip_energy
        ),
        initial_usable_battery_energy_kwh=float(
            initial_usable
        ),
        initial_driving_range_km=float(
            initial_range
        ),
        enroute_battery_energy_required_kwh=float(
            enroute_battery
        ),
        enroute_energy_from_charger_kwh=(
            total_charger_energy
        ),
        recharge_window_battery_energy_kwh=float(
            recharge_window
        ),
        number_of_charging_stops=int(
            stops
        ),
        charging_required=bool(
            stops
            > 0
        ),
        battery_energy_per_stop_kwh=tuple(
            battery_per_stop
        ),
        charger_energy_per_stop_kwh=tuple(
            charger_per_stop
        ),
        charger_powers_kw=(
            powers
        ),
        charging_times_per_stop_h=tuple(
            charging_times_h
        ),
        total_charging_time_h=(
            total_time_h
        ),
        total_charging_time_min=float(
            total_time_min
        ),
        stop_rows=tuple(
            stop_rows
        ),
    )


def calculate_charging_time(
    trip_distance_km: float,
    vehicle_id: str = DEFAULT_VEHICLE_ID,
    vehicle_specs: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ]
    | None = None,
    departure_soc_override: float | None = None,
) -> ChargingResult:
    """
    Reusable charging calculation for historical model modules.

    Normal/final pipeline behaviour:
        departure_soc_override=None
        -> fixed 90% departure-SOC assumption.

    The optional override is retained ONLY for backward compatibility with
    existing historical model code. It is not used by run_all().
    """
    departure_soc = (
        ASSUMED_DEPARTURE_SOC
        if departure_soc_override
        is None
        else float(
            departure_soc_override
        )
    )

    _validate_soc(
        departure_soc,
        "Departure SOC",
    )

    spec = (
        _mapping_to_specification(
            vehicle_id=(
                vehicle_id
            ),
            vehicle_specs=(
                vehicle_specs
            ),
            departure_soc=(
                departure_soc
            ),
        )
    )

    return _calculate_from_spec(
        trip_distance_km=(
            trip_distance_km
        ),
        spec=(
            spec
        ),
    )


def calculate_charging_time_from_prediction(
    prediction_input: Mapping[
        str,
        Any,
    ],
    vehicle_specs: Mapping[
        str,
        Mapping[
            str,
            Any,
        ],
    ]
    | None = None,
) -> ChargingResult:
    if "trip_distance_km" not in prediction_input:
        raise KeyError(
            "Prediction input must contain 'trip_distance_km'."
        )

    vehicle_id = str(
        prediction_input.get(
            "vehicle_id",
            DEFAULT_VEHICLE_ID,
        )
    )

    return calculate_charging_time(
        trip_distance_km=float(
            prediction_input[
                "trip_distance_km"
            ]
        ),
        vehicle_id=(
            vehicle_id
        ),
        vehicle_specs=(
            vehicle_specs
        ),
    )


def find_raw_workbook(
    raw_input_dir: Path,
    source_file_name: str,
) -> Path:
    direct = (
        raw_input_dir
        / source_file_name
    )

    if direct.is_file():
        return direct

    matches = [
        path
        for path in raw_input_dir.rglob(
            source_file_name
        )
        if path.is_file()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Raw workbook {source_file_name!r} was not found under:\n"
            f"{raw_input_dir}"
        )

    if len(
        matches
    ) > 1:
        raise RuntimeError(
            "Multiple raw workbooks have the same file name; "
            "cannot safely choose:\n  - "
            + "\n  - ".join(
                str(
                    path
                )
                for path in matches
            )
        )

    return matches[
        0
    ]


def _safe_trip_id(
    value: Any,
) -> str:
    text = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        _clean_text(
            value
        ),
    ).strip(
        "_"
    )

    return (
        text[
            :120
        ]
        or "ICE_TRIP"
    )


def _result_columns(
    result: ChargingResult,
) -> dict[str, Any]:
    return {
        "Assumed Departure SOC": (
            result.start_soc
        ),
        "Minimum SOC": (
            result.minimum_soc
        ),
        "Charge Target SOC": (
            result.charge_target_soc
        ),
        "Charging Required": (
            result.charging_required
        ),
        "Trip Energy Required (kWh)": (
            result.trip_energy_required_kwh
        ),
        "Initial Usable Battery Energy (kWh)": (
            result.initial_usable_battery_energy_kwh
        ),
        "Initial Driving Range (km)": (
            result.initial_driving_range_km
        ),
        "En-route Battery Energy Required (kWh)": (
            result.enroute_battery_energy_required_kwh
        ),
        "Energy Drawn From Charger (kWh)": (
            result.enroute_energy_from_charger_kwh
        ),
        "Recharge Window (kWh)": (
            result.recharge_window_battery_energy_kwh
        ),
        "Charging Stops": (
            result.number_of_charging_stops
        ),
        "Charging Time (min)": (
            result.total_charging_time_min
        ),
        "Charging Time (h)": (
            result.total_charging_time_h
        ),
    }


def calculate_trip_charging(
    master_row: pd.Series,
    raw_workbook: Path,
) -> tuple[
    dict[
        str,
        Any,
    ],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    spec = (
        read_vehicle_charger_specs(
            raw_workbook
        )
    )

    distance = float(
        master_row[
            "ICE Total Distance (km)"
        ]
    )

    # Single authoritative operational scenario.
    result = (
        _calculate_from_spec(
            trip_distance_km=(
                distance
            ),
            spec=(
                spec
            ),
        )
    )

    summary: dict[
        str,
        Any,
    ] = {
        "Trip ID": (
            master_row[
                "Trip ID"
            ]
        ),
        "Source ICE File": (
            master_row[
                "Source ICE File"
            ]
        ),
        "Raw Workbook": str(
            raw_workbook
        ),
        "Specification Sheet": (
            spec.source_sheet
        ),
        "Start Timestamp": (
            master_row[
                "Start Timestamp"
            ]
        ),
        "End Timestamp": (
            master_row[
                "End Timestamp"
            ]
        ),
        "Start Latitude": (
            master_row[
                "Start Latitude"
            ]
        ),
        "Start Longitude": (
            master_row[
                "Start Longitude"
            ]
        ),
        "End Latitude": (
            master_row[
                "End Latitude"
            ]
        ),
        "End Longitude": (
            master_row[
                "End Longitude"
            ]
        ),
        "Start Address": (
            master_row[
                "Start Address"
            ]
        ),
        "End Address": (
            master_row[
                "End Address"
            ]
        ),
        "ICE Total Distance (km)": (
            distance
        ),
        "ICE Total TAT (min)": (
            master_row.get(
                "ICE Total TAT (min)",
                float(
                    "nan"
                ),
            )
        ),
        "Vehicle ID": (
            spec.vehicle_id
        ),
        "Battery Capacity (kWh)": (
            spec.battery_capacity_kwh
        ),
        "Energy Consumption (kWh/km)": (
            spec.energy_consumption_kwh_per_km
        ),
        "Charging Efficiency": (
            spec.charging_efficiency
        ),
        "Charger Power (kW)": (
            spec.charger_power_kw
        ),
        "Charging Model Version": (
            SCRIPT_VERSION
        ),
        "SOC Handling": (
            "SINGLE ASSUMED DEPARTURE SOC"
        ),
        "Location/Time Handling": (
            "RETAINED FOR AUDIT; NOT USED IN V1 CHARGING PHYSICS"
        ),
        "Pair Class Handling": (
            "NOT APPLICABLE / NOT READ / NOT RECALCULATED / "
            "NOT OVERWRITTEN"
        ),
    }

    summary.update(
        _result_columns(
            result
        )
    )

    stops = pd.DataFrame(
        [
            {
                "Trip ID": (
                    master_row[
                        "Trip ID"
                    ]
                ),
                **row,
            }
            for row in (
                result.stop_rows
            )
        ],
        columns=[
            "Trip ID",
            "Stop Number",
            "Assumed Departure SOC",
            "Battery Energy Added (kWh)",
            "Energy Drawn From Charger (kWh)",
            "Charger Power (kW)",
            "Charging Time (min)",
            "Is Partial Final Stop",
        ],
    )

    specs = pd.DataFrame(
        [
            {
                "Specification": (
                    "Vehicle ID"
                ),
                "Value": (
                    spec.vehicle_id
                ),
                "Units / Note": "",
            },
            {
                "Specification": (
                    "Battery Capacity"
                ),
                "Value": (
                    spec.battery_capacity_kwh
                ),
                "Units / Note": "kWh",
            },
            {
                "Specification": (
                    "Energy Consumption"
                ),
                "Value": (
                    spec.energy_consumption_kwh_per_km
                ),
                "Units / Note": "kWh/km",
            },
            {
                "Specification": (
                    "Charging Efficiency"
                ),
                "Value": (
                    spec.charging_efficiency
                ),
                "Units / Note": "fraction",
            },
            {
                "Specification": (
                    "Charger Power"
                ),
                "Value": (
                    spec.charger_power_kw
                ),
                "Units / Note": "kW",
            },
            {
                "Specification": (
                    "Source Worksheet"
                ),
                "Value": (
                    spec.source_sheet
                ),
                "Units / Note": (
                    "original raw ICE workbook"
                ),
            },
        ]
    )

    assumptions = pd.DataFrame(
        [
            {
                "Assumption": (
                    "Assumed Departure SOC"
                ),
                "Value": (
                    ASSUMED_DEPARTURE_SOC
                ),
            },
            {
                "Assumption": (
                    "Minimum SOC"
                ),
                "Value": (
                    MINIMUM_SOC
                ),
            },
            {
                "Assumption": (
                    "En-route Charge Target SOC"
                ),
                "Value": (
                    CHARGE_TARGET_SOC
                ),
            },
            {
                "Assumption": (
                    "Departure SOC sensitivity scenarios"
                ),
                "Value": (
                    "NOT USED IN MAIN PREDICTION PIPELINE"
                ),
            },
            {
                "Assumption": (
                    "Location / clock-time use"
                ),
                "Value": (
                    "Audit only in Version 1"
                ),
            },
        ]
    )

    return (
        summary,
        stops,
        specs,
        assumptions,
    )


def write_trip_result(
    output_path: Path,
    summary: dict[
        str,
        Any,
    ],
    stops: pd.DataFrame,
    specs: pd.DataFrame,
    assumptions: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(
            [
                summary
            ]
        ).to_excel(
            writer,
            sheet_name="01_Charging_Summary",
            index=False,
        )

        stops.to_excel(
            writer,
            sheet_name="02_Charging_Stops",
            index=False,
        )

        specs.to_excel(
            writer,
            sheet_name="03_Vehicle_Charger_Specs",
            index=False,
        )

        assumptions.to_excel(
            writer,
            sheet_name="04_SOC_Assumptions",
            index=False,
        )


def run_all(
    ice_master_path: Path,
    raw_input_dir: Path,
    output_dir: Path,
) -> dict[
    str,
    Any,
]:
    print(
        "=" * 92
    )
    print(
        "COMMON EV CHARGING-TIME PIPELINE"
    )
    print(
        "=" * 92
    )
    print(
        f"Version      : {SCRIPT_VERSION}"
    )
    print(
        f"ICE master   : {ice_master_path}"
    )
    print(
        f"Raw ICE input: {raw_input_dir}"
    )
    print(
        f"Output       : {output_dir}"
    )
    print(
        "SOC          : assumed departure 90% | minimum 20% | target 80%"
    )
    print(
        "Scenarios    : ONE operational charging calculation per trip"
    )
    print(
        "Location/time: retained for audit; NOT used in Version-1 charging physics"
    )
    print(
        "Pair Class   : NOT APPLICABLE / NOT READ / "
        "NOT RECALCULATED / NOT OVERWRITTEN"
    )
    print(
        "=" * 92
    )

    if not ice_master_path.is_file():
        raise FileNotFoundError(
            "ICE preprocessing master file was not found:\n"
            f"{ice_master_path}\n"
            "Run ice_processing.py successfully first."
        )

    if not raw_input_dir.is_dir():
        raise FileNotFoundError(
            "Raw ICE input folder was not found:\n"
            f"{raw_input_dir}"
        )

    master = pd.read_excel(
        ice_master_path
    )

    missing = [
        column
        for column in REQUIRED_MASTER_COLUMNS
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            "ICE master is missing required columns: "
            + ", ".join(
                missing
            )
        )

    if master.empty:
        raise ValueError(
            "ICE preprocessing master contains no trips."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries: list[
        dict[
            str,
            Any,
        ]
    ] = []

    all_stops: list[
        pd.DataFrame
    ] = []

    status_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, master_row in (
        master.iterrows()
    ):
        trip_id = _safe_trip_id(
            master_row[
                "Trip ID"
            ]
        )

        source_name = _clean_text(
            master_row[
                "Source ICE File"
            ]
        )

        print(
            f"\n[{index + 1}/{len(master)}] "
            f"{trip_id} | {source_name}"
        )

        try:
            raw_workbook = (
                find_raw_workbook(
                    raw_input_dir,
                    source_name,
                )
            )

            (
                summary,
                stops,
                specs,
                assumptions,
            ) = (
                calculate_trip_charging(
                    master_row,
                    raw_workbook,
                )
            )

            trip_output = (
                output_dir
                / f"{trip_id}_Charging_Result.xlsx"
            )

            write_trip_result(
                trip_output,
                summary,
                stops,
                specs,
                assumptions,
            )

            summaries.append(
                summary
            )

            if not stops.empty:
                all_stops.append(
                    stops
                )

            print(
                "  Specs      : "
                f"{summary['Battery Capacity (kWh)']:.1f} kWh | "
                f"{summary['Energy Consumption (kWh/km)']:.3f} kWh/km | "
                f"eta={summary['Charging Efficiency']:.3f} | "
                f"{summary['Charger Power (kW)']:.1f} kW"
            )

            print(
                f"  Distance   : "
                f"{summary['ICE Total Distance (km)']:.3f} km"
            )

            print(
                "  Charging   : "
                f"{summary['Charging Stops']} stop(s) | "
                f"{summary['Charging Time (min)']:.2f} min"
            )

            print(
                f"  Saved      : {trip_output}"
            )

            status_rows.append(
                {
                    "Trip ID": (
                        master_row[
                            "Trip ID"
                        ]
                    ),
                    "Source ICE File": (
                        source_name
                    ),
                    "Status": (
                        "SUCCESS"
                    ),
                    "Error": "",
                }
            )

        except Exception as error:
            message = str(
                error
            )

            print(
                f"  FAILED     : {message}"
            )

            traceback.print_exc()

            status_rows.append(
                {
                    "Trip ID": (
                        master_row[
                            "Trip ID"
                        ]
                    ),
                    "Source ICE File": (
                        source_name
                    ),
                    "Status": (
                        "FAILED"
                    ),
                    "Error": (
                        message
                    ),
                }
            )

    master_output = (
        output_dir
        / "All_ICE_Charging_Results.xlsx"
    )

    status_output = (
        output_dir
        / "Charging_Run_Status.xlsx"
    )

    if summaries:
        summary_frame = pd.DataFrame(
            summaries
        )

        if all_stops:
            stops_frame = pd.concat(
                all_stops,
                ignore_index=True,
            )

        else:
            stops_frame = pd.DataFrame(
                columns=[
                    "Trip ID",
                    "Stop Number",
                    "Assumed Departure SOC",
                    "Battery Energy Added (kWh)",
                    "Energy Drawn From Charger (kWh)",
                    "Charger Power (kW)",
                    "Charging Time (min)",
                    "Is Partial Final Stop",
                ]
            )

        with pd.ExcelWriter(
            master_output,
            engine="openpyxl",
        ) as writer:
            summary_frame.to_excel(
                writer,
                sheet_name="Charging_Results",
                index=False,
            )

            stops_frame.to_excel(
                writer,
                sheet_name="Charging_Stops",
                index=False,
            )

    pd.DataFrame(
        status_rows
    ).to_excel(
        status_output,
        index=False,
    )

    print(
        "\n"
        + "=" * 92
    )
    print(
        "CHARGING PIPELINE COMPLETED"
    )
    print(
        "=" * 92
    )
    print(
        f"Successful trips: {len(summaries)} / {len(master)}"
    )

    if summaries:
        print(
            f"Master result   : {master_output}"
        )

    print(
        f"Run status      : {status_output}"
    )

    return {
        "successful_trips": int(
            len(
                summaries
            )
        ),
        "total_trips": int(
            len(
                master
            )
        ),
        "master_output": (
            master_output
            if summaries
            else None
        ),
        "status_output": (
            status_output
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate one common Version-1 EV charging time "
            "using a fixed 90% departure-SOC assumption."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--ice-master",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--raw-input-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    project_root = (
        args.project_root.resolve()
        if args.project_root
        is not None
        else detect_project_root(
            Path(
                __file__
            )
        )
    )

    ice_master_path = (
        args.ice_master.resolve()
        if args.ice_master
        is not None
        else (
            project_root
            / "Results"
            / "Prediction"
            / "ICE_Ready"
            / "ICE_Prediction_Input_Master.xlsx"
        )
    )

    raw_input_dir = (
        args.raw_input_dir.resolve()
        if args.raw_input_dir
        is not None
        else (
            project_root
            / "Raw_ICE_Input_Data"
        )
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        is not None
        else (
            project_root
            / "Results"
            / "Prediction"
            / "Charging"
        )
    )

    run_all(
        ice_master_path,
        raw_input_dir,
        output_dir,
    )


if __name__ == "__main__":
    main()
