"""
TAT MS INTERCITY
EV CHARGING TIME ESTIMATION

Prototype charging module based on the finalized Method 2.

Design:
- This module DOES NOT ask for inputs in the VS Code terminal.
- Trip distance is obtained from the prediction pipeline.
- EV/charger specifications are supplied as vehicle-specific input data.
- The same function can be called repeatedly for many vehicles.
- Prototype default specifications are included temporarily because the
  actual vehicle database/specification file is not available yet.

Method:
E_required = D * E_EV
E_charger = E_required / eta_charging
E_usable = C_battery * SOC_window
Range = E_usable / E_EV
N_stops = ceil(D / Range)
T_charging,i = E_charger,i / P_charger,i
T_total_charging = sum(T_charging,i)

Later, the prototype specification source can be replaced by:
- Excel/CSV vehicle specification file
- database
- vehicle metadata supplied with the prediction input
without changing the calculation logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# PROTOTYPE VEHICLE SPECIFICATIONS
# ---------------------------------------------------------------------------
# TEMPORARY ONLY.
# Replace these values later with the actual vehicle specifications.
# The prediction pipeline should select specifications using vehicle_id.
# ---------------------------------------------------------------------------

PROTOTYPE_VEHICLE_SPECS: dict[str, dict[str, Any]] = {
    "EV_PROTOTYPE_01": {
        "battery_capacity_kwh": 350.0,
        "soc_window": 0.60,  # 20%-80% = 60%
        "energy_consumption_kwh_per_km": 1.20,
        "charging_efficiency": 0.90,
        "charger_power_kw": 180.0,
    },
}


@dataclass(frozen=True)
class ChargingSpecification:
    battery_capacity_kwh: float
    soc_window: float
    energy_consumption_kwh_per_km: float
    charging_efficiency: float
    charger_powers_kw: tuple[float, ...]


@dataclass(frozen=True)
class ChargingResult:
    vehicle_id: str
    trip_distance_km: float
    energy_required_kwh: float
    energy_from_charger_kwh: float
    usable_battery_energy_kwh: float
    driving_range_km: float
    number_of_charging_stops: int
    charging_times_per_stop_h: tuple[float, ...]
    total_charging_time_h: float
    total_charging_time_min: float


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_specification(spec: ChargingSpecification) -> None:
    _validate_positive(spec.battery_capacity_kwh, "Battery capacity")
    _validate_positive(
        spec.energy_consumption_kwh_per_km,
        "Energy consumption",
    )
    _validate_positive(spec.charging_efficiency, "Charging efficiency")

    if not 0 < spec.soc_window <= 1:
        raise ValueError("SOC window must be between 0 and 1.")

    if not spec.charger_powers_kw:
        raise ValueError("At least one charger power is required.")

    for index, power in enumerate(spec.charger_powers_kw, start=1):
        _validate_positive(power, f"Charger power at stop {index}")


def get_vehicle_specification(
    vehicle_id: str,
    vehicle_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> ChargingSpecification:
    """
    Get vehicle-specific charging specifications.

    In the prototype, values come from PROTOTYPE_VEHICLE_SPECS.
    Later this can receive specifications loaded from Excel/CSV/database.
    """
    source = (
        PROTOTYPE_VEHICLE_SPECS
        if vehicle_specs is None
        else vehicle_specs
    )

    if vehicle_id not in source:
        raise KeyError(
            f"No charging specification found for vehicle: {vehicle_id}"
        )

    data = source[vehicle_id]

    # Support either one common charger power or a list of powers.
    charger_power = data.get("charger_power_kw")
    charger_powers = data.get("charger_powers_kw")

    if charger_powers is None:
        if charger_power is None:
            raise ValueError(
                f"No charger power specified for vehicle: {vehicle_id}"
            )
        charger_powers = (float(charger_power),)
    else:
        charger_powers = tuple(float(x) for x in charger_powers)

    specification = ChargingSpecification(
        battery_capacity_kwh=float(data["battery_capacity_kwh"]),
        soc_window=float(data["soc_window"]),
        energy_consumption_kwh_per_km=float(
            data["energy_consumption_kwh_per_km"]
        ),
        charging_efficiency=float(data["charging_efficiency"]),
        charger_powers_kw=charger_powers,
    )

    _validate_specification(specification)
    return specification


def calculate_charging_time(
    trip_distance_km: float,
    vehicle_id: str,
    vehicle_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> ChargingResult:
    """
    Calculate EV charging time for one prediction.

    trip_distance_km should normally come directly from the prediction
    pipeline, so the module does not ask the user for distance.

    vehicle_id selects the appropriate EV specifications.
    """
    _validate_positive(trip_distance_km, "Trip distance")

    specification = get_vehicle_specification(
        vehicle_id=vehicle_id,
        vehicle_specs=vehicle_specs,
    )

    # Eq. (12): energy required for the trip.
    energy_required_kwh = (
        trip_distance_km
        * specification.energy_consumption_kwh_per_km
    )

    # Eq. (13): energy drawn from charger.
    energy_from_charger_kwh = (
        energy_required_kwh
        / specification.charging_efficiency
    )

    # Eq. (15): usable battery energy.
    usable_battery_energy_kwh = (
        specification.battery_capacity_kwh
        * specification.soc_window
    )

    # Eq. (16): driving range.
    driving_range_km = (
        usable_battery_energy_kwh
        / specification.energy_consumption_kwh_per_km
    )

    # Eq. (17): number of charging stops.
    number_of_charging_stops = max(
        1,
        math.ceil(trip_distance_km / driving_range_km),
    )

    # Prototype assumption:
    # distribute the total required charger energy equally among stops.
    # If actual stop-wise energy data becomes available later, replace this
    # allocation with the measured/calculated energy for each stop.
    energy_per_stop_kwh = (
        energy_from_charger_kwh / number_of_charging_stops
    )

    powers = list(specification.charger_powers_kw)

    # If only one charger power is provided, use it at every stop.
    if len(powers) == 1:
        powers = powers * number_of_charging_stops

    # If fewer powers than stops are supplied, repeat the last known power.
    elif len(powers) < number_of_charging_stops:
        powers.extend(
            [powers[-1]]
            * (number_of_charging_stops - len(powers))
        )

    powers = powers[:number_of_charging_stops]

    # Eq. (14): charging time at each stop.
    charging_times_per_stop_h = tuple(
        energy_per_stop_kwh / power
        for power in powers
    )

    # Eq. (18): total charging time.
    total_charging_time_h = sum(charging_times_per_stop_h)
    total_charging_time_min = total_charging_time_h * 60.0

    return ChargingResult(
        vehicle_id=vehicle_id,
        trip_distance_km=trip_distance_km,
        energy_required_kwh=energy_required_kwh,
        energy_from_charger_kwh=energy_from_charger_kwh,
        usable_battery_energy_kwh=usable_battery_energy_kwh,
        driving_range_km=driving_range_km,
        number_of_charging_stops=number_of_charging_stops,
        charging_times_per_stop_h=charging_times_per_stop_h,
        total_charging_time_h=total_charging_time_h,
        total_charging_time_min=total_charging_time_min,
    )


def calculate_charging_time_from_prediction(
    prediction_input: Mapping[str, Any],
    vehicle_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> ChargingResult:
    """
    Adapter used by ML/DL/Time-Series prediction pipelines.

    Expected prediction_input:
        {
            "vehicle_id": "...",
            "trip_distance_km": <distance obtained from pipeline>
        }

    Later, prediction_input can contain additional vehicle metadata.
    """
    if "vehicle_id" not in prediction_input:
        raise KeyError("Prediction input must contain 'vehicle_id'.")

    if "trip_distance_km" not in prediction_input:
        raise KeyError(
            "Prediction input must contain 'trip_distance_km'."
        )

    return calculate_charging_time(
        trip_distance_km=float(prediction_input["trip_distance_km"]),
        vehicle_id=str(prediction_input["vehicle_id"]),
        vehicle_specs=vehicle_specs,
    )


def result_to_dict(result: ChargingResult) -> dict[str, Any]:
    """Convert the result to a dictionary for Excel/JSON/report output."""
    return {
        "Vehicle ID": result.vehicle_id,
        "Trip Distance (km)": result.trip_distance_km,
        "Energy Required (kWh)": result.energy_required_kwh,
        "Energy From Charger (kWh)": result.energy_from_charger_kwh,
        "Usable Battery Energy (kWh)": result.usable_battery_energy_kwh,
        "Driving Range (km)": result.driving_range_km,
        "Number of Charging Stops": result.number_of_charging_stops,
        "Charging Time (h)": result.total_charging_time_h,
        "Charging Time (min)": result.total_charging_time_min,
    }


def example_prototype_run() -> ChargingResult:
    """
    Small prototype test.

    This is deliberately a function, not an interactive terminal input.
    """
    prediction_output = {
        "vehicle_id": "EV_PROTOTYPE_01",
        # In the real pipeline this value will come from the model/pipeline.
        "trip_distance_km": 180.0,
    }

    return calculate_charging_time_from_prediction(prediction_output)


if __name__ == "__main__":
    # Prototype verification only.
    # No user input is requested.
    result = example_prototype_run()

    print("=" * 68)
    print("TAT MS INTERCITY")
    print("EV CHARGING TIME PIPELINE - PROTOTYPE TEST")
    print("=" * 68)
    print(f"Vehicle ID              : {result.vehicle_id}")
    print(f"Trip distance           : {result.trip_distance_km:.2f} km")
    print(f"Energy required         : {result.energy_required_kwh:.2f} kWh")
    print(
        f"Energy from charger    : "
        f"{result.energy_from_charger_kwh:.2f} kWh"
    )
    print(
        f"Usable battery energy  : "
        f"{result.usable_battery_energy_kwh:.2f} kWh"
    )
    print(f"Driving range          : {result.driving_range_km:.2f} km")
    print(f"Charging stops         : {result.number_of_charging_stops}")
    print(
        f"Total charging time   : "
        f"{result.total_charging_time_h:.3f} h "
        f"({result.total_charging_time_min:.2f} min)"
    )
    print("=" * 68)
    print("Prototype charging calculation completed.")

def example_prototype_run() -> ChargingResult:
    """
    Prototype verification using temporary vehicle specifications.
    No terminal input is required.
    """
    prediction_input = {
        "vehicle_id": "EV_PROTOTYPE_01",
        # Temporary test distance.
        # In the real prediction pipeline this will come automatically
        # from the model/prediction data.
        "trip_distance_km": 180.0,
    }

    return calculate_charging_time_from_prediction(prediction_input)


if __name__ == "__main__":
    result = example_prototype_run()

    print("\n" + "=" * 68)
    print("TAT MS INTERCITY")
    print("EV CHARGING TIME PIPELINE - PROTOTYPE TEST")
    print("=" * 68)
    print(f"Vehicle ID              : {result.vehicle_id}")
    print(f"Trip distance           : {result.trip_distance_km:.2f} km")
    print(f"Energy required         : {result.energy_required_kwh:.2f} kWh")
    print(
        f"Energy from charger    : "
        f"{result.energy_from_charger_kwh:.2f} kWh"
    )
    print(
        f"Usable battery energy  : "
        f"{result.usable_battery_energy_kwh:.2f} kWh"
    )
    print(f"Driving range          : {result.driving_range_km:.2f} km")
    print(f"Charging stops         : {result.number_of_charging_stops}")
    print(
        f"Total charging time    : "
        f"{result.total_charging_time_h:.3f} h "
        f"({result.total_charging_time_min:.2f} min)"
    )
    print("=" * 68)
    print("Prototype charging calculation completed.")
