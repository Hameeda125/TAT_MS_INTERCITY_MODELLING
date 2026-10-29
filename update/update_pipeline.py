r"""
TAT MS INTERCITY MODELLING — MASTER UPDATE PIPELINE
===================================================

Version:
    2026-10-04_MODELLING_MASTER_PIPELINE_V3_DYNAMIC_PREFLIGHT

Save as:
    D:\Hameeda_Projects\TAT_MS_INTERCITY_MODELLING\update\update_pipeline.py

Recommended complete run:
    python .\update\update_pipeline.py --mode FULL --model ALL --scope ALL --device cuda --inference-device cpu

Preflight only:
    python .\update\update_pipeline.py --mode FULL --model ALL --scope ALL --device cuda --inference-device cpu --check-only

Train/evaluate all ML + DL only:
    python .\update\update_pipeline.py --mode TRAIN --model ALL --scope ALL --device cuda

Use existing trained models for new raw ICE trip(s):
    python .\update\update_pipeline.py --mode PREDICT --model ALL --scope ALL --inference-device cpu

PIPELINE ORDER
--------------
TRAIN:
    1. Full ML
    2. Phase-wise ML
    3. Full DL
    4. Phase-wise DL

PREDICT:
    5. New raw ICE preprocessing
    6. Trained-model prediction
    7. Physical EV charging calculation
    8. Final EV assembly

IMPORTANT
---------
- This wrapper does NOT change model logic.
- It always executes the current scripts from the project folders.
- Pair Class is never recalculated or overwritten here.
- STRICT/NONSTRICT/COMBINED remain model-training/model-selection scopes only.
- ML remains CPU-side.
- DL training uses --device cuda or cpu.
- DL inference defaults to CPU.
- The wrapper is fail-fast by default.
- Exact hard-coded child-script VERSION markers are intentionally NOT used here.
  Instead, the wrapper validates:
      * script existence
      * Python syntax
      * required CLI option strings
      * raw ICE input availability when prediction is requested
      * CUDA availability when CUDA is explicitly requested
      * freshness of critical prediction outputs
- A SHA-256 fingerprint of every selected child script is written to the run log.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import traceback

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


VERSION = "2026-10-04_MODELLING_MASTER_PIPELINE_V3_DYNAMIC_PREFLIGHT"

# This file is intended to live in:
#   <PROJECT_ROOT>\update\update_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = PROJECT_ROOT / "Results"
SYSTEM_DIR = RESULTS_ROOT / "System"

RAW_ICE_DIR = PROJECT_ROOT / "Raw_ICE_Input_Data"

FULL_ML_SCRIPT = PROJECT_ROOT / "src" / "ml" / "full_ml.py"
PHASE_ML_SCRIPT = PROJECT_ROOT / "src" / "ml" / "phase_wise_ml.py"
FULL_DL_SCRIPT = PROJECT_ROOT / "src" / "dl" / "full_dl.py"
PHASE_DL_SCRIPT = PROJECT_ROOT / "src" / "dl" / "phase_wise_dl.py"

ICE_PROCESSING_SCRIPT = (
    PROJECT_ROOT / "src" / "ice_raw_data" / "ice_processing.py"
)

MODEL_PREDICTION_SCRIPT = (
    PROJECT_ROOT / "src" / "prediction" / "model_prediction.py"
)

CHARGING_SCRIPT = (
    PROJECT_ROOT / "src" / "prediction" / "charging_time.py"
)

FINAL_ASSEMBLY_SCRIPT = (
    PROJECT_ROOT / "src" / "prediction" / "final_tat_assembly.py"
)

ICE_READY_DIR = RESULTS_ROOT / "Prediction" / "ICE_Ready"

ICE_MASTER = ICE_READY_DIR / "ICE_Prediction_Input_Master.xlsx"

ICE_PARAMETER_TABLE = (
    RESULTS_ROOT / "Prediction" / "ICE_17_Parameters_Table.xlsx"
)

MODEL_PREDICTIONS = (
    RESULTS_ROOT / "Prediction" / "Model_Predictions.xlsx"
)

CHARGING_DIR = RESULTS_ROOT / "Prediction" / "Charging"

CHARGING_RESULTS = (
    CHARGING_DIR / "All_ICE_Charging_Results.xlsx"
)

FINAL_OUTPUT_DIR = RESULTS_ROOT / "Prediction" / "Final_TAT"


VALID_MODELS = (
    "FULL_ML",
    "PHASE_ML",
    "ML",
    "FULL_DL",
    "PHASE_DL",
    "DL",
    "ALL",
)

VALID_SCOPES = (
    "STRICT",
    "NONSTRICT",
    "COMBINED",
    "ALL",
)

VALID_MODES = (
    "FULL",
    "TRAIN",
    "PREDICT",
)

VALID_TRAINING_DEVICES = (
    "cuda",
    "cpu",
)

VALID_INFERENCE_DEVICES = (
    "cpu",
    "cuda",
    "auto",
)


# We intentionally validate interface strings rather than exact child-script
# version markers. This lets you update a child script without having to edit
# this master wrapper every time, while still failing early when a CLI contract
# required by the wrapper has actually changed.
SCRIPT_REQUIRED_FLAGS: dict[Path, tuple[str, ...]] = {
    FULL_ML_SCRIPT: (
        "--scope",
    ),
    PHASE_ML_SCRIPT: (
        "--scope",
    ),
    FULL_DL_SCRIPT: (
        "--scope",
        "--device",
    ),
    PHASE_DL_SCRIPT: (
        "--scope",
        "--device",
    ),
    ICE_PROCESSING_SCRIPT: (
        "--input-dir",
        "--output-dir",
    ),
    MODEL_PREDICTION_SCRIPT: (
        "--models",
        "--scope",
        "--device",
        "--project-root",
        "--input",
        "--input-sheet",
        "--output",
    ),
    CHARGING_SCRIPT: (
        "--project-root",
        "--ice-master",
        "--raw-input-dir",
        "--output-dir",
    ),
    FINAL_ASSEMBLY_SCRIPT: (
        "--predictions",
        "--prediction-sheet",
        "--charging",
        "--charging-sheet",
        "--model",
        "--scope",
        "--output",
    ),
}


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    expected_output: Path | None = None
    device_mode: str = "cpu"


def normalize_choice(
    value: str,
    allowed: Iterable[str],
    label: str,
) -> str:
    normalized = str(value).strip().upper()

    if normalized not in set(allowed):
        raise ValueError(
            f"Invalid {label}: {value!r}. "
            f"Allowed: {', '.join(allowed)}"
        )

    return normalized


def selected_model_sources(
    model_selector: str,
) -> tuple[str, ...]:
    selector = normalize_choice(
        model_selector,
        VALID_MODELS,
        "model selector",
    )

    mapping = {
        "FULL_ML": ("FULL_ML",),
        "PHASE_ML": ("PHASE_ML",),
        "ML": ("FULL_ML", "PHASE_ML"),
        "FULL_DL": ("FULL_DL",),
        "PHASE_DL": ("PHASE_DL",),
        "DL": ("FULL_DL", "PHASE_DL"),
        "ALL": (
            "FULL_ML",
            "PHASE_ML",
            "FULL_DL",
            "PHASE_DL",
        ),
    }

    return mapping[selector]


def includes_dl(
    model_selector: str,
) -> bool:
    return bool(
        set(selected_model_sources(model_selector))
        & {
            "FULL_DL",
            "PHASE_DL",
        }
    )


def python_command(
    script: Path,
    *args: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-u",
        str(script),
        *(str(item) for item in args),
    )


def final_output_path(
    model_selector: str,
    scope: str,
) -> Path:
    model_selector = normalize_choice(
        model_selector,
        VALID_MODELS,
        "model selector",
    )

    scope = normalize_choice(
        scope,
        VALID_SCOPES,
        "scope",
    )

    if (
        model_selector == "ALL"
        and scope == "ALL"
    ):
        return (
            FINAL_OUTPUT_DIR
            / "Final_EV_TAT_Results.xlsx"
        )

    return (
        FINAL_OUTPUT_DIR
        / f"Final_EV_TAT_{model_selector}_{scope}.xlsx"
    )


def build_stages(
    *,
    mode: str,
    model_selector: str,
    scope: str,
    device: str,
    inference_device: str,
) -> list[Stage]:
    mode = normalize_choice(
        mode,
        VALID_MODES,
        "mode",
    )

    model_selector = normalize_choice(
        model_selector,
        VALID_MODELS,
        "model selector",
    )

    scope = normalize_choice(
        scope,
        VALID_SCOPES,
        "scope",
    )

    device = str(device).strip().lower()

    if device not in VALID_TRAINING_DEVICES:
        raise ValueError(
            "--device must be cuda or cpu."
        )

    inference_device = (
        str(inference_device)
        .strip()
        .lower()
    )

    if (
        inference_device
        not in VALID_INFERENCE_DEVICES
    ):
        raise ValueError(
            "--inference-device must be cpu, cuda, or auto."
        )

    selected = selected_model_sources(
        model_selector
    )

    stages: list[Stage] = []

    # ------------------------------------------------------------------
    # TRAINING / LOPO EVALUATION
    # ------------------------------------------------------------------
    if mode in {
        "FULL",
        "TRAIN",
    }:
        if "FULL_ML" in selected:
            stages.append(
                Stage(
                    name=(
                        "Full ML | 15 ICE inputs -> 15 EV outputs | LOPO"
                    ),
                    command=python_command(
                        FULL_ML_SCRIPT,
                        "--scope",
                        scope,
                    ),
                )
            )

        if "PHASE_ML" in selected:
            stages.append(
                Stage(
                    name=(
                        "Phase-wise ML | six independent phases | LOPO"
                    ),
                    command=python_command(
                        PHASE_ML_SCRIPT,
                        "--scope",
                        scope,
                    ),
                )
            )

        if "FULL_DL" in selected:
            stages.append(
                Stage(
                    name=(
                        "Full DL | fixed/predeclared 15x15 DNN | LOPO"
                    ),
                    command=python_command(
                        FULL_DL_SCRIPT,
                        "--scope",
                        scope,
                        "--device",
                        device,
                    ),
                    device_mode=device,
                )
            )

        if "PHASE_DL" in selected:
            stages.append(
                Stage(
                    name=(
                        "Phase-wise DL | six fixed/predeclared DNNs | LOPO"
                    ),
                    command=python_command(
                        PHASE_DL_SCRIPT,
                        "--scope",
                        scope,
                        "--device",
                        device,
                    ),
                    device_mode=device,
                )
            )

    # ------------------------------------------------------------------
    # NEW-ICE INFERENCE
    # ------------------------------------------------------------------
    if mode in {
        "FULL",
        "PREDICT",
    }:
        stages.append(
            Stage(
                name=(
                    "New raw ICE preprocessing -> 17 ICE parameters"
                ),
                command=python_command(
                    ICE_PROCESSING_SCRIPT,
                    "--input-dir",
                    str(RAW_ICE_DIR),
                    "--output-dir",
                    str(ICE_READY_DIR),
                ),
                expected_output=ICE_PARAMETER_TABLE,
            )
        )

        # Prediction remains before charging. Both depend on ICE preprocessing,
        # and the final assembly depends on both outputs.
        prediction_device = (
            inference_device
            if includes_dl(model_selector)
            else "cpu"
        )

        stages.append(
            Stage(
                name=(
                    "Apply trained model(s) to every processed ICE trip"
                ),
                command=python_command(
                    MODEL_PREDICTION_SCRIPT,
                    "--models",
                    model_selector,
                    "--scope",
                    scope,
                    "--device",
                    prediction_device,
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--input",
                    str(ICE_PARAMETER_TABLE),
                    "--input-sheet",
                    "ICE_17_Parameters",
                    "--output",
                    str(MODEL_PREDICTIONS),
                ),
                expected_output=MODEL_PREDICTIONS,
                device_mode=prediction_device,
            )
        )

        stages.append(
            Stage(
                name=(
                    "Physical EV charging calculation for every ICE trip"
                ),
                command=python_command(
                    CHARGING_SCRIPT,
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--ice-master",
                    str(ICE_MASTER),
                    "--raw-input-dir",
                    str(RAW_ICE_DIR),
                    "--output-dir",
                    str(CHARGING_DIR),
                ),
                expected_output=CHARGING_RESULTS,
            )
        )

        final_output = final_output_path(
            model_selector,
            scope,
        )

        stages.append(
            Stage(
                name=(
                    "Final EV assembly | TAT + distance + moving speed"
                ),
                command=python_command(
                    FINAL_ASSEMBLY_SCRIPT,
                    "--predictions",
                    str(MODEL_PREDICTIONS),
                    "--prediction-sheet",
                    "01_Model_Predictions",
                    "--charging",
                    str(CHARGING_RESULTS),
                    "--charging-sheet",
                    "Charging_Results",
                    "--model",
                    model_selector,
                    "--scope",
                    scope,
                    "--output",
                    str(final_output),
                ),
                expected_output=final_output,
            )
        )

    return stages


def required_scripts(
    stages: list[Stage],
) -> list[Path]:
    scripts: list[Path] = []
    seen: set[Path] = set()

    for stage in stages:
        if len(stage.command) < 3:
            continue

        script = Path(stage.command[2])

        if script not in seen:
            scripts.append(script)
            seen.add(script)

    return scripts


def verify_stage_files(
    stages: list[Stage],
) -> None:
    missing = [
        path
        for path in required_scripts(stages)
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required pipeline script(s) were not found:\n"
            + "\n".join(
                f" - {path}"
                for path in missing
            )
        )


def read_script_text(
    script: Path,
) -> str:
    try:
        return script.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as error:
        raise RuntimeError(
            f"Could not read pipeline script:\n{script}\n{error}"
        ) from error


def verify_python_syntax(
    stages: list[Stage],
) -> None:
    failures: list[str] = []

    for script in required_scripts(stages):
        text = read_script_text(script)

        try:
            compile(
                text,
                str(script),
                "exec",
            )
        except SyntaxError as error:
            failures.append(
                (
                    f"{script}\n"
                    f"  line {error.lineno}: {error.msg}"
                )
            )

    if failures:
        raise RuntimeError(
            "Python syntax preflight failed:\n\n"
            + "\n\n".join(failures)
        )


def verify_script_interfaces(
    stages: list[Stage],
) -> None:
    failures: list[str] = []

    for script in required_scripts(stages):
        required_flags = SCRIPT_REQUIRED_FLAGS.get(
            script,
            (),
        )

        if not required_flags:
            continue

        text = read_script_text(script)

        missing = [
            flag
            for flag in required_flags
            if flag not in text
        ]

        if missing:
            failures.append(
                (
                    f"{script}\n"
                    + "\n".join(
                        f"  missing required CLI option string: {flag}"
                        for flag in missing
                    )
                )
            )

    if failures:
        raise RuntimeError(
            "Pipeline child-script CLI contract verification failed.\n"
            "The master wrapper and one or more child scripts are no longer "
            "using the same command-line interface:\n\n"
            + "\n\n".join(failures)
        )


def verify_raw_ice_input() -> None:
    if not RAW_ICE_DIR.is_dir():
        raise FileNotFoundError(
            "Raw ICE input folder was not found:\n"
            f"{RAW_ICE_DIR}"
        )

    workbooks = sorted(
        path
        for path in RAW_ICE_DIR.glob("*.xlsx")
        if not path.name.startswith("~$")
    )

    if not workbooks:
        raise FileNotFoundError(
            "No .xlsx raw ICE workbook was found in:\n"
            f"{RAW_ICE_DIR}"
        )


def configure_runtime_environment(
    gpu_index: int,
    *,
    expose_gpu: bool,
) -> None:
    if gpu_index < 0:
        raise ValueError(
            "--gpu-index must be >= 0."
        )

    if expose_gpu:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(gpu_index)

    os.environ.setdefault(
        "PYTHONUTF8",
        "1",
    )

    os.environ.setdefault(
        "PYTHONIOENCODING",
        "utf-8",
    )

    os.environ.setdefault(
        "PYTHONUNBUFFERED",
        "1",
    )


def verify_cuda_available() -> dict[str, str]:
    try:
        import torch
    except Exception as error:
        raise RuntimeError(
            "CUDA was explicitly requested but PyTorch could not be imported "
            f"from the active venv:\n{error}"
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested but torch.cuda.is_available() "
            "is False."
        )

    return {
        "torch_version": str(
            torch.__version__
        ),
        "cuda_runtime": str(
            torch.version.cuda
        ),
        "device_name": str(
            torch.cuda.get_device_name(0)
        ),
        "device_count_visible": str(
            torch.cuda.device_count()
        ),
    }


def format_duration(
    seconds: float,
) -> str:
    seconds = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def display_command(
    command: tuple[str, ...],
) -> str:
    return subprocess.list2cmdline(
        list(command)
    )


def file_signature(
    path: Path | None,
) -> tuple[int, int] | None:
    if (
        path is None
        or not path.is_file()
    ):
        return None

    stat = path.stat()

    return (
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def build_child_environment(
    *,
    gpu_index: int,
    expose_gpu: bool,
) -> dict[str, str]:
    env = os.environ.copy()

    if expose_gpu:
        env[
            "CUDA_VISIBLE_DEVICES"
        ] = str(gpu_index)

    env[
        "PYTHONUTF8"
    ] = "1"

    env[
        "PYTHONIOENCODING"
    ] = "utf-8"

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    return env


def run_stage(
    *,
    stage_number: int,
    total_stages: int,
    stage: Stage,
    log_file,
    env: dict[str, str],
) -> dict[str, object]:
    started = time.monotonic()

    previous_output_signature = (
        file_signature(
            stage.expected_output
        )
    )

    header = (
        "\n"
        + "=" * 100
        + "\n"
        + f"STAGE {stage_number}/{total_stages} | {stage.name}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Command: {display_command(stage.command)}"
        + "\n"
    )

    print(
        header,
        flush=True,
    )

    log_file.write(header)
    log_file.flush()

    process = subprocess.Popen(
        list(stage.command),
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(
            line,
            end="",
            flush=True,
        )
        log_file.write(line)

    exit_code = process.wait()

    elapsed = (
        time.monotonic()
        - started
    )

    status = (
        "SUCCESS"
        if exit_code == 0
        else "FAILED"
    )

    output_status = "N/A"

    if (
        exit_code == 0
        and stage.expected_output
        is not None
    ):
        current_signature = file_signature(
            stage.expected_output
        )

        if current_signature is None:
            status = "FAILED"
            exit_code = 97
            output_status = "MISSING"

            message = (
                "\nStage process returned success, but the expected "
                "output file was not created:\n"
                f"{stage.expected_output}\n"
            )

            print(
                message,
                flush=True,
            )
            log_file.write(message)

        elif (
            previous_output_signature
            is not None
            and current_signature
            == previous_output_signature
        ):
            status = "FAILED"
            exit_code = 98
            output_status = "STALE_NOT_UPDATED"

            message = (
                "\nStage process returned success, but the expected "
                "output file was not updated during this run:\n"
                f"{stage.expected_output}\n"
            )

            print(
                message,
                flush=True,
            )
            log_file.write(message)

        else:
            output_status = "VERIFIED_FRESH"

    footer = (
        "\n"
        + "-" * 100
        + "\n"
        + f"STAGE {stage_number}/{total_stages} {status}"
        + f" | {stage.name}"
        + f" | elapsed {format_duration(elapsed)}"
        + (
            f" | output {output_status}"
            if stage.expected_output is not None
            else ""
        )
        + "\n"
        + "-" * 100
        + "\n"
    )

    print(
        footer,
        flush=True,
    )

    log_file.write(footer)
    log_file.flush()

    return {
        "stage": stage_number,
        "name": stage.name,
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "expected_output": (
            str(stage.expected_output)
            if stage.expected_output is not None
            else ""
        ),
        "output_status": output_status,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the TAT MS INTERCITY ML/DL training and "
            "new-ICE inference pipeline in dependency order."
        )
    )

    parser.add_argument(
        "--mode",
        choices=list(VALID_MODES),
        default="FULL",
        help=(
            "FULL=train/evaluate + new-ICE inference; "
            "TRAIN=train/evaluate only; "
            "PREDICT=use existing trained models."
        ),
    )

    parser.add_argument(
        "--model",
        choices=list(VALID_MODELS),
        default="ALL",
        help=(
            "Model family/families to train and/or apply."
        ),
    )

    parser.add_argument(
        "--scope",
        choices=list(VALID_SCOPES),
        default="ALL",
        help=(
            "Historical training/model scope."
        ),
    )

    parser.add_argument(
        "--device",
        choices=list(VALID_TRAINING_DEVICES),
        default="cuda",
        help=(
            "Device for DL TRAINING. Default: cuda."
        ),
    )

    parser.add_argument(
        "--inference-device",
        choices=list(VALID_INFERENCE_DEVICES),
        default="cpu",
        help=(
            "Device for DL INFERENCE in model_prediction.py. "
            "Default: cpu."
        ),
    )

    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help=(
            "Physical GPU index exposed to child processes. Default 0."
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate paths, syntax, CLI contracts, raw input when needed, "
            "and explicitly requested CUDA without executing stages."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Diagnostic only. Normal safe behavior is fail-fast."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    mode = normalize_choice(
        args.mode,
        VALID_MODES,
        "mode",
    )

    model_selector = normalize_choice(
        args.model,
        VALID_MODELS,
        "model selector",
    )

    scope = normalize_choice(
        args.scope,
        VALID_SCOPES,
        "scope",
    )

    training_device = (
        str(args.device)
        .strip()
        .lower()
    )

    inference_device = (
        str(args.inference_device)
        .strip()
        .lower()
    )

    stages = build_stages(
        mode=mode,
        model_selector=model_selector,
        scope=scope,
        device=training_device,
        inference_device=inference_device,
    )

    if not stages:
        raise RuntimeError(
            "No pipeline stages were selected."
        )

    # --------------------------------------------------------------
    # PREFLIGHT
    # --------------------------------------------------------------
    verify_stage_files(stages)
    verify_python_syntax(stages)
    verify_script_interfaces(stages)

    if mode in {
        "FULL",
        "PREDICT",
    }:
        verify_raw_ice_input()

    explicit_cuda_needed = any(
        stage.device_mode == "cuda"
        for stage in stages
    )

    gpu_may_be_used = any(
        stage.device_mode in {
            "cuda",
            "auto",
        }
        for stage in stages
    )

    configure_runtime_environment(
        args.gpu_index,
        expose_gpu=gpu_may_be_used,
    )

    cuda_info = None

    if explicit_cuda_needed:
        cuda_info = verify_cuda_available()

    SYSTEM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        SYSTEM_DIR
        / f"update_modelling_pipeline_{timestamp}.log"
    )

    print("=" * 100)
    print(
        "TAT MS INTERCITY — MODELLING MASTER PIPELINE"
    )
    print("=" * 100)
    print(
        f"Version          : {VERSION}"
    )
    print(
        f"Project root     : {PROJECT_ROOT}"
    )
    print(
        f"Python           : {sys.executable}"
    )
    print(
        f"Mode             : {mode}"
    )
    print(
        f"Model            : {model_selector}"
    )
    print(
        f"Scope            : {scope}"
    )
    print(
        f"DL training      : {training_device}"
    )
    print(
        f"DL inference     : {inference_device}"
    )

    if cuda_info is not None:
        print(
            f"GPU index        : {args.gpu_index}"
        )
        print(
            f"GPU              : {cuda_info['device_name']}"
        )
        print(
            f"PyTorch          : {cuda_info['torch_version']}"
        )
        print(
            f"CUDA runtime     : {cuda_info['cuda_runtime']}"
        )

    print(
        "\nArchitecture:"
    )
    print(
        "  Pair Class       -> PRESERVED ONLY"
    )
    print(
        "  Model training   -> LOPO OOS evaluation"
    )
    print(
        "  Charging         -> separate physical module"
    )
    print(
        "  Final assembly   -> TAT + distance + moving speed"
    )

    print(
        "\nSelected child scripts:"
    )

    for script in required_scripts(stages):
        print(
            f"  - {script.relative_to(PROJECT_ROOT)}"
        )

    print(
        "\nStages:"
    )

    for number, stage in enumerate(
        stages,
        start=1,
    ):
        suffix = ""

        if stage.device_mode == "cuda":
            suffix = " [CUDA]"
        elif stage.device_mode == "auto":
            suffix = " [AUTO DEVICE]"

        print(
            f"  {number}. {stage.name}{suffix}"
        )

    if args.check_only:
        print(
            "\nCHECK-ONLY PASSED"
        )
        print(
            "All selected scripts exist, compile successfully, "
            "and expose the CLI strings required by this wrapper."
        )
        print(
            "No stage was executed."
        )
        return

    env = build_child_environment(
        gpu_index=args.gpu_index,
        expose_gpu=gpu_may_be_used,
    )

    overall_started = time.monotonic()

    summaries: list[
        dict[str, object]
    ] = []

    with log_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as log_file:
        log_file.write(
            "TAT MS INTERCITY — MODELLING MASTER PIPELINE\n"
        )
        log_file.write(
            f"Version: {VERSION}\n"
        )
        log_file.write(
            f"Started: {datetime.now().isoformat()}\n"
        )
        log_file.write(
            f"Project root: {PROJECT_ROOT}\n"
        )
        log_file.write(
            f"Python: {sys.executable}\n"
        )
        log_file.write(
            f"Mode: {mode}\n"
        )
        log_file.write(
            f"Model: {model_selector}\n"
        )
        log_file.write(
            f"Scope: {scope}\n"
        )
        log_file.write(
            f"DL training device: {training_device}\n"
        )
        log_file.write(
            f"DL inference device: {inference_device}\n"
        )

        if cuda_info is not None:
            log_file.write(
                f"GPU: {cuda_info['device_name']}\n"
            )
            log_file.write(
                f"CUDA runtime: {cuda_info['cuda_runtime']}\n"
            )

        log_file.write(
            "\nSelected child script fingerprints:\n"
        )

        for script in required_scripts(stages):
            log_file.write(
                f"  {script.relative_to(PROJECT_ROOT)}"
                f" | sha256={sha256_file(script)}\n"
            )

        log_file.flush()

        for stage_number, stage in enumerate(
            stages,
            start=1,
        ):
            summary = run_stage(
                stage_number=stage_number,
                total_stages=len(stages),
                stage=stage,
                log_file=log_file,
                env=env,
            )

            summaries.append(summary)

            if (
                summary[
                    "status"
                ]
                != "SUCCESS"
                and not args.continue_on_error
            ):
                message = (
                    "\nPIPELINE STOPPED SAFELY.\n"
                    f"Stage {stage_number}/{len(stages)} failed: "
                    f"{stage.name}\n"
                    "Later dependent stages were not executed.\n"
                    f"Run log: {log_path}\n"
                )

                print(
                    message,
                    flush=True,
                )
                log_file.write(message)

                raise RuntimeError(
                    f"Stage {stage_number}/{len(stages)} failed: "
                    f"{stage.name}"
                )

    elapsed = (
        time.monotonic()
        - overall_started
    )

    failed = [
        item
        for item in summaries
        if item[
            "status"
        ]
        != "SUCCESS"
    ]

    print(
        "\n"
        + "=" * 100
    )
    print(
        "MODELLING MASTER PIPELINE SUMMARY"
    )
    print(
        "=" * 100
    )

    for item in summaries:
        print(
            f"Stage {item['stage']}/{len(stages)}"
            f" | {item['status']}"
            f" | {item['name']}"
            f" | {format_duration(float(item['elapsed_seconds']))}"
        )

    print(
        f"\nTotal elapsed: {format_duration(elapsed)}"
    )
    print(
        f"Run log      : {log_path}"
    )

    if failed:
        raise RuntimeError(
            f"{len(failed)} stage(s) failed. "
            "See the run log above."
        )

    print(
        "\nALL REQUESTED MODELLING / INFERENCE STAGES "
        "COMPLETED SUCCESSFULLY"
    )

    if mode in {
        "FULL",
        "PREDICT",
    }:
        print(
            "Final result : "
            f"{final_output_path(model_selector, scope)}"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            "\nTAT MODELLING PIPELINE FAILED"
        )
        print(
            f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()
        raise
