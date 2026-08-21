#!/usr/bin/env python3

"""
Pressure-defined gait event detection for BWS gait analysis.

Purpose
-------
Uses ONLY plantar-pressure data to define gait cycles and pressure-based
events. IMU data are deliberately NOT used for event detection.

Outputs per trial
-----------------
<trial>_pressure_signals.npz
<trial>_gait_cycles.csv
<trial>_events_metadata.json

QC:
<trial>_pressure_events.png
<trial>_sensor_heatmaps.png

Pressure layout
---------------
pressure[:, 0:64]   = LEFT
pressure[:, 64:128] = RIGHT

Coordinate convention
---------------------
Sensor coordinate map:
    x = transverse coordinate [mm]
    y = longitudinal coordinate [mm]

For the supplied cell maps:
    low y  = heel
    high y = toe

IMPORTANT:
The coordinate array ordering is assumed to correspond directly to
pressure-channel ordering. This assumption is explicitly recorded in
metadata and should be verified against manufacturer documentation or
a physical sensor activation test.
"""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/alya/Desktop/optimal-bws")

INPUT_FOLDER = (
    PROJECT_ROOT
    / "filtered_data"
    / "filtered_second_batch"
)

PATIENT_CSV = (
    PROJECT_ROOT
    / "documents"
    / "patient_information_summary.csv"
)

SENSORS_S_FILE = (
    PROJECT_ROOT
    / "documents"
    / "sensorsS.ts"
)

SENSORS_M_FILE = (
    PROJECT_ROOT
    / "documents"
    / "sensorsM.ts"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "filtered_data"
    / "pressure_events_second_batch_new"
)

QC_FOLDER = (
    OUTPUT_FOLDER
    / "qc"
)


# ============================================================
# ALGORITHM VERSION
# ============================================================

ALGORITHM_VERSION = "pressure-events-v2.0"


# ============================================================
# WHOLE-FOOT CONTACT CONFIGURATION
# ============================================================

# Adaptive thresholds relative to robust stance/loading level.
# ============================================================
# WHOLE-FOOT CONTACT CONFIGURATION
# ============================================================

# Threshold positions within the trial-specific pressure range
# from unloaded baseline -> loaded stance level.
ON_FRACTION = 0.25
OFF_FRACTION = 0.12

# Sensor-level activity threshold
SENSOR_ACTIVE_THRESHOLD_KPA = 3.125

# Require at least this many active sensors for contact onset.
MIN_ACTIVE_SENSORS_ON = 3

# Contact can terminate once active sensor count falls to this
# value OR pressure falls below the OFF threshold.
MIN_ACTIVE_SENSORS_OFF = 2

# Persistence required for transition
CONTACT_DEBOUNCE_S = 0.05


# ============================================================
# REGIONAL EVENT CONFIGURATION
# ============================================================

# Zones are defined from normalized longitudinal coordinate:
#
# heel -> 0 ---------------------------------------- 1 -> toe

HEEL_END = 0.30
FOREFOOT_START = 0.65

# Regional contact requires BOTH pressure and active sensors.
REGION_PRESSURE_FRACTION = 0.08
REGION_MIN_ACTIVE_SENSORS = 1

# Minimum stable duration before accepting a regional event.
REGION_MIN_DURATION_S = 0.05

# Foot-flat proxy:
# heel and forefoot simultaneously loaded.
FOOT_FLAT_MIN_DURATION_S = 0.05

# Heel-off:
# heel unloaded while forefoot remains loaded.
HEEL_OFF_MIN_DURATION_S = 0.05

# Forefoot dominant:
# CoP longitudinal position above this normalized coordinate.
FOREFOOT_DOMINANT_COP = 0.65
FOREFOOT_DOMINANT_MIN_DURATION_S = 0.05


# ============================================================
# CYCLE QC
# ============================================================

MIN_CYCLE_DURATION_S = 0.40
MAX_CYCLE_DURATION_S = 5.00

MIN_STANCE_DURATION_S = 0.15

# CoP requires enough total load to avoid unstable division.
COP_MIN_TOTAL_PRESSURE = 5.0


# ============================================================
# SENSOR MAP PARSING
# ============================================================

def parse_ts_coordinate_array(text: str, name: str) -> np.ndarray:
    """
    Extract LEFT_MM or RIGHT_MM from sensorsS.ts / sensorsM.ts.

    Supports:
        export const LEFT_MM: [number, number][] = [
            [x, y],
            ...
        ]

    with or without a semicolon after the array.
    """

    # Find declaration
    pattern = rf"export\s+const\s+{re.escape(name)}\b"

    match = re.search(pattern, text)

    if not match:
        raise ValueError(
            f"Could not find {name} in sensor map."
        )

    # Find "=" after declaration
    equals_idx = text.find("=", match.end())

    if equals_idx == -1:
        raise ValueError(
            f"Could not find '=' for {name}."
        )

    # Find actual array opening bracket after "="
    array_start = text.find("[", equals_idx)

    if array_start == -1:
        raise ValueError(
            f"Could not find opening '[' for {name}."
        )

    # --------------------------------------------------------
    # Find matching closing bracket.
    #
    # We cannot simply search for the next "]" because every
    # sensor coordinate [x, y] also contains brackets.
    # --------------------------------------------------------

    depth = 0
    array_end = None

    for i in range(array_start, len(text)):

        if text[i] == "[":
            depth += 1

        elif text[i] == "]":
            depth -= 1

            if depth == 0:
                array_end = i
                break

    if array_end is None:
        raise ValueError(
            f"Could not find matching closing ']' for {name}."
        )

    body = text[array_start + 1:array_end]

    # Extract coordinate pairs
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    pairs = re.findall(
        rf"\[\s*({number})\s*,\s*({number})\s*\]",
        body,
    )

    coords = np.asarray(
        [
            [float(x), float(y)]
            for x, y in pairs
        ],
        dtype=float,
    )

    if coords.shape != (64, 2):
        raise ValueError(
            f"{name}: expected 64 coordinate pairs, "
            f"found {len(coords)}; shape={coords.shape}"
        )

    return coords


def load_sensor_map(path: Path) -> dict:

    text = path.read_text(
        encoding="utf-8"
    )

    left = parse_ts_coordinate_array(
        text,
        "LEFT_MM",
    )

    right = parse_ts_coordinate_array(
        text,
        "RIGHT_MM",
    )

    file_hash = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    return {
        "left": left,
        "right": right,
        "sha256": file_hash,
        "filename": path.name,
    }


def load_all_sensor_maps():

    return {
        "S": load_sensor_map(
            SENSORS_S_FILE
        ),
        "M": load_sensor_map(
            SENSORS_M_FILE
        ),
    }


# ============================================================
# PATIENT INFORMATION
# ============================================================

def normalize_header(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalize_insole_size(value: str) -> str:

    text = (
        str(value)
        .strip()
        .lower()
    )

    mapping = {
        "s": "S",
        "small": "S",
        "sensorss": "S",

        "m": "M",
        "medium": "M",
        "sensorsm": "M",
    }

    if text not in mapping:
        raise ValueError(
            f"Unknown insole size: {value!r}"
        )

    return mapping[text]


def load_patient_insole_sizes(path: Path) -> dict:
    """
    Expected columns include:
        Patient
        Insole size
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Patient summary not found:\n{path}"
        )

    mapping = {}

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(
                "Patient CSV has no header."
            )

        normalized = {
            normalize_header(name): name
            for name in reader.fieldnames
        }

        patient_candidates = [
            "patient",
            "patient_id",
            "subject",
            "subject_id",
        ]

        size_candidates = [
            "insole_size",
            "size",
            "sensor_size",
        ]

        patient_column = next(
            (
                normalized[x]
                for x in patient_candidates
                if x in normalized
            ),
            None,
        )

        size_column = next(
            (
                normalized[x]
                for x in size_candidates
                if x in normalized
            ),
            None,
        )

        if patient_column is None:
            raise ValueError(
                "Could not find patient column."
            )

        if size_column is None:
            raise ValueError(
                "Could not find insole-size column."
            )

        for row in reader:

            patient = (
                str(row[patient_column])
                .strip()
                .lower()
            )

            raw_size = (
                str(row[size_column])
                .strip()
            )

            if not patient or not raw_size:
                continue

            mapping[patient] = (
                normalize_insole_size(
                    raw_size
                )
            )

    return mapping


# ============================================================
# TRIAL NAME
# ============================================================

def canonical_trial_name(path: Path) -> str:

    stem = path.stem

    for suffix in (
        "_filtered",
        "_pressure_filtered",
        "_cleaned",
    ):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    return stem


def parse_trial_name(trial_name: str):

    match = re.fullmatch(
        r"(.+?)_(\d+)_bws",
        trial_name,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot parse trial name: "
            f"{trial_name}"
        )

    patient_id = (
        match.group(1)
        .lower()
    )

    bws_percent = int(
        match.group(2)
    )

    return (
        patient_id,
        bws_percent,
    )


# ============================================================
# LOAD FILTERED PRESSURE
# ============================================================

def load_filtered_trial(path: Path):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "time",
            "pressure",
        }

        missing = required.difference(
            data.files
        )

        if missing:
            raise ValueError(
                f"{path.name}: missing "
                f"{sorted(missing)}"
            )

        time = np.asarray(
            data["time"],
            dtype=float,
        ).reshape(-1)

        pressure = np.asarray(
            data["pressure"],
            dtype=float,
        )

        if "sampling_rate" in data.files:
            sampling_rate = float(
                np.asarray(
                    data["sampling_rate"]
                ).reshape(-1)[0]
            )

        elif "fs" in data.files:
            sampling_rate = float(
                np.asarray(
                    data["fs"]
                ).reshape(-1)[0]
            )

        else:
            sampling_rate = (
                1.0
                / np.median(
                    np.diff(time)
                )
            )

    if pressure.ndim != 2:
        raise ValueError(
            f"{path.name}: pressure must be 2-D."
        )

    if pressure.shape[0] != len(time):

        if pressure.shape[1] == len(time):
            pressure = pressure.T

        else:
            raise ValueError(
                f"{path.name}: time/pressure "
                f"shape mismatch."
            )

    if pressure.shape[1] != 128:
        raise ValueError(
            f"{path.name}: expected 128 "
            f"pressure channels, got "
            f"{pressure.shape[1]}"
        )

    if len(time) < 2:
        raise ValueError(
            "Not enough time samples."
        )

    if not np.all(
        np.diff(time) > 0
    ):
        raise ValueError(
            "Time must be strictly increasing."
        )

    return (
        time,
        pressure,
        sampling_rate,
    )


# ============================================================
# GEOMETRIC ZONES
# ============================================================

def build_sensor_geometry(
    coords: np.ndarray
):

    x = coords[:, 0]
    y = coords[:, 1]

    y_min = float(
        np.min(y)
    )

    y_max = float(
        np.max(y)
    )

    if y_max <= y_min:
        raise ValueError(
            "Invalid longitudinal coordinates."
        )

    # Explicit convention:
    # low y = heel
    # high y = toe
    longitudinal = (
        (y - y_min)
        / (y_max - y_min)
    )

    heel_mask = (
        longitudinal < HEEL_END
    )

    midfoot_mask = (
        (longitudinal >= HEEL_END)
        & (longitudinal < FOREFOOT_START)
    )

    forefoot_mask = (
        longitudinal >= FOREFOOT_START
    )

    return {
        "x": x,
        "y": y,
        "longitudinal": longitudinal,

        "heel_mask": heel_mask,
        "midfoot_mask": midfoot_mask,
        "forefoot_mask": forefoot_mask,

        "y_min": y_min,
        "y_max": y_max,
    }


# ============================================================
# SAMPLE-LEVEL PRESSURE FEATURES
# ============================================================

def region_features(
    foot_pressure: np.ndarray,
    sensor_mask: np.ndarray,
):

    region = foot_pressure[
        :,
        sensor_mask,
    ]

    pressure_sum = np.sum(
        region,
        axis=1,
    )

    active_count = np.sum(
        region
        >= SENSOR_ACTIVE_THRESHOLD_KPA,
        axis=1,
    )

    return (
        pressure_sum,
        active_count,
    )


def calculate_foot_signals(
    foot_pressure: np.ndarray,
    coords: np.ndarray,
):

    geometry = build_sensor_geometry(
        coords
    )

    p = np.nan_to_num(
        foot_pressure,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    p = np.maximum(
        p,
        0.0,
    )

    total = np.sum(
        p,
        axis=1,
    )

    active_count = np.sum(
        p >= SENSOR_ACTIVE_THRESHOLD_KPA,
        axis=1,
    )

    # --------------------------------------------------------
    # CoP
    # --------------------------------------------------------

    weighted_x = np.sum(
        p * geometry["x"][None, :],
        axis=1,
    )

    weighted_y = np.sum(
        p * geometry["y"][None, :],
        axis=1,
    )

    cop_x = np.full(
        len(total),
        np.nan,
        dtype=float,
    )

    cop_y = np.full(
        len(total),
        np.nan,
        dtype=float,
    )

    valid_cop = (
        total >= COP_MIN_TOTAL_PRESSURE
    )

    cop_x[valid_cop] = (
        weighted_x[valid_cop]
        / total[valid_cop]
    )

    cop_y[valid_cop] = (
        weighted_y[valid_cop]
        / total[valid_cop]
    )

    cop_longitudinal = np.full(
        len(total),
        np.nan,
        dtype=float,
    )

    cop_longitudinal[valid_cop] = (
        (
            cop_y[valid_cop]
            - geometry["y_min"]
        )
        / (
            geometry["y_max"]
            - geometry["y_min"]
        )
    )

    # --------------------------------------------------------
    # Zones
    # --------------------------------------------------------

    (
        pressure_heel,
        active_heel,
    ) = region_features(
        p,
        geometry["heel_mask"],
    )

    (
        pressure_midfoot,
        active_midfoot,
    ) = region_features(
        p,
        geometry["midfoot_mask"],
    )

    (
        pressure_forefoot,
        active_forefoot,
    ) = region_features(
        p,
        geometry["forefoot_mask"],
    )

    return {
        "total_pressure": total,
        "active_sensor_count": active_count,

        "cop_x": cop_x,
        "cop_y": cop_y,
        "cop_longitudinal": cop_longitudinal,

        "pressure_heel": pressure_heel,
        "pressure_midfoot": pressure_midfoot,
        "pressure_forefoot": pressure_forefoot,

        "active_count_heel": active_heel,
        "active_count_midfoot": active_midfoot,
        "active_count_forefoot": active_forefoot,

        "geometry": geometry,
    }


# ============================================================
# ADAPTIVE CONTACT THRESHOLDS
# ============================================================

def robust_loading_level(
    total_pressure: np.ndarray
) -> float:

    positive = total_pressure[
        total_pressure > 0
    ]

    if len(positive) == 0:
        return 0.0

    # Upper half of non-zero loading values gives a robust
    # approximation of actual loaded/stance pressure.
    cutoff = np.percentile(
        positive,
        50,
    )

    loaded = positive[
        positive >= cutoff
    ]

    if len(loaded) == 0:
        loaded = positive

    return float(
        np.median(loaded)
    )


def calculate_contact_thresholds(
    total_pressure: np.ndarray,
):
    """
    Estimate adaptive whole-foot contact thresholds from both the
    unloaded and loaded pressure distributions.

    This is preferable to using a fraction of stance pressure alone,
    because swing may have a non-zero pressure baseline.
    """

    x = np.asarray(
        total_pressure,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        raise ValueError(
            "No finite total-pressure samples."
        )

    # Robust lower and upper levels.
    #
    # 10th percentile approximates the unloaded/swing level.
    # 90th percentile approximates loaded stance.
    low_level = float(
        np.percentile(
            x,
            10,
        )
    )

    high_level = float(
        np.percentile(
            x,
            90,
        )
    )

    dynamic_range = (
        high_level
        - low_level
    )

    if dynamic_range <= 0:
        raise ValueError(
            "Pressure signal has no usable dynamic range."
        )

    on_threshold = (
    low_level
    + ON_FRACTION
    * dynamic_range
    )

    off_threshold = (
    low_level
    + OFF_FRACTION
    * dynamic_range
    )

    # Safety check
    if off_threshold >= on_threshold:
        raise ValueError(
            "OFF threshold must be lower than ON threshold."
        )

    return {
        "unloaded_level": low_level,
        "loaded_level": high_level,
        "dynamic_range": dynamic_range,
        "on_threshold": float(on_threshold),
        "off_threshold": float(off_threshold),
    }


# ============================================================
# DEBOUNCED WHOLE-FOOT CONTACT
# ============================================================

def build_contact_mask(
    total_pressure,
    active_sensor_count,
    sampling_rate,
    on_threshold,
    off_threshold,
):
    """
    Hysteretic whole-foot contact detector.

    ON:
        pressure >= ON threshold
        AND enough active sensors

    OFF:
        pressure <= OFF threshold
        OR too few active sensors

    Both transitions must persist for CONTACT_DEBOUNCE_S.
    """

    total_pressure = np.asarray(
        total_pressure,
        dtype=float,
    )

    active_sensor_count = np.asarray(
        active_sensor_count,
        dtype=int,
    )

    n = len(
        total_pressure
    )

    contact = np.zeros(
        n,
        dtype=bool,
    )

    debounce_samples = max(
        1,
        int(
            round(
                CONTACT_DEBOUNCE_S
                * sampling_rate
            )
        ),
    )

    state = False

    candidate_state = None
    candidate_start = None
    candidate_count = 0

    for i in range(n):

        pressure_i = (
            total_pressure[i]
        )

        active_i = (
            active_sensor_count[i]
        )

        # ----------------------------------------------------
        # Desired state
        # ----------------------------------------------------

        if not state:

            # Start contact only if BOTH criteria indicate load
            desired_state = (
                pressure_i >= on_threshold
                and
                active_i >= MIN_ACTIVE_SENSORS_ON
            )

        else:

            # End contact if EITHER criterion strongly indicates
            # unloading.
            should_turn_off = (
                pressure_i <= off_threshold
                or
                active_i <= MIN_ACTIVE_SENSORS_OFF
            )

            desired_state = (
                not should_turn_off
            )

        # ----------------------------------------------------
        # Debounce
        # ----------------------------------------------------

        if desired_state == state:

            candidate_state = None
            candidate_start = None
            candidate_count = 0

        else:

            if candidate_state == desired_state:

                candidate_count += 1

            else:

                candidate_state = (
                    desired_state
                )

                candidate_start = i
                candidate_count = 1

            if (
                candidate_count
                >= debounce_samples
            ):

                state = (
                    candidate_state
                )

                # Put the transition at the beginning of the
                # stable interval rather than after debounce.
                if candidate_start is not None:

                    contact[
                        candidate_start:i + 1
                    ] = state

                candidate_state = None
                candidate_start = None
                candidate_count = 0

        contact[i] = state

    return contact

# ============================================================
# BINARY TRANSITIONS
# ============================================================

def contact_transitions(
    mask: np.ndarray,
):

    x = mask.astype(
        np.int8
    )

    diff = np.diff(
        x
    )

    ic_indices = (
        np.where(
            diff == 1
        )[0]
        + 1
    )

    to_indices = (
        np.where(
            diff == -1
        )[0]
        + 1
    )

    # If recording begins in stance, do not invent an IC at
    # sample zero. It is an incomplete cycle and is excluded
    # naturally from IC-to-next-IC cycles.

    return (
        ic_indices,
        to_indices,
    )


# ============================================================
# STABLE BOOLEAN EVENT
# ============================================================

def first_stable_true(
    mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    min_samples: int,
):

    start_idx = max(
        0,
        start_idx,
    )

    end_idx = min(
        len(mask),
        end_idx,
    )

    count = 0

    for i in range(
        start_idx,
        end_idx,
    ):

        if mask[i]:
            count += 1

            if count >= min_samples:
                return (
                    i
                    - count
                    + 1
                )

        else:
            count = 0

    return None


# ============================================================
# REGIONAL CONTACT
# ============================================================

def build_region_contact(
    region_pressure,
    region_active_count,
    whole_foot_dynamic_range,
):
    """
    Pressure-defined regional contact.

    Regional loading is intentionally conservative: only a small
    fraction of whole-foot dynamic pressure is required because a
    single anatomical region carries only part of total plantar load.
    """

    pressure_threshold = max(
        SENSOR_ACTIVE_THRESHOLD_KPA,
        REGION_PRESSURE_FRACTION
        * whole_foot_dynamic_range,
    )

    mask = (
        (
            region_pressure
            >= pressure_threshold
        )
        &
        (
            region_active_count
            >= REGION_MIN_ACTIVE_SENSORS
        )
    )

    return (
        mask,
        float(
            pressure_threshold
        ),
    )


# ============================================================
# EVENT HELPERS
# ============================================================

def find_to_between(
    to_indices,
    ic_idx,
    next_ic_idx,
):

    candidates = to_indices[
        (to_indices > ic_idx)
        & (to_indices < next_ic_idx)
    ]

    if len(candidates) == 0:
        return None

    return int(
        candidates[0]
    )


def determine_ic_region(
    heel_contact,
    forefoot_contact,
    ic_idx,
    window_samples,
    simultaneous_tolerance_samples,
):
    heel_valid = (
        heel_contact is not None
        and heel_contact <= ic_idx + window_samples
    )

    forefoot_valid = (
        forefoot_contact is not None
        and forefoot_contact <= ic_idx + window_samples
    )

    if heel_valid and not forefoot_valid:
        return "heel"

    if forefoot_valid and not heel_valid:
        return "forefoot"

    if not heel_valid and not forefoot_valid:
        return "unknown"

    difference = heel_contact - forefoot_contact

    if abs(difference) <= simultaneous_tolerance_samples:
        return "mixed"

    if difference < 0:
        return "heel"

    return "forefoot"


# ============================================================
# ANALYZE ONE GAIT CYCLE
# ============================================================

def analyze_cycle(
    cycle_id,
    foot,
    time,
    signals,
    contact_mask,
    ic_idx,
    next_ic_idx,
    to_idx,
    sampling_rate,
    thresholds,
):

    flags = []

    cycle_duration = (
        time[next_ic_idx]
        - time[ic_idx]
    )

    cycle_valid = True

    if (
        cycle_duration
        < MIN_CYCLE_DURATION_S
        or cycle_duration
        > MAX_CYCLE_DURATION_S
    ):
        flags.append(
            "IMPLAUSIBLE_CYCLE_DURATION"
        )
        cycle_valid = False

    if to_idx is None:

        flags.append(
            "MISSING_TO"
        )

        stance_duration = np.nan
        swing_duration = np.nan
        stance_percent = np.nan
        swing_percent = np.nan

        stance_end_idx = next_ic_idx

    else:

        stance_end_idx = to_idx

        stance_duration = float(
            time[to_idx]
            - time[ic_idx]
        )

        swing_duration = float(
            time[next_ic_idx]
            - time[to_idx]
        )

        if (
            stance_duration
            < MIN_STANCE_DURATION_S
        ):
            flags.append(
                "SHORT_STANCE"
            )
            cycle_valid = False

        stance_percent = (
            100.0
            * stance_duration
            / cycle_duration
        )

        swing_percent = (
            100.0
            * swing_duration
            / cycle_duration
        )

    # --------------------------------------------------------
    # Regional contact masks
    # --------------------------------------------------------

    heel_contact_mask = (
        signals[
            "heel_contact_mask"
        ]
    )

    forefoot_contact_mask = (
        signals[
            "forefoot_contact_mask"
        ]
    )

    regional_samples = max(
        1,
        int(
            round(
                REGION_MIN_DURATION_S
                * sampling_rate
            )
        ),
    )

    footflat_samples = max(
        1,
        int(
            round(
                FOOT_FLAT_MIN_DURATION_S
                * sampling_rate
            )
        ),
    )

    heeloff_samples = max(
        1,
        int(
            round(
                HEEL_OFF_MIN_DURATION_S
                * sampling_rate
            )
        ),
    )

    forefoot_dom_samples = max(
        1,
        int(
            round(
                FOREFOOT_DOMINANT_MIN_DURATION_S
                * sampling_rate
            )
        ),
    )

    # --------------------------------------------------------
    # Heel contact
    # --------------------------------------------------------

    heel_contact_idx = (
        first_stable_true(
            heel_contact_mask,
            ic_idx,
            stance_end_idx,
            regional_samples,
        )
    )

    # --------------------------------------------------------
    # Forefoot contact
    # --------------------------------------------------------

    forefoot_contact_idx = (
        first_stable_true(
            forefoot_contact_mask,
            ic_idx,
            stance_end_idx,
            regional_samples,
        )
    )

    # --------------------------------------------------------
    # Initial contact region
    # --------------------------------------------------------

    ic_window_samples = max(
    regional_samples,
    int(round(0.10 * sampling_rate)),
)

    simultaneous_tolerance_samples = max(
    1,
    int(round(0.03 * sampling_rate)),
)

    ic_region = determine_ic_region(
        heel_contact_idx,
        forefoot_contact_idx,
        ic_idx,
        ic_window_samples,
        simultaneous_tolerance_samples,
)

    if ic_region == "forefoot":
        flags.append(
            "FOREFOOT_INITIAL_CONTACT"
        )

    elif ic_region == "unknown":
        flags.append(
            "UNKNOWN_IC_REGION"
        )

    # --------------------------------------------------------
    # Foot-flat proxy
    # --------------------------------------------------------

    both_loaded = (
        heel_contact_mask
        & forefoot_contact_mask
        & contact_mask
    )

    foot_flat_idx = (
        first_stable_true(
            both_loaded,
            ic_idx,
            stance_end_idx,
            footflat_samples,
        )
    )

    # --------------------------------------------------------
    # Heel off
    #
    # Heel not loaded + forefoot loaded, after heel contact
    # if heel contact exists.
    # --------------------------------------------------------

    heel_off_candidate = (
        (~heel_contact_mask)
        & forefoot_contact_mask
        & contact_mask
    )

    heel_off_search_start = (
        heel_contact_idx + 1
        if heel_contact_idx is not None
        else ic_idx
    )

    heel_off_idx = (
        first_stable_true(
            heel_off_candidate,
            heel_off_search_start,
            stance_end_idx,
            heeloff_samples,
        )
    )

    # If heel was never loaded, "heel off" has no meaningful
    # pressure-defined interpretation.
    if heel_contact_idx is None:
        heel_off_idx = None

    # --------------------------------------------------------
    # Forefoot-dominant phase
    # --------------------------------------------------------

    cop_long = (
        signals[
            "cop_longitudinal"
        ]
    )

    forefoot_dom_mask = (
        np.isfinite(
            cop_long
        )
        & (
            cop_long
            >= FOREFOOT_DOMINANT_COP
        )
        & contact_mask
    )

    forefoot_dominant_idx = (
        first_stable_true(
            forefoot_dom_mask,
            ic_idx,
            stance_end_idx,
            forefoot_dom_samples,
        )
    )

    # --------------------------------------------------------
    # Missing-event flags
    # --------------------------------------------------------

    if heel_contact_idx is None:
        flags.append(
            "NO_HEEL_CONTACT"
        )

    if forefoot_contact_idx is None:
        flags.append(
            "NO_FOREFOOT_CONTACT"
        )

    if foot_flat_idx is None:
        flags.append(
            "NO_FOOT_FLAT_PROXY"
        )

    if heel_off_idx is None:
        flags.append(
            "NO_HEEL_OFF"
        )

    if forefoot_dominant_idx is None:
        flags.append(
            "NO_FOREFOOT_DOMINANT_PHASE"
        )

    # --------------------------------------------------------
    # CoP progression
    # --------------------------------------------------------

    if to_idx is not None:

        stance_cop_y = (
            signals["cop_y"][
                ic_idx:to_idx + 1
            ]
        )

        valid = np.isfinite(
            stance_cop_y
        )

        if np.sum(valid) >= 2:

            valid_y = (
                stance_cop_y[
                    valid
                ]
            )

            # Longitudinal excursion from posterior-most to
            # anterior-most observed CoP during stance.
            cop_progression_mm = float(
                np.nanmax(valid_y)
                - np.nanmin(valid_y)
            )

            cop_progression_valid = True

        else:

            cop_progression_mm = np.nan
            cop_progression_valid = False
            flags.append(
                "COP_INSUFFICIENT_LOAD"
            )

    else:

        cop_progression_mm = np.nan
        cop_progression_valid = False

    if not flags:
        flags = ["OK"]

    def event_time(index):
        if index is None:
            return np.nan

        return float(
            time[index]
        )

    return {
        "foot": foot,
        "cycle_id": cycle_id,

        "ic_time":
            float(time[ic_idx]),

        "toe_off_time":
            event_time(to_idx),

        "next_ic_time":
            float(time[next_ic_idx]),

        "stance_duration_s":
            stance_duration,

        "swing_duration_s":
            swing_duration,

        "cycle_duration_s":
            float(cycle_duration),

        "stance_percent":
            stance_percent,

        "swing_percent":
            swing_percent,

        "ic_region":
            ic_region,

        "heel_contact_time":
            event_time(
                heel_contact_idx
            ),

        "forefoot_contact_time":
            event_time(
                forefoot_contact_idx
            ),

        "foot_flat_proxy_time":
            event_time(
                foot_flat_idx
            ),

        "heel_off_time":
            event_time(
                heel_off_idx
            ),

        "forefoot_dominant_time":
            event_time(
                forefoot_dominant_idx
            ),

        "heel_contact_valid":
            heel_contact_idx
            is not None,

        "forefoot_contact_valid":
            forefoot_contact_idx
            is not None,

        "foot_flat_proxy_valid":
            foot_flat_idx
            is not None,

        "heel_off_valid":
            heel_off_idx
            is not None,

        "forefoot_dominant_valid":
            forefoot_dominant_idx
            is not None,

        "cop_progression_mm":
            cop_progression_mm,

        "cop_progression_valid":
            cop_progression_valid,

        "cycle_valid":
            cycle_valid,

        "quality_flags":
            ";".join(flags),
    }


# ============================================================
# ANALYZE ONE FOOT
# ============================================================

def analyze_foot(
    foot,
    time,
    foot_pressure,
    coords,
    sampling_rate,
):

    signals = calculate_foot_signals(
        foot_pressure,
        coords,
    )

    # --------------------------------------------------------
    # Adaptive whole-foot thresholds
    # --------------------------------------------------------

    thresholds = (
        calculate_contact_thresholds(
            signals["total_pressure"]
        )
    )

    # --------------------------------------------------------
    # Whole-foot contact mask
    # --------------------------------------------------------

    contact_mask = (
        build_contact_mask(
            signals["total_pressure"],
            signals["active_sensor_count"],
            sampling_rate,
            thresholds["on_threshold"],
            thresholds["off_threshold"],
        )
    )

    (
        ic_indices,
        to_indices,
    ) = contact_transitions(
        contact_mask
    )

    # --------------------------------------------------------
    # Regional masks
    # --------------------------------------------------------

    regional_reference = (
        thresholds["dynamic_range"]
    )

    (
        heel_contact_mask,
        heel_threshold,
    ) = build_region_contact(
        signals["pressure_heel"],
        signals["active_count_heel"],
        regional_reference,
    )

    (
        forefoot_contact_mask,
        forefoot_threshold,
    ) = build_region_contact(
        signals["pressure_forefoot"],
        signals["active_count_forefoot"],
        regional_reference,
    )

    signals[
        "heel_contact_mask"
    ] = heel_contact_mask

    signals[
        "forefoot_contact_mask"
    ] = forefoot_contact_mask

    thresholds[
        "heel_region_threshold"
    ] = heel_threshold

    thresholds[
        "forefoot_region_threshold"
    ] = forefoot_threshold

    # --------------------------------------------------------
    # Cycles
    # --------------------------------------------------------

    cycles = []

    for i in range(
        len(ic_indices) - 1
    ):

        ic_idx = int(
            ic_indices[i]
        )

        next_ic_idx = int(
            ic_indices[i + 1]
        )

        to_idx = find_to_between(
            to_indices,
            ic_idx,
            next_ic_idx,
        )

        row = analyze_cycle(
            cycle_id=i + 1,
            foot=foot,
            time=time,
            signals=signals,
            contact_mask=contact_mask,
            ic_idx=ic_idx,
            next_ic_idx=next_ic_idx,
            to_idx=to_idx,
            sampling_rate=sampling_rate,
            thresholds=thresholds,
        )

        cycles.append(row)

    # --------------------------------------------------------
    # Store event indices and contact
    # --------------------------------------------------------

    signals[
        "contact_mask"
    ] = contact_mask

    signals[
        "ic_indices"
    ] = ic_indices

    signals[
        "to_indices"
    ] = to_indices

    return (
        signals,
        thresholds,
        cycles,
    )


# ============================================================
# SAVE SAMPLE-LEVEL SIGNALS
# ============================================================

def save_pressure_signals(
    path,
    time,
    left,
    right,
):

    np.savez_compressed(
        path,

        time=time,

        left_total_pressure=
            left["total_pressure"],

        right_total_pressure=
            right["total_pressure"],

        left_active_sensor_count=
            left["active_sensor_count"],

        right_active_sensor_count=
            right["active_sensor_count"],

        left_contact_mask=
            left["contact_mask"],

        right_contact_mask=
            right["contact_mask"],

        left_CoP_x=
            left["cop_x"],

        left_CoP_y=
            left["cop_y"],

        right_CoP_x=
            right["cop_x"],

        right_CoP_y=
            right["cop_y"],

        left_CoP_longitudinal=
            left["cop_longitudinal"],

        right_CoP_longitudinal=
            right["cop_longitudinal"],

        left_pressure_heel=
            left["pressure_heel"],

        left_pressure_midfoot=
            left["pressure_midfoot"],

        left_pressure_forefoot=
            left["pressure_forefoot"],

        right_pressure_heel=
            right["pressure_heel"],

        right_pressure_midfoot=
            right["pressure_midfoot"],

        right_pressure_forefoot=
            right["pressure_forefoot"],

        left_active_count_heel=
            left["active_count_heel"],

        left_active_count_midfoot=
            left["active_count_midfoot"],

        left_active_count_forefoot=
            left["active_count_forefoot"],

        right_active_count_heel=
            right["active_count_heel"],

        right_active_count_midfoot=
            right["active_count_midfoot"],

        right_active_count_forefoot=
            right["active_count_forefoot"],
    )


# ============================================================
# SAVE CYCLE TABLE
# ============================================================

def save_cycles_csv(
    path,
    trial_name,
    patient_id,
    bws_percent,
    rows,
):

    output_rows = []

    for row in rows:

        output_rows.append(
            {
                "trial_name":
                    trial_name,

                "patient_id":
                    patient_id,

                "bws_percent":
                    bws_percent,

                **row,
            }
        )

    if not output_rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    path,
    trial_name,
    patient_id,
    bws_percent,
    sampling_rate,
    insole_size,
    sensor_map,
    left_thresholds,
    right_thresholds,
):

    metadata = {
        "trial_name":
            trial_name,

        "patient_id":
            patient_id,

        "bws_percent":
            bws_percent,

        "sampling_rate_hz":
            sampling_rate,

        "pressure_units":
            "kPa",

        "insole_size":
            insole_size,

        "algorithm_version":
            ALGORITHM_VERSION,

        "coordinate_map": {
            "filename":
                sensor_map[
                    "filename"
                ],

            "sha256":
                sensor_map[
                    "sha256"
                ],

            "coordinate_units":
                "mm",

            "coordinate_order":
                "[x, y]",

            "longitudinal_axis":
                "y",

            "heel_direction":
                "low_y",

            "toe_direction":
                "high_y",

            "orientation_source":
                "verified_from_sensor-map geometry",

            "pressure_channel_mapping":
                (
                    "Assumed direct array order: "
                    "pressure left channel i -> "
                    "LEFT_MM[i]; pressure right "
                    "channel i -> RIGHT_MM[i]. "
                    "Requires external/manufacturer "
                    "or physical validation."
                ),
        },

        "zone_definitions": {
            "method":
                "normalized longitudinal "
                "sensor coordinate",

            "heel":
                f"0 <= y_norm < "
                f"{HEEL_END}",

            "midfoot":
                f"{HEEL_END} <= y_norm < "
                f"{FOREFOOT_START}",

            "forefoot":
                f"{FOREFOOT_START} <= "
                f"y_norm <= 1",

            "heel_end":
                HEEL_END,

            "forefoot_start":
                FOREFOOT_START,
        },

        "contact_detection": {
            "on_fraction":
                ON_FRACTION,

            "off_fraction":
                OFF_FRACTION,

            "sensor_active_threshold_kpa":
                SENSOR_ACTIVE_THRESHOLD_KPA,

            "minimum_active_sensors_on":
                MIN_ACTIVE_SENSORS_ON,

            "minimum_active_sensors_off":
                MIN_ACTIVE_SENSORS_OFF,

            "debounce_s":
                CONTACT_DEBOUNCE_S,

            "left_thresholds":
                left_thresholds,

            "right_thresholds":
                right_thresholds,
        },

        "regional_events": {
            "region_pressure_fraction":
                REGION_PRESSURE_FRACTION,

            "region_min_active_sensors":
                REGION_MIN_ACTIVE_SENSORS,

            "region_min_duration_s":
                REGION_MIN_DURATION_S,

            "foot_flat_min_duration_s":
                FOOT_FLAT_MIN_DURATION_S,

            "heel_off_min_duration_s":
                HEEL_OFF_MIN_DURATION_S,

            "forefoot_dominant_cop_normalized":
                FOREFOOT_DOMINANT_COP,

            "forefoot_dominant_min_duration_s":
                FOREFOOT_DOMINANT_MIN_DURATION_S,
        },

        "cycle_definition":
            "IC_i -> IC_(i+1)",

        "stance_definition":
            "IC -> TO",

        "swing_definition":
            "TO -> next IC",

        "imu_sync": {
            "performed":
                False,

            "note":
                (
                    "All event times are in the "
                    "original pressure timebase. "
                    "Pressure and IMU must be "
                    "explicitly synchronized in a "
                    "separate processing step."
                ),
        },
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# QC OVERVIEW
# ============================================================

def plot_pressure_qc(
    path,
    trial_name,
    time,
    left,
    right,
):

    for foot_name, sig in (
        ("Left", left),
        ("Right", right),
    ):

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 11),
            sharex=True,
        )

        # Total pressure
        axes[0].plot(
            time,
            sig["total_pressure"],
            linewidth=0.8,
        )

        axes[0].set_ylabel(
            "Total pressure"
        )

        # Regional pressure
        axes[1].plot(
            time,
            sig["pressure_heel"],
            label="Heel",
        )

        axes[1].plot(
            time,
            sig["pressure_midfoot"],
            label="Midfoot",
        )

        axes[1].plot(
            time,
            sig["pressure_forefoot"],
            label="Forefoot",
        )

        axes[1].legend()

        axes[1].set_ylabel(
            "Regional pressure"
        )

        # Contact
        axes[2].plot(
            time,
            sig[
                "contact_mask"
            ].astype(float),
        )

        axes[2].set_ylabel(
            "Contact"
        )

        # CoP
        axes[3].plot(
            time,
            sig[
                "cop_longitudinal"
            ],
        )

        axes[3].set_ylabel(
            "CoP longitudinal"
        )

        axes[3].set_xlabel(
            "Pressure timebase (s)"
        )

        # IC and TO on all axes
        for idx in sig[
            "ic_indices"
        ]:

            for ax in axes:
                ax.axvline(
                    time[idx],
                    linestyle="--",
                    linewidth=0.8,
                )

        for idx in sig[
            "to_indices"
        ]:

            for ax in axes:
                ax.axvline(
                    time[idx],
                    linestyle=":",
                    linewidth=0.8,
                )

        fig.suptitle(
            f"{trial_name} | {foot_name}"
        )

        fig.tight_layout()

        output = (
            path
            / (
                f"{trial_name}_"
                f"{foot_name.lower()}_"
                f"pressure_events.png"
            )
        )

        fig.savefig(
            output,
            dpi=160,
        )

        plt.close(fig)


# ============================================================
# HEATMAP QC
# ============================================================

def plot_sensor_heatmaps(
    path,
    trial_name,
    time,
    pressure,
    sensor_map,
    left_signals,
    right_signals,
):

    examples = []

    for foot_name, offset, signals, coords in (
        (
            "Left",
            0,
            left_signals,
            sensor_map["left"],
        ),
        (
            "Right",
            64,
            right_signals,
            sensor_map["right"],
        ),
    ):

        ic = signals[
            "ic_indices"
        ]

        to = signals[
            "to_indices"
        ]

        count = 0

        for ic_idx in ic:

            to_candidates = to[
                to > ic_idx
            ]

            if len(to_candidates) == 0:
                continue

            to_idx = int(
                to_candidates[0]
            )

            if to_idx <= ic_idx:
                continue

            mid_idx = int(
                (ic_idx + to_idx)
                // 2
            )

            examples.append(
                (
                    foot_name,
                    offset,
                    coords,
                    mid_idx,
                )
            )

            count += 1

            if count >= 3:
                break

    if not examples:
        return

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 9),
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (
        foot_name,
        offset,
        coords,
        sample_idx,
    ) in zip(
        axes,
        examples,
    ):

        values = pressure[
            sample_idx,
            offset:offset + 64,
        ]

        ax.axis("on")

        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            s=90,
        )

        ax.set_aspect(
            "equal"
        )

        ax.set_title(
            f"{foot_name} | "
            f"{time[sample_idx]:.2f} s"
        )

        ax.set_xlabel(
            "x (mm)"
        )

        ax.set_ylabel(
            "y (mm)"
        )

        fig.colorbar(
            sc,
            ax=ax,
            label="kPa",
        )

    fig.suptitle(
        f"{trial_name} | "
        f"sensor pressure QC"
    )

    fig.tight_layout()

    fig.savefig(
        path
        / f"{trial_name}_sensor_heatmaps.png",
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# TERMINAL QC SUMMARY
# ============================================================

def print_cycle_summary(
    trial_name,
    rows,
):

    print(
        f"\n  Cycle QC: {trial_name}"
    )

    for foot in (
        "left",
        "right",
    ):

        subset = [
            x
            for x in rows
            if x["foot"] == foot
        ]

        if not subset:
            print(
                f"    {foot}: no cycles"
            )
            continue

        valid = sum(
            bool(x["cycle_valid"])
            for x in subset
        )

        regions = {
            "heel": 0,
            "forefoot": 0,
            "mixed": 0,
            "unknown": 0,
        }

        for row in subset:
            regions[
                row["ic_region"]
            ] += 1

        n = len(subset)

        print(
            f"    {foot}: "
            f"{valid}/{n} valid cycles"
        )

        print(
            "      IC regions: "
            f"heel={regions['heel']/n:.1%}, "
            f"forefoot={regions['forefoot']/n:.1%}, "
            f"mixed={regions['mixed']/n:.1%}, "
            f"unknown={regions['unknown']/n:.1%}"
        )


# ============================================================
# PROCESS TRIAL
# ============================================================

def process_trial(
    input_path,
    patient_sizes,
    sensor_maps,
):

    trial_name = (
        canonical_trial_name(
            input_path
        )
    )

    (
        patient_id,
        bws_percent,
    ) = parse_trial_name(
        trial_name
    )

    if patient_id not in patient_sizes:
        raise ValueError(
            f"No insole size found for "
            f"patient {patient_id!r}"
        )

    insole_size = (
        patient_sizes[
            patient_id
        ]
    )

    sensor_map = (
        sensor_maps[
            insole_size
        ]
    )

    (
        time,
        pressure,
        sampling_rate,
    ) = load_filtered_trial(
        input_path
    )

    left_pressure = (
        pressure[:, :64]
    )

    right_pressure = (
        pressure[:, 64:]
    )

    (
        left,
        left_thresholds,
        left_cycles,
    ) = analyze_foot(
        foot="left",
        time=time,
        foot_pressure=left_pressure,
        coords=sensor_map["left"],
        sampling_rate=sampling_rate,
    )

    (
        right,
        right_thresholds,
        right_cycles,
    ) = analyze_foot(
        foot="right",
        time=time,
        foot_pressure=right_pressure,
        coords=sensor_map["right"],
        sampling_rate=sampling_rate,
    )

    all_cycles = (
        left_cycles
        + right_cycles
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    signals_path = (
        OUTPUT_FOLDER
        / (
            f"{trial_name}_"
            f"pressure_signals.npz"
        )
    )

    cycles_path = (
        OUTPUT_FOLDER
        / (
            f"{trial_name}_"
            f"gait_cycles.csv"
        )
    )

    metadata_path = (
        OUTPUT_FOLDER
        / (
            f"{trial_name}_"
            f"events_metadata.json"
        )
    )

    save_pressure_signals(
        signals_path,
        time,
        left,
        right,
    )

    save_cycles_csv(
        cycles_path,
        trial_name,
        patient_id,
        bws_percent,
        all_cycles,
    )

    save_metadata(
        metadata_path,
        trial_name,
        patient_id,
        bws_percent,
        sampling_rate,
        insole_size,
        sensor_map,
        left_thresholds,
        right_thresholds,
    )

    plot_pressure_qc(
        QC_FOLDER,
        trial_name,
        time,
        left,
        right,
    )

    plot_sensor_heatmaps(
        QC_FOLDER,
        trial_name,
        time,
        pressure,
        sensor_map,
        left,
        right,
    )

    print_cycle_summary(
        trial_name,
        all_cycles,
    )

    return len(
        all_cycles
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading patient information..."
    )

    patient_sizes = (
        load_patient_insole_sizes(
            PATIENT_CSV
        )
    )

    print(
        f"Loaded insole sizes for "
        f"{len(patient_sizes)} patients."
    )

    for patient, size in sorted(
        patient_sizes.items()
    ):
        print(
            f"  {patient}: {size}"
        )

    print(
        "\nLoading sensor maps..."
    )

    sensor_maps = (
        load_all_sensor_maps()
    )

    print(
        "  S map: OK"
    )

    print(
        "  M map: OK"
    )

    files = sorted(
        INPUT_FOLDER.glob(
            "*_filtered.npz"
        )
    )

    if not files:
        raise RuntimeError(
            f"No filtered pressure files "
            f"found in:\n{INPUT_FOLDER}"
        )

    print(
        f"\nFound {len(files)} trials."
    )

    successful = 0
    failed = 0
    total_cycles = 0

    for path in files:

        print()
        print("=" * 70)
        print(
            f"Processing: {path.name}"
        )
        print("=" * 70)

        try:

            n_cycles = (
                process_trial(
                    path,
                    patient_sizes,
                    sensor_maps,
                )
            )

            total_cycles += (
                n_cycles
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"ERROR: {exc}"
            )

    print()
    print("=" * 70)
    print(
        "PRESSURE EVENT DETECTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Successful trials : {successful}"
    )

    print(
        f"Failed trials     : {failed}"
    )

    print(
        f"Gait cycles       : {total_cycles}"
    )

    print(
        f"\nOutputs:\n"
        f"{OUTPUT_FOLDER}"
    )

    print(
        f"\nQC plots:\n"
        f"{QC_FOLDER}"
    )


if __name__ == "__main__":
    main()