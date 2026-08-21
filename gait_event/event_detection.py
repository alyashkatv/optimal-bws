#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/filtered_second_batch"
)   

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/events_second_batch"
)

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128


# ============================================================
# CONTACT DETECTION PARAMETERS
# ============================================================

# A sensor is considered individually active when its pressure
# exceeds this value.
#
# This is NOT by itself enough to declare whole-foot contact.
SENSOR_ACTIVE_THRESHOLD_KPA = 3.125


# Minimum number of simultaneously active sensors required
# to support whole-foot contact.
MIN_ACTIVE_SENSORS_ON = 3

# Slightly lower requirement for staying in contact.
MIN_ACTIVE_SENSORS_OFF = 2


# ------------------------------------------------------------
# Adaptive whole-foot loading threshold
#
# We estimate a low-loading baseline and high-loading level
# separately for every foot and every trial.
#
# ON threshold is higher than OFF threshold -> hysteresis.
# ------------------------------------------------------------

ON_FRACTION = 0.15
OFF_FRACTION = 0.08


# ------------------------------------------------------------
# Temporal debounce
#
# These remove very brief false contact / false swing periods.
# Importantly these are NOT normal-gait assumptions.
# ------------------------------------------------------------

MIN_CONTACT_DURATION_S = 0.10
MIN_SWING_DURATION_S = 0.10


# ============================================================
# FILE NAME HANDLING
# ============================================================

def canonical_trial_stem(path):
    stem = path.stem

    for suffix in (
        "_filtered",
        "_cleaned",
        "_cropped",
    ):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem


def parse_trial_name(path):
    """
    Examples:

        ab_0_bws_filtered.npz
        ar_30_bws_filtered.npz
        yer_ground_filtered.npz

    Returns:
        patient_id, condition
    """

    stem = canonical_trial_stem(
        path
    ).lower()

    match = re.fullmatch(
        r"([a-z]+)_(?:(\d+)_bws|(ground))",
        stem,
    )

    if not match:
        return None

    patient = match.group(1)

    if match.group(2) is not None:
        condition = int(
            match.group(2)
        )
    else:
        condition = "ground"

    return patient, condition


# ============================================================
# LOAD FILTERED NPZ
# ============================================================

def load_filtered_npz(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "time",
            "pressure",
            "sampling_rate",
        }

        missing = required.difference(
            data.files
        )

        if missing:
            raise ValueError(
                f"{path.name}: missing keys "
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

        fs = float(
            np.asarray(
                data["sampling_rate"]
            ).reshape(-1)[0]
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if pressure.ndim != 2:
        raise ValueError(
            f"{path.name}: pressure must be 2-D, "
            f"got {pressure.shape}"
        )

    # Allow 128 x samples representation too
    if (
        pressure.shape[1]
        != TOTAL_SENSORS
    ):
        if (
            pressure.shape[0]
            == TOTAL_SENSORS
        ):
            pressure = pressure.T

        else:
            raise ValueError(
                f"{path.name}: expected 128 sensors, "
                f"got shape {pressure.shape}"
            )

    if len(time) != len(pressure):
        raise ValueError(
            f"{path.name}: "
            f"time length != pressure length"
        )

    if (
        not np.isfinite(fs)
        or fs <= 0
    ):
        raise ValueError(
            f"{path.name}: invalid "
            f"sampling rate {fs}"
        )

    # Butterworth filtering can introduce
    # small negative values around transitions.
    #
    # Negative physical pressure is impossible,
    # so contact detection uses a clipped copy.
    pressure_for_detection = np.clip(
        pressure,
        0,
        None,
    )

    return (
        time,
        pressure_for_detection,
        fs,
    )


# ============================================================
# BASIC FOOT-LEVEL SIGNALS
# ============================================================

def calculate_foot_signals(
    foot_pressure
):
    """
    foot_pressure:
        samples x 64

    Returns:
        total_pressure
        average_pressure
        active_sensor_count
    """

    total_pressure = np.sum(
        foot_pressure,
        axis=1,
    )

    average_pressure = np.mean(
        foot_pressure,
        axis=1,
    )

    active_sensor_count = np.sum(
        foot_pressure
        >= SENSOR_ACTIVE_THRESHOLD_KPA,
        axis=1,
    )

    return {
        "total_pressure": total_pressure,
        "average_pressure": average_pressure,
        "active_sensor_count":
            active_sensor_count,
    }


# ============================================================
# ADAPTIVE CONTACT THRESHOLDS
# ============================================================

def calculate_contact_thresholds(
    total_pressure
):
    """
    Estimate thresholds independently for
    each foot/trial.

    Percentiles are used rather than absolute values because
    loading can differ considerably with:
        - patient
        - BWS level
        - affected side
        - walking speed
    """

    low = float(
        np.percentile(
            total_pressure,
            5,
        )
    )

    high = float(
        np.percentile(
            total_pressure,
            95,
        )
    )

    dynamic_range = (
        high - low
    )

    if dynamic_range <= 0:
        raise ValueError(
            "No meaningful loading variation "
            "found in this foot."
        )

    on_threshold = (
        low
        + ON_FRACTION
        * dynamic_range
    )

    off_threshold = (
        low
        + OFF_FRACTION
        * dynamic_range
    )

    return {
        "baseline": low,
        "high": high,
        "on": on_threshold,
        "off": off_threshold,
    }


# ============================================================
# HYSTERESIS CONTACT STATE
# ============================================================

def build_contact_mask(
    total_pressure,
    thresholds,
):
    """
    Whole-foot contact detection using summed plantar pressure.

    False = swing / unloaded
    True  = stance / loaded

    Hysteresis:
        - enter contact above ON threshold
        - leave contact below OFF threshold
    """

    n = len(total_pressure)

    contact = np.zeros(
        n,
        dtype=bool,
    )

    state = False

    for i in range(n):

        if not state:
            # Swing -> stance
            if total_pressure[i] >= thresholds["on"]:
                state = True

        else:
            # Stance -> swing
            if total_pressure[i] <= thresholds["off"]:
                state = False

        contact[i] = state

    return contact

# ============================================================
# BINARY SEGMENT UTILITIES
# ============================================================

def find_segments(mask):
    """
    Return:
        [(start, end, state), ...]

    end is inclusive.
    """

    if len(mask) == 0:
        return []

    transitions = np.where(
        np.diff(
            mask.astype(int)
        ) != 0
    )[0]

    starts = np.concatenate(
        (
            [0],
            transitions + 1,
        )
    )

    ends = np.concatenate(
        (
            transitions,
            [len(mask) - 1],
        )
    )

    return [
        (
            int(start),
            int(end),
            bool(mask[start]),
        )
        for start, end
        in zip(starts, ends)
    ]


# ============================================================
# REMOVE VERY SHORT CONTACT / SWING SEGMENTS
# ============================================================

def clean_contact_mask(
    contact_mask,
    fs,
):
    """
    Removes only extremely brief states.

    This is temporal debounce, NOT exclusion of
    slow pathological gait.
    """

    cleaned = (
        contact_mask
        .copy()
    )

    min_contact_samples = max(
        1,
        int(
            round(
                MIN_CONTACT_DURATION_S
                * fs
            )
        ),
    )

    min_swing_samples = max(
        1,
        int(
            round(
                MIN_SWING_DURATION_S
                * fs
            )
        ),
    )

    # --------------------------------------------------------
    # Repeat twice because fixing one short segment can create
    # a new merge with neighboring segments.
    # --------------------------------------------------------

    for _ in range(2):

        segments = find_segments(
            cleaned
        )

        for (
            start,
            end,
            state,
        ) in segments:

            length = (
                end - start + 1
            )

            if state:
                # Very short false stance
                if (
                    length
                    < min_contact_samples
                ):
                    cleaned[
                        start:end + 1
                    ] = False

            else:
                # Very short false swing
                if (
                    length
                    < min_swing_samples
                ):
                    cleaned[
                        start:end + 1
                    ] = True

    return cleaned


# ============================================================
# EXTRACT INITIAL CONTACT AND TOE OFF
# ============================================================

def extract_events(
    contact_mask
):
    """
    False -> True = Initial Contact (IC)
    True -> False = Toe Off (TO)

    Returns sample indices.
    """

    transitions = np.diff(
        contact_mask.astype(int)
    )

    initial_contacts = (
        np.where(
            transitions == 1
        )[0]
        + 1
    )

    toe_offs = (
        np.where(
            transitions == -1
        )[0]
        + 1
    )

    # If recording begins in stance, do not invent an IC.
    # That stance is incomplete and validation can ignore it.

    return (
        initial_contacts.astype(int),
        toe_offs.astype(int),
    )


# ============================================================
# PAIR IC -> TO INTO STANCES
# ============================================================

def build_stances(
    initial_contacts,
    toe_offs,
):
    """
    For every IC, find the first TO occurring afterward
    but before the next IC.

    Returns:
        Nx2 array:
            [IC_index, TO_index]
    """

    stances = []

    for i, ic in enumerate(
        initial_contacts
    ):

        if (
            i + 1
            < len(initial_contacts)
        ):
            next_ic = (
                initial_contacts[
                    i + 1
                ]
            )
        else:
            next_ic = np.inf

        valid_to = toe_offs[
            (toe_offs > ic)
            & (toe_offs < next_ic)
        ]

        if len(valid_to) == 0:
            continue

        to = int(
            valid_to[0]
        )

        stances.append(
            (
                int(ic),
                to,
            )
        )

    if not stances:
        return np.empty(
            (0, 2),
            dtype=int,
        )

    return np.asarray(
        stances,
        dtype=int,
    )


# ============================================================
# DETECT EVENTS FOR ONE FOOT
# ============================================================

def detect_foot_events(
    foot_pressure,
    time,
    fs,
):
    signals = (
        calculate_foot_signals(
            foot_pressure
        )
    )

    thresholds = (
        calculate_contact_thresholds(
            signals[
                "total_pressure"
            ]
        )
    )

    raw_contact = build_contact_mask(
    signals["total_pressure"],
    thresholds,
    )

    clean_contact = (
        clean_contact_mask(
            raw_contact,
            fs,
        )
    )

    (
        ic_indices,
        to_indices,
    ) = extract_events(
        clean_contact
    )

    stances = build_stances(
        ic_indices,
        to_indices,
    )

    ic_times = time[
        ic_indices
    ]

    to_times = time[
        to_indices
    ]

    if len(stances):
        stance_times = np.column_stack(
            (
                time[
                    stances[:, 0]
                ],
                time[
                    stances[:, 1]
                ],
            )
        )

    else:
        stance_times = np.empty(
            (0, 2),
            dtype=float,
        )

    return {
        "total_pressure":
            signals["total_pressure"],

        "average_pressure":
            signals["average_pressure"],

        "active_sensor_count":
            signals[
                "active_sensor_count"
            ],

        "raw_contact_mask":
            raw_contact,

        "contact_mask":
            clean_contact,

        "ic_indices":
            ic_indices,

        "to_indices":
            to_indices,

        "ic_times":
            ic_times,

        "to_times":
            to_times,

        "stances_indices":
            stances,

        "stances_times":
            stance_times,

        "threshold_baseline":
            thresholds["baseline"],

        "threshold_high":
            thresholds["high"],

        "threshold_on":
            thresholds["on"],

        "threshold_off":
            thresholds["off"],
    }


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(
    path
):
    parsed = parse_trial_name(
        path
    )

    if parsed is None:
        raise ValueError(
            f"Unrecognized trial name: "
            f"{path.name}"
        )

    patient, condition = parsed

    (
        time,
        pressure,
        fs,
    ) = load_filtered_npz(
        path
    )

    left_pressure = (
        pressure[:, :64]
    )

    right_pressure = (
        pressure[:, 64:]
    )

    left = detect_foot_events(
        left_pressure,
        time,
        fs,
    )

    right = detect_foot_events(
        right_pressure,
        time,
        fs,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    trial_name = (
        canonical_trial_stem(
            path
        )
    )

    output_path = (
        OUTPUT_FOLDER
        / f"{trial_name}_events.npz"
    )

    condition_string = str(
        condition
    )

    np.savez_compressed(
        output_path,

        # -----------------------------------------
        # General
        # -----------------------------------------
        source_file=path.name,
        patient=patient,
        condition=condition_string,
        sampling_rate=fs,
        time=time,

        # -----------------------------------------
        # LEFT
        # -----------------------------------------
        left_total_pressure=
            left["total_pressure"],

        left_average_pressure=
            left["average_pressure"],

        left_active_sensor_count=
            left["active_sensor_count"],

        left_raw_contact_mask=
            left["raw_contact_mask"],

        left_contact_mask=
            left["contact_mask"],

        left_ic_indices=
            left["ic_indices"],

        left_to_indices=
            left["to_indices"],

        left_ic_times=
            left["ic_times"],

        left_to_times=
            left["to_times"],

        left_stances_indices=
            left["stances_indices"],

        left_stances_times=
            left["stances_times"],

        left_threshold_baseline=
            left["threshold_baseline"],

        left_threshold_high=
            left["threshold_high"],

        left_threshold_on=
            left["threshold_on"],

        left_threshold_off=
            left["threshold_off"],

        # -----------------------------------------
        # RIGHT
        # -----------------------------------------
        right_total_pressure=
            right["total_pressure"],

        right_average_pressure=
            right["average_pressure"],

        right_active_sensor_count=
            right["active_sensor_count"],

        right_raw_contact_mask=
            right["raw_contact_mask"],

        right_contact_mask=
            right["contact_mask"],

        right_ic_indices=
            right["ic_indices"],

        right_to_indices=
            right["to_indices"],

        right_ic_times=
            right["ic_times"],

        right_to_times=
            right["to_times"],

        right_stances_indices=
            right["stances_indices"],

        right_stances_times=
            right["stances_times"],

        right_threshold_baseline=
            right["threshold_baseline"],

        right_threshold_high=
            right["threshold_high"],

        right_threshold_on=
            right["threshold_on"],

        right_threshold_off=
            right["threshold_off"],

        # -----------------------------------------
        # Algorithm parameters
        # -----------------------------------------
        sensor_active_threshold_kpa=
            SENSOR_ACTIVE_THRESHOLD_KPA,

        min_active_sensors_on=
            MIN_ACTIVE_SENSORS_ON,

        min_active_sensors_off=
            MIN_ACTIVE_SENSORS_OFF,

        on_fraction=
            ON_FRACTION,

        off_fraction=
            OFF_FRACTION,

        min_contact_duration_s=
            MIN_CONTACT_DURATION_S,

        min_swing_duration_s=
            MIN_SWING_DURATION_S,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    if condition == "ground":
        label = "Ground"
    else:
        label = (
            f"{condition}% BWS"
        )

    print(
        f"{patient.upper():>5} | "
        f"{label:<9} | "
        f"L: IC={len(left['ic_indices']):3d}, "
        f"TO={len(left['to_indices']):3d}, "
        f"stances={len(left['stances_indices']):3d} | "
        f"R: IC={len(right['ic_indices']):3d}, "
        f"TO={len(right['to_indices']):3d}, "
        f"stances={len(right['stances_indices']):3d}"
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        INPUT_FOLDER.glob(
            "*_filtered.npz"
        )
    )

    if not files:
        raise RuntimeError(
            f"No filtered NPZ files found in:\n"
            f"{INPUT_FOLDER}"
        )

    print(
        f"Found {len(files)} filtered trials.\n"
    )

    print(
        "Patient | Condition | "
        "Left events | Right events"
    )

    print(
        "-" * 90
    )

    successful = 0
    failed = 0

    for path in files:

        try:

            process_trial(
                path
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"\nERROR: {path.name}"
            )

            print(
                f"  {exc}"
            )

    print(
        "\n===================================="
    )

    print(
        "EVENT DETECTION COMPLETE"
    )

    print(
        "===================================="
    )

    print(
        f"Successful : {successful}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"\nEvent files:\n"
        f"{OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()