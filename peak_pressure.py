#!/usr/bin/env python3

from pathlib import Path
import re
import csv

import numpy as np


# ============================================================
# CONFIG
# ============================================================

# Change this when running another batch
INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/filtered_first_batch"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/pressure_analysis"
)

OUTPUT_CSV = (
    OUTPUT_FOLDER
    / "peak_pressure_by_trial_first.csv"
)

SENSORS_PER_INSOLE = 64

# If L/R peak pressures differ by less than this percentage,
# classify them as approximately equal rather than forcing
# a dominant side.
DOMINANCE_THRESHOLD_PERCENT = 5.0


# ============================================================
# FILE NAME PARSING
# ============================================================

def parse_trial_name(path):
    """
    Expected examples:

        ab_0_bws_filtered.npz
        al_20_bws_filtered.npz
        yer_5_bws_filtered.npz
        yers_ground_filtered.npz

    Returns:
        patient
        condition
        bws_percent
    """

    stem = path.stem.lower()

    # Remove suffix
    stem = re.sub(
        r"_filtered$",
        "",
        stem,
    )

    # Ground condition
    match = re.match(
        r"^(.+?)_ground$",
        stem,
    )

    if match:
        return (
            match.group(1).upper(),
            "Ground",
            np.nan,
        )

    # BWS condition
    match = re.match(
        r"^(.+?)_(\d+)_bws$",
        stem,
    )

    if match:
        patient = (
            match.group(1).upper()
        )

        bws = int(
            match.group(2)
        )

        return (
            patient,
            f"{bws}% BWS",
            bws,
        )

    return None


# ============================================================
# LOAD PRESSURE
# ============================================================

def load_pressure(path):
    """
    Expected filtered NPZ structure:

        pressure -> shape (N, 128)

    First 64 channels  = left insole
    Last 64 channels   = right insole
    """

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        if "pressure" not in data:
            raise KeyError(
                "NPZ does not contain "
                "'pressure'."
            )

        pressure = np.asarray(
            data["pressure"],
            dtype=float,
        )

        time = (
            np.asarray(
                data["time"],
                dtype=float,
            )
            if "time" in data
            else None
        )

    if pressure.ndim != 2:
        raise ValueError(
            f"Expected 2-D pressure array, "
            f"got {pressure.shape}"
        )

    expected_channels = (
        2 * SENSORS_PER_INSOLE
    )

    if (
        pressure.shape[1]
        != expected_channels
    ):
        raise ValueError(
            f"Expected {expected_channels} "
            f"pressure channels, "
            f"got {pressure.shape[1]}"
        )

    left = pressure[
        :,
        :SENSORS_PER_INSOLE,
    ]

    right = pressure[
        :,
        SENSORS_PER_INSOLE:,
    ]

    return (
        left,
        right,
        time,
    )


# ============================================================
# PRESSURE METRICS
# ============================================================

def calculate_pressure_metrics(
    pressure,
):
    """
    Calculate peak loading metrics for one foot.

    pressure shape:
        samples x 64 sensors
    """

    # --------------------------------------------------------
    # Total pressure across the entire insole at each sample
    # --------------------------------------------------------

    total_pressure = np.nansum(
        pressure,
        axis=1,
    )

    peak_total_pressure = float(
        np.nanmax(
            total_pressure
        )
    )

    peak_total_index = int(
        np.nanargmax(
            total_pressure
        )
    )

    # --------------------------------------------------------
    # Mean pressure across all sensors at each sample
    # --------------------------------------------------------

    mean_pressure = np.nanmean(
        pressure,
        axis=1,
    )

    peak_mean_pressure = float(
        np.nanmax(
            mean_pressure
        )
    )

    # --------------------------------------------------------
    # Highest pressure recorded by ANY individual sensor
    # --------------------------------------------------------

    peak_sensor_pressure = float(
        np.nanmax(
            pressure
        )
    )

    return {
        "peak_total_pressure":
            peak_total_pressure,

        "peak_mean_pressure_kpa":
            peak_mean_pressure,

        "peak_sensor_pressure_kpa":
            peak_sensor_pressure,

        "peak_index":
            peak_total_index,
    }


# ============================================================
# DOMINANCE
# ============================================================

def determine_dominant_leg(
    left_peak,
    right_peak,
):
    """
    Determine pressure-dominant leg.

    Difference is normalized by the mean of L/R:

        |L-R| / ((L+R)/2) * 100

    If difference < threshold:
        Balanced
    """

    if (
        not np.isfinite(left_peak)
        or not np.isfinite(right_peak)
    ):
        return (
            "Unknown",
            np.nan,
        )

    mean_peak = (
        left_peak
        + right_peak
    ) / 2.0

    if mean_peak <= 0:
        return (
            "Unknown",
            np.nan,
        )

    difference_percent = (
        abs(
            left_peak
            - right_peak
        )
        / mean_peak
        * 100.0
    )

    if (
        difference_percent
        < DOMINANCE_THRESHOLD_PERCENT
    ):
        dominant = "Balanced"

    elif left_peak > right_peak:
        dominant = "Left"

    else:
        dominant = "Right"

    return (
        dominant,
        difference_percent,
    )


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(
    path,
    patient,
    condition,
    bws_percent,
):

    left, right, time = (
        load_pressure(
            path
        )
    )

    left_metrics = (
        calculate_pressure_metrics(
            left
        )
    )

    right_metrics = (
        calculate_pressure_metrics(
            right
        )
    )

    dominant_leg, (
        difference_percent
    ) = determine_dominant_leg(
        left_metrics[
            "peak_total_pressure"
        ],
        right_metrics[
            "peak_total_pressure"
        ],
    )

    # Time at which peak loading occurred
    if time is not None:

        left_peak_time = float(
            time[
                left_metrics[
                    "peak_index"
                ]
            ]
        )

        right_peak_time = float(
            time[
                right_metrics[
                    "peak_index"
                ]
            ]
        )

    else:

        left_peak_time = np.nan
        right_peak_time = np.nan

    return {
        "patient":
            patient,

        "condition":
            condition,

        "bws_percent":
            bws_percent,

        # ---------------- LEFT ----------------

        "left_peak_total_pressure":
            left_metrics[
                "peak_total_pressure"
            ],

        "left_peak_mean_pressure_kpa":
            left_metrics[
                "peak_mean_pressure_kpa"
            ],

        "left_peak_sensor_pressure_kpa":
            left_metrics[
                "peak_sensor_pressure_kpa"
            ],

        "left_peak_time_s":
            left_peak_time,

        # ---------------- RIGHT ----------------

        "right_peak_total_pressure":
            right_metrics[
                "peak_total_pressure"
            ],

        "right_peak_mean_pressure_kpa":
            right_metrics[
                "peak_mean_pressure_kpa"
            ],

        "right_peak_sensor_pressure_kpa":
            right_metrics[
                "peak_sensor_pressure_kpa"
            ],

        "right_peak_time_s":
            right_peak_time,

        # ---------------- COMPARISON ----------------

        "dominant_leg":
            dominant_leg,

        "peak_pressure_difference_percent":
            difference_percent,
    }


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

    print(
        f"Found {len(files)} "
        f"filtered trials."
    )

    rows = []

    successful = 0
    failed = 0

    for path in files:

        parsed = parse_trial_name(
            path
        )

        if parsed is None:

            print(
                f"SKIP: cannot parse "
                f"{path.name}"
            )

            continue

        (
            patient,
            condition,
            bws_percent,
        ) = parsed

        try:

            row = process_trial(
                path,
                patient,
                condition,
                bws_percent,
            )

            rows.append(
                row
            )

            successful += 1

            print(
                f"{patient:>5} | "
                f"{condition:<10} | "
                f"L={row['left_peak_mean_pressure_kpa']:7.2f} kPa | "
                f"R={row['right_peak_mean_pressure_kpa']:7.2f} kPa | "
                f"{row['dominant_leg']}"
            )

        except Exception as e:

            failed += 1

            print(
                f"ERROR {path.name}: "
                f"{e}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    if rows:

        rows = sorted(
            rows,
            key=lambda x: (
                x["patient"],
                (
                    999
                    if not np.isfinite(
                        x["bws_percent"]
                    )
                    else x[
                        "bws_percent"
                    ]
                ),
            ),
        )

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    print()
    print("=" * 60)
    print("PRESSURE ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"\nOutput:\n{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()