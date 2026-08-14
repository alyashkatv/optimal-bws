#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

FILTERED_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/filtered_data/filtered_second_batch"
)

DOCUMENTS_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/documents"
)

PATIENT_INFO_CSV = DOCUMENTS_FOLDER / "patient_information_summary.csv"
SENSORS_S_TS = DOCUMENTS_FOLDER / "sensorsS.ts"
SENSORS_M_TS = DOCUMENTS_FOLDER / "sensorsM.ts"

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/regional_loading_analysis"
)

SUMMARY_CSV = OUTPUT_FOLDER / "regional_loading_summary.csv"

ABSOLUTE_PLOTS_FOLDER = OUTPUT_FOLDER / "absolute"
NORMALIZED_PLOTS_FOLDER = OUTPUT_FOLDER / "normalized"


# ============================================================
# CONSTANTS
# ============================================================

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128

# Anatomical regions defined from normalized AP coordinate.
#
# 0.0 = posterior-most sensor
# 1.0 = anterior-most sensor
#
# In sensorsS.ts and sensorsM.ts, the raw Y coordinate runs in
# the opposite direction: large Y is at the heel and small Y is
# at the toes. assign_regions() reverses that axis before applying
# these limits.
#
# These are approximate geometrical regions rather than
# clinically validated anatomical masks.
REGION_LIMITS = {
    "heel": (0.00, 0.25),
    "midfoot": (0.25, 0.50),
    "forefoot": (0.50, 0.80),
    "toe": (0.80, 1.01),
}

REGION_ORDER = [
    "heel",
    "midfoot",
    "forefoot",
    "toe",
]


# ============================================================
# PATIENT METADATA
# ============================================================

def load_patient_sizes(csv_path):
    patient_sizes = {}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            patient = row.get("Patient", "").strip().lower()
            size = row.get("Insole size", "").strip().upper()

            if not patient:
                continue

            if size not in {"S", "M"}:
                print(
                    f"WARNING: no valid insole size for "
                    f"patient '{patient}'"
                )
                continue

            patient_sizes[patient] = size

    return patient_sizes


# ============================================================
# SENSOR MATRIX PARSING
# ============================================================

def extract_ts_array(text, array_name):
    """
    Extract coordinate arrays such as:

    export const LEFT_MM: [number, number][] = [
      [30.528, 19],
      [67.325, 226.09],
      ...
    ]
    """

    # Find the beginning of:
    # export const LEFT_MM ... = [
    start_pattern = rf"export\s+const\s+{re.escape(array_name)}[^=]*=\s*\["

    start_match = re.search(
        start_pattern,
        text,
        flags=re.DOTALL,
    )

    if not start_match:
        raise ValueError(
            f"Could not find array {array_name}"
        )

    # Start immediately after the opening [ of the outer array
    start = start_match.end()

    # Find the array terminator:
    # ]
    # followed by export const ... or end of file
    end_match = re.search(
        r"\]\s*(?=\n\s*export\s+const|\Z)",
        text[start:],
        flags=re.DOTALL,
    )

    if not end_match:
        raise ValueError(
            f"Could not find end of array {array_name}"
        )

    block = text[
        start:
        start + end_match.start()
    ]

    # Extract every [x, y] pair
    pairs = re.findall(
        r"\[\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*,\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*\]",
        block,
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
            f"{array_name}: expected 64 coordinates, "
            f"found {len(coords)}"
        )

    return coords

def load_sensor_layout(ts_path):
    text = ts_path.read_text(
        encoding="utf-8"
    )

    side_match = re.search(
        r"SENSOR_SIDE_MM\s*=\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
    )

    if not side_match:
        raise ValueError(
            f"Could not find SENSOR_SIDE_MM in "
            f"{ts_path.name}"
        )

    sensor_side_mm = float(
        side_match.group(1)
    )

    left_coords = extract_ts_array(
        text,
        "LEFT_MM",
    )

    right_coords = extract_ts_array(
        text,
        "RIGHT_MM",
    )

    sensor_area_mm2 = sensor_side_mm ** 2
    sensor_area_m2 = sensor_area_mm2 * 1e-6

    return {
        "left": left_coords,
        "right": right_coords,
        "sensor_side_mm": sensor_side_mm,
        "sensor_area_mm2": sensor_area_mm2,
        "sensor_area_m2": sensor_area_m2,
    }


def load_all_layouts():
    return {
        "S": load_sensor_layout(
            SENSORS_S_TS
        ),
        "M": load_sensor_layout(
            SENSORS_M_TS
        ),
    }


# ============================================================
# REGION ASSIGNMENT
# ============================================================

def assign_regions(coords):
    """
    Assign every sensor to a plantar region based on its
    normalized anterior-posterior coordinate.

    Returns:
        dict:
            heel -> boolean mask, shape (64,)
            midfoot -> ...
            forefoot -> ...
            toe -> ...
    """

    y = coords[:, 1]

    y_min = np.min(y)
    y_max = np.max(y)

    if y_max <= y_min:
        raise ValueError(
            "Invalid AP sensor coordinates."
        )

    # TypeScript layout convention:
    #     y_max = posterior/heel
    #     y_min = anterior/toes
    # Convert it to the anatomical convention used by REGION_LIMITS:
    #     0.0 = posterior/heel
    #     1.0 = anterior/toes
    y_norm = (
        (y_max - y)
        / (y_max - y_min)
    )

    masks = {}

    for region, (low, high) in REGION_LIMITS.items():
        masks[region] = (
            (y_norm >= low)
            & (y_norm < high)
        )

    assignment_count = np.sum(
        np.stack(list(masks.values())),
        axis=0,
    )

    if not np.all(assignment_count == 1):
        raise ValueError(
            "Region limits must assign every sensor to exactly one region."
        )

    return masks


# ============================================================
# NPZ READING
# ============================================================

def load_filtered_npz(path):
    data = np.load(
        path,
        allow_pickle=True,
    )

    if "pressure" not in data:
        raise ValueError(
            f"{path.name}: no 'pressure' array.\n"
            f"Available keys: {list(data.keys())}"
        )

    pressure = np.asarray(
        data["pressure"],
        dtype=float,
    )

    if pressure.ndim != 2:
        raise ValueError(
            f"{path.name}: expected 2D pressure array, "
            f"got {pressure.shape}"
        )

    # Allow either:
    #     samples x 128
    # or:
    #     128 x samples
    if pressure.shape[1] == TOTAL_SENSORS:
        pass

    elif pressure.shape[0] == TOTAL_SENSORS:
        pressure = pressure.T

    else:
        raise ValueError(
            f"{path.name}: expected 128 sensor channels, "
            f"got shape {pressure.shape}"
        )

    if "time" in data:
        time = np.asarray(
            data["time"],
            dtype=float,
        )
    else:
        sampling_rate = float(
            np.asarray(
                data["sampling_rate"]
            ).squeeze()
        )

        time = (
            np.arange(
                pressure.shape[0]
            )
            / sampling_rate
        )

    if "sampling_rate" in data:
        sampling_rate = float(
            np.asarray(
                data["sampling_rate"]
            ).squeeze()
        )
    else:
        sampling_rate = 1.0 / np.median(
            np.diff(time)
        )

    left = pressure[:, :64]
    right = pressure[:, 64:]

    return (
        time,
        left,
        right,
        sampling_rate,
    )


# ============================================================
# FILE NAME PARSING
# ============================================================

def parse_trial_name(path):
    """
    Example:
        ab_20_bws_filtered.npz

    returns:
        patient = "ab"
        bws = 20
    """

    stem = path.stem.lower()

    match = re.match(
        r"^([a-z]+)_(\d+)_bws",
        stem,
    )

    if not match:
        return None, None

    patient = match.group(1)
    bws = int(match.group(2))

    return patient, bws


# ============================================================
# FORCE CALCULATION
# ============================================================

def pressure_to_force(
    pressure_kpa,
    sensor_area_m2,
):
    """
    pressure:
        kPa

    area:
        m²

    Since:
        1 kPa = 1000 N/m²

    force = pressure * area
    """

    return (
        pressure_kpa
        * 1000.0
        * sensor_area_m2
    )


def calculate_regional_loading(
    sensors,
    region_masks,
    sensor_area_m2,
):
    """
    sensors:
        samples x 64 pressure array

    Returns the whole-recording mean force for each region.
    """

    force = pressure_to_force(
        sensors,
        sensor_area_m2,
    )

    result = {}

    for region in REGION_ORDER:
        mask = region_masks[region]

        if not np.any(mask):
            result[region] = np.nan
            continue

        # Sum sensors inside this region at every sample.
        regional_force_time = np.sum(
            force[:, mask],
            axis=1,
        )

        # Mean over the entire recording.
        result[region] = float(
            np.mean(
                regional_force_time
            )
        )

    result["total"] = float(
        np.mean(
            np.sum(force, axis=1)
        )
    )

    return result


# ============================================================
# PROCESS TRIAL
# ============================================================

def process_trial(
    path,
    patient_sizes,
    layouts,
):
    patient, bws = parse_trial_name(
        path
    )

    if patient is None:
        print(
            f"SKIP: could not parse trial "
            f"name {path.name}"
        )
        return None

    if patient not in patient_sizes:
        print(
            f"SKIP: no insole size for "
            f"{patient}"
        )
        return None

    size = patient_sizes[patient]
    layout = layouts[size]

    (
        time,
        left,
        right,
        sampling_rate,
    ) = load_filtered_npz(path)

    left_masks = assign_regions(
        layout["left"]
    )

    right_masks = assign_regions(
        layout["right"]
    )

    left_result = calculate_regional_loading(
        left,
        left_masks,
        layout["sensor_area_m2"],
    )

    right_result = calculate_regional_loading(
        right,
        right_masks,
        layout["sensor_area_m2"],
    )

    row = {
        "file": path.name,
        "patient": patient,
        "bws_percent": bws,
        "insole_size": size,
        "sampling_rate_hz": sampling_rate,
        "duration_s": (
            time[-1] - time[0]
            if len(time) > 1
            else 0
        ),
    }

    for region in (
        REGION_ORDER
        + ["total"]
    ):
        left_value = left_result[
            region
        ]

        right_value = right_result[
            region
        ]

        bilateral_value = (
            left_value
            + right_value
        )

        row[
            f"left_{region}_mean_force_n"
        ] = left_value

        row[
            f"right_{region}_mean_force_n"
        ] = right_value

        row[
            f"bilateral_{region}_mean_force_n"
        ] = bilateral_value

    print(
        f"{patient.upper()} "
        f"{bws:>2}% BWS | "
        f"L={left_result['total']:.2f} N | "
        f"R={right_result['total']:.2f} N | "
        f"bilateral="
        f"{left_result['total'] + right_result['total']:.2f} N"
    )

    return row


# ============================================================
# NORMALIZATION
# ============================================================

def add_normalized_values(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(
            row["patient"],
            [],
        ).append(row)

    for patient, patient_rows in grouped.items():

        baseline_rows = [
            r
            for r in patient_rows
            if r["bws_percent"] == 0
        ]

        if not baseline_rows:
            print(
                f"WARNING: {patient} has no "
                f"0% BWS baseline."
            )
            continue

        baseline = baseline_rows[0]

        for row in patient_rows:

            for side in (
                "left",
                "right",
                "bilateral",
            ):
                for region in (
                    REGION_ORDER
                    + ["total"]
                ):

                    key = (
                        f"{side}_{region}_"
                        f"mean_force_n"
                    )

                    norm_key = (
                        f"{side}_{region}_"
                        f"relative_to_0_percent"
                    )

                    baseline_value = baseline[
                        key
                    ]

                    current_value = row[
                        key
                    ]

                    if (
                        baseline_value is None
                        or not np.isfinite(
                            baseline_value
                        )
                        or baseline_value == 0
                    ):
                        row[norm_key] = np.nan

                    else:
                        row[norm_key] = (
                            current_value
                            / baseline_value
                            * 100.0
                        )


# ============================================================
# PLOTS
# ============================================================

def plot_absolute_patient(
    patient,
    patient_rows,
):
    rows = sorted(
        patient_rows,
        key=lambda r: r["bws_percent"],
    )

    bws = np.array(
        [
            r["bws_percent"]
            for r in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    for region in REGION_ORDER:
        values = np.array(
            [
                r[
                    f"bilateral_{region}_"
                    f"mean_force_n"
                ]
                for r in rows
            ]
        )

        ax.plot(
            bws,
            values,
            marker="o",
            label=region.capitalize(),
        )

    ax.set_title(
        f"{patient.upper()} — Regional plantar loading"
    )

    ax.set_xlabel(
        "BWS level (%)"
    )

    ax.set_ylabel(
        "Mean bilateral estimated force (N)"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    output = (
        ABSOLUTE_PLOTS_FOLDER
        / f"{patient}_regional_absolute.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_normalized_patient(
    patient,
    patient_rows,
):
    rows = sorted(
        patient_rows,
        key=lambda r: r["bws_percent"],
    )

    bws = np.array(
        [
            r["bws_percent"]
            for r in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    for region in REGION_ORDER:
        key = (
            f"bilateral_{region}_"
            f"relative_to_0_percent"
        )

        values = np.array(
            [
                r.get(
                    key,
                    np.nan,
                )
                for r in rows
            ]
        )

        ax.plot(
            bws,
            values,
            marker="o",
            label=region.capitalize(),
        )

    ax.axhline(
        100,
        linestyle=":",
        linewidth=1,
    )

    ax.set_title(
        f"{patient.upper()} — "
        f"Regional loading relative to 0% BWS"
    )

    ax.set_xlabel(
        "BWS level (%)"
    )

    ax.set_ylabel(
        "Regional loading relative to 0% BWS (%)"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    output = (
        NORMALIZED_PLOTS_FOLDER
        / f"{patient}_regional_normalized.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SAVE CSV
# ============================================================

def save_summary(rows):
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    # Some normalization fields are added after processing,
    # so collect all possible keys.
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    ABSOLUTE_PLOTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    NORMALIZED_PLOTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_sizes = load_patient_sizes(
        PATIENT_INFO_CSV
    )

    print("\nPatient insole sizes:")

    for patient in sorted(
        patient_sizes
    ):
        print(
            f"  {patient}: "
            f"{patient_sizes[patient]}"
        )

    layouts = load_all_layouts()

    print("\nSensor layouts:")

    for size, layout in layouts.items():
        print(
            f"  {size}: "
            f"{layout['sensor_area_mm2']:.2f} "
            f"mm²/sensor"
        )

    files = sorted(
        FILTERED_FOLDER.glob(
            "*_bws_filtered.npz"
        )
    )

    if not files:
        raise RuntimeError(
            f"No filtered NPZ files found in:\n"
            f"{FILTERED_FOLDER}"
        )

    print(
        f"\nFound {len(files)} files.\n"
    )

    rows = []

    successful = 0
    failed = 0

    for path in files:

        try:
            row = process_trial(
                path,
                patient_sizes,
                layouts,
            )

            if row is None:
                continue

            rows.append(row)
            successful += 1

        except Exception as e:
            failed += 1

            print(
                f"ERROR: {path.name}: "
                f"{e}"
            )

    add_normalized_values(
        rows
    )

    save_summary(
        rows
    )

    grouped = {}

    for row in rows:
        grouped.setdefault(
            row["patient"],
            [],
        ).append(row)

    for patient, patient_rows in grouped.items():

        plot_absolute_patient(
            patient,
            patient_rows,
        )

        plot_normalized_patient(
            patient,
            patient_rows,
        )

    print(
        "\n===================================="
    )

    print("Finished")

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
        f"\nSummary:\n{SUMMARY_CSV}"
    )

    print(
        f"\nAbsolute plots:\n"
        f"{ABSOLUTE_PLOTS_FOLDER}"
    )

    print(
        f"\nNormalized plots:\n"
        f"{NORMALIZED_PLOTS_FOLDER}"
    )


if __name__ == "__main__":
    main()
