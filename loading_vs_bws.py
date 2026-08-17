#!/usr/bin/env python3

from pathlib import Path
import csv
import re
import struct

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Change this to the batch you want to analyse.
INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/filtered_first_batch"
)

# Folder containing raw .insoleX files.
# Used only if a filtered NPZ is not available.
RAW_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/first_batch"
)

DOCUMENTS_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/documents" 
)

PATIENT_CSV = DOCUMENTS_FOLDER / "patient_information_summary.csv"

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/loading_analysis"
)

TRIAL_PLOTS_FOLDER = OUTPUT_FOLDER / "trial_loading"
PATIENT_PLOTS_FOLDER = OUTPUT_FOLDER / "patient_bws_comparison"

SUMMARY_CSV = OUTPUT_FOLDER / "loading_summary_first.csv"


SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128
DATA_START = 72


# Sensor dimensions taken from sensorsM.ts and sensorsS.ts.
SENSOR_SIDE_MM = {
    "M": 8.0,
    "S": 7.35,
}

SENSOR_AREA_MM2 = {
    size: side_mm ** 2
    for size, side_mm in SENSOR_SIDE_MM.items()
}


# ============================================================
# PATIENT METADATA
# ============================================================

def load_patient_sizes(csv_path):
    """
    Returns:

        {
            "ab": "M",
            "bo": "S",
            ...
        }

    Patients without a recorded valid size are skipped.
    """

    patient_sizes = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            patient = row.get("Patient", "").strip().lower()
            size = row.get("Insole size", "").strip().upper()

            if not patient:
                continue

            if size in SENSOR_AREA_MM2:
                patient_sizes[patient] = size
            else:
                print(
                    f"WARNING: no valid insole size for patient "
                    f"{patient!r}"
                )

    return patient_sizes


# ============================================================
# FILE NAME PARSING
# ============================================================

def strip_processing_suffixes(stem):
    """
    Converts:

        ab_20_bws_filtered
        ab_20_bws_cleaned
        ab_20_bws

    all to:

        ab_20_bws
    """

    suffixes = (
        "_filtered",
        "_cleaned",
        "_cropped",
        "_baseline_corrected",
    )

    changed = True

    while changed:
        changed = False

        for suffix in suffixes:
            if stem.lower().endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True

    return stem


def parse_trial_name(path):
    """
    Supports names such as:

        ab_0_bws.npz
        ab_10_bws_filtered.npz
        yer_30_bws.insoleX

    Returns:

        patient, bws_percentage

    Ground trials return bws=None.
    """

    stem = strip_processing_suffixes(path.stem.lower())

    match = re.match(
        r"^([a-z]+)_(\d+)_bws$",
        stem,
    )

    if match:
        patient = match.group(1)
        bws = int(match.group(2))
        return patient, bws

    match = re.match(
        r"^([a-z]+)_ground$",
        stem,
    )

    if match:
        patient = match.group(1)
        return patient, None

    raise ValueError(
        f"Cannot parse trial name: {path.name}"
    )


def canonical_trial_id(path):
    """
    Creates a common identifier independent of file type.

    Example:

        ab_25_bws_filtered.npz
        ab_25_bws.insoleX

    -> ab_25_bws
    """

    return strip_processing_suffixes(
        path.stem.lower()
    )


# ============================================================
# RAW INSOLEX READER
# ============================================================

def read_insolex(path):
    """
    Returns:
        time
        left_sensors   shape = (samples, 64)
        right_sensors  shape = (samples, 64)
        sampling_rate
    """

    with open(path, "rb") as f:
        raw = f.read()

    date_length = raw[2]
    metadata_offset = 3 + date_length + 1

    sampling_rate = struct.unpack_from(
        "<f",
        raw,
        metadata_offset,
    )[0]

    n_samples = struct.unpack_from(
        "<I",
        raw,
        metadata_offset + 4,
    )[0]

    expected_size = (
        DATA_START
        + n_samples * TOTAL_SENSORS * 4
    )

    if len(raw) != expected_size:
        raise ValueError(
            f"{path.name}: unexpected file size. "
            f"Expected {expected_size}, got {len(raw)}"
        )

    pressure = np.frombuffer(
        raw,
        dtype="<f4",
        count=n_samples * TOTAL_SENSORS,
        offset=DATA_START,
    )

    pressure = pressure.reshape(
        n_samples,
        TOTAL_SENSORS,
    ).astype(float)

    left = pressure[:, :64]
    right = pressure[:, 64:]

    time = (
        np.arange(n_samples, dtype=float)
        / sampling_rate
    )

    return (
        time,
        left,
        right,
        float(sampling_rate),
    )


# ============================================================
# NPZ READER
# ============================================================

def first_existing_key(data, candidates):
    for key in candidates:
        if key in data:
            return key

    return None


def read_npz(path):
    with np.load(path, allow_pickle=True) as data:

        if "pressure" not in data:
            raise ValueError(
                f"{path.name}: missing 'pressure'. "
                f"Available keys: {list(data.keys())}"
            )

        pressure = np.asarray(
            data["pressure"],
            dtype=float,
        )

        # -----------------------------------------
        # Read sensor configuration
        # -----------------------------------------

        sensors_per_insole = int(
            np.asarray(
                data.get(
                    "sensors_per_insole",
                    SENSORS_PER_INSOLE,
                )
            ).reshape(-1)[0]
        )

        total_sensors = int(
            np.asarray(
                data.get(
                    "total_sensors",
                    TOTAL_SENSORS,
                )
            ).reshape(-1)[0]
        )

        if pressure.ndim != 2:
            raise ValueError(
                f"{path.name}: pressure must be 2-D, "
                f"got shape {pressure.shape}"
            )

        # Expected orientation:
        # samples × sensors
        if pressure.shape[1] != total_sensors:

            if pressure.shape[0] == total_sensors:
                pressure = pressure.T

            else:
                raise ValueError(
                    f"{path.name}: expected "
                    f"{total_sensors} sensor columns, "
                    f"got shape {pressure.shape}"
                )

        if total_sensors != 2 * sensors_per_insole:
            raise ValueError(
                f"{path.name}: inconsistent sensor metadata: "
                f"{sensors_per_insole} per insole but "
                f"{total_sensors} total."
            )

        # -----------------------------------------
        # Split left and right insoles
        # -----------------------------------------

        left = pressure[
            :, :sensors_per_insole
        ]

        right = pressure[
            :, sensors_per_insole:
        ]

        # -----------------------------------------
        # Sampling rate
        # -----------------------------------------

        if "sampling_rate" in data:
            sampling_rate = float(
                np.asarray(
                    data["sampling_rate"]
                ).reshape(-1)[0]
            )
        else:
            sampling_rate = None

        # -----------------------------------------
        # Time
        # -----------------------------------------

        if "time" in data:

            time = np.asarray(
                data["time"],
                dtype=float,
            ).reshape(-1)

            if len(time) != len(pressure):
                raise ValueError(
                    f"{path.name}: time length "
                    f"{len(time)} != pressure length "
                    f"{len(pressure)}"
                )

        else:

            if sampling_rate is None:
                raise ValueError(
                    f"{path.name}: neither time nor "
                    f"sampling_rate available."
                )

            time = (
                np.arange(len(pressure))
                / sampling_rate
            )

        if sampling_rate is None:

            if len(time) < 2:
                raise ValueError(
                    f"{path.name}: cannot infer sampling rate."
                )

            sampling_rate = float(
                1.0 / np.median(np.diff(time))
            )

    return (
        time,
        left,
        right,
        sampling_rate,
    )


# ============================================================
# SELECT BEST FILE FOR EACH TRIAL
# ============================================================

def discover_trials():
    """
    For each trial:

        filtered NPZ > other NPZ > raw .insoleX

    This lets you run the same script on batches in which
    some trials are filtered and some are still raw.
    """

    candidates = {}

    # --------------------------------------------------------
    # NPZ files
    # --------------------------------------------------------

    if INPUT_FOLDER.exists():

        for path in INPUT_FOLDER.glob("*.npz"):
            trial_id = canonical_trial_id(path)

            candidates.setdefault(
                trial_id,
                [],
            ).append(path)

    # --------------------------------------------------------
    # Raw files
    # --------------------------------------------------------

    if RAW_FOLDER.exists():

        for path in RAW_FOLDER.glob("*.insoleX"):
            trial_id = canonical_trial_id(path)

            candidates.setdefault(
                trial_id,
                [],
            ).append(path)

    selected = []

    for trial_id, paths in candidates.items():

        npz_paths = [
            p for p in paths
            if p.suffix.lower() == ".npz"
        ]

        if npz_paths:

            def npz_priority(p):
                stem = p.stem.lower()

                if stem.endswith("_filtered"):
                    return (0, p.name.lower())

                if stem.endswith(
                    ("_cleaned", "_cropped")
                ):
                    return (1, p.name.lower())

                return (2, p.name.lower())

            chosen = sorted(
                npz_paths,
                key=npz_priority,
            )[0]

        else:

            chosen = sorted(
                paths,
                key=lambda p: p.name.lower(),
            )[0]

        selected.append(chosen)

    return sorted(
        selected,
        key=lambda p: p.name.lower(),
    )


# ============================================================
# LOADING CALCULATION
# ============================================================

def calculate_loading(
    left_sensors,
    right_sensors,
    sensor_area_mm2,
):
    """
    Pressure sum:
        kPa

    Estimated plantar force:
        N

    Conversion:

        kPa * mm²
        = 1000 N/m² * 1e-6 m²
        = 0.001 N
    """

    # Sum pressure across all 64 sensors.
    left_pressure_sum = np.sum(
        left_sensors,
        axis=1,
    )

    right_pressure_sum = np.sum(
        right_sensors,
        axis=1,
    )

    bilateral_pressure_sum = (
        left_pressure_sum
        + right_pressure_sum
    )

    # --------------------------------------------------------
    # Estimated force
    # --------------------------------------------------------

    conversion = (
        sensor_area_mm2
        * 0.001
    )

    left_force = (
        left_pressure_sum
        * conversion
    )

    right_force = (
        right_pressure_sum
        * conversion
    )

    bilateral_force = (
        left_force
        + right_force
    )

    return {
        "left_pressure_sum": left_pressure_sum,
        "right_pressure_sum": right_pressure_sum,
        "bilateral_pressure_sum": bilateral_pressure_sum,
        "left_force": left_force,
        "right_force": right_force,
        "bilateral_force": bilateral_force,
    }


# ============================================================
# PER-TRIAL PLOT
# ============================================================

def plot_trial_loading(
    time,
    loading,
    patient,
    bws,
    insole_size,
    source_path,
):
    fig, ax = plt.subplots(
        figsize=(15, 6),
        constrained_layout=True,
    )

    ax.plot(
        time,
        loading["left_force"],
        label="Left",
        linewidth=1,
    )

    ax.plot(
        time,
        loading["right_force"],
        label="Right",
        linewidth=1,
    )

    ax.plot(
        time,
        loading["bilateral_force"],
        label="Bilateral",
        linewidth=1.2,
    )

    if bws is None:
        condition = "Ground"
    else:
        condition = f"{bws}% BWS"

    ax.set_title(
        f"{patient.upper()} — {condition} — "
        f"Insole {insole_size}\n"
        f"{source_path.name}"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(
        "Estimated plantar force (N)"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    output_path = (
        TRIAL_PLOTS_FOLDER
        / f"{canonical_trial_id(source_path)}_loading.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# TRIAL SUMMARY
# ============================================================

def summarize_trial(
    source_path,
    patient,
    bws,
    insole_size,
    sampling_rate,
    time,
    loading,
):
    """
    These are whole-recording descriptive values.

    They deliberately include the entire recording.
    They are NOT stance-normalized gait outcomes.
    """

    left = loading["left_force"]
    right = loading["right_force"]
    bilateral = loading["bilateral_force"]

    return {
        "patient": patient,
        "condition": (
            "ground"
            if bws is None
            else f"{bws}_bws"
        ),
        "bws_percent": (
            ""
            if bws is None
            else bws
        ),
        "insole_size": insole_size,
        "sensor_area_mm2": SENSOR_AREA_MM2[
            insole_size
        ],
        "source_file": source_path.name,
        "source_type": source_path.suffix,
        "sampling_rate_hz": sampling_rate,
        "duration_s": (
            time[-1] - time[0]
            if len(time) > 1
            else 0
        ),

        "left_force_mean_n": np.mean(left),
        "left_force_median_n": np.median(left),
        "left_force_max_n": np.max(left),

        "right_force_mean_n": np.mean(right),
        "right_force_median_n": np.median(right),
        "right_force_max_n": np.max(right),

        "bilateral_force_mean_n": np.mean(
            bilateral
        ),
        "bilateral_force_median_n": np.median(
            bilateral
        ),
        "bilateral_force_max_n": np.max(
            bilateral
        ),
    }


# ============================================================
# PATIENT-LEVEL BWS COMPARISON
# ============================================================

def plot_patient_bws(rows):
    """
    Generates one graph per patient:

        BWS %
            vs
        whole-recording mean estimated plantar force.

    Ground trials are intentionally excluded because they are
    a different walking condition.
    """

    patients = sorted(
        set(row["patient"] for row in rows)
    )

    for patient in patients:

        patient_rows = [
            row
            for row in rows
            if (
                row["patient"] == patient
                and row["bws_percent"] != ""
            )
        ]

        if not patient_rows:
            continue

        patient_rows.sort(
            key=lambda row: int(
                row["bws_percent"]
            )
        )

        bws = np.array(
            [
                int(row["bws_percent"])
                for row in patient_rows
            ]
        )

        left = np.array(
            [
                float(row["left_force_mean_n"])
                for row in patient_rows
            ]
        )

        right = np.array(
            [
                float(row["right_force_mean_n"])
                for row in patient_rows
            ]
        )

        bilateral = np.array(
            [
                float(
                    row[
                        "bilateral_force_mean_n"
                    ]
                )
                for row in patient_rows
            ]
        )

        insole_size = patient_rows[0][
            "insole_size"
        ]

        fig, ax = plt.subplots(
            figsize=(9, 6),
            constrained_layout=True,
        )

        ax.plot(
            bws,
            left,
            marker="o",
            label="Left",
        )

        ax.plot(
            bws,
            right,
            marker="o",
            label="Right",
        )

        ax.plot(
            bws,
            bilateral,
            marker="o",
            linewidth=2,
            label="Bilateral",
        )

        ax.set_title(
            f"{patient.upper()} — "
            f"Whole-recording plantar loading\n"
            f"Insole size {insole_size}"
        )

        ax.set_xlabel("BWS level (%)")
        ax.set_ylabel(
            "Mean estimated plantar force (N)"
        )

        ax.set_xticks(bws)
        ax.grid(alpha=0.3)
        ax.legend()

        output_path = (
            PATIENT_PLOTS_FOLDER
            / f"{patient}_loading_vs_bws.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# OPTIONAL: NORMALIZED BWS GRAPH
# ============================================================

def plot_patient_normalized_bws(rows):
    """
    Same comparison, but normalized to each patient's 0% BWS.

    0% BWS = 100%.

    This is particularly useful because absolute force depends
    strongly on body mass and insole size.
    """

    patients = sorted(
        set(row["patient"] for row in rows)
    )

    for patient in patients:

        patient_rows = [
            row
            for row in rows
            if (
                row["patient"] == patient
                and row["bws_percent"] != ""
            )
        ]

        if not patient_rows:
            continue

        zero_rows = [
            row
            for row in patient_rows
            if int(row["bws_percent"]) == 0
        ]

        if not zero_rows:
            print(
                f"WARNING: {patient}: no 0% BWS "
                f"trial, skipping normalized plot."
            )
            continue

        baseline = float(
            zero_rows[0][
                "bilateral_force_mean_n"
            ]
        )

        if baseline <= 0:
            continue

        patient_rows.sort(
            key=lambda row: int(
                row["bws_percent"]
            )
        )

        bws = np.array(
            [
                int(row["bws_percent"])
                for row in patient_rows
            ]
        )

        bilateral = np.array(
            [
                float(
                    row[
                        "bilateral_force_mean_n"
                    ]
                )
                for row in patient_rows
            ]
        )

        percentage_of_baseline = (
            bilateral
            / baseline
            * 100.0
        )

        expected_remaining_load = (
            100.0 - bws
        )

        fig, ax = plt.subplots(
            figsize=(9, 6),
            constrained_layout=True,
        )

        ax.plot(
            bws,
            percentage_of_baseline,
            marker="o",
            label="Measured plantar loading",
        )

        ax.plot(
            bws,
            expected_remaining_load,
            marker="o",
            linestyle="--",
            label="Nominal remaining body weight",
        )

        ax.axhline(
            100,
            linewidth=1,
            linestyle=":",
        )

        ax.set_title(
            f"{patient.upper()} — "
            f"Plantar loading relative to 0% BWS"
        )

        ax.set_xlabel("BWS level (%)")
        ax.set_ylabel(
            "Loading relative to 0% BWS (%)"
        )

        ax.set_xticks(bws)
        ax.grid(alpha=0.3)
        ax.legend()

        output_path = (
            PATIENT_PLOTS_FOLDER
            / f"{patient}_loading_vs_bws_normalized.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# CSV OUTPUT
# ============================================================

def save_summary(rows):
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

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

    TRIAL_PLOTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    PATIENT_PLOTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Patient metadata
    # --------------------------------------------------------

    patient_sizes = load_patient_sizes(
        PATIENT_CSV
    )

    print("\nPatient insole sizes:")

    for patient, size in sorted(
        patient_sizes.items()
    ):
        print(
            f"  {patient}: {size} "
            f"({SENSOR_AREA_MM2[size]:.2f} mm²/sensor)"
        )

    # --------------------------------------------------------
    # Discover trials
    # --------------------------------------------------------

    files = discover_trials()

    if not files:
        raise RuntimeError(
            "No usable NPZ or .insoleX files found."
        )

    print(
        f"\nFound {len(files)} unique trials.\n"
    )

    rows = []

    successful = 0
    skipped = 0
    failed = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for path in files:

        print(
            f"Processing: {path.name}"
        )

        try:
            patient, bws = parse_trial_name(
                path
            )

            # ----------------------------------------------
            # Need insole size to estimate force.
            # ----------------------------------------------

            if patient not in patient_sizes:

                print(
                    f"  SKIP: no valid insole size "
                    f"for patient {patient}"
                )

                skipped += 1
                continue

            insole_size = patient_sizes[
                patient
            ]

            sensor_area_mm2 = SENSOR_AREA_MM2[
                insole_size
            ]

            # ----------------------------------------------
            # Read preferred source
            # ----------------------------------------------

            if path.suffix.lower() == ".npz":

                (
                    time,
                    left_sensors,
                    right_sensors,
                    sampling_rate,
                ) = read_npz(path)

                print("  source: NPZ")

            elif path.suffix.lower() == ".insolex":

                (
                    time,
                    left_sensors,
                    right_sensors,
                    sampling_rate,
                ) = read_insolex(path)

                print("  source: raw insoleX")

            else:
                raise ValueError(
                    f"Unsupported file: {path}"
                )

            # ----------------------------------------------
            # Loading
            # ----------------------------------------------

            loading = calculate_loading(
                left_sensors,
                right_sensors,
                sensor_area_mm2,
            )

            # ----------------------------------------------
            # Plot
            # ----------------------------------------------

            plot_trial_loading(
                time=time,
                loading=loading,
                patient=patient,
                bws=bws,
                insole_size=insole_size,
                source_path=path,
            )

            # ----------------------------------------------
            # Summary
            # ----------------------------------------------

            row = summarize_trial(
                source_path=path,
                patient=patient,
                bws=bws,
                insole_size=insole_size,
                sampling_rate=sampling_rate,
                time=time,
                loading=loading,
            )

            rows.append(row)

            print(
                f"  mean force: "
                f"L={row['left_force_mean_n']:.1f} N, "
                f"R={row['right_force_mean_n']:.1f} N, "
                f"bilateral="
                f"{row['bilateral_force_mean_n']:.1f} N"
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"  ERROR: {exc}"
            )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    save_summary(rows)

    plot_patient_bws(rows)

    plot_patient_normalized_bws(rows)

    print(
        "\n===================================="
    )
    print("Finished")
    print(
        "===================================="
    )

    print(f"Successful : {successful}")
    print(f"Skipped    : {skipped}")
    print(f"Failed     : {failed}")

    print(
        f"\nSummary:\n{SUMMARY_CSV}"
    )

    print(
        f"\nTrial plots:\n"
        f"{TRIAL_PLOTS_FOLDER}"
    )

    print(
        f"\nPatient BWS plots:\n"
        f"{PATIENT_PLOTS_FOLDER}"
    )


if __name__ == "__main__":
    main()