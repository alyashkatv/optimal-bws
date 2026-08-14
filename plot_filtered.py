from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


INPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/filtered_data/filtered_alya_test"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/pressure_graphs_filtered/alya_test"
)

SUMMARY_CSV = OUTPUT_FOLDER / "pressure_summary_filtered.csv"

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128


def read_filtered_npz(path):
    data = np.load(
        path,
        allow_pickle=False
    )

    if "pressure" not in data:
        raise ValueError(
            f"{path.name}: missing 'pressure'"
        )

    if "time" not in data:
        raise ValueError(
            f"{path.name}: missing 'time'"
        )

    if "sampling_rate" not in data:
        raise ValueError(
            f"{path.name}: missing 'sampling_rate'"
        )

    pressure = np.asarray(
        data["pressure"],
        dtype=float
    )

    time = np.asarray(
        data["time"],
        dtype=float
    )

    sampling_rate = float(
        data["sampling_rate"]
    )

    if pressure.ndim != 2:
        raise ValueError(
            f"{path.name}: pressure must be 2-D"
        )

    if pressure.shape[1] != TOTAL_SENSORS:
        raise ValueError(
            f"{path.name}: expected "
            f"{TOTAL_SENSORS} sensors, "
            f"got {pressure.shape[1]}"
        )

    if len(time) != len(pressure):
        raise ValueError(
            f"{path.name}: time and pressure lengths do not match"
        )

    recording_datetime = ""

    if "recording_datetime" in data:
        value = data["recording_datetime"]

        if np.ndim(value) == 0:
            recording_datetime = str(
                value.item()
            )

    source_file = ""

    if "source_file" in data:
        value = data["source_file"]

        if np.ndim(value) == 0:
            source_file = str(
                value.item()
            )

    source_type = ""

    if "source_type" in data:
        value = data["source_type"]

        if np.ndim(value) == 0:
            source_type = str(
                value.item()
            )

    # Split 128 sensors into left/right
    left_sensors = pressure[
        :,
        :SENSORS_PER_INSOLE
    ]

    right_sensors = pressure[
        :,
        SENSORS_PER_INSOLE:
    ]

    # Whole-foot average pressure at every time sample
    left_pressure = np.mean(
        left_sensors,
        axis=1
    )

    right_pressure = np.mean(
        right_sensors,
        axis=1
    )

    return (
        time,
        left_pressure,
        right_pressure,
        recording_datetime,
        sampling_rate,
        len(time),
        source_file,
        source_type
    )


def plot_recording(file_path):

    (
        time,
        left,
        right,
        recording_datetime,
        sampling_rate,
        n_samples,
        source_file,
        source_type
    ) = read_filtered_npz(
        file_path
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 6),
        constrained_layout=True
    )

    ax.plot(
        time,
        left,
        label="Left insole",
        linewidth=1
    )

    ax.plot(
        time,
        right,
        label="Right insole",
        linewidth=1
    )

    # Remove "_filtered" from displayed name
    display_name = file_path.stem.replace(
        "_filtered",
        ""
    )

    ax.set_title(
        f"{display_name} — filtered"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Average pressure (kPa)"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    output_path = (
        OUTPUT_FOLDER
        / f"{display_name}_filtered.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    row = {
        "file": file_path.name,
        "source_file": source_file,
        "source_type": source_type,
        "recording_datetime": recording_datetime,
        "sampling_rate_hz": sampling_rate,
        "n_samples": n_samples,
        "duration_s": n_samples / sampling_rate,

        "left_mean_kpa": np.mean(left),
        "left_min_kpa": np.min(left),
        "left_max_kpa": np.max(left),
        "left_range_kpa": np.ptp(left),
        "left_std_kpa": np.std(left),

        "right_mean_kpa": np.mean(right),
        "right_min_kpa": np.min(right),
        "right_max_kpa": np.max(right),
        "right_range_kpa": np.ptp(right),
        "right_std_kpa": np.std(right),
    }

    print(
        f"{display_name}: "
        f"L mean={row['left_mean_kpa']:.2f}, "
        f"L max={row['left_max_kpa']:.2f}, "
        f"R mean={row['right_mean_kpa']:.2f}, "
        f"R max={row['right_max_kpa']:.2f}"
    )

    print(
        f"  Source: {source_type} "
        f"({source_file})"
    )

    print(
        f"  Saved: {output_path}"
    )

    return row


def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
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
        f"Found {len(files)} filtered files."
    )

    summary_rows = []

    successful = 0
    failed = 0

    for file_path in files:

        try:
            row = plot_recording(
                file_path
            )

            summary_rows.append(
                row
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR processing "
                f"{file_path.name}: {e}"
            )

    # ---------------------------------------------------------
    # Save summary CSV
    # ---------------------------------------------------------

    if summary_rows:

        fieldnames = list(
            summary_rows[0].keys()
        )

        with open(
            SUMMARY_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "Finished"
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
        f"\nPlots:\n"
        f"{OUTPUT_FOLDER}"
    )

    print(
        f"\nSummary table:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()