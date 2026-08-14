from pathlib import Path
import struct
import csv

import numpy as np
import matplotlib.pyplot as plt


INPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/raw_data/second_batch"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/pressure_graphs_second_batch_raw"
)

SUMMARY_CSV = OUTPUT_FOLDER / "pressure_summary_second_batch.csv"

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128
DATA_START = 72


def read_insolex(path):
    with open(path, "rb") as f:
        raw = f.read()

    date_length = raw[2]

    recording_datetime = raw[
        3:3 + date_length
    ].decode("ascii", errors="replace")

    metadata_offset = 3 + date_length + 1

    sampling_rate = struct.unpack_from(
        "<f",
        raw,
        metadata_offset
    )[0]

    n_samples = struct.unpack_from(
        "<I",
        raw,
        metadata_offset + 4
    )[0]

    expected_size = (
        DATA_START
        + n_samples * TOTAL_SENSORS * 4
    )

    if len(raw) != expected_size:
        raise ValueError(
            f"Unexpected file size: "
            f"expected {expected_size}, got {len(raw)}"
        )

    pressure = np.frombuffer(
        raw,
        dtype="<f4",
        count=n_samples * TOTAL_SENSORS,
        offset=DATA_START
    )

    pressure = pressure.reshape(
        n_samples,
        TOTAL_SENSORS
    )

    left_sensors = pressure[:, :64]
    right_sensors = pressure[:, 64:]

    # Average across 64 sensors at each time sample
    left_pressure = np.mean(left_sensors, axis=1)
    right_pressure = np.mean(right_sensors, axis=1)

    time = np.arange(n_samples) / sampling_rate

    return (
        time,
        left_pressure,
        right_pressure,
        recording_datetime,
        sampling_rate,
        n_samples
    )


def plot_recording(file_path):

    (
        time,
        left,
        right,
        recording_datetime,
        sampling_rate,
        n_samples
    ) = read_insolex(file_path)

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

    ax.set_title(file_path.name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average pressure (kPa)")
    ax.grid(alpha=0.3)
    ax.legend()

    output_path = (
        OUTPUT_FOLDER / f"{file_path.stem}.png"
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
        f"{file_path.name}: "
        f"L mean={row['left_mean_kpa']:.2f}, "
        f"L max={row['left_max_kpa']:.2f}, "
        f"R mean={row['right_mean_kpa']:.2f}, "
        f"R max={row['right_max_kpa']:.2f}"
    )

    return row


def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        INPUT_FOLDER.glob("*.insoleX")
    )

    if not files:
        raise RuntimeError(
            f"No .insoleX files found in:\n"
            f"{INPUT_FOLDER}"
        )

    print(f"Found {len(files)} files.")

    summary_rows = []

    successful = 0
    failed = 0

    for file_path in files:

        try:
            row = plot_recording(file_path)

            summary_rows.append(row)
            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR processing "
                f"{file_path.name}: {e}"
            )

    # ---------------------------------------------------------
    # Save summary table
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
            writer.writerows(summary_rows)

    print("\n====================================")
    print("Finished")
    print("====================================")

    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")

    print(f"\nPlots:")
    print(OUTPUT_FOLDER)

    print(f"\nSummary table:")
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()