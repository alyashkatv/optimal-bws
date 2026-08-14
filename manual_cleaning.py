from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt


INPUT_FILE = Path(
    "/home/alya/Desktop/test_insoles/first_batch/ab_25_bws.insoleX"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/test_first_batch/ab_25_cleaned"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128
DATA_START = 72

# ---------------------------------------------------------
# MANUAL QC SETTINGS FOR THIS RECORDING
# ---------------------------------------------------------

# Disturbed interval to completely exclude
BAD_INTERVAL_START_S = 68.0
BAD_INTERVAL_END_S = 90.0

# Isolated spike region
# SPIKE_START_S = 52.0
# SPIKE_END_S = 52.2


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
    ).copy()

    pressure = pressure.reshape(
        n_samples,
        TOTAL_SENSORS
    )

    time = np.arange(n_samples) / sampling_rate

    return (
        time,
        pressure,
        sampling_rate,
        recording_datetime
    )


def remove_right_pressure_spike(
    pressure,
    time,
    search_start_s=45.0,
    search_end_s=60.0,
    threshold_kpa=40.0,
):
    """
    Detect the extreme right-insole spike within a known search region
    and interpolate across it.

    threshold_kpa refers to whole-foot average pressure.
    """

    cleaned = pressure.copy()

    # Whole-foot right average
    right_average = np.mean(
        cleaned[:, 64:],
        axis=1
    )

    # Only search within the relevant part of this recording
    search_mask = (
        (time >= search_start_s)
        & (time <= search_end_s)
    )

    # Find samples with obviously impossible/extreme average pressure
    spike_mask = (
        search_mask
        & (right_average > threshold_kpa)
    )

    spike_indices = np.where(spike_mask)[0]

    if len(spike_indices) == 0:
        print("No right-foot spike detected.")
        return cleaned

    start_idx = spike_indices[0]
    end_idx = spike_indices[-1]

    print(
        f"Spike detected at "
        f"{time[start_idx]:.3f}–{time[end_idx]:.3f} s"
    )

    # Add a small margin around the affected samples
    margin_samples = 2

    start_idx = max(
        start_idx - margin_samples,
        1
    )

    end_idx = min(
        end_idx + margin_samples,
        len(time) - 2
    )

    # Values immediately before and after the corrupted region
    before = cleaned[start_idx - 1].copy()
    after = cleaned[end_idx + 1].copy()

    n_bad = end_idx - start_idx + 1

    # Interpolate EACH of the 128 sensor channels
    for sensor in range(cleaned.shape[1]):

        cleaned[
            start_idx:end_idx + 1,
            sensor
        ] = np.linspace(
            before[sensor],
            after[sensor],
            n_bad + 2
        )[1:-1]

    return cleaned


def remove_bad_interval(
    time,
    pressure,
    start_s,
    end_s
):
    """
    Completely remove the disturbed interval.
    """

    keep_mask = (
        (time < start_s)
        | (time > end_s)
    )

    cleaned_pressure = pressure[keep_mask]

    return cleaned_pressure


def main():

    (
        time,
        pressure,
        fs,
        recording_datetime
    ) = read_insolex(INPUT_FILE)

    print(f"File: {INPUT_FILE.name}")
    print(f"Sampling rate: {fs:.2f} Hz")
    print(f"Original duration: {time[-1]:.2f} s")

    # ---------------------------------------------------------
    # Original average pressure
    # ---------------------------------------------------------

    original_left = np.mean(
        pressure[:, :64],
        axis=1
    )

    original_right = np.mean(
        pressure[:, 64:],
        axis=1
    )

    # ---------------------------------------------------------
    # STEP 1:
    # Replace isolated spike using interpolation
    # ---------------------------------------------------------

    pressure_no_spike = remove_right_pressure_spike(
        pressure,
        time,
        search_start_s=45.0,
        search_end_s=60.0,
        threshold_kpa=40.0
    )

    # ---------------------------------------------------------
    # STEP 2:
    # Remove disturbed section
    # ---------------------------------------------------------

    cleaned_pressure = remove_bad_interval(
        time,
        pressure_no_spike,
        BAD_INTERVAL_START_S,
        BAD_INTERVAL_END_S
    )

    # Create a new continuous time vector
    cleaned_time = (
        np.arange(len(cleaned_pressure))
        / fs
    )

    cleaned_left = np.mean(
        cleaned_pressure[:, :64],
        axis=1
    )

    cleaned_right = np.mean(
        cleaned_pressure[:, 64:],
        axis=1
    )

    print(
        f"Cleaned duration: "
        f"{cleaned_time[-1]:.2f} s"
    )

    print(
        f"Removed disturbed interval: "
        f"{BAD_INTERVAL_START_S:.1f}–"
        f"{BAD_INTERVAL_END_S:.1f} s"
    )

    # print(
    #     f"Interpolated spike interval: "
    #     f"{SPIKE_START_S:.1f}–"
    #     f"{SPIKE_END_S:.1f} s"
    # )

    # ---------------------------------------------------------
    # Save cleaned sensor data
    # ---------------------------------------------------------

    npz_output = (
        OUTPUT_FOLDER
        / "ab_25_bws_cleaned.npz"
    )

    np.savez_compressed(
        npz_output,
        pressure=cleaned_pressure,
        time=cleaned_time,
        sampling_rate=fs,
        original_file=INPUT_FILE.name,
        recording_datetime=recording_datetime,
        removed_interval=np.array([
            BAD_INTERVAL_START_S,
            BAD_INTERVAL_END_S
        ]),
        # interpolated_spike=np.array([
        #     SPIKE_START_S,
        #     SPIKE_END_S
        # ])
    )

    # ---------------------------------------------------------
    # Save CSV of average pressure
    # ---------------------------------------------------------

    csv_output = (
        OUTPUT_FOLDER
        / "ab_25_bws_cleaned_average_pressure.csv"
    )

    csv_data = np.column_stack([
        cleaned_time,
        cleaned_left,
        cleaned_right
    ])

    np.savetxt(
        csv_output,
        csv_data,
        delimiter=",",
        header=(
            "time_s,"
            "left_average_pressure_kpa,"
            "right_average_pressure_kpa"
        ),
        comments=""
    )

    # ---------------------------------------------------------
    # BEFORE plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 6),
        constrained_layout=True
    )

    ax.plot(
        time,
        original_left,
        label="Left"
    )

    ax.plot(
        time,
        original_right,
        label="Right"
    )

    ax.axvspan(
        BAD_INTERVAL_START_S,
        BAD_INTERVAL_END_S,
        alpha=0.2,
        label="Excluded interval"
    )

    # ax.axvspan(
    #     SPIKE_START_S,
    #     SPIKE_END_S,
    #     alpha=0.2,
    #     label="Spike interval"
    # )

    ax.set_title(
        "AB 25% BWS — Original"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Average pressure (kPa)")
    ax.grid(alpha=0.3)
    ax.legend()

    before_output = (
        OUTPUT_FOLDER
        / "ab_25_before_cleaning.png"
    )

    plt.savefig(
        before_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # AFTER plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 6),
        constrained_layout=True
    )

    ax.plot(
        cleaned_time,
        cleaned_left,
        label="Left"
    )

    ax.plot(
        cleaned_time,
        cleaned_right,
        label="Right"
    )

    ax.set_title(
        "AB 25% BWS — Cleaned"
    )

    ax.set_xlabel(
        "Cleaned time (s)"
    )

    ax.set_ylabel(
        "Average pressure (kPa)"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    after_output = (
        OUTPUT_FOLDER
        / "ab_25_after_cleaning.png"
    )

    plt.savefig(
        after_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved:")
    print(npz_output)
    print(csv_output)
    print(before_output)
    print(after_output)


if __name__ == "__main__":
    main()