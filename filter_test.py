from pathlib import Path
import struct
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


INPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/first_batch"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/filter_test_individual_sensors"
)

SELECTED_FILES = [
    "bo_25_bws.insoleX",
    "en_30_bws.insoleX",
    "ab_15_bws.insoleX",
    "ab_10_bws.insoleX",
]

# We already ruled out the very low cutoffs in the whole-foot test.
CUTOFFS_HZ = [
    7,
    10,
    15,
    20,
]

FILTER_ORDER = 4

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128
DATA_START = 72

# Number of representative sensors to inspect per foot
N_SELECTED_SENSORS = 6

# Length of zoomed comparison window
ZOOM_LENGTH_S = 8.0


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
            f"{path.name}: expected {expected_size} bytes, "
            f"got {len(raw)} bytes."
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
        recording_datetime,
    )


def lowpass_filter(signal, fs, cutoff_hz, order=4):
    sos = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return sosfiltfilt(
        sos,
        signal
    )


def calculate_metrics(raw_signal, filtered_signal):
    diff = filtered_signal - raw_signal

    rmse = np.sqrt(
        np.mean(diff ** 2)
    )

    mae = np.mean(
        np.abs(diff)
    )

    raw_range = (
        np.max(raw_signal)
        - np.min(raw_signal)
    )

    if raw_range > 0:
        normalized_rmse = rmse / raw_range
    else:
        normalized_rmse = np.nan

    if np.std(raw_signal) > 0 and np.std(filtered_signal) > 0:
        correlation = np.corrcoef(
            raw_signal,
            filtered_signal
        )[0, 1]
    else:
        correlation = np.nan

    raw_peak = np.max(
        raw_signal
    )

    filtered_peak = np.max(
        filtered_signal
    )

    if raw_peak != 0:
        peak_change_percent = (
            100
            * (filtered_peak - raw_peak)
            / raw_peak
        )
    else:
        peak_change_percent = np.nan

    # Simple estimate of how much high-frequency variation was removed
    raw_diff_std = np.std(
        np.diff(raw_signal)
    )

    filtered_diff_std = np.std(
        np.diff(filtered_signal)
    )

    if raw_diff_std > 0:
        noise_reduction_percent = (
            100
            * (
                raw_diff_std
                - filtered_diff_std
            )
            / raw_diff_std
        )
    else:
        noise_reduction_percent = np.nan

    return {
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "mae": mae,
        "correlation": correlation,
        "raw_peak_kpa": raw_peak,
        "filtered_peak_kpa": filtered_peak,
        "peak_change_percent": peak_change_percent,
        "raw_diff_std": raw_diff_std,
        "filtered_diff_std": filtered_diff_std,
        "noise_reduction_percent": noise_reduction_percent,
    }


def estimate_sensor_noise(signal):
    """
    Rough relative noise estimate.

    Uses the standard deviation of the first difference.
    This is not a full PSD-based measure, but is useful for
    selecting a visibly/noisily behaving sensor.
    """

    return np.std(
        np.diff(signal)
    )


def select_representative_sensors(sensor_matrix):
    """
    sensor_matrix shape:
        samples x 64

    Returns unique sensor indices chosen to represent
    different signal characteristics.
    """

    means = np.mean(
        sensor_matrix,
        axis=0
    )

    peaks = np.max(
        sensor_matrix,
        axis=0
    )

    noise = np.array([
        estimate_sensor_noise(
            sensor_matrix[:, i]
        )
        for i in range(sensor_matrix.shape[1])
    ])

    selected = []

    def add(index):
        index = int(index)

        if index not in selected:
            selected.append(index)

    # Highest mean-loading sensor
    add(
        np.argmax(means)
    )

    # Highest peak-pressure sensor
    add(
        np.argmax(peaks)
    )

    # Noisiest sensor
    add(
        np.argmax(noise)
    )

    # Lower-pressure sensor, but avoid completely inactive sensors if possible
    positive_mean_indices = np.where(
        means > np.percentile(means, 20)
    )[0]

    if len(positive_mean_indices) > 0:
        low_idx = positive_mean_indices[
            np.argmin(
                means[positive_mean_indices]
            )
        ]
        add(low_idx)

    # Sensor around median loading
    median_target = np.median(means)

    add(
        np.argmin(
            np.abs(
                means - median_target
            )
        )
    )

    # Add high-noise sensors until enough unique channels exist
    noise_order = np.argsort(
        noise
    )[::-1]

    for idx in noise_order:
        add(idx)

        if len(selected) >= N_SELECTED_SENSORS:
            break

    return selected[:N_SELECTED_SENSORS]


def plot_sensor_comparison(
    time,
    raw_signal,
    filtered_signals,
    trial_name,
    foot,
    sensor_idx,
    output_path,
    start_s,
    end_s,
):
    mask = (
        (time >= start_s)
        & (time <= end_s)
    )

    fig, ax = plt.subplots(
        figsize=(15, 6),
        constrained_layout=True
    )

    ax.plot(
        time[mask],
        raw_signal[mask],
        label="Raw",
        linewidth=1.4
    )

    for cutoff, filtered_signal in filtered_signals.items():
        ax.plot(
            time[mask],
            filtered_signal[mask],
            label=f"{cutoff} Hz",
            linewidth=1.1
        )

    ax.set_title(
        f"{trial_name} — {foot} foot — "
        f"sensor {sensor_idx + 1}"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Pressure (kPa)"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        ncol=3
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_raw_vs_cutoff(
    time,
    raw_signal,
    filtered_signal,
    cutoff,
    trial_name,
    foot,
    sensor_idx,
    output_path,
    start_s,
    end_s,
):
    mask = (
        (time >= start_s)
        & (time <= end_s)
    )

    fig, ax = plt.subplots(
        figsize=(15, 6),
        constrained_layout=True
    )

    ax.plot(
        time[mask],
        raw_signal[mask],
        label="Raw",
        linewidth=1.4
    )

    ax.plot(
        time[mask],
        filtered_signal[mask],
        label=f"{cutoff} Hz",
        linewidth=1.4
    )

    ax.set_title(
        f"{trial_name} — {foot} foot — "
        f"sensor {sensor_idx + 1} — "
        f"Raw vs {cutoff} Hz"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Pressure (kPa)"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


def process_foot(
    time,
    sensor_matrix,
    fs,
    trial_name,
    foot,
    trial_output,
):
    selected_sensors = select_representative_sensors(
        sensor_matrix
    )

    print(
        f"  {foot}: selected sensors "
        f"{[i + 1 for i in selected_sensors]}"
    )

    foot_output = (
        trial_output
        / foot
    )

    foot_output.mkdir(
        parents=True,
        exist_ok=True
    )

    duration = time[-1]

    zoom_start = max(
        0,
        duration / 2 - ZOOM_LENGTH_S / 2
    )

    zoom_end = min(
        duration,
        zoom_start + ZOOM_LENGTH_S
    )

    rows = []

    for sensor_idx in selected_sensors:
        raw_signal = sensor_matrix[
            :,
            sensor_idx
        ]

        filtered_signals = {}

        for cutoff in CUTOFFS_HZ:
            if cutoff >= fs / 2:
                continue

            filtered_signal = lowpass_filter(
                raw_signal,
                fs,
                cutoff,
                FILTER_ORDER
            )

            filtered_signals[
                cutoff
            ] = filtered_signal

            metrics = calculate_metrics(
                raw_signal,
                filtered_signal
            )

            rows.append({
                "trial": trial_name,
                "foot": foot,
                "sensor_index_zero_based": sensor_idx,
                "sensor_number_one_based": sensor_idx + 1,
                "sampling_frequency_hz": fs,
                "cutoff_hz": cutoff,
                "filter_order": FILTER_ORDER,
                "sensor_mean_kpa": np.mean(raw_signal),
                "sensor_peak_kpa": np.max(raw_signal),
                "sensor_noise_score": estimate_sensor_noise(raw_signal),
                **metrics,
            })

        # All candidate filters on one zoomed plot
        plot_sensor_comparison(
            time,
            raw_signal,
            filtered_signals,
            trial_name,
            foot,
            sensor_idx,
            foot_output
            / f"sensor_{sensor_idx + 1:02d}_all_cutoffs.png",
            zoom_start,
            zoom_end
        )

        # Individual raw-vs-filter plots
        for cutoff in CUTOFFS_HZ:
            if cutoff not in filtered_signals:
                continue

            plot_raw_vs_cutoff(
                time,
                raw_signal,
                filtered_signals[cutoff],
                cutoff,
                trial_name,
                foot,
                sensor_idx,
                foot_output
                / (
                    f"sensor_{sensor_idx + 1:02d}"
                    f"_raw_vs_{cutoff}hz.png"
                ),
                zoom_start,
                zoom_end
            )

        # Special AB10 loading-change window
        if trial_name == "ab_10_bws.insoleX":
            plot_sensor_comparison(
                time,
                raw_signal,
                filtered_signals,
                trial_name,
                foot,
                sensor_idx,
                foot_output
                / (
                    f"sensor_{sensor_idx + 1:02d}"
                    "_55_75s_loading_change.png"
                ),
                55,
                75
            )

    return rows


def process_trial(file_path):
    (
        time,
        pressure,
        fs,
        recording_datetime,
    ) = read_insolex(
        file_path
    )

    print(
        f"\nProcessing {file_path.name}"
    )

    print(
        f"  Sampling rate: {fs:.2f} Hz"
    )

    print(
        f"  Duration: {time[-1]:.2f} s"
    )

    trial_output = (
        OUTPUT_FOLDER
        / file_path.stem
    )

    trial_output.mkdir(
        parents=True,
        exist_ok=True
    )

    left_sensors = pressure[
        :,
        :64
    ]

    right_sensors = pressure[
        :,
        64:
    ]

    rows = []

    rows.extend(
        process_foot(
            time,
            left_sensors,
            fs,
            file_path.name,
            "left",
            trial_output
        )
    )

    rows.extend(
        process_foot(
            time,
            right_sensors,
            fs,
            file_path.name,
            "right",
            trial_output
        )
    )

    return rows


def main():
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    for filename in SELECTED_FILES:
        file_path = (
            INPUT_FOLDER
            / filename
        )

        if not file_path.exists():
            print(
                f"WARNING: file not found: "
                f"{file_path}"
            )
            continue

        rows = process_trial(
            file_path
        )

        all_rows.extend(
            rows
        )

    if not all_rows:
        raise RuntimeError(
            "No files were processed."
        )

    csv_path = (
        OUTPUT_FOLDER
        / "individual_sensor_filter_summary.csv"
    )

    fieldnames = list(
        all_rows[0].keys()
    )

    with open(
        csv_path,
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
            all_rows
        )

    print("\n====================================")
    print("Individual sensor filtering test done")
    print("====================================")

    print(
        f"Results saved to:\n"
        f"{OUTPUT_FOLDER}"
    )

    print(
        f"\nSummary CSV:\n"
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()