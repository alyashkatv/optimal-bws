from pathlib import Path
import struct

import numpy as np
from scipy.signal import butter, sosfiltfilt


# ============================================================
# Configuration
# ============================================================

INPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/raw_data/alya_test"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/test_insoles/filtered_alya_test"
)

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128
DATA_START = 72

FILTER_ORDER = 4
CUTOFF_HZ = 20.0


# ============================================================
# Read raw COMETA .insoleX
# ============================================================

def read_insolex(path):
    with open(path, "rb") as f:
        raw = f.read()

    date_length = raw[2]

    recording_datetime = raw[
        3:3 + date_length
    ].decode(
        "ascii",
        errors="replace"
    )

    metadata_offset = (
        3
        + date_length
        + 1
    )

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
        + n_samples
        * TOTAL_SENSORS
        * 4
    )

    if len(raw) != expected_size:
        raise ValueError(
            f"{path.name}: unexpected file size.\n"
            f"Expected: {expected_size}\n"
            f"Actual:   {len(raw)}"
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

    time = (
        np.arange(n_samples)
        / sampling_rate
    )

    return (
        time,
        pressure,
        sampling_rate,
        recording_datetime
    )


# ============================================================
# Read cleaned NPZ
# ============================================================

def read_cleaned_npz(path):
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
            f"{path.name}: time and pressure "
            f"lengths do not match"
        )

    recording_datetime = ""

    if "recording_datetime" in data:
        value = data[
            "recording_datetime"
        ]

        if np.ndim(value) == 0:
            recording_datetime = str(
                value.item()
            )

    return (
        time,
        pressure,
        sampling_rate,
        recording_datetime
    )


# ============================================================
# Source-selection logic
# ============================================================

def load_trial(insolex_path):
    """
    Prefer:

        trial_cleaned.npz

    over:

        trial.insoleX

    if both exist.
    """

    cleaned_path = (
        insolex_path.parent
        / (
            insolex_path.stem
            + "_cleaned.npz"
        )
    )

    if cleaned_path.exists():

        print(
            f"  SOURCE: CLEANED NPZ "
            f"({cleaned_path.name})"
        )

        (
            time,
            pressure,
            fs,
            recording_datetime
        ) = read_cleaned_npz(
            cleaned_path
        )

        source_type = "cleaned_npz"
        source_path = cleaned_path

    else:

        print(
            f"  SOURCE: RAW INSOLEX "
            f"({insolex_path.name})"
        )

        (
            time,
            pressure,
            fs,
            recording_datetime
        ) = read_insolex(
            insolex_path
        )

        source_type = "raw_insolex"
        source_path = insolex_path

    return (
        time,
        pressure,
        fs,
        recording_datetime,
        source_type,
        source_path
    )


# ============================================================
# Filtering
# ============================================================

def filter_pressure(
    pressure,
    fs,
    cutoff_hz=CUTOFF_HZ,
    order=FILTER_ORDER
):
    nyquist = fs / 2

    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Cutoff {cutoff_hz} Hz must be "
            f"below Nyquist frequency "
            f"{nyquist} Hz"
        )

    sos = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    # IMPORTANT:
    #
    # pressure shape:
    #     samples × 128
    #
    # axis=0 means each of the 128 sensors
    # is filtered independently through time.

    filtered = sosfiltfilt(
        sos,
        pressure,
        axis=0
    )

    return filtered


# ============================================================
# Process one trial
# ============================================================

def process_trial(insolex_path):

    print(
        "\n========================================"
    )

    print(
        f"Trial: {insolex_path.stem}"
    )

    (
        time,
        pressure,
        fs,
        recording_datetime,
        source_type,
        source_path
    ) = load_trial(
        insolex_path
    )

    print(
        f"  Sampling rate : {fs:.2f} Hz"
    )

    print(
        f"  Samples       : {len(time)}"
    )

    print(
        f"  Duration      : "
        f"{len(time) / fs:.2f} s"
    )

    print(
        f"  Filter        : "
        f"{FILTER_ORDER}th-order "
        f"Butterworth, "
        f"{CUTOFF_HZ:g} Hz, "
        f"zero-phase"
    )

    # --------------------------------------------------------
    # Apply filter to ALL 128 individual sensors
    # --------------------------------------------------------

    filtered_pressure = filter_pressure(
        pressure,
        fs
    )

    # --------------------------------------------------------
    # Save analysis-ready NPZ
    # --------------------------------------------------------

    output_path = (
        OUTPUT_FOLDER
        / (
            insolex_path.stem
            + "_filtered.npz"
        )
    )

    np.savez_compressed(
        output_path,

        # Main data
        time=time,
        pressure=filtered_pressure,
        sampling_rate=fs,

        # Provenance
        original_trial=insolex_path.name,
        source_file=source_path.name,
        source_type=source_type,
        recording_datetime=recording_datetime,

        # Filtering metadata
        filter_type="Butterworth low-pass",
        filter_order=FILTER_ORDER,
        cutoff_hz=CUTOFF_HZ,
        zero_phase=True,

        # Dimensions
        sensors_per_insole=SENSORS_PER_INSOLE,
        total_sensors=TOTAL_SENSORS
    )

    print(
        f"  SAVED: {output_path.name}"
    )

    return {
        "trial": insolex_path.stem,
        "source": source_path.name,
        "source_type": source_type,
        "output": output_path.name,
        "fs": fs,
        "samples": len(time),
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find raw trial names.
    #
    # We use the .insoleX files as the trial index.
    # load_trial() then checks whether a cleaned NPZ exists.
    # --------------------------------------------------------

    files = sorted(
        INPUT_FOLDER.glob(
            "*.insoleX"
        )
    )

    if not files:
        raise RuntimeError(
            f"No .insoleX files found in:\n"
            f"{INPUT_FOLDER}"
        )

    print(
        f"Found {len(files)} trials."
    )

    successful = []
    failed = []

    for file_path in files:

        try:
            result = process_trial(
                file_path
            )

            successful.append(
                result
            )

        except Exception as e:

            print(
                f"\nERROR: "
                f"{file_path.name}"
            )

            print(
                f"  {e}"
            )

            failed.append(
                (
                    file_path.name,
                    str(e)
                )
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n\n========================================"
    )

    print(
        "FILTERING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Successful : {len(successful)}"
    )

    print(
        f"Failed     : {len(failed)}"
    )

    print(
        f"\nOutput folder:\n"
        f"{OUTPUT_FOLDER}"
    )

    print(
        "\nSource selection:"
    )

    for result in successful:

        marker = (
            "CLEANED"
            if result["source_type"]
            == "cleaned_npz"
            else "RAW"
        )

        print(
            f"  [{marker:7}] "
            f"{result['trial']} "
            f"<- {result['source']}"
        )

    if failed:

        print(
            "\nFailed trials:"
        )

        for filename, error in failed:

            print(
                f"  {filename}: "
                f"{error}"
            )


if __name__ == "__main__":
    main()