#!/usr/bin/env python3

from pathlib import Path
import shutil
import numpy as np


DATA_DIR = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/filtered_second_batch"
)

# Seconds to remove from beginning/end of each filtered trial.
CROP_RULES = {
    "al_0_bws_filtered.npz": {
        "start": 5.0,
        "end": 0.0,
    },
    "yers_0_bws_filtered.npz": {
        "start": 18.0,
        "end": 0.0,
    },
    "yer_5_bws_filtered.npz": {
        "start": 0.0,
        "end": 10.0,
    },
    "dm_0_bws_filtered.npz": {
        "start": 5.0,
        "end": 0.0,
    },
}


def get_time_axis(data):
    """
    Return time array and sampling rate.

    Expected filtered NPZ structure includes:
        time
        pressure
        sampling_rate
        ...
    """
    if "time" not in data:
        raise KeyError("NPZ does not contain a 'time' array.")

    time = np.asarray(data["time"]).squeeze()

    if time.ndim != 1:
        raise ValueError(
            f"'time' must be 1-D, got shape {time.shape}"
        )

    if len(time) < 2:
        raise ValueError("Time array is too short.")

    if "sampling_rate" in data:
        fs = float(np.asarray(data["sampling_rate"]).squeeze())
    else:
        dt = np.median(np.diff(time))
        if dt <= 0:
            raise ValueError("Could not infer sampling rate from time array.")
        fs = 1.0 / dt

    return time, fs


def crop_array_if_time_aligned(arr, n_samples, start_idx, end_idx):
    """
    Crop arrays whose first dimension corresponds to time.

    Examples:
        time:       (N,)
        pressure:   (N, 128)
        force:      (N, 2)

    Metadata/scalars are left unchanged.
    """
    arr = np.asarray(arr)

    if arr.ndim >= 1 and arr.shape[0] == n_samples:
        return arr[start_idx:end_idx]

    return arr


def crop_file(path, start_seconds, end_seconds):
    print(f"\nProcessing: {path.name}")

    if not path.exists():
        print("  ERROR: file not found")
        return False

    with np.load(path, allow_pickle=True) as data:
        contents = {
            key: data[key]
            for key in data.files
        }

    time, fs = get_time_axis(contents)

    n_samples = len(time)
    duration = time[-1] - time[0]

    print(f"  Original samples : {n_samples}")
    print(f"  Sampling rate    : {fs:.3f} Hz")
    print(f"  Original duration: {duration:.3f} s")
    print(f"  Crop beginning   : {start_seconds:.3f} s")
    print(f"  Crop end         : {end_seconds:.3f} s")

    start_idx = int(round(start_seconds * fs))

    if end_seconds > 0:
        end_idx = n_samples - int(round(end_seconds * fs))
    else:
        end_idx = n_samples

    start_idx = max(0, start_idx)
    end_idx = min(n_samples, end_idx)

    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid crop for {path.name}: "
            f"start_idx={start_idx}, end_idx={end_idx}"
        )

    cropped = {}

    for key, value in contents.items():
        cropped[key] = crop_array_if_time_aligned(
            value,
            n_samples=n_samples,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    # Reset cropped time so every recording starts at t = 0.
    cropped_time = np.asarray(cropped["time"], dtype=float)
    cropped["time"] = cropped_time - cropped_time[0]

    # Store provenance directly inside the NPZ.
    cropped["manual_crop_applied"] = np.array(True)
    cropped["manual_crop_start_seconds"] = np.array(
        start_seconds,
        dtype=float,
    )
    cropped["manual_crop_end_seconds"] = np.array(
        end_seconds,
        dtype=float,
    )

    new_n = len(cropped["time"])
    new_duration = cropped["time"][-1] - cropped["time"][0]

    print(f"  New samples      : {new_n}")
    print(f"  New duration     : {new_duration:.3f} s")

    # Create backup only once.
    backup_path = path.with_name(
        path.stem + "_before_manual_crop.npz"
    )

    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print(f"  Backup           : {backup_path.name}")
    else:
        print(
            f"  Backup already exists: {backup_path.name}"
        )

    # Save to temporary file first, then replace original.
    temp_path = path.with_name(
        path.stem + "_cropping_tmp.npz"
    )

    np.savez_compressed(temp_path, **cropped)

    temp_path.replace(path)

    print(f"  OVERWRITTEN      : {path.name}")

    return True


def main():
    print("=" * 70)
    print("MANUAL CROPPING OF FILTERED INSOLE TRIALS")
    print("=" * 70)
    print(f"Directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Directory does not exist:\n{DATA_DIR}"
        )

    successful = 0
    failed = 0

    for filename, rule in CROP_RULES.items():
        path = DATA_DIR / filename

        try:
            ok = crop_file(
                path,
                start_seconds=rule["start"],
                end_seconds=rule["end"],
            )

            if ok:
                successful += 1
            else:
                failed += 1

        except Exception as exc:
            failed += 1
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed    : {failed}")


if __name__ == "__main__":
    main()