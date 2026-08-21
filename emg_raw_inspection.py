#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import ezc3d
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/raw_data/second_batch/emg"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/emg_raw_inspection/second_batch"
)

SUMMARY_CSV = (

    OUTPUT_FOLDER / "raw_emg_summary.csv"
)

ZOOM_START_S = 10.0
ZOOM_DURATION_S = 10.0

# Number of channels expected in your recordings.
# The script still detects EMG channels from labels.
EXPECTED_EMG_CHANNELS = 10


# ============================================================
# HELPERS
# ============================================================

def canonical_trial_name(path):
    return path.stem.lower()


def detect_emg_channels(labels):
    """
    Find EMG channels from C3D analog labels.

    Works for:
        Emg_1
        Emg_2
        R.Rectus Fem.
        L.Tib.Ant.
        etc.

    For second batch, the first 10 channels are EMG.
    For first batch, labels are Emg_1 ... Emg_11.
    """

    emg_indices = []

    for i, label in enumerate(labels):

        text = str(label).strip().lower()

        if (
            text.startswith("emg")
            or "rectus" in text
            or "semitend" in text
            or "tib.ant" in text
            or "tib ant" in text
            or "gastro" in text
            or "soleus" in text
        ):
            emg_indices.append(i)

    return emg_indices


def clean_label(label):
    label = str(label).strip()

    replacements = {
        "R.Rectus Fem.": "R Rectus femoris",
        "L.Rectus Fem.": "L Rectus femoris",
        "R.Semitend.": "R Semitendinosus",
        "L.Semitend.": "L Semitendinosus",
        "R.Tib.Ant.": "R Tibialis anterior",
        "L.Tib.Ant.": "L Tibialis anterior",
        "R.Med. Gastro": "R Medial gastrocnemius",
        "L.Med. Gastro": "L Medial gastrocnemius",
        "R.Soleus": "R Soleus",
        "L.Soleus": "L Soleus",
    }

    return replacements.get(
        label,
        label,
    )


def calculate_raw_metrics(signal):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    finite = signal[
        np.isfinite(signal)
    ]

    if len(finite) == 0:
        return {
            "min_uv": np.nan,
            "max_uv": np.nan,
            "mean_uv": np.nan,
            "std_uv": np.nan,
            "rms_uv": np.nan,
            "zero_fraction": np.nan,
            "near_max_fraction": np.nan,
        }

    abs_signal = np.abs(finite)

    max_abs = np.max(
        abs_signal
    )

    rms = np.sqrt(
        np.mean(
            finite ** 2
        )
    )

    zero_fraction = np.mean(
        np.isclose(
            finite,
            0.0,
            atol=1e-12,
        )
    )

    # Crude clipping/saturation indicator:
    # fraction of points extremely close to the observed maximum.
    if max_abs > 0:
        near_max_fraction = np.mean(
            abs_signal
            >= 0.999 * max_abs
        )
    else:
        near_max_fraction = 0.0

    return {
        "min_uv":
            float(np.min(finite)),

        "max_uv":
            float(np.max(finite)),

        "mean_uv":
            float(np.mean(finite)),

        "std_uv":
            float(np.std(finite)),

        "rms_uv":
            float(rms),

        "zero_fraction":
            float(zero_fraction),

        "near_max_fraction":
            float(near_max_fraction),
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_overview(
    time,
    emg,
    labels,
    output_path,
    title,
):
    """
    Whole-recording plot.
    One subplot per EMG channel.
    """

    n_channels = emg.shape[0]

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(
            14,
            max(
                10,
                1.8 * n_channels,
            ),
        ),
        sharex=True,
        constrained_layout=True,
    )

    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(
        axes
    ):

        ax.plot(
            time,
            emg[i],
            linewidth=0.5,
        )

        ax.set_ylabel(
            labels[i],
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )

        ax.grid(
            alpha=0.2
        )

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        title
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_zoom(
    time,
    emg,
    labels,
    output_path,
    title,
    start_s,
    duration_s,
):
    """
    Short raw-EMG window for closer inspection.
    """

    end_s = (
        start_s
        + duration_s
    )

    mask = (
        (time >= start_s)
        & (time <= end_s)
    )

    if np.sum(mask) < 2:
        return

    zoom_time = time[mask]
    zoom_emg = emg[:, mask]

    n_channels = zoom_emg.shape[0]

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(
            14,
            max(
                10,
                1.8 * n_channels,
            ),
        ),
        sharex=True,
        constrained_layout=True,
    )

    if n_channels == 1:
        axes = [axes]

    for i, ax in enumerate(
        axes
    ):

        ax.plot(
            zoom_time,
            zoom_emg[i],
            linewidth=0.7,
        )

        ax.set_ylabel(
            labels[i],
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )

        ax.grid(
            alpha=0.2
        )

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        title
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    path,
    summary_rows,
):

    c3d = ezc3d.c3d(
        str(path)
    )

    analogs = np.asarray(
        c3d["data"]["analogs"][0],
        dtype=float,
    )

    fs = float(
        c3d["header"]["analogs"][
            "frame_rate"
        ]
    )

    labels = list(
        c3d["parameters"]["ANALOG"][
            "LABELS"
        ]["value"]
    )

    units = list(
        c3d["parameters"]["ANALOG"][
            "UNITS"
        ]["value"]
    )

    emg_indices = (
        detect_emg_channels(
            labels
        )
    )

    if not emg_indices:
        raise RuntimeError(
            f"No EMG channels detected in "
            f"{path.name}"
        )

    emg = analogs[
        emg_indices,
        :
    ]

    emg_labels = [
        clean_label(
            labels[i]
        )
        for i
        in emg_indices
    ]

    n_samples = emg.shape[1]

    time = (
        np.arange(
            n_samples,
            dtype=float,
        )
        / fs
    )

    duration = (
        n_samples
        / fs
    )

    print()
    print(
        f"FILE: {path.name}"
    )

    print(
        f"  EMG channels: "
        f"{len(emg_indices)}"
    )

    print(
        f"  Sampling rate: "
        f"{fs:.1f} Hz"
    )

    print(
        f"  Duration: "
        f"{duration:.2f} s"
    )

    if (
        len(emg_indices)
        != EXPECTED_EMG_CHANNELS
    ):
        print(
            f"  WARNING: expected "
            f"{EXPECTED_EMG_CHANNELS} EMG "
            f"channels, found "
            f"{len(emg_indices)}"
        )

    for idx, label in zip(
        emg_indices,
        emg_labels,
    ):
        unit = (
            units[idx]
            if idx < len(units)
            else ""
        )

        print(
            f"    {idx:2d}: "
            f"{label} [{unit}]"
        )

    # --------------------------------------------------------
    # Per-channel QC summary
    # --------------------------------------------------------

    for local_idx, (
        original_idx,
        label,
    ) in enumerate(
        zip(
            emg_indices,
            emg_labels,
        )
    ):

        metrics = (
            calculate_raw_metrics(
                emg[
                    local_idx
                ]
            )
        )

        summary_rows.append(
            {
                "file":
                    path.name,

                "trial":
                    canonical_trial_name(
                        path
                    ),

                "channel_index":
                    original_idx,

                "channel_label":
                    label,

                "sampling_rate_hz":
                    fs,

                "duration_s":
                    duration,

                **metrics,
            }
        )

    # --------------------------------------------------------
    # Plot paths
    # --------------------------------------------------------

   # --------------------------------------------------------
# Plot paths
# --------------------------------------------------------

       # --------------------------------------------------------
# Plot paths
# --------------------------------------------------------

    overview_path = (
        OUTPUT_FOLDER
        / f"{path.stem}_raw_emg_overview.png"
    )

    zoom_path = (
        OUTPUT_FOLDER
        / f"{path.stem}_raw_emg_zoom.png"
    )
    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_overview(
        time,
        emg,
        emg_labels,
        overview_path,
        title=(
            f"{path.stem} — "
            f"Raw EMG overview"
        ),
    )

    effective_zoom_start = (
        min(
            ZOOM_START_S,
            max(
                0.0,
                duration
                - ZOOM_DURATION_S,
            ),
        )
    )

    plot_zoom(
        time,
        emg,
        emg_labels,
        zoom_path,
        title=(
            f"{path.stem} — "
            f"Raw EMG "
            f"({effective_zoom_start:.1f}–"
            f"{effective_zoom_start + ZOOM_DURATION_S:.1f} s)"
        ),
        start_s=
            effective_zoom_start,
        duration_s=
            ZOOM_DURATION_S,
    )


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
            "*.c3d"
        )
    )

    if not files:
        raise RuntimeError(
            f"No C3D files found in:\n"
            f"{INPUT_FOLDER}"
        )

    print(
        f"Found {len(files)} "
        f"C3D files."
    )

    summary_rows = []

    successful = 0
    failed = 0

    for path in files:

        try:

            process_file(
                path,
                summary_rows,
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR: "
                f"{path.name}: {e}"
            )

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    if summary_rows:

        fieldnames = list(
            summary_rows[0].keys()
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
            writer.writerows(
                summary_rows
            )

    print()
    print("=" * 60)
    print(
        "RAW EMG INSPECTION COMPLETE"
    )
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"\nPlots:\n"
        f"{OUTPUT_FOLDER}"
    )

    print(
        f"\nSummary CSV:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()