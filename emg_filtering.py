#!/usr/bin/env python3

from pathlib import Path
import csv

import ezc3d
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    sosfiltfilt,
    iirnotch,
    filtfilt,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/raw_data/second_batch/emg"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/emg_second_batch"
)

PLOT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/emg_filter_validation/second_batch"
)

QC_CSV = OUTPUT_FOLDER / "emg_filtering_qc.csv"


# ------------------------------------------------------------
# EMG filtering settings
# ------------------------------------------------------------

BANDPASS_LOW_HZ = 20.0
BANDPASS_HIGH_HZ = 450.0
BANDPASS_ORDER = 4

# Leave False initially.
# Turn on only if PSD inspection shows clear 50 Hz contamination.
USE_NOTCH = False

NOTCH_FREQ_HZ = 50.0
NOTCH_Q = 30.0

# Linear envelope
ENVELOPE_LOW_PASS_HZ = 6.0
ENVELOPE_ORDER = 4


# ------------------------------------------------------------
# Artifact QC
# ------------------------------------------------------------

# IMPORTANT:
# We FLAG extreme samples; we do not remove them automatically.
#
# Robust z-score based on median absolute deviation.
ARTIFACT_Z_THRESHOLD = 12.0

# Expand each detected artifact slightly so the QC mask covers
# its immediate neighborhood.
ARTIFACT_PADDING_MS = 10.0


# ------------------------------------------------------------
# Validation plots
# ------------------------------------------------------------

ZOOM_START_S = 10.0
ZOOM_DURATION_S = 10.0


# ============================================================
# CHANNEL DETECTION
# ============================================================

def detect_emg_channels(labels, units):
    """
    Detect EMG channels.

    Works with your second-batch labels such as:
        R.Rectus Fem.
        L.Semitend.
        R.Tib.Ant.
        ...

    It also supports generic first-batch labels:
        Emg_1
        Emg_2
        ...
    """

    indices = []

    muscle_keywords = [
        "rectus",
        "semitend",
        "tib.ant",
        "tib ant",
        "gastro",
        "soleus",
    ]

    for i, label in enumerate(labels):

        text = str(label).strip().lower()

        unit = (
            str(units[i]).strip().lower()
            if i < len(units)
            else ""
        )

        is_named_emg = (
            text.startswith("emg")
            or any(
                keyword in text
                for keyword in muscle_keywords
            )
        )

        is_uv = (
            unit in {
                "uv",
                "µv",
                "μv",
            }
        )

        if is_named_emg or is_uv:
            indices.append(i)

    return indices


def clean_label(label):

    label = str(label).strip()

    replacements = {
        "R.Rectus Fem.":
            "R Rectus femoris",

        "L.Rectus Fem.":
            "L Rectus femoris",

        "R.Semitend.":
            "R Semitendinosus",

        "L.Semitend.":
            "L Semitendinosus",

        "R.Tib.Ant.":
            "R Tibialis anterior",

        "L.Tib.Ant.":
            "L Tibialis anterior",

        "R.Med. Gastro":
            "R Medial gastrocnemius",

        "L.Med. Gastro":
            "L Medial gastrocnemius",

        "R.Soleus":
            "R Soleus",

        "L.Soleus":
            "L Soleus",
    }

    return replacements.get(
        label,
        label,
    )


# ============================================================
# FILTERS
# ============================================================

def remove_dc(signal):
    """
    Remove channel DC offset using the median.

    Median is used instead of mean because it is more robust
    to large transient spikes.
    """

    return signal - np.nanmedian(signal)


def bandpass_filter(signal, fs):

    nyquist = fs / 2.0

    if BANDPASS_HIGH_HZ >= nyquist:
        raise ValueError(
            f"Band-pass upper cutoff "
            f"{BANDPASS_HIGH_HZ} Hz "
            f"must be below Nyquist "
            f"{nyquist:.1f} Hz."
        )

    sos = butter(
        BANDPASS_ORDER,
        [
            BANDPASS_LOW_HZ,
            BANDPASS_HIGH_HZ,
        ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(
        sos,
        signal,
    )


def notch_filter(signal, fs):

    b, a = iirnotch(
        w0=NOTCH_FREQ_HZ,
        Q=NOTCH_Q,
        fs=fs,
    )

    return filtfilt(
        b,
        a,
        signal,
    )


def envelope_filter(rectified, fs):

    sos = butter(
        ENVELOPE_ORDER,
        ENVELOPE_LOW_PASS_HZ,
        btype="lowpass",
        fs=fs,
        output="sos",
    )

    return sosfiltfilt(
        sos,
        rectified,
    )


# ============================================================
# ARTIFACT DETECTION
# ============================================================

def robust_artifact_mask(signal, fs):
    """
    Flag unusually extreme samples using median/MAD.

    This DOES NOT change the EMG signal.

    It is only a QC indicator.
    """

    signal = np.asarray(
        signal,
        dtype=float,
    )

    median = np.nanmedian(signal)

    abs_dev = np.abs(
        signal - median
    )

    mad = np.nanmedian(
        abs_dev
    )

    if (
        not np.isfinite(mad)
        or mad <= 1e-12
    ):
        return np.zeros(
            signal.shape,
            dtype=bool,
        )

    # Robust equivalent of standard deviation
    robust_sigma = (
        1.4826 * mad
    )

    robust_z = (
        np.abs(signal - median)
        / robust_sigma
    )

    mask = (
        robust_z
        > ARTIFACT_Z_THRESHOLD
    )

    # Expand mask around spikes
    padding_samples = int(
        round(
            ARTIFACT_PADDING_MS
            / 1000.0
            * fs
        )
    )

    if (
        padding_samples > 0
        and np.any(mask)
    ):

        kernel = np.ones(
            2 * padding_samples + 1,
            dtype=int,
        )

        expanded = np.convolve(
            mask.astype(int),
            kernel,
            mode="same",
        )

        mask = expanded > 0

    return mask


# ============================================================
# PROCESS ONE CHANNEL
# ============================================================

def process_channel(raw, fs):

    raw = np.asarray(
        raw,
        dtype=float,
    )

    # --------------------------------------------------------
    # 1. Remove DC
    # --------------------------------------------------------

    demeaned = remove_dc(
        raw
    )

    # --------------------------------------------------------
    # 2. Band-pass
    # --------------------------------------------------------

    filtered = bandpass_filter(
        demeaned,
        fs,
    )

    # --------------------------------------------------------
    # 3. Optional 50 Hz notch
    # --------------------------------------------------------

    if USE_NOTCH:

        filtered = notch_filter(
            filtered,
            fs,
        )

    # --------------------------------------------------------
    # 4. Artifact flagging
    # --------------------------------------------------------

    artifact_mask = (
        robust_artifact_mask(
            filtered,
            fs,
        )
    )

    # --------------------------------------------------------
    # 5. Full-wave rectification
    # --------------------------------------------------------

    rectified = np.abs(
        filtered
    )

    # --------------------------------------------------------
    # 6. Linear envelope
    # --------------------------------------------------------

    envelope = envelope_filter(
        rectified,
        fs,
    )

    # Numerical zero-phase filtering can create tiny negative
    # values close to zero.
    envelope = np.maximum(
        envelope,
        0.0,
    )

    return {
        "demeaned":
            demeaned,

        "filtered":
            filtered,

        "rectified":
            rectified,

        "envelope":
            envelope,

        "artifact_mask":
            artifact_mask,
    }


# ============================================================
# QC METRICS
# ============================================================

def calculate_qc(
    raw,
    filtered,
    artifact_mask,
):

    raw = np.asarray(
        raw,
        dtype=float,
    )

    filtered = np.asarray(
        filtered,
        dtype=float,
    )

    artifact_mask = np.asarray(
        artifact_mask,
        dtype=bool,
    )

    return {
        "raw_min_uv":
            float(np.nanmin(raw)),

        "raw_max_uv":
            float(np.nanmax(raw)),

        "raw_rms_uv":
            float(
                np.sqrt(
                    np.nanmean(
                        raw ** 2
                    )
                )
            ),

        "filtered_rms_uv":
            float(
                np.sqrt(
                    np.nanmean(
                        filtered ** 2
                    )
                )
            ),

        "artifact_fraction":
            float(
                np.mean(
                    artifact_mask
                )
            ),

        "artifact_percent":
            float(
                100.0
                * np.mean(
                    artifact_mask
                )
            ),
    }


# ============================================================
# VALIDATION PLOT
# ============================================================

def plot_validation(
    time,
    raw,
    filtered,
    envelope,
    artifact_mask,
    labels,
    output_path,
    title,
):

    start_s = ZOOM_START_S

    duration = (
        time[-1]
        if len(time)
        else 0
    )

    if (
        start_s
        + ZOOM_DURATION_S
        > duration
    ):
        start_s = max(
            0.0,
            duration
            - ZOOM_DURATION_S,
        )

    end_s = (
        start_s
        + ZOOM_DURATION_S
    )

    mask = (
        (time >= start_s)
        & (time <= end_s)
    )

    n_channels = (
        raw.shape[0]
    )

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(
            15,
            max(
                12,
                2.2 * n_channels,
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

        t = time[mask]

        ax.plot(
            t,
            raw[i, mask],
            linewidth=0.45,
            alpha=0.45,
            label="Raw",
        )

        ax.plot(
            t,
            filtered[i, mask],
            linewidth=0.6,
            label="Band-pass filtered",
        )

        ax.plot(
            t,
            envelope[i, mask],
            linewidth=1.2,
            label="Envelope",
        )

        local_artifacts = (
            artifact_mask[
                i,
                mask
            ]
        )

        if np.any(
            local_artifacts
        ):

            ax.scatter(
                t[
                    local_artifacts
                ],
                filtered[
                    i,
                    mask
                ][
                    local_artifacts
                ],
                s=4,
                marker=".",
                label="Artifact flag",
            )

        ax.set_ylabel(
            labels[i],
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
        )

        ax.grid(
            alpha=0.2
        )

        if i == 0:
            ax.legend(
                loc="upper right",
                fontsize=7,
            )

    axes[-1].set_xlabel(
        "Time (s)"
    )

    fig.suptitle(
        f"{title}\n"
        f"{start_s:.1f}–"
        f"{end_s:.1f} s"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# PROCESS C3D
# ============================================================

def process_file(
    path,
    qc_rows,
):

    print()
    print(
        f"Processing: {path.name}"
    )

    c3d = ezc3d.c3d(
        str(path)
    )

    analogs = np.asarray(
        c3d[
            "data"
        ][
            "analogs"
        ][0],
        dtype=float,
    )

    fs = float(
        c3d[
            "header"
        ][
            "analogs"
        ][
            "frame_rate"
        ]
    )

    labels = list(
        c3d[
            "parameters"
        ][
            "ANALOG"
        ][
            "LABELS"
        ][
            "value"
        ]
    )

    units = list(
        c3d[
            "parameters"
        ][
            "ANALOG"
        ][
            "UNITS"
        ][
            "value"
        ]
    )

    emg_indices = (
        detect_emg_channels(
            labels,
            units,
        )
    )

    if not emg_indices:

        raise RuntimeError(
            "No EMG channels detected."
        )

    raw_emg = analogs[
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

    n_channels, n_samples = (
        raw_emg.shape
    )

    time = (
        np.arange(
            n_samples
        )
        / fs
    )

    print(
        f"  Sampling rate: "
        f"{fs:.1f} Hz"
    )

    print(
        f"  EMG channels: "
        f"{n_channels}"
    )

    print(
        f"  Duration: "
        f"{n_samples / fs:.2f} s"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    demeaned = np.zeros_like(
        raw_emg
    )

    filtered = np.zeros_like(
        raw_emg
    )

    rectified = np.zeros_like(
        raw_emg
    )

    envelope = np.zeros_like(
        raw_emg
    )

    artifact_mask = np.zeros(
        raw_emg.shape,
        dtype=bool,
    )

    # --------------------------------------------------------
    # Process channels
    # --------------------------------------------------------

    for ch in range(
        n_channels
    ):

        result = (
            process_channel(
                raw_emg[ch],
                fs,
            )
        )

        demeaned[ch] = (
            result[
                "demeaned"
            ]
        )

        filtered[ch] = (
            result[
                "filtered"
            ]
        )

        rectified[ch] = (
            result[
                "rectified"
            ]
        )

        envelope[ch] = (
            result[
                "envelope"
            ]
        )

        artifact_mask[ch] = (
            result[
                "artifact_mask"
            ]
        )

        qc = calculate_qc(
            raw_emg[ch],
            filtered[ch],
            artifact_mask[ch],
        )

        qc_rows.append(
            {
                "file":
                    path.name,

                "trial":
                    path.stem,

                "channel_index":
                    emg_indices[ch],

                "channel_label":
                    emg_labels[ch],

                "sampling_rate_hz":
                    fs,

                "duration_s":
                    n_samples / fs,

                "bandpass_low_hz":
                    BANDPASS_LOW_HZ,

                "bandpass_high_hz":
                    BANDPASS_HIGH_HZ,

                "notch_used":
                    USE_NOTCH,

                "envelope_lowpass_hz":
                    ENVELOPE_LOW_PASS_HZ,

                **qc,
            }
        )

        print(
            f"    "
            f"{emg_labels[ch]:25s} "
            f"artifact="
            f"{qc['artifact_percent']:.3f}%"
        )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    output_path = (
        OUTPUT_FOLDER
        / (
            f"{path.stem}"
            f"_emg_filtered.npz"
        )
    )

    np.savez_compressed(
        output_path,

        time=time,

        fs=np.array(
            fs
        ),

        channel_indices=np.asarray(
            emg_indices,
            dtype=int,
        ),

        channel_labels=np.asarray(
            emg_labels,
            dtype=str,
        ),

        raw=raw_emg,

        demeaned=demeaned,

        filtered=filtered,

        rectified=rectified,

        envelope=envelope,

        artifact_mask=artifact_mask,

        bandpass_low_hz=np.array(
            BANDPASS_LOW_HZ
        ),

        bandpass_high_hz=np.array(
            BANDPASS_HIGH_HZ
        ),

        notch_used=np.array(
            USE_NOTCH
        ),

        notch_freq_hz=np.array(
            NOTCH_FREQ_HZ
        ),

        envelope_lowpass_hz=np.array(
            ENVELOPE_LOW_PASS_HZ
        ),
    )

    print(
        f"  Saved: "
        f"{output_path.name}"
    )

    # --------------------------------------------------------
    # Validation plot
    # --------------------------------------------------------

    plot_path = (
        PLOT_FOLDER
        / (
            f"{path.stem}"
            f"_emg_filter_validation.png"
        )
    )

    plot_validation(
        time=time,
        raw=raw_emg,
        filtered=filtered,
        envelope=envelope,
        artifact_mask=artifact_mask,
        labels=emg_labels,
        output_path=plot_path,
        title=(
            f"{path.stem} — "
            f"EMG filtering validation"
        ),
    )


# ============================================================
# SAVE QC CSV
# ============================================================

def save_qc_csv(rows):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        QC_CSV,
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


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_FOLDER.mkdir(
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

    print()
    print(
        "Filtering settings:"
    )

    print(
        f"  Band-pass: "
        f"{BANDPASS_LOW_HZ}–"
        f"{BANDPASS_HIGH_HZ} Hz"
    )

    print(
        f"  50 Hz notch: "
        f"{USE_NOTCH}"
    )

    print(
        f"  Envelope LPF: "
        f"{ENVELOPE_LOW_PASS_HZ} Hz"
    )

    qc_rows = []

    successful = 0
    failed = 0

    for path in files:

        try:

            process_file(
                path,
                qc_rows,
            )

            successful += 1

        except Exception as e:

            failed += 1

            print(
                f"\nERROR processing "
                f"{path.name}: {e}"
            )

    save_qc_csv(
        qc_rows
    )

    print()
    print("=" * 60)
    print(
        "EMG FILTERING FINISHED"
    )
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        "\nFiltered NPZ files:"
    )

    print(
        OUTPUT_FOLDER
    )

    print(
        "\nValidation plots:"
    )

    print(
        PLOT_FOLDER
    )

    print(
        "\nQC table:"
    )

    print(
        QC_CSV
    )


if __name__ == "__main__":
    main()