#!/usr/bin/env python3

from pathlib import Path
import csv

import ezc3d
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch


# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/raw_data/second_batch/emg"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/emg_psd_qc/second_batch"
)

SUMMARY_CSV = OUTPUT_FOLDER / "emg_psd_summary.csv"


# ------------------------------------------------------------
# Welch PSD settings
# ------------------------------------------------------------

WELCH_SEGMENT_SECONDS = 2.0
WELCH_OVERLAP_FRACTION = 0.5


# ------------------------------------------------------------
# Spectral bands for QC
# ------------------------------------------------------------

LOW_FREQ_MAX_HZ = 20.0

POWERLINE_FREQ_HZ = 50.0
POWERLINE_HALF_WIDTH_HZ = 1.0

HIGH_FREQ_MIN_HZ = 450.0

EMG_ANALYSIS_LOW_HZ = 20.0
EMG_ANALYSIS_HIGH_HZ = 450.0


# ------------------------------------------------------------
# QC thresholds
#
# These are intended as screening flags, not hard rejection
# criteria.
# ------------------------------------------------------------

REVIEW_LOW_FREQ_PERCENT = 20.0
POOR_LOW_FREQ_PERCENT = 35.0

REVIEW_POWERLINE_PERCENT = 3.0
POOR_POWERLINE_PERCENT = 7.0

REVIEW_HIGH_FREQ_PERCENT = 10.0
POOR_HIGH_FREQ_PERCENT = 20.0


# ============================================================
# CHANNEL DETECTION
# ============================================================

def detect_emg_channels(labels, units):
    """
    Detect EMG channels from either named second-batch labels
    or generic first-batch names such as Emg_1.
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

        is_uv = unit in {
            "uv",
            "µv",
            "μv",
        }

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
# TRIAL NAME PARSING
# ============================================================

def parse_trial_name(stem):
    """
    Examples:
        al_0_bws
        ar_25_bws
        ti_30_bws

    Returns:
        patient, bws
    """

    parts = stem.split("_")

    patient = (
        parts[0].upper()
        if len(parts) >= 1
        else stem.upper()
    )

    bws = np.nan

    for part in parts[1:]:

        try:
            bws = float(part)
            break

        except ValueError:
            continue

    return patient, bws


# ============================================================
# PSD CALCULATION
# ============================================================

def calculate_psd(signal, fs):

    signal = np.asarray(
        signal,
        dtype=float,
    )

    signal = signal - np.nanmedian(
        signal
    )

    nperseg = int(
        round(
            WELCH_SEGMENT_SECONDS
            * fs
        )
    )

    nperseg = min(
        nperseg,
        len(signal),
    )

    if nperseg < 8:
        raise ValueError(
            "Signal too short for Welch PSD."
        )

    noverlap = int(
        round(
            nperseg
            * WELCH_OVERLAP_FRACTION
        )
    )

    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    return freqs, psd


# ============================================================
# SPECTRAL METRICS
# ============================================================

def integrate_band(
    freqs,
    psd,
    f_low,
    f_high,
):

    mask = (
        (freqs >= f_low)
        & (freqs <= f_high)
    )

    if np.sum(mask) < 2:
        return 0.0

    return float(
        np.trapezoid(
            psd[mask],
            freqs[mask],
        )
    )


def relative_band_power(
    freqs,
    psd,
    f_low,
    f_high,
):

    total_power = float(
        np.trapezoid(
            psd,
            freqs,
        )
    )

    if total_power <= 0:
        return np.nan

    band_power = integrate_band(
        freqs,
        psd,
        f_low,
        f_high,
    )

    return (
        100.0
        * band_power
        / total_power
    )


def dominant_frequency(
    freqs,
    psd,
    f_low,
    f_high,
):

    mask = (
        (freqs >= f_low)
        & (freqs <= f_high)
    )

    if not np.any(mask):
        return np.nan

    local_freqs = freqs[mask]
    local_psd = psd[mask]

    idx = int(
        np.argmax(
            local_psd
        )
    )

    return float(
        local_freqs[idx]
    )


def mean_frequency(
    freqs,
    psd,
    f_low,
    f_high,
):

    mask = (
        (freqs >= f_low)
        & (freqs <= f_high)
    )

    if np.sum(mask) < 2:
        return np.nan

    f = freqs[mask]
    p = psd[mask]

    denominator = np.trapezoid(
        p,
        f,
    )

    if denominator <= 0:
        return np.nan

    numerator = np.trapezoid(
        f * p,
        f,
    )

    return float(
        numerator
        / denominator
    )


def median_frequency(
    freqs,
    psd,
    f_low,
    f_high,
):

    mask = (
        (freqs >= f_low)
        & (freqs <= f_high)
    )

    if np.sum(mask) < 2:
        return np.nan

    f = freqs[mask]
    p = psd[mask]

    cumulative = np.zeros_like(
        p,
        dtype=float,
    )

    if len(p) > 1:

        cumulative[1:] = np.cumsum(
            0.5
            * (
                p[:-1]
                + p[1:]
            )
            * np.diff(f)
        )

    total_power = cumulative[-1]

    if total_power <= 0:
        return np.nan

    target = (
        total_power / 2.0
    )

    idx = int(
        np.searchsorted(
            cumulative,
            target,
        )
    )

    idx = min(
        idx,
        len(f) - 1,
    )

    return float(
        f[idx]
    )


# ============================================================
# QC FLAG
# ============================================================

def classify_qc(
    low_freq_pct,
    powerline_pct,
    high_freq_pct,
):

    if (
        low_freq_pct >= POOR_LOW_FREQ_PERCENT
        or powerline_pct >= POOR_POWERLINE_PERCENT
        or high_freq_pct >= POOR_HIGH_FREQ_PERCENT
    ):
        return "POOR"

    if (
        low_freq_pct >= REVIEW_LOW_FREQ_PERCENT
        or powerline_pct >= REVIEW_POWERLINE_PERCENT
        or high_freq_pct >= REVIEW_HIGH_FREQ_PERCENT
    ):
        return "REVIEW"

    return "GOOD"


# ============================================================
# METRICS FOR ONE CHANNEL
# ============================================================

def calculate_metrics(
    freqs,
    psd,
):

    low_freq_pct = relative_band_power(
        freqs,
        psd,
        0.0,
        LOW_FREQ_MAX_HZ,
    )

    powerline_pct = relative_band_power(
        freqs,
        psd,
        POWERLINE_FREQ_HZ
        - POWERLINE_HALF_WIDTH_HZ,
        POWERLINE_FREQ_HZ
        + POWERLINE_HALF_WIDTH_HZ,
    )

    high_freq_pct = relative_band_power(
        freqs,
        psd,
        HIGH_FREQ_MIN_HZ,
        freqs[-1],
    )

    dominant_hz = dominant_frequency(
        freqs,
        psd,
        EMG_ANALYSIS_LOW_HZ,
        EMG_ANALYSIS_HIGH_HZ,
    )

    mean_hz = mean_frequency(
        freqs,
        psd,
        EMG_ANALYSIS_LOW_HZ,
        EMG_ANALYSIS_HIGH_HZ,
    )

    median_hz = median_frequency(
        freqs,
        psd,
        EMG_ANALYSIS_LOW_HZ,
        EMG_ANALYSIS_HIGH_HZ,
    )

    qc = classify_qc(
        low_freq_pct,
        powerline_pct,
        high_freq_pct,
    )

    return {
        "dominant_frequency_hz":
            dominant_hz,

        "mean_frequency_hz":
            mean_hz,

        "median_frequency_hz":
            median_hz,

        "power_below_20hz_pct":
            low_freq_pct,

        "power_around_50hz_pct":
            powerline_pct,

        "power_above_450hz_pct":
            high_freq_pct,

        "qc":
            qc,
    }


# ============================================================
# PSD PLOT
# ============================================================

def plot_trial_psd(
    trial_name,
    channel_results,
    output_path,
):

    n_channels = len(
        channel_results
    )

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(
            14,
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

    for ax, result in zip(
        axes,
        channel_results,
    ):

        freqs = result["freqs"]
        psd = result["psd"]
        label = result["label"]
        metrics = result["metrics"]

        ax.semilogy(
            freqs,
            psd,
            linewidth=0.8,
        )

        ax.axvline(
            20,
            linestyle="--",
            linewidth=0.8,
        )

        ax.axvline(
            50,
            linestyle="--",
            linewidth=0.8,
        )

        ax.axvline(
            450,
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_ylabel(
            label,
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
        )

        ax.grid(
            alpha=0.2
        )

        text = (
            f"QC={metrics['qc']} | "
            f"<20 Hz="
            f"{metrics['power_below_20hz_pct']:.1f}% | "
            f"50 Hz="
            f"{metrics['power_around_50hz_pct']:.2f}% | "
            f">450 Hz="
            f"{metrics['power_above_450hz_pct']:.1f}%"
        )

        ax.text(
            0.99,
            0.92,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )

        ax.set_xlim(
            0,
            min(
                500,
                freqs[-1],
            ),
        )

    axes[-1].set_xlabel(
        "Frequency (Hz)"
    )

    fig.suptitle(
        f"{trial_name} — Raw EMG PSD QC"
    )

    plt.savefig(
        output_path,
        dpi=180,
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

    emg_indices = detect_emg_channels(
        labels,
        units,
    )

    if not emg_indices:
        raise RuntimeError(
            "No EMG channels detected."
        )

    patient, bws = parse_trial_name(
        path.stem
    )

    channel_results = []

    for channel_index in emg_indices:

        label = clean_label(
            labels[
                channel_index
            ]
        )

        signal = analogs[
            channel_index
        ]

        freqs, psd = calculate_psd(
            signal,
            fs,
        )

        metrics = calculate_metrics(
            freqs,
            psd,
        )

        channel_results.append(
            {
                "label":
                    label,

                "freqs":
                    freqs,

                "psd":
                    psd,

                "metrics":
                    metrics,
            }
        )

        summary_rows.append(
            {
                "file":
                    path.name,

                "trial":
                    path.stem,

                "patient":
                    patient,

                "bws_percent":
                    bws,

                "channel_index":
                    channel_index,

                "muscle":
                    label,

                "sampling_rate_hz":
                    fs,

                "duration_s":
                    len(signal) / fs,

                **metrics,
            }
        )

        print(
            f"  {label:25s} "
            f"QC={metrics['qc']:6s} "
            f"<20Hz="
            f"{metrics['power_below_20hz_pct']:6.2f}% "
            f"50Hz="
            f"{metrics['power_around_50hz_pct']:6.2f}% "
            f">450Hz="
            f"{metrics['power_above_450hz_pct']:6.2f}%"
        )

    plot_path = (
        OUTPUT_FOLDER
        / f"{path.stem}_psd.png"
    )

    plot_trial_psd(
        trial_name=path.stem,
        channel_results=channel_results,
        output_path=plot_path,
    )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary_csv(
    rows,
):

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
        f"Found {len(files)} C3D files."
    )

    print(
        f"Input: {INPUT_FOLDER}"
    )

    print(
        f"Output: {OUTPUT_FOLDER}"
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
                f"ERROR processing "
                f"{path.name}: {e}"
            )

    save_summary_csv(
        summary_rows
    )

    print()
    print("=" * 60)
    print("PSD QC FINISHED")
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print()
    print(
        f"PSD plots:\n"
        f"{OUTPUT_FOLDER}"
    )

    print()
    print(
        f"Summary CSV:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()  