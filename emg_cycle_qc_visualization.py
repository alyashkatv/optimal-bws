#!/usr/bin/env python3

from pathlib import Path
import csv
import random

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

EMG_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/emg_second_batch"
)

QC_CSV = Path(
    "/home/alya/Desktop/optimal-bws/emg_cycle_qc/second_batch/"
    "emg_cycle_qc_by_muscle.csv"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/emg_cycle_qc_visualization/second_batch"
)


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

# Number of examples to save for each QC category.
N_EXAMPLES_PER_CLASS = 15

# Use fixed seed so repeated runs give the same examples.
RANDOM_SEED = 42


# ------------------------------------------------------------
# Padding around the gait cycle in the plot
# ------------------------------------------------------------

PADDING_S = 0.15


# ============================================================
# LOAD QC TABLE
# ============================================================

def load_qc_rows(path):

    if not path.exists():
        raise FileNotFoundError(
            f"QC CSV not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    # Convert useful numeric columns
    for row in rows:

        row["bws_percent"] = float(
            row["bws_percent"]
        )

        row["cycle_number"] = int(
            row["cycle_number"]
        )

        row["channel_index"] = int(
            row["channel_index"]
        )

        row["cycle_start_s"] = float(
            row["cycle_start_s"]
        )

        row["cycle_end_s"] = float(
            row["cycle_end_s"]
        )

        row["cycle_duration_s"] = float(
            row["cycle_duration_s"]
        )

        row["artifact_fraction"] = float(
            row["artifact_fraction"]
        )

        row["artifact_percent"] = float(
            row["artifact_percent"]
        )

        row["rms_uv"] = float(
            row["rms_uv"]
        )

        row["peak_abs_uv"] = float(
            row["peak_abs_uv"]
        )

    return rows


# ============================================================
# LOAD FILTERED EMG NPZ
# ============================================================

def load_emg_trial(trial):

    path = (
        EMG_FOLDER
        / f"{trial}_emg_filtered.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Filtered EMG file not found:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        time = np.asarray(
            data["time"],
            dtype=float,
        ).reshape(-1)

        filtered = np.asarray(
            data["filtered"],
            dtype=float,
        )

        envelope = np.asarray(
            data["envelope"],
            dtype=float,
        )

        artifact_mask = np.asarray(
            data["artifact_mask"],
            dtype=bool,
        )

        labels = [
            str(x)
            for x in np.asarray(
                data["channel_labels"]
            ).reshape(-1)
        ]

    # Expected channels x samples
    if filtered.shape[1] != len(time):

        if filtered.shape[0] == len(time):

            filtered = filtered.T
            envelope = envelope.T
            artifact_mask = artifact_mask.T

        else:
            raise ValueError(
                f"{path.name}: invalid shape "
                f"{filtered.shape}"
            )

    return {
        "time": time,
        "filtered": filtered,
        "envelope": envelope,
        "artifact_mask": artifact_mask,
        "labels": labels,
    }


# ============================================================
# SELECT EXAMPLES
# ============================================================

def select_examples(rows):

    random.seed(
        RANDOM_SEED
    )

    selected = []

    for qc_class in (
        "KEEP",
        "REVIEW",
        "REJECT",
    ):

        class_rows = [
            row
            for row in rows
            if row["qc"] == qc_class
        ]

        if not class_rows:
            continue

        n = min(
            N_EXAMPLES_PER_CLASS,
            len(class_rows),
        )

        selected.extend(
            random.sample(
                class_rows,
                n,
            )
        )

    return selected


# ============================================================
# TIME WINDOW
# ============================================================

def get_plot_indices(
    time,
    cycle_start,
    cycle_end,
):

    plot_start = max(
        0.0,
        cycle_start - PADDING_S,
    )

    plot_end = min(
        time[-1],
        cycle_end + PADDING_S,
    )

    start_idx = int(
        np.searchsorted(
            time,
            plot_start,
            side="left",
        )
    )

    end_idx = int(
        np.searchsorted(
            time,
            plot_end,
            side="right",
        )
    )

    return (
        start_idx,
        end_idx,
        plot_start,
        plot_end,
    )


# ============================================================
# PLOT ONE EXAMPLE
# ============================================================

def plot_qc_example(
    row,
    emg,
    output_path,
):

    time = emg["time"]

    channel = (
        row["channel_index"]
    )

    # Important:
    # channel_index in the QC CSV is local EMG index,
    # because emg_cycle_qc.py used enumerate(labels).
    if (
        channel < 0
        or channel >= emg["filtered"].shape[0]
    ):
        raise IndexError(
            f"Channel {channel} outside "
            f"EMG shape {emg['filtered'].shape}"
        )

    (
        start_idx,
        end_idx,
        plot_start,
        plot_end,
    ) = get_plot_indices(
        time,
        row["cycle_start_s"],
        row["cycle_end_s"],
    )

    t = time[
        start_idx:end_idx
    ]

    filtered = emg[
        "filtered"
    ][
        channel,
        start_idx:end_idx,
    ]

    envelope = emg[
        "envelope"
    ][
        channel,
        start_idx:end_idx,
    ]

    artifact = emg[
        "artifact_mask"
    ][
        channel,
        start_idx:end_idx,
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(14, 5),
        constrained_layout=True,
    )

    ax.plot(
        t,
        filtered,
        linewidth=0.7,
        label="Filtered EMG",
    )

    ax.plot(
        t,
        envelope,
        linewidth=1.5,
        label="Envelope",
    )

    # Artifact points
    if np.any(
        artifact
    ):

        ax.scatter(
            t[artifact],
            filtered[artifact],
            s=10,
            marker=".",
            label="Artifact flagged",
        )

    # --------------------------------------------------------
    # IC boundaries
    # --------------------------------------------------------

    ax.axvline(
        row["cycle_start_s"],
        linestyle="--",
        linewidth=1.2,
        label="IC start",
    )

    ax.axvline(
        row["cycle_end_s"],
        linestyle="--",
        linewidth=1.2,
        label="Next IC",
    )

    # Shade actual gait cycle
    ax.axvspan(
        row["cycle_start_s"],
        row["cycle_end_s"],
        alpha=0.08,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlim(
        plot_start,
        plot_end,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "EMG (µV)"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="upper right"
    )

    title = (
        f"{row['patient']} | "
        f"{row['bws_percent']:.0f}% BWS | "
        f"{row['side']} | "
        f"{row['muscle']} | "
        f"Cycle {row['cycle_number']} | "
        f"QC={row['qc']}"
    )

    subtitle = (
        f"Artifact={row['artifact_percent']:.2f}% | "
        f"RMS={row['rms_uv']:.2f} µV | "
        f"Peak={row['peak_abs_uv']:.2f} µV | "
        f"Duration={row['cycle_duration_s']:.2f} s"
    )

    fig.suptitle(
        title
    )

    ax.set_title(
        subtitle,
        fontsize=9,
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SAVE INDEX CSV
# ============================================================

def save_selected_rows(
    rows,
    path,
):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
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

    rows = load_qc_rows(
        QC_CSV
    )

    print(
        f"Loaded {len(rows)} "
        f"muscle-cycle QC rows."
    )

    selected = (
        select_examples(
            rows
        )
    )

    print(
        f"Selected {len(selected)} "
        f"examples."
    )

    # Cache trials so each NPZ is only loaded once
    trial_cache = {}

    successful = 0
    failed = 0

    for i, row in enumerate(
        selected,
        start=1,
    ):

        trial = row[
            "trial"
        ]

        try:

            if trial not in (
                trial_cache
            ):

                trial_cache[
                    trial
                ] = load_emg_trial(
                    trial
                )

            emg = trial_cache[
                trial
            ]

            safe_muscle = (
                row["muscle"]
                .replace(" ", "_")
                .replace("/", "_")
                .replace(".", "")
            )

            filename = (
                f"{row['qc']}_"
                f"{row['patient']}_"
                f"{row['bws_percent']:.0f}bws_"
                f"{row['side']}_"
                f"{safe_muscle}_"
                f"cycle{row['cycle_number']:03d}.png"
            )

            output_path = (
                OUTPUT_FOLDER
                / filename
            )

            plot_qc_example(
                row,
                emg,
                output_path,
            )

            successful += 1

            print(
                f"{i:3d}/{len(selected)} "
                f"{filename}"
            )

        except Exception as exc:

            failed += 1

            print(
                f"ERROR: "
                f"{trial}, "
                f"{row['muscle']}, "
                f"cycle "
                f"{row['cycle_number']}: "
                f"{exc}"
            )

    selected_csv = (
        OUTPUT_FOLDER
        / "selected_qc_examples.csv"
    )

    save_selected_rows(
        selected,
        selected_csv,
    )

    print()
    print("=" * 70)
    print(
        "EMG CYCLE QC VISUALIZATION COMPLETE"
    )
    print("=" * 70)

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
        f"\nSelected examples table:\n"
        f"{selected_csv}"
    )


if __name__ == "__main__":
    main()