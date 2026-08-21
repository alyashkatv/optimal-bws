
#!/usr/bin/env python3

from pathlib import Path
import csv
import random
from collections import defaultdict

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
    "/home/alya/Desktop/optimal-bws/emg_manual_qc_validation/second_batch"
)

PLOTS_FOLDER = OUTPUT_FOLDER / "plots"

REVIEW_CSV = OUTPUT_FOLDER / "manual_qc_review.csv"
KEY_CSV = OUTPUT_FOLDER / "manual_qc_key.csv"

TARGET_N = 240
QC_CLASSES = ("KEEP", "REVIEW", "REJECT")
RANDOM_SEED = 42

# Number of neighboring cycles shown on each side of the target.
N_NEIGHBOR_CYCLES = 1

# Small padding beyond the outer cycles.
OUTER_PADDING_S = 0.10


# ============================================================
# HELPERS
# ============================================================

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def safe_int(x, default=-1):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


# ============================================================
# LOAD QC TABLE
# ============================================================

def load_qc_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"QC CSV not found:\n{path}")

    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("QC CSV is empty.")

    required = {
        "patient",
        "bws_percent",
        "trial",
        "side",
        "cycle_number",
        "muscle",
        "channel_index",
        "cycle_start_s",
        "cycle_end_s",
        "cycle_duration_s",
        "artifact_percent",
        "qc",
    }

    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(
            f"QC CSV missing required columns: {sorted(missing)}"
        )

    for row in rows:
        row["bws_percent"] = safe_float(row["bws_percent"])
        row["cycle_number"] = safe_int(row["cycle_number"])
        row["channel_index"] = safe_int(row["channel_index"])
        row["cycle_start_s"] = safe_float(row["cycle_start_s"])
        row["cycle_end_s"] = safe_float(row["cycle_end_s"])
        row["cycle_duration_s"] = safe_float(row["cycle_duration_s"])
        row["artifact_percent"] = safe_float(row["artifact_percent"])
        row["qc"] = str(row["qc"]).strip().upper()

    return rows


# ============================================================
# LOAD FILTERED EMG
# ============================================================

def load_emg_trial(trial):
    path = EMG_FOLDER / f"{trial}_emg_filtered.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Filtered EMG file not found:\n{path}"
        )

    with np.load(path, allow_pickle=False) as data:
        time = np.asarray(data["time"], dtype=float).reshape(-1)
        filtered = np.asarray(data["filtered"], dtype=float)
        envelope = np.asarray(data["envelope"], dtype=float)
        labels = [
            str(x)
            for x in np.asarray(data["channel_labels"]).reshape(-1)
        ]

    if filtered.shape[1] != len(time):
        if filtered.shape[0] == len(time):
            filtered = filtered.T
            envelope = envelope.T
        else:
            raise ValueError(
                f"{path.name}: invalid EMG shape {filtered.shape}"
            )

    return {
        "time": time,
        "filtered": filtered,
        "envelope": envelope,
        "labels": labels,
    }


# ============================================================
# BALANCED SAMPLING
# ============================================================

def sample_balanced(rows, target_n):
    rng = random.Random(RANDOM_SEED)

    rows = [
        row for row in rows
        if row["qc"] in QC_CLASSES
    ]

    by_qc = defaultdict(list)
    for row in rows:
        by_qc[row["qc"]].append(row)

    target_per_class = {
        qc: target_n // len(QC_CLASSES)
        for qc in QC_CLASSES
    }

    remainder = target_n - sum(target_per_class.values())
    for qc in QC_CLASSES[:remainder]:
        target_per_class[qc] += 1

    selected = []

    for qc in QC_CLASSES:
        class_rows = by_qc.get(qc, [])
        if not class_rows:
            continue

        by_muscle = defaultdict(list)
        for row in class_rows:
            by_muscle[row["muscle"]].append(row)

        for group in by_muscle.values():
            rng.shuffle(group)

        muscles = list(by_muscle.keys())
        rng.shuffle(muscles)

        n_target = min(
            target_per_class[qc],
            len(class_rows),
        )

        chosen = []
        ptr = {m: 0 for m in muscles}

        while len(chosen) < n_target:
            added = False

            for muscle in muscles:
                idx = ptr[muscle]
                group = by_muscle[muscle]

                if idx < len(group):
                    chosen.append(group[idx])
                    ptr[muscle] += 1
                    added = True

                    if len(chosen) >= n_target:
                        break

            if not added:
                break

        selected.extend(chosen)

    if len(selected) < target_n:
        used = {
            (
                r["trial"],
                r["side"],
                r["cycle_number"],
                r["muscle"],
            )
            for r in selected
        }

        remaining = [
            r for r in rows
            if (
                r["trial"],
                r["side"],
                r["cycle_number"],
                r["muscle"],
            )
            not in used
        ]

        rng.shuffle(remaining)
        selected.extend(
            remaining[: target_n - len(selected)]
        )

    rng.shuffle(selected)
    return selected[:target_n]


# ============================================================
# BUILD CYCLE LOOKUP
# ============================================================

def build_cycle_lookup(rows):
    """
    Organize rows by:
        trial + side + muscle

    so previous / target / next cycles can be retrieved.
    """

    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["trial"],
            row["side"],
            row["muscle"],
        )
        grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(
            key=lambda r: (
                r["cycle_start_s"],
                r["cycle_number"],
            )
        )

    return grouped


def get_context_rows(target_row, lookup):
    key = (
        target_row["trial"],
        target_row["side"],
        target_row["muscle"],
    )

    group = lookup.get(key, [])

    if not group:
        return [target_row], 0

    target_idx = None

    for i, row in enumerate(group):
        if (
            row["cycle_number"] == target_row["cycle_number"]
            and np.isclose(
                row["cycle_start_s"],
                target_row["cycle_start_s"],
            )
        ):
            target_idx = i
            break

    if target_idx is None:
        return [target_row], 0

    start = max(
        0,
        target_idx - N_NEIGHBOR_CYCLES,
    )

    end = min(
        len(group),
        target_idx + N_NEIGHBOR_CYCLES + 1,
    )

    context = group[start:end]
    target_pos = target_idx - start

    return context, target_pos


# ============================================================
# PLOTTING
# ============================================================

def plot_context_example(
    example_id,
    target_row,
    context_rows,
    target_pos,
    emg,
    output_path,
):
    """
    Show previous + target + next cycle.

    IMPORTANT:
    - no automatic QC label
    - no artifact percentage
    - no algorithm-flagged dots
    - target cycle is shaded
    """

    channel = target_row["channel_index"]

    if channel < 0 or channel >= emg["filtered"].shape[0]:
        raise IndexError(
            f"Channel {channel} outside EMG shape "
            f"{emg['filtered'].shape}"
        )

    time = emg["time"]

    first = context_rows[0]
    last = context_rows[-1]

    plot_start = max(
        float(time[0]),
        first["cycle_start_s"] - OUTER_PADDING_S,
    )

    plot_end = min(
        float(time[-1]),
        last["cycle_end_s"] + OUTER_PADDING_S,
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

    t = time[start_idx:end_idx]

    filtered = emg["filtered"][
        channel,
        start_idx:end_idx,
    ]

    envelope = emg["envelope"][
        channel,
        start_idx:end_idx,
    ]

    fig, ax = plt.subplots(
        figsize=(15, 5.5),
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
        linewidth=1.6,
        label="Envelope",
    )

    # Draw boundaries for every visible cycle.
    boundaries = []

    for row in context_rows:
        boundaries.extend(
            [
                row["cycle_start_s"],
                row["cycle_end_s"],
            ]
        )

    # Deduplicate near-identical boundaries.
    unique_boundaries = []
    for x in sorted(boundaries):
        if (
            not unique_boundaries
            or not np.isclose(x, unique_boundaries[-1])
        ):
            unique_boundaries.append(x)

    for x in unique_boundaries:
        ax.axvline(
            x,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

    # Highlight only the target cycle.
    ax.axvspan(
        target_row["cycle_start_s"],
        target_row["cycle_end_s"],
        alpha=0.10,
        label="Target cycle",
    )

    # Add small text labels above cycles.
    y_top = ax.get_ylim()[1]

    labels = []
    for i, row in enumerate(context_rows):
        if i < target_pos:
            label = "Previous"
        elif i == target_pos:
            label = "TARGET"
        else:
            label = "Next"

        mid = 0.5 * (
            row["cycle_start_s"]
            + row["cycle_end_s"]
        )

        labels.append((mid, label))

    # Need current ylim after plotting.
    ymin, ymax = ax.get_ylim()
    text_y = ymax - 0.04 * (ymax - ymin)

    for mid, label in labels:
        ax.text(
            mid,
            text_y,
            label,
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold" if label == "TARGET" else "normal",
        )

    ax.set_xlim(plot_start, plot_end)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG (µV)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    ax.set_title(
        f"Manual EMG QC — Example {example_id:03d}"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# OUTPUT CSVs
# ============================================================

def save_review_csv(successful_items, path):
    fieldnames = [
        "example_id",
        "plot_file",
        "manual_qc",
        "notes",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in successful_items:
            writer.writerow(
                {
                    "example_id":
                        f"{item['example_id']:03d}",

                    "plot_file":
                        f"{item['example_id']:03d}.png",

                    "manual_qc":
                        "",

                    "notes":
                        "",
                }
            )


def save_key_csv(successful_items, path):
    fieldnames = [
        "example_id",
        "patient",
        "bws_percent",
        "trial",
        "side",
        "cycle_number",
        "muscle",
        "channel_index",
        "cycle_start_s",
        "cycle_end_s",
        "cycle_duration_s",
        "artifact_percent",
        "auto_qc",
        "n_context_cycles_shown",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in successful_items:
            row = item["row"]

            writer.writerow(
                {
                    "example_id":
                        f"{item['example_id']:03d}",

                    "patient":
                        row["patient"],

                    "bws_percent":
                        row["bws_percent"],

                    "trial":
                        row["trial"],

                    "side":
                        row["side"],

                    "cycle_number":
                        row["cycle_number"],

                    "muscle":
                        row["muscle"],

                    "channel_index":
                        row["channel_index"],

                    "cycle_start_s":
                        row["cycle_start_s"],

                    "cycle_end_s":
                        row["cycle_end_s"],

                    "cycle_duration_s":
                        row["cycle_duration_s"],

                    "artifact_percent":
                        row["artifact_percent"],

                    "auto_qc":
                        row["qc"],

                    "n_context_cycles_shown":
                        len(item["context_rows"]),
                }
            )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_qc_rows(QC_CSV)

    print(f"Loaded {len(rows)} muscle-cycle QC rows.")

    selected = sample_balanced(
        rows,
        TARGET_N,
    )

    lookup = build_cycle_lookup(rows)

    print(f"Selected {len(selected)} examples.")

    selected_counts = {
        qc: sum(
            row["qc"] == qc
            for row in selected
        )
        for qc in QC_CLASSES
    }

    for qc in QC_CLASSES:
        print(
            f"  {qc:6s}: "
            f"{selected_counts[qc]}"
        )

    trial_cache = {}
    successful_items = []
    failed = 0

    for example_id, row in enumerate(
        selected,
        start=1,
    ):
        try:
            trial = row["trial"]

            if trial not in trial_cache:
                trial_cache[trial] = load_emg_trial(
                    trial
                )

            emg = trial_cache[trial]

            context_rows, target_pos = (
                get_context_rows(
                    row,
                    lookup,
                )
            )

            plot_path = (
                PLOTS_FOLDER
                / f"{example_id:03d}.png"
            )

            plot_context_example(
                example_id,
                row,
                context_rows,
                target_pos,
                emg,
                plot_path,
            )

            successful_items.append(
                {
                    "example_id":
                        example_id,

                    "row":
                        row,

                    "context_rows":
                        context_rows,
                }
            )

            print(
                f"[{example_id:03d}/{len(selected):03d}] "
                f"saved {plot_path.name} "
                f"({len(context_rows)} cycles shown)"
            )

        except Exception as exc:
            failed += 1

            print(
                f"[{example_id:03d}] FAILED: "
                f"{row['trial']} | "
                f"{row['muscle']} | "
                f"{exc}"
            )

    save_review_csv(
        successful_items,
        REVIEW_CSV,
    )

    save_key_csv(
        successful_items,
        KEY_CSV,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Plots:      {PLOTS_FOLDER}")
    print(f"Review CSV: {REVIEW_CSV}")
    print(f"Key CSV:    {KEY_CSV}")
    print(f"Failed:     {failed}")
    print()
    print("Manual labels:")
    print("  VALID")
    print("  ARTIFACT")
    print("  UNCERTAIN")
    print()
    print(
        "During review, use only the plots and manual_qc_review.csv."
    )
    print(
        "Do not open manual_qc_key.csv until labeling is complete."
    )


if __name__ == "__main__":
    main()