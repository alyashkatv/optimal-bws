#!/usr/bin/env python3

from pathlib import Path
import csv
import re

import numpy as np


# ============================================================
# CONFIG
# ============================================================

EMG_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/emg_second_batch"
)

EVENTS_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/events_second_batch"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/emg_cycle_qc/second_batch"
)

CYCLE_CSV = (
    OUTPUT_FOLDER
    / "emg_cycle_qc_by_muscle.csv"
)

SUMMARY_CSV = (
    OUTPUT_FOLDER
    / "emg_cycle_qc_summary.csv"
)


# ------------------------------------------------------------
# QC thresholds
#
# These are screening thresholds.
# We are NOT automatically deleting data here.
# ------------------------------------------------------------

REVIEW_ARTIFACT_FRACTION = 0.02
REJECT_ARTIFACT_FRACTION = 0.05

# Reject obviously implausible IC-to-IC intervals.
# Kept broad because your participants include slow CP gait.
MIN_GAIT_CYCLE_S = 0.40
MAX_GAIT_CYCLE_S = 5.00


# ============================================================
# NAME HANDLING
# ============================================================

def canonical_trial_stem(path):
    """
    Convert:

        ar_20_bws_emg_filtered.npz

    into:

        ar_20_bws
    """

    stem = path.stem

    suffixes = [
        "_emg_filtered",
        "_filtered",
        "_cleaned",
        "_cropped",
    ]

    for suffix in suffixes:
        if stem.lower().endswith(
            suffix
        ):
            stem = stem[
                :-len(suffix)
            ]
            break

    return stem


def parse_trial_name(stem):
    """
    Examples:

        al_0_bws
        ar_25_bws
        ti_30_bws

    Returns:
        patient, bws_percent
    """

    match = re.fullmatch(
        r"([A-Za-z]+)_(\d+)_bws",
        stem,
    )

    if not match:
        return (
            stem.upper(),
            np.nan,
        )

    patient = (
        match.group(1)
        .upper()
    )

    bws = int(
        match.group(2)
    )

    return patient, bws


# ============================================================
# MUSCLE SIDE
# ============================================================

def muscle_side(label):
    """
    Infer left/right from cleaned channel label.
    """

    text = (
        str(label)
        .strip()
        .lower()
    )

    if (
        text.startswith("l ")
        or text.startswith("left ")
        or text.startswith("l.")
    ):
        return "Left"

    if (
        text.startswith("r ")
        or text.startswith("right ")
        or text.startswith("r.")
    ):
        return "Right"

    return "Unknown"


# ============================================================
# LOAD FILTERED EMG
# ============================================================

def load_filtered_emg(path):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "time",
            "filtered",
            "artifact_mask",
            "channel_labels",
        }

        missing = (
            required
            .difference(
                data.files
            )
        )

        if missing:
            raise ValueError(
                f"{path.name}: "
                f"missing keys "
                f"{sorted(missing)}"
            )

        time = np.asarray(
            data["time"],
            dtype=float,
        ).reshape(-1)

        filtered = np.asarray(
            data["filtered"],
            dtype=float,
        )

        artifact_mask = np.asarray(
            data[
                "artifact_mask"
            ],
            dtype=bool,
        )

        labels = [
            str(x)
            for x
            in np.asarray(
                data[
                    "channel_labels"
                ]
            ).reshape(-1)
        ]

        if "fs" in data.files:
            fs = float(
                np.asarray(
                    data["fs"]
                ).reshape(-1)[0]
            )

        elif (
            "sampling_rate"
            in data.files
        ):
            fs = float(
                np.asarray(
                    data[
                        "sampling_rate"
                    ]
                ).reshape(-1)[0]
            )

        else:
            if len(time) < 2:
                raise ValueError(
                    f"{path.name}: "
                    f"cannot infer sampling rate"
                )

            dt = np.median(
                np.diff(time)
            )

            fs = 1.0 / dt

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if filtered.ndim != 2:
        raise ValueError(
            f"{path.name}: "
            f"filtered EMG must be 2-D"
        )

    if (
        artifact_mask.shape
        != filtered.shape
    ):
        raise ValueError(
            f"{path.name}: "
            f"artifact mask shape "
            f"{artifact_mask.shape} "
            f"does not match EMG "
            f"{filtered.shape}"
        )

    # Expected:
    # channels x samples
    if (
        filtered.shape[1]
        != len(time)
    ):
        if (
            filtered.shape[0]
            == len(time)
        ):
            filtered = (
                filtered.T
            )

            artifact_mask = (
                artifact_mask.T
            )

        else:
            raise ValueError(
                f"{path.name}: "
                f"time/EMG dimensions "
                f"do not match"
            )

    if (
        len(labels)
        != filtered.shape[0]
    ):
        raise ValueError(
            f"{path.name}: "
            f"{len(labels)} labels but "
            f"{filtered.shape[0]} channels"
        )

    return {
        "time":
            time,

        "filtered":
            filtered,

        "artifact_mask":
            artifact_mask,

        "labels":
            labels,

        "fs":
            fs,
    }


# ============================================================
# LOAD EVENTS
# ============================================================

def load_events(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Event file not found:\n"
            f"{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "left_ic_times",
            "right_ic_times",
            "left_to_times",
            "right_to_times",
        }

        missing = (
            required
            .difference(
                data.files
            )
        )

        if missing:
            raise ValueError(
                f"{path.name}: "
                f"missing event keys "
                f"{sorted(missing)}"
            )

        return {
            "left_ic_times":
                np.asarray(
                    data[
                        "left_ic_times"
                    ],
                    dtype=float,
                ).reshape(-1),

            "right_ic_times":
                np.asarray(
                    data[
                        "right_ic_times"
                    ],
                    dtype=float,
                ).reshape(-1),

            "left_to_times":
                np.asarray(
                    data[
                        "left_to_times"
                    ],
                    dtype=float,
                ).reshape(-1),

            "right_to_times":
                np.asarray(
                    data[
                        "right_to_times"
                    ],
                    dtype=float,
                ).reshape(-1),
        }


# ============================================================
# BUILD IC -> NEXT IC CYCLES
# ============================================================

def build_gait_cycles(
    ic_times
):
    """
    Build complete gait cycles:

        IC_i -> IC_(i+1)

    Returns:
        list of:
            (cycle_number,
             start_time,
             end_time,
             duration)
    """

    ic_times = np.asarray(
        ic_times,
        dtype=float,
    )

    ic_times = ic_times[
        np.isfinite(
            ic_times
        )
    ]

    if len(ic_times) < 2:
        return []

    cycles = []

    cycle_number = 1

    for i in range(
        len(ic_times) - 1
    ):

        start = float(
            ic_times[i]
        )

        end = float(
            ic_times[i + 1]
        )

        duration = (
            end - start
        )

        if (
            duration
            < MIN_GAIT_CYCLE_S
            or duration
            > MAX_GAIT_CYCLE_S
        ):
            continue

        cycles.append(
            (
                cycle_number,
                start,
                end,
                duration,
            )
        )

        cycle_number += 1

    return cycles


# ============================================================
# TIME -> INDICES
# ============================================================

def time_window_to_indices(
    time,
    start_s,
    end_s,
):

    start_idx = int(
        np.searchsorted(
            time,
            start_s,
            side="left",
        )
    )

    end_idx = int(
        np.searchsorted(
            time,
            end_s,
            side="right",
        )
    )

    start_idx = max(
        0,
        min(
            start_idx,
            len(time) - 1,
        ),
    )

    end_idx = max(
        start_idx + 1,
        min(
            end_idx,
            len(time),
        ),
    )

    return (
        start_idx,
        end_idx,
    )


# ============================================================
# CYCLE METRICS
# ============================================================

def calculate_cycle_metrics(
    signal,
    artifact_mask,
):

    signal = np.asarray(
        signal,
        dtype=float,
    )

    artifact_mask = np.asarray(
        artifact_mask,
        dtype=bool,
    )

    finite = np.isfinite(
        signal
    )

    if not np.any(
        finite
    ):
        return None

    x = signal[
        finite
    ]

    local_artifacts = (
        artifact_mask[
            finite
        ]
    )

    rms = float(
        np.sqrt(
            np.mean(
                x ** 2
            )
        )
    )

    peak_abs = float(
        np.max(
            np.abs(
                x
            )
        )
    )

    mean_abs = float(
        np.mean(
            np.abs(
                x
            )
        )
    )

    artifact_fraction = float(
        np.mean(
            local_artifacts
        )
    )

    return {
        "rms_uv":
            rms,

        "peak_abs_uv":
            peak_abs,

        "mean_abs_uv":
            mean_abs,

        "artifact_fraction":
            artifact_fraction,

        "artifact_percent":
            100.0
            * artifact_fraction,
    }


# ============================================================
# QC CLASSIFICATION
# ============================================================

def classify_cycle(
    artifact_fraction,
):

    if (
        artifact_fraction
        >= REJECT_ARTIFACT_FRACTION
    ):
        return "REJECT"

    if (
        artifact_fraction
        >= REVIEW_ARTIFACT_FRACTION
    ):
        return "REVIEW"

    return "KEEP"


# ============================================================
# PROCESS ONE SIDE
# ============================================================

def process_side(
    patient,
    bws,
    trial,
    side,
    ic_times,
    time,
    filtered,
    artifact_mask,
    labels,
):

    rows = []

    cycles = build_gait_cycles(
        ic_times
    )

    side_channel_indices = [
        i
        for i, label
        in enumerate(labels)
        if muscle_side(
            label
        ) == side
    ]

    if not side_channel_indices:
        print(
            f"  WARNING: "
            f"no {side} EMG channels"
        )

        return rows

    for (
        cycle_number,
        start_s,
        end_s,
        duration_s,
    ) in cycles:

        (
            start_idx,
            end_idx,
        ) = time_window_to_indices(
            time,
            start_s,
            end_s,
        )

        for channel_index in (
            side_channel_indices
        ):

            signal = (
                filtered[
                    channel_index,
                    start_idx:end_idx,
                ]
            )

            local_mask = (
                artifact_mask[
                    channel_index,
                    start_idx:end_idx,
                ]
            )

            metrics = (
                calculate_cycle_metrics(
                    signal,
                    local_mask,
                )
            )

            if metrics is None:
                continue

            qc = classify_cycle(
                metrics[
                    "artifact_fraction"
                ]
            )

            rows.append(
                {
                    "patient":
                        patient,

                    "bws_percent":
                        bws,

                    "trial":
                        trial,

                    "side":
                        side,

                    "cycle_number":
                        cycle_number,

                    "muscle":
                        labels[
                            channel_index
                        ],

                    "channel_index":
                        channel_index,

                    "cycle_start_s":
                        start_s,

                    "cycle_end_s":
                        end_s,

                    "cycle_duration_s":
                        duration_s,

                    **metrics,

                    "qc":
                        qc,
                }
            )

    return rows


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(
    emg_path
):

    trial = (
        canonical_trial_stem(
            emg_path
        )
    )

    patient, bws = (
        parse_trial_name(
            trial
        )
    )

    event_path = (
        EVENTS_FOLDER
        / f"{trial}_events.npz"
    )

    print()
    print(
        f"Processing: {trial}"
    )

    print(
        f"  EMG:    "
        f"{emg_path.name}"
    )

    print(
        f"  Events: "
        f"{event_path.name}"
    )

    emg = load_filtered_emg(
        emg_path
    )

    events = load_events(
        event_path
    )

    rows = []

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    rows.extend(
        process_side(
            patient=
                patient,

            bws=
                bws,

            trial=
                trial,

            side=
                "Left",

            ic_times=
                events[
                    "left_ic_times"
                ],

            time=
                emg[
                    "time"
                ],

            filtered=
                emg[
                    "filtered"
                ],

            artifact_mask=
                emg[
                    "artifact_mask"
                ],

            labels=
                emg[
                    "labels"
                ],
        )
    )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    rows.extend(
        process_side(
            patient=
                patient,

            bws=
                bws,

            trial=
                trial,

            side=
                "Right",

            ic_times=
                events[
                    "right_ic_times"
                ],

            time=
                emg[
                    "time"
                ],

            filtered=
                emg[
                    "filtered"
                ],

            artifact_mask=
                emg[
                    "artifact_mask"
                ],

            labels=
                emg[
                    "labels"
                ],
        )
    )

    return rows


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    rows
):

    if not rows:
        return []

    grouped = {}

    for row in rows:

        key = (
            row["patient"],
            row["bws_percent"],
            row["trial"],
            row["side"],
            row["muscle"],
        )

        grouped.setdefault(
            key,
            [],
        )

        grouped[key].append(
            row
        )

    summary_rows = []

    for (
        patient,
        bws,
        trial,
        side,
        muscle,
    ), group in grouped.items():

        n_total = len(
            group
        )

        n_keep = sum(
            row["qc"] == "KEEP"
            for row in group
        )

        n_review = sum(
            row["qc"] == "REVIEW"
            for row in group
        )

        n_reject = sum(
            row["qc"] == "REJECT"
            for row in group
        )

        keep_fraction = (
            n_keep
            / n_total
            if n_total > 0
            else np.nan
        )

        artifact_values = np.asarray(
            [
                row[
                    "artifact_percent"
                ]
                for row in group
            ],
            dtype=float,
        )

        rms_keep = np.asarray(
            [
                row["rms_uv"]
                for row in group
                if row["qc"]
                == "KEEP"
            ],
            dtype=float,
        )

        summary_rows.append(
            {
                "patient":
                    patient,

                "bws_percent":
                    bws,

                "trial":
                    trial,

                "side":
                    side,

                "muscle":
                    muscle,

                "n_cycles_total":
                    n_total,

                "n_keep":
                    n_keep,

                "n_review":
                    n_review,

                "n_reject":
                    n_reject,

                "keep_fraction":
                    keep_fraction,

                "keep_percent":
                    100.0
                    * keep_fraction,

                "mean_artifact_percent":
                    float(
                        np.nanmean(
                            artifact_values
                        )
                    ),

                "median_artifact_percent":
                    float(
                        np.nanmedian(
                            artifact_values
                        )
                    ),

                "mean_rms_keep_uv":
                    (
                        float(
                            np.mean(
                                rms_keep
                            )
                        )
                        if len(
                            rms_keep
                        )
                        else np.nan
                    ),

                "median_rms_keep_uv":
                    (
                        float(
                            np.median(
                                rms_keep
                            )
                        )
                        if len(
                            rms_keep
                        )
                        else np.nan
                    ),
            }
        )

    return summary_rows


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path,
    rows,
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

    files = sorted(
        EMG_FOLDER.glob(
            "*_emg_filtered.npz"
        )
    )

    if not files:
        raise RuntimeError(
            f"No filtered EMG files "
            f"found in:\n"
            f"{EMG_FOLDER}"
        )

    print(
        f"Found {len(files)} "
        f"filtered EMG trials."
    )

    all_rows = []

    successful = 0
    failed = 0

    for path in files:

        try:

            rows = process_trial(
                path
            )

            all_rows.extend(
                rows
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"\nERROR: "
                f"{path.name}"
            )

            print(
                f"  {exc}"
            )

    # --------------------------------------------------------
    # Save cycle-level table
    # --------------------------------------------------------

    save_csv(
        CYCLE_CSV,
        all_rows,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_rows = (
        build_summary(
            all_rows
        )
    )

    save_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    if all_rows:

        n_keep = sum(
            row["qc"] == "KEEP"
            for row in all_rows
        )

        n_review = sum(
            row["qc"] == "REVIEW"
            for row in all_rows
        )

        n_reject = sum(
            row["qc"] == "REJECT"
            for row in all_rows
        )

        n_total = len(
            all_rows
        )

        print()
        print("=" * 70)
        print(
            "EMG GAIT-CYCLE QC COMPLETE"
        )
        print("=" * 70)

        print(
            f"Trials successful : "
            f"{successful}"
        )

        print(
            f"Trials failed     : "
            f"{failed}"
        )

        print()

        print(
            f"Muscle-cycles total : "
            f"{n_total}"
        )

        print(
            f"KEEP               : "
            f"{n_keep} "
            f"({100*n_keep/n_total:.1f}%)"
        )

        print(
            f"REVIEW             : "
            f"{n_review} "
            f"({100*n_review/n_total:.1f}%)"
        )

        print(
            f"REJECT             : "
            f"{n_reject} "
            f"({100*n_reject/n_total:.1f}%)"
        )

    print()
    print(
        f"Cycle-level QC:\n"
        f"{CYCLE_CSV}"
    )

    print()
    print(
        f"Summary:\n"
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()