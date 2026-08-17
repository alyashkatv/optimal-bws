#!/usr/bin/env python3

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/alya/Desktop/optimal-bws")

EVENT_DIRS = [
    PROJECT_ROOT / "filtered_data" / "events_first_batch",
    PROJECT_ROOT / "filtered_data" / "events_second_batch",
    PROJECT_ROOT / "filtered_data" / "events_alya_test",
]

OUTPUT_DIR = PROJECT_ROOT / "gait_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_OUTPUT = OUTPUT_DIR / "spatiotemporal_metrics_by_trial.csv"
PATIENT_OUTPUT = OUTPUT_DIR / "spatiotemporal_metrics_by_patient_bws.csv"


# ============================================================
# SETTINGS
# ============================================================

# Reject obviously incomplete or physiologically impossible cycles.
#
# These are QC bounds, NOT definitions of normal gait.
# They are deliberately broad because some participants walk
# very slowly.
MIN_STRIDE_TIME_S = 0.40
MAX_STRIDE_TIME_S = 10.0

MIN_STANCE_TIME_S = 0.15
MAX_STANCE_TIME_S = 8.0

MIN_SWING_TIME_S = 0.10
MAX_SWING_TIME_S = 8.0

MIN_STEP_TIME_S = 0.10
MAX_STEP_TIME_S = 6.0

MIN_VALID_STRIDES_PER_FOOT = 3


# ============================================================
# EVENT DETECTOR ADAPTER
# ============================================================

def get_events(event_path):
    """
    Load previously detected and validated gait events.

    Returns event times in seconds:
        left_ic
        left_to
        right_ic
        right_to
    """

    with np.load(event_path, allow_pickle=False) as data:

        required = {
            "left_ic_times",
            "left_to_times",
            "right_ic_times",
            "right_to_times",
        }

        missing = required.difference(data.files)

        if missing:
            raise ValueError(
                f"{event_path.name}: missing event arrays: "
                f"{sorted(missing)}"
            )

        left_ic = np.asarray(
            data["left_ic_times"],
            dtype=float,
        )

        left_to = np.asarray(
            data["left_to_times"],
            dtype=float,
        )

        right_ic = np.asarray(
            data["right_ic_times"],
            dtype=float,
        )

        right_to = np.asarray(
            data["right_to_times"],
            dtype=float,
        )

    return (
        left_ic,
        left_to,
        right_ic,
        right_to,
    )

# ============================================================
# FILE/PATIENT INFORMATION
# ============================================================

def parse_trial_name(path):
    """
    Examples:
        ab_0_bws_filtered.npz
        al_25_bws_filtered.npz
        yer_ground_filtered.npz

    Returns:
        patient
        condition
        bws_percent
    """

    stem = path.stem.lower()

    stem = re.sub(r"_(filtered|events)$", "", stem)

    m = re.match(
        r"^(?P<patient>[a-z]+)_(?P<condition>ground|\d+_bws)$",
        stem,
    )

    if not m:
        raise ValueError(f"Cannot parse trial filename: {path.name}")

    patient = m.group("patient").upper()
    condition = m.group("condition")

    if condition == "ground":
        return patient, "Ground", np.nan

    bws = int(condition.replace("_bws", ""))

    return patient, f"{bws}% BWS", bws


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(np.mean(x))


def safe_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    return float(np.std(x, ddof=1))


def coefficient_of_variation(x):
    """
    CV (%) = SD / mean * 100
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    mean = np.mean(x)

    if mean <= 0:
        return np.nan

    return float(100.0 * np.std(x, ddof=1) / mean)


def symmetry_index(left, right):
    """
    Absolute bilateral symmetry index:

        SI = |L - R| / ((L + R)/2) * 100

    0% = perfect symmetry
    higher = greater asymmetry
    """

    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan

    denominator = 0.5 * (left + right)

    if denominator <= 0:
        return np.nan

    return float(abs(left - right) / denominator * 100.0)


def filter_duration(values, lower, upper):
    values = np.asarray(values, dtype=float)

    return values[
        np.isfinite(values)
        & (values >= lower)
        & (values <= upper)
    ]


# ============================================================
# CYCLE CONSTRUCTION
# ============================================================

def construct_cycles(ic, to):
    """
    Constructs complete gait cycles:

        IC_i -> TO_i -> IC_(i+1)

    Returns a DataFrame containing:
        ic
        to
        next_ic
        stride_time
        stance_time
        swing_time
        stance_percent
        swing_percent
    """

    ic = np.sort(np.asarray(ic, dtype=float))
    to = np.sort(np.asarray(to, dtype=float))

    rows = []

    if len(ic) < 2 or len(to) == 0:
        return pd.DataFrame()

    for i in range(len(ic) - 1):

        current_ic = ic[i]
        next_ic = ic[i + 1]

        # TO must occur within this gait cycle.
        candidates = to[
            (to > current_ic)
            & (to < next_ic)
        ]

        if len(candidates) == 0:
            continue

        # Normally there should be exactly one.
        # Use the first valid TO after IC.
        current_to = candidates[0]

        stride_time = next_ic - current_ic
        stance_time = current_to - current_ic
        swing_time = next_ic - current_to

        if not (
            MIN_STRIDE_TIME_S <= stride_time <= MAX_STRIDE_TIME_S
        ):
            continue

        if not (
            MIN_STANCE_TIME_S <= stance_time <= MAX_STANCE_TIME_S
        ):
            continue

        if not (
            MIN_SWING_TIME_S <= swing_time <= MAX_SWING_TIME_S
        ):
            continue

        stance_pct = 100.0 * stance_time / stride_time
        swing_pct = 100.0 * swing_time / stride_time

        rows.append(
            {
                "ic": current_ic,
                "to": current_to,
                "next_ic": next_ic,
                "stride_time": stride_time,
                "stance_time": stance_time,
                "swing_time": swing_time,
                "stance_percent": stance_pct,
                "swing_percent": swing_pct,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP TIMES
# ============================================================

def calculate_step_times(left_ic, right_ic):
    """
    Step time:

    Left step:
        previous RIGHT IC -> LEFT IC

    Right step:
        previous LEFT IC -> RIGHT IC

    Returns:
        left_step_times
        right_step_times
    """

    left_ic = np.sort(np.asarray(left_ic, dtype=float))
    right_ic = np.sort(np.asarray(right_ic, dtype=float))

    left_steps = []
    right_steps = []

    # LEFT step time:
    # latest contralateral IC before each left IC
    for t in left_ic:
        previous = right_ic[right_ic < t]

        if len(previous):
            dt = t - previous[-1]

            if MIN_STEP_TIME_S <= dt <= MAX_STEP_TIME_S:
                left_steps.append(dt)

    # RIGHT step time
    for t in right_ic:
        previous = left_ic[left_ic < t]

        if len(previous):
            dt = t - previous[-1]

            if MIN_STEP_TIME_S <= dt <= MAX_STEP_TIME_S:
                right_steps.append(dt)

    return (
        np.asarray(left_steps, dtype=float),
        np.asarray(right_steps, dtype=float),
    )


# ============================================================
# CONTACT STATE / SUPPORT TIMES
# ============================================================

def build_contact_intervals(ic, to):
    """
    Returns stance intervals:

        [(IC_1, TO_1), (IC_2, TO_2), ...]
    """

    ic = np.sort(np.asarray(ic, dtype=float))
    to = np.sort(np.asarray(to, dtype=float))

    intervals = []

    for start in ic:

        candidates = to[to > start]

        if len(candidates) == 0:
            continue

        end = candidates[0]

        if end > start:
            intervals.append((start, end))

    return intervals


def contact_state_at(t, intervals):
    for start, end in intervals:
        if start <= t < end:
            return True

    return False


def calculate_support_percentages(
    left_ic,
    left_to,
    right_ic,
    right_to,
):
    """
    Determine how much of the common valid recording interval is:

        left single support
        right single support
        double support

    Uses exact event-to-event intervals, not sample-based
    approximation.
    """

    left_intervals = build_contact_intervals(left_ic, left_to)
    right_intervals = build_contact_intervals(right_ic, right_to)

    if not left_intervals or not right_intervals:
        return {
            "single_support_left_s": np.nan,
            "single_support_right_s": np.nan,
            "double_support_s": np.nan,
            "single_support_left_percent": np.nan,
            "single_support_right_percent": np.nan,
            "double_support_percent": np.nan,
        }

    start_time = max(
        left_intervals[0][0],
        right_intervals[0][0],
    )

    end_time = min(
        left_intervals[-1][1],
        right_intervals[-1][1],
    )

    if end_time <= start_time:
        return {
            "single_support_left_s": np.nan,
            "single_support_right_s": np.nan,
            "double_support_s": np.nan,
            "single_support_left_percent": np.nan,
            "single_support_right_percent": np.nan,
            "double_support_percent": np.nan,
        }

    boundaries = [start_time, end_time]

    for start, end in left_intervals + right_intervals:
        if start_time < start < end_time:
            boundaries.append(start)

        if start_time < end < end_time:
            boundaries.append(end)

    boundaries = np.unique(boundaries)
    boundaries.sort()

    left_single = 0.0
    right_single = 0.0
    double_support = 0.0

    for a, b in zip(boundaries[:-1], boundaries[1:]):

        midpoint = 0.5 * (a + b)

        left_contact = contact_state_at(
            midpoint,
            left_intervals,
        )

        right_contact = contact_state_at(
            midpoint,
            right_intervals,
        )

        dt = b - a

        if left_contact and right_contact:
            double_support += dt

        elif left_contact and not right_contact:
            left_single += dt

        elif right_contact and not left_contact:
            right_single += dt

    total = end_time - start_time

    return {
        "single_support_left_s":
            left_single,

        "single_support_right_s":
            right_single,

        "double_support_s":
            double_support,

        "single_support_left_percent":
            100.0 * left_single / total,

        "single_support_right_percent":
            100.0 * right_single / total,

        "double_support_percent":
            100.0 * double_support / total,
    }


# ============================================================
# CADENCE
# ============================================================

def calculate_cadence(left_ic, right_ic):
    """
    Cadence = total detected steps / analysis duration * 60.

    Uses all IC events after restricting to the common event
    interval.
    """

    all_ic = np.sort(
        np.concatenate(
            [
                np.asarray(left_ic, dtype=float),
                np.asarray(right_ic, dtype=float),
            ]
        )
    )

    if len(all_ic) < 3:
        return np.nan

    duration = all_ic[-1] - all_ic[0]

    if duration <= 0:
        return np.nan

    # Number of step intervals rather than number of timestamps.
    n_steps = len(all_ic) - 1

    return float(60.0 * n_steps / duration)


# ============================================================
# ONE TRIAL
# ============================================================

def analyse_trial(npz_path):

    patient, condition, bws = parse_trial_name(npz_path)

    (
        left_ic,
        left_to,
        right_ic,
        right_to,
    ) = get_events(npz_path)

    left_cycles = construct_cycles(
        left_ic,
        left_to,
    )

    right_cycles = construct_cycles(
        right_ic,
        right_to,
    )

    left_step_times, right_step_times = calculate_step_times(
        left_ic,
        right_ic,
    )

    support = calculate_support_percentages(
        left_ic,
        left_to,
        right_ic,
        right_to,
    )

    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    stride_l = safe_mean(
        left_cycles.get("stride_time", [])
    )
    stride_r = safe_mean(
        right_cycles.get("stride_time", [])
    )

    stance_l = safe_mean(
        left_cycles.get("stance_time", [])
    )
    stance_r = safe_mean(
        right_cycles.get("stance_time", [])
    )

    swing_l = safe_mean(
        left_cycles.get("swing_time", [])
    )
    swing_r = safe_mean(
        right_cycles.get("swing_time", [])
    )

    stance_pct_l = safe_mean(
        left_cycles.get("stance_percent", [])
    )
    stance_pct_r = safe_mean(
        right_cycles.get("stance_percent", [])
    )

    swing_pct_l = safe_mean(
        left_cycles.get("swing_percent", [])
    )
    swing_pct_r = safe_mean(
        right_cycles.get("swing_percent", [])
    )

    step_l = safe_mean(left_step_times)
    step_r = safe_mean(right_step_times)

    # --------------------------------------------------------
    # Variability
    # --------------------------------------------------------

    stride_cv_l = coefficient_of_variation(
        left_cycles.get("stride_time", [])
    )

    stride_cv_r = coefficient_of_variation(
        right_cycles.get("stride_time", [])
    )

    stance_cv_l = coefficient_of_variation(
        left_cycles.get("stance_time", [])
    )

    stance_cv_r = coefficient_of_variation(
        right_cycles.get("stance_time", [])
    )

    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    step_si = symmetry_index(
        step_l,
        step_r,
    )

    stance_si = symmetry_index(
        stance_l,
        stance_r,
    )

    single_support_si = symmetry_index(
        support["single_support_left_percent"],
        support["single_support_right_percent"],
    )

    # --------------------------------------------------------
    # Trial QC
    # --------------------------------------------------------

    n_left = len(left_cycles)
    n_right = len(right_cycles)

    if (
        n_left >= MIN_VALID_STRIDES_PER_FOOT
        and n_right >= MIN_VALID_STRIDES_PER_FOOT
    ):
        qc = "pass"
    else:
        qc = "review"

    return {
        "patient": patient,
        "condition": condition,
        "bws_percent": bws,
        "source_file": npz_path.name,

        # Event counts
        "ic_count_left": len(left_ic),
        "to_count_left": len(left_to),
        "ic_count_right": len(right_ic),
        "to_count_right": len(right_to),

        # Complete cycles
        "valid_strides_left": n_left,
        "valid_strides_right": n_right,

        # Temporal metrics
        "stride_time_left_s": stride_l,
        "stride_time_right_s": stride_r,

        "step_time_left_s": step_l,
        "step_time_right_s": step_r,

        "stance_time_left_s": stance_l,
        "stance_time_right_s": stance_r,

        "swing_time_left_s": swing_l,
        "swing_time_right_s": swing_r,

        "stance_percent_left": stance_pct_l,
        "stance_percent_right": stance_pct_r,

        "swing_percent_left": swing_pct_l,
        "swing_percent_right": swing_pct_r,

        # Support
        **support,

        # Cadence
        "cadence_steps_per_min":
            calculate_cadence(left_ic, right_ic),

        # Symmetry
        "step_time_SI_percent": step_si,
        "stance_time_SI_percent": stance_si,
        "single_support_SI_percent":
            single_support_si,

        # Variability
        "stride_time_CV_left_percent":
            stride_cv_l,

        "stride_time_CV_right_percent":
            stride_cv_r,

        "stance_time_CV_left_percent":
            stance_cv_l,

        "stance_time_CV_right_percent":
            stance_cv_r,

        # QC
        "gait_metrics_qc": qc,
    }


# ============================================================
# PATIENT/BWS SUMMARY
# ============================================================

def make_patient_summary(df):
    """
    Since there is currently normally one trial per patient/BWS
    condition, this mostly gives a cleaner comparison table.

    If later you have repeated trials, numeric columns are
    averaged automatically.
    """

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    exclude = {
        "bws_percent",
        "ic_count_left",
        "to_count_left",
        "ic_count_right",
        "to_count_right",
        "valid_strides_left",
        "valid_strides_right",
    }

    metric_cols = [
        x for x in numeric_cols
        if x not in exclude
    ]

    summary = (
        df.groupby(
            [
                "patient",
                "condition",
                "bws_percent",
            ],
            dropna=False,
        )[metric_cols]
        .mean()
        .reset_index()
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    files = []

    for directory in EVENT_DIRS:

        if not directory.exists():
            print(
                f"WARNING: directory does not exist: "
                f"{directory}"
            )
            continue

        files.extend(
            sorted(directory.glob("*_events.npz")) 
        )

    # Remove duplicates if any.
    files = sorted(set(files))

    print(f"Found {len(files)} filtered trials.")

    if not files:
        print("No filtered NPZ files found.")
        return

    rows = []

    failed = []

    for path in files:

        print(f"Processing: {path.name}")

        try:
            result = analyse_trial(path)
            rows.append(result)

            print(
                f"  {result['patient']:>5} | "
                f"{result['condition']:<8} | "
                f"L strides={result['valid_strides_left']:>3} | "
                f"R strides={result['valid_strides_right']:>3} | "
                f"QC={result['gait_metrics_qc']}"
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            failed.append(
                {
                    "file": path.name,
                    "error": str(exc),
                }
            )

    if not rows:
        print(
            "\nNo trials were successfully analysed."
        )
        return

    df = pd.DataFrame(rows)

    # Patient first, then BWS.
    df = df.sort_values(
        ["patient", "bws_percent"],
        na_position="last",
    )

    df.to_csv(
        TRIAL_OUTPUT,
        index=False,
        float_format="%.6f",
    )

    patient_summary = make_patient_summary(df)

    patient_summary.to_csv(
        PATIENT_OUTPUT,
        index=False,
        float_format="%.6f",
    )

    print()
    print("=" * 60)
    print("GAIT ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"Successful : {len(rows)}")
    print(f"Failed     : {len(failed)}")

    print()
    print("Trial-level metrics:")
    print(TRIAL_OUTPUT)

    print()
    print("Patient/BWS comparison:")
    print(PATIENT_OUTPUT)

    if failed:

        failure_path = (
            OUTPUT_DIR / "gait_metrics_failures.csv"
        )

        pd.DataFrame(failed).to_csv(
            failure_path,
            index=False,
        )

        print()
        print("Failures:")
        print(failure_path)


if __name__ == "__main__":
    main()