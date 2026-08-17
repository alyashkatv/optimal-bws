#!/usr/bin/env python3

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FILTERED_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/filtered_second_batch"
)

EVENTS_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/events_second_batch"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/event_validation_second_batch"
)

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128


# ============================================================
# FILE NAME UTILITIES
# ============================================================

def canonical_trial_stem(path):
    stem = path.stem

    for suffix in (
        "_filtered",
        "_events",
        "_cleaned",
        "_cropped",
    ):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem


def parse_trial_name(path):
    stem = canonical_trial_stem(
        path
    ).lower()

    match = re.fullmatch(
        r"([a-z]+)_(?:(\d+)_bws|(ground))",
        stem,
    )

    if not match:
        return None

    patient = match.group(1)

    if match.group(2) is not None:
        condition = int(
            match.group(2)
        )
    else:
        condition = "ground"

    return patient, condition


def condition_label(condition):
    if condition == "ground":
        return "Ground"

    return f"{condition}% BWS"


# ============================================================
# LOAD FILTERED PRESSURE
# ============================================================

def load_filtered_npz(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        pressure = np.asarray(
            data["pressure"],
            dtype=float,
        )

        time = np.asarray(
            data["time"],
            dtype=float,
        ).reshape(-1)

        fs = float(
            np.asarray(
                data["sampling_rate"]
            ).reshape(-1)[0]
        )

    if pressure.ndim != 2:
        raise ValueError(
            f"{path.name}: pressure must be 2-D"
        )

    if pressure.shape[1] != TOTAL_SENSORS:

        if pressure.shape[0] == TOTAL_SENSORS:
            pressure = pressure.T

        else:
            raise ValueError(
                f"{path.name}: expected 128 sensors, "
                f"got {pressure.shape}"
            )

    if len(time) != len(pressure):
        raise ValueError(
            f"{path.name}: time and pressure lengths differ"
        )

    pressure = np.clip(
        pressure,
        0,
        None,
    )

    left = pressure[:, :64]
    right = pressure[:, 64:]

    return (
        time,
        left,
        right,
        fs,
    )


# ============================================================
# LOAD EVENT DETECTION OUTPUT
# ============================================================

def load_events(path):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        result = {
            "left_total_pressure": np.asarray(
                data["left_total_pressure"],
                dtype=float,
            ),

            "right_total_pressure": np.asarray(
                data["right_total_pressure"],
                dtype=float,
            ),

            "left_active_sensor_count": np.asarray(
                data["left_active_sensor_count"],
                dtype=float,
            ),

            "right_active_sensor_count": np.asarray(
                data["right_active_sensor_count"],
                dtype=float,
            ),

            "left_contact_mask": np.asarray(
                data["left_contact_mask"],
                dtype=bool,
            ),

            "right_contact_mask": np.asarray(
                data["right_contact_mask"],
                dtype=bool,
            ),

            "left_ic_times": np.asarray(
                data["left_ic_times"],
                dtype=float,
            ),

            "left_to_times": np.asarray(
                data["left_to_times"],
                dtype=float,
            ),

            "right_ic_times": np.asarray(
                data["right_ic_times"],
                dtype=float,
            ),

            "right_to_times": np.asarray(
                data["right_to_times"],
                dtype=float,
            ),

            "left_threshold_on": float(
                np.asarray(
                    data["left_threshold_on"]
                ).reshape(-1)[0]
            ),

            "left_threshold_off": float(
                np.asarray(
                    data["left_threshold_off"]
                ).reshape(-1)[0]
            ),

            "right_threshold_on": float(
                np.asarray(
                    data["right_threshold_on"]
                ).reshape(-1)[0]
            ),

            "right_threshold_off": float(
                np.asarray(
                    data["right_threshold_off"]
                ).reshape(-1)[0]
            ),
        }

    return result


# ============================================================
# CONTACT SHADING
# ============================================================

def shade_contact_regions(
    ax,
    time,
    contact_mask,
):
    """
    Shade periods where the detector says the foot
    is in contact with the ground.
    """

    if len(contact_mask) == 0:
        return

    transitions = np.diff(
        contact_mask.astype(int)
    )

    starts = list(
        np.where(
            transitions == 1
        )[0] + 1
    )

    ends = list(
        np.where(
            transitions == -1
        )[0] + 1
    )

    if contact_mask[0]:
        starts.insert(
            0,
            0,
        )

    if contact_mask[-1]:
        ends.append(
            len(contact_mask) - 1
        )

    for start, end in zip(
        starts,
        ends,
    ):
        ax.axvspan(
            time[start],
            time[end],
            alpha=0.08,
        )


# ============================================================
# EVENT MARKERS
# ============================================================

def plot_event_markers(
    ax,
    ic_times,
    to_times,
):
    first_ic = True
    first_to = True

    for t in ic_times:

        ax.axvline(
            t,
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            label=(
                "Initial Contact (IC)"
                if first_ic
                else None
            ),
        )

        first_ic = False

    for t in to_times:

        ax.axvline(
            t,
            linestyle=":",
            linewidth=0.8,
            alpha=0.75,
            label=(
                "Toe Off (TO)"
                if first_to
                else None
            ),
        )

        first_to = False


# ============================================================
# PLOT ONE FOOT
# ============================================================

def plot_foot_validation(
    ax,
    time,
    total_pressure,
    active_sensor_count,
    contact_mask,
    ic_times,
    to_times,
    threshold_on,
    threshold_off,
    foot_name,
):
    # --------------------------------------------------------
    # Main loading signal
    # --------------------------------------------------------

    ax.plot(
        time,
        total_pressure,
        linewidth=1,
        label="Summed pressure",
    )

    # --------------------------------------------------------
    # Contact thresholds
    # --------------------------------------------------------

    ax.axhline(
        threshold_on,
        linestyle="--",
        linewidth=1,
        label="Contact ON threshold",
    )

    ax.axhline(
        threshold_off,
        linestyle=":",
        linewidth=1,
        label="Contact OFF threshold",
    )

    # --------------------------------------------------------
    # Contact periods
    # --------------------------------------------------------

    shade_contact_regions(
        ax,
        time,
        contact_mask,
    )

    # --------------------------------------------------------
    # IC / TO
    # --------------------------------------------------------

    plot_event_markers(
        ax,
        ic_times,
        to_times,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(
        f"{foot_name} — "
        f"IC={len(ic_times)}, "
        f"TO={len(to_times)}"
    )

    ax.set_ylabel(
        "Summed pressure across 64 sensors (kPa)"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="upper right"
    )


# ============================================================
# ACTIVE SENSOR PLOT
# ============================================================

def plot_active_sensor_count(
    ax,
    time,
    left_count,
    right_count,
):
    ax.plot(
        time,
        left_count,
        linewidth=1,
        label="Left active sensors",
    )

    ax.plot(
        time,
        right_count,
        linewidth=1,
        label="Right active sensors",
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Active sensor count"
    )

    ax.set_ylim(
        0,
        64,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()


# ============================================================
# PLOT ONE TRIAL
# ============================================================

def process_trial(
    filtered_path,
    event_path,
):
    parsed = parse_trial_name(
        filtered_path
    )

    if parsed is None:
        raise ValueError(
            f"Cannot parse trial name "
            f"{filtered_path.name}"
        )

    patient, condition = parsed

    (
        time,
        left_pressure,
        right_pressure,
        fs,
    ) = load_filtered_npz(
        filtered_path
    )

    events = load_events(
        event_path
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if len(
        events["left_total_pressure"]
    ) != len(time):

        raise ValueError(
            f"{filtered_path.name}: event signal length "
            f"does not match filtered signal length"
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(16, 11),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[
            2,
            2,
            1,
        ],
    )

    ax_left = fig.add_subplot(
        gs[0]
    )

    ax_right = fig.add_subplot(
        gs[1],
        sharex=ax_left,
    )

    ax_active = fig.add_subplot(
        gs[2],
        sharex=ax_left,
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    plot_foot_validation(
        ax=ax_left,
        time=time,
        total_pressure=events[
            "left_total_pressure"
        ],
        active_sensor_count=events[
            "left_active_sensor_count"
        ],
        contact_mask=events[
            "left_contact_mask"
        ],
        ic_times=events[
            "left_ic_times"
        ],
        to_times=events[
            "left_to_times"
        ],
        threshold_on=events[
            "left_threshold_on"
        ],
        threshold_off=events[
            "left_threshold_off"
        ],
        foot_name="LEFT",
    )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    plot_foot_validation(
        ax=ax_right,
        time=time,
        total_pressure=events[
            "right_total_pressure"
        ],
        active_sensor_count=events[
            "right_active_sensor_count"
        ],
        contact_mask=events[
            "right_contact_mask"
        ],
        ic_times=events[
            "right_ic_times"
        ],
        to_times=events[
            "right_to_times"
        ],
        threshold_on=events[
            "right_threshold_on"
        ],
        threshold_off=events[
            "right_threshold_off"
        ],
        foot_name="RIGHT",
    )

    # --------------------------------------------------------
    # ACTIVE SENSOR COUNT
    # --------------------------------------------------------

    plot_active_sensor_count(
        ax=ax_active,
        time=time,
        left_count=events[
            "left_active_sensor_count"
        ],
        right_count=events[
            "right_active_sensor_count"
        ],
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    fig.suptitle(
        f"{patient.upper()} — "
        f"{condition_label(condition)} — "
        f"Event detection validation\n"
        f"{filtered_path.name}",
        fontsize=14,
    )

    output_path = (
        OUTPUT_FOLDER
        / (
            canonical_trial_stem(
                filtered_path
            )
            + "_event_validation.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{patient.upper():>5} | "
        f"{condition_label(condition):<10} | "
        f"L IC={len(events['left_ic_times']):3d}, "
        f"TO={len(events['left_to_times']):3d} | "
        f"R IC={len(events['right_ic_times']):3d}, "
        f"TO={len(events['right_to_times']):3d}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered_files = sorted(
        FILTERED_FOLDER.glob(
            "*_filtered.npz"
        )
    )

    if not filtered_files:
        raise RuntimeError(
            f"No filtered trials found in:\n"
            f"{FILTERED_FOLDER}"
        )

    print(
        f"Found {len(filtered_files)} "
        f"filtered trials.\n"
    )

    successful = 0
    failed = 0

    for filtered_path in filtered_files:

        trial_stem = canonical_trial_stem(
            filtered_path
        )

        event_path = (
            EVENTS_FOLDER
            / f"{trial_stem}_events.npz"
        )

        if not event_path.exists():

            print(
                f"SKIP: no event file for "
                f"{filtered_path.name}"
            )

            continue

        try:

            process_trial(
                filtered_path,
                event_path,
            )

            successful += 1

        except Exception as exc:

            failed += 1

            print(
                f"\nERROR: "
                f"{filtered_path.name}"
            )

            print(
                f"  {exc}"
            )

    print(
        "\n===================================="
    )

    print(
        "EVENT VALIDATION PLOTS COMPLETE"
    )

    print(
        "===================================="
    )

    print(
        f"Successful : {successful}"
    )

    print(
        f"Failed     : {failed}"
    )

    print(
        f"\nPlots:\n"
        f"{OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()