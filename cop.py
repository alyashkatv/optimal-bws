from pathlib import Path
import csv
import re
import struct

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/filtered_first_batch"
)

EVENTS_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/filtered_data/events_first_batch"
)

OUTPUT_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/cop_plots"
)

STANCE_METRICS_CSV = (
    OUTPUT_FOLDER / "cop_metrics_by_stance.csv"
)

TRIAL_METRICS_CSV = (
    OUTPUT_FOLDER / "cop_metrics_by_trial.csv"
)

DOCUMENTS_FOLDER = Path(
    "/home/alya/Desktop/optimal-bws/documents"
)

PATIENT_INFO_FILE = (
    DOCUMENTS_FOLDER / "patient_information_summary.csv"
)

SENSORS_M_FILE = (
    DOCUMENTS_FOLDER / "sensorsM.ts"
)

SENSORS_S_FILE = (
    DOCUMENTS_FOLDER / "sensorsS.ts"
)

SENSORS_PER_INSOLE = 64
TOTAL_SENSORS = 128    
DATA_START = 72


def load_gait_events(event_path):
    """
    Load previously detected and validated gait events.

    Returns stance intervals as event TIMES in seconds:
        left_stances_times
        right_stances_times
    """

    if not event_path.exists():
        raise FileNotFoundError(
            f"Event file not found:\n{event_path}"
        )

    with np.load(
        event_path,
        allow_pickle=False,
    ) as data:

        required = {
            "left_stances_times",
            "right_stances_times",
            "left_ic_times",
            "left_to_times",
            "right_ic_times",
            "right_to_times",
        }

        missing = required.difference(
            data.files
        )

        if missing:
            raise ValueError(
                f"{event_path.name}: missing event arrays: "
                f"{sorted(missing)}"
            )

        left_stances_times = np.asarray(
            data["left_stances_times"],
            dtype=float,
        )

        right_stances_times = np.asarray(
            data["right_stances_times"],
            dtype=float,
        )

        left_ic_times = np.asarray(
            data["left_ic_times"],
            dtype=float,
        )

        left_to_times = np.asarray(
            data["left_to_times"],
            dtype=float,
        )

        right_ic_times = np.asarray(
            data["right_ic_times"],
            dtype=float,
        )

        right_to_times = np.asarray(
            data["right_to_times"],
            dtype=float,
        )

    return {
        "left_stances_times": left_stances_times,
        "right_stances_times": right_stances_times,
        "left_ic_times": left_ic_times,
        "left_to_times": left_to_times,
        "right_ic_times": right_ic_times,
        "right_to_times": right_to_times,
    }


def stance_times_to_indices(
    stance_times,
    time,
):
    """
    Convert Nx2 [IC_time, TO_time] stance intervals
    into sample indices for the filtered pressure recording.
    """

    stance_times = np.asarray(
        stance_times,
        dtype=float,
    )

    if stance_times.size == 0:
        return []

    if (
        stance_times.ndim != 2
        or stance_times.shape[1] != 2
    ):
        raise ValueError(
            f"Expected stance_times shape (N, 2), "
            f"got {stance_times.shape}"
        )

    stances = []

    for ic_time, to_time in stance_times:

        if (
            not np.isfinite(ic_time)
            or not np.isfinite(to_time)
            or to_time <= ic_time
        ):
            continue

        start_idx = int(
            np.searchsorted(
                time,
                ic_time,
                side="left",
            )
        )

        end_idx = int(
            np.searchsorted(
                time,
                to_time,
                side="right",
            )
            - 1
        )

        start_idx = max(
            0,
            min(start_idx, len(time) - 1),
        )

        end_idx = max(
            0,
            min(end_idx, len(time) - 1),
        )

        if end_idx > start_idx:
            stances.append(
                (
                    start_idx,
                    end_idx,
                )
            )

    return stances

# ------------------------------------------------------------
# COP
# ------------------------------------------------------------

MIN_COP_PRESSURE_SUM = 1e-6


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# LOAD PATIENT INFORMATION
# ============================================================

def load_patient_sizes(path):
    """
    Read patient_information_summary.csv and return:

        {
            "ab": "M",
            "bo": "S",
            ...
        }

    The function tries to identify patient-ID and insole-size
    columns automatically.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Patient information file not found:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path.name}: no CSV header found."
            )

        # Normalize headers
        header_map = {
            normalize_name(name): name
            for name in reader.fieldnames
        }

        # Possible names for patient column
        patient_candidates = [
            "patient_id",
            "patient",
            "subject_id",
            "subject",
            "patient_code",
            "code",
            "id",
        ]

        # Possible names for insole-size column
        size_candidates = [
            "insole_size",
            "size",
            "sensor_size",
            "sensors_size",
            "insole",
            "insole_type",
            "sensor_layout",
        ]

        patient_column = None
        size_column = None

        for candidate in patient_candidates:
            if candidate in header_map:
                patient_column = header_map[candidate]
                break

        for candidate in size_candidates:
            if candidate in header_map:
                size_column = header_map[candidate]
                break

        if patient_column is None:
            raise ValueError(
                "Could not identify patient-ID column.\n"
                f"Columns found: {reader.fieldnames}"
            )

        if size_column is None:
            raise ValueError(
                "Could not identify insole-size column.\n"
                f"Columns found: {reader.fieldnames}"
            )

        patient_sizes = {}

        for row in reader:

            patient = (
                row[patient_column]
                .strip()
                .lower()
            )

            size_raw = (
                row[size_column]
                .strip()
                .upper()
            )

            if not patient:
                continue

            # Accept values such as:
            # S
            # Small
            # sensorsS
            # M
            # Medium
            # sensorsM

            # No size recorded yet -> skip this patient.
# This is NOT an error unless we actually try to process
# a trial belonging to this patient.
            if not size_raw:
                print(
                    f"  WARNING: no insole size recorded for "
                    f"{patient}"
                )
                continue

            if (
                size_raw == "S"
                or size_raw.startswith("SMALL")
                or "SENSORSS" in size_raw
            ):
                size = "S"

            elif (
                size_raw == "M"
                or size_raw.startswith("MEDIUM")
                or "SENSORSM" in size_raw
            ):
                size = "M"

            else:
                print(
                    f"  WARNING: unrecognized insole size "
                    f"'{size_raw}' for {patient}; skipping"
                )
                continue

            patient_sizes[patient] = size

    print("\nPatient insole sizes:")

    for patient, size in sorted(
        patient_sizes.items()
    ):
        print(
            f"  {patient}: {size}"
        )

    return patient_sizes


# ============================================================
# LOAD SENSOR COORDINATES FROM TYPESCRIPT
# ============================================================

def load_sensor_matrix_ts(path):
    """
    Read LEFT_MM and RIGHT_MM arrays from sensorsS.ts / sensorsM.ts.

    Returns:
        left_x, left_y, right_x, right_y
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Sensor matrix file not found:\n{path}"
        )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    def extract_array(array_name):

        start_idx = None

        # Find declaration:
        # export const LEFT_MM ...
        for i, line in enumerate(lines):

            if (
                f"export const {array_name}"
                in line
            ):
                start_idx = i + 1
                break

        if start_idx is None:
            raise ValueError(
                f"{path.name}: could not find "
                f"{array_name}"
            )

        coords = []

        pair_pattern = re.compile(
            r"""
            \[
            \s*
            (-?\d+(?:\.\d+)?)
            \s*,\s*
            (-?\d+(?:\.\d+)?)
            \s*
            \]
            """,
            re.VERBOSE
        )

        # Read lines until the array closing bracket
        for line in lines[start_idx:]:

            stripped = line.strip()

            if stripped.startswith("]"):
                break

            match = pair_pattern.search(
                line
            )

            if match:
                x = float(
                    match.group(1)
                )

                y = float(
                    match.group(2)
                )

                coords.append(
                    (x, y)
                )

        if len(coords) != SENSORS_PER_INSOLE:
            raise ValueError(
                f"{path.name}: {array_name} "
                f"contains {len(coords)} sensors; "
                f"expected {SENSORS_PER_INSOLE}"
            )

        coords = np.asarray(
            coords,
            dtype=float
        )

        return (
            coords[:, 0],
            coords[:, 1]
        )

    left_x, left_y = extract_array(
        "LEFT_MM"
    )

    right_x, right_y = extract_array(
        "RIGHT_MM"
    )

    print(
        f"\nLoaded {path.name}:"
    )

    print(
        f"  LEFT_MM : {len(left_x)} sensors"
    )

    print(
        f"    X range: "
        f"{left_x.min():.2f} – "
        f"{left_x.max():.2f} mm"
    )

    print(
        f"    Y range: "
        f"{left_y.min():.2f} – "
        f"{left_y.max():.2f} mm"
    )

    print(
        f"  RIGHT_MM: {len(right_x)} sensors"
    )

    print(
        f"    X range: "
        f"{right_x.min():.2f} – "
        f"{right_x.max():.2f} mm"
    )

    print(
        f"    Y range: "
        f"{right_y.min():.2f} – "
        f"{right_y.max():.2f} mm"
    )

    return (
        left_x,
        left_y,
        right_x,
        right_y
    )
    
# ============================================================
# LOAD BOTH INSOLE LAYOUTS
# ============================================================
def load_all_sensor_layouts():

    (
        left_x_m,
        left_y_m,
        right_x_m,
        right_y_m
    ) = load_sensor_matrix_ts(
        SENSORS_M_FILE
    )

    (
        left_x_s,
        left_y_s,
        right_x_s,
        right_y_s
    ) = load_sensor_matrix_ts(
        SENSORS_S_FILE
    )

    return {
        "M": {
            "left": (
                left_x_m,
                left_y_m
            ),
            "right": (
                right_x_m,
                right_y_m
            ),
        },

        "S": {
            "left": (
                left_x_s,
                left_y_s
            ),
            "right": (
                right_x_s,
                right_y_s
            ),
        },
    }

# ============================================================
# PARSE TRIAL NAME
# ============================================================

def canonical_trial_stem(path):
    """Return the raw recording stem for either a raw or derived file."""
    stem = path.stem
    for suffix in ("_cleaned", "_cropped", "_filtered"):
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def parse_trial_name(path):
    """
    Examples:

        ab_0_bws_filtered.npz
        ab_10_bws_filtered.npz
        bo_25_bws_filtered.npz

    Returns:

        patient_id, bws
    """

    match = re.fullmatch(
        r"([A-Za-z]+)_(?:(\d+)_bws|(ground))",
        canonical_trial_stem(path),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    patient_id = (
        match.group(1)
        .lower()
    )

    condition = (
        int(match.group(2))
        if match.group(2) is not None
        else "ground"
    )

    return (
        patient_id,
        condition
    )


# ============================================================
# READ NPZ OR RAW INSOLEX
# ============================================================

def validate_recording(path, time, pressure, fs):
    """Validate and normalize arrays returned by either file format."""
    time = np.asarray(time, dtype=float).reshape(-1)
    pressure = np.asarray(pressure, dtype=float)
    fs = float(fs)

    if fs <= 0 or not np.isfinite(fs):
        raise ValueError(f"{path.name}: invalid sampling rate {fs}")
    if pressure.ndim != 2:
        raise ValueError(f"{path.name}: pressure must be 2-D.")
    if pressure.shape[1] != TOTAL_SENSORS:
        raise ValueError(
            f"{path.name}: expected {TOTAL_SENSORS} channels, "
            f"found {pressure.shape[1]}"
        )
    if len(time) != len(pressure):
        raise ValueError(f"{path.name}: time and pressure lengths do not match.")
    if len(time) == 0:
        raise ValueError(f"{path.name}: recording contains no samples.")
    return time, pressure, fs


def load_npz(path):

    data = np.load(
        path,
        allow_pickle=False
    )

    required = {"pressure", "time", "sampling_rate"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path.name}: missing NPZ keys: {sorted(missing)}")

    return validate_recording(
        path, data["time"], data["pressure"], data["sampling_rate"]
    )


def load_insolex(path):
    """Read a COMETA .insoleX pressure recording."""
    raw = path.read_bytes()
    if len(raw) < DATA_START:
        raise ValueError(f"{path.name}: file is too short")
    date_length = raw[2]
    metadata_offset = 3 + date_length + 1
    if metadata_offset + 8 > len(raw):
        raise ValueError(f"{path.name}: incomplete metadata")
    fs = struct.unpack_from("<f", raw, metadata_offset)[0]
    n_samples = struct.unpack_from("<I", raw, metadata_offset + 4)[0]
    expected_size = DATA_START + n_samples * TOTAL_SENSORS * 4
    if len(raw) != expected_size:
        raise ValueError(
            f"{path.name}: unexpected size; expected {expected_size}, got {len(raw)}"
        )
    pressure = np.frombuffer(
        raw, dtype="<f4", count=n_samples * TOTAL_SENSORS, offset=DATA_START
    ).copy().reshape(n_samples, TOTAL_SENSORS)
    time = np.arange(n_samples, dtype=float) / fs
    return validate_recording(path, time, pressure, fs)


def load_recording(path):
    if path.suffix.lower() == ".npz":
        return load_npz(path)
    if path.suffix.lower() == ".insolex":
        return load_insolex(path)
    raise ValueError(f"Unsupported recording format: {path.name}")


def discover_trials(folder):
    """Find recordings, using one NPZ in preference to each matching insoleX."""
    candidates = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in {".npz", ".insolex"}:
            candidates.setdefault(canonical_trial_stem(path).lower(), []).append(path)

    selected = []
    for paths in candidates.values():
        npz_paths = [p for p in paths if p.suffix.lower() == ".npz"]
        if npz_paths:
            # Cleaning/cropping outputs take precedence over other derived NPZs.
            def npz_priority(p):
                stem = p.stem.lower()

                if stem.endswith("_filtered"):
                    return (0, p.name.lower())

                if stem.endswith(("_cleaned", "_cropped")):
                    return (1, p.name.lower())

                return (2, p.name.lower())
            chosen = sorted(npz_paths, key=npz_priority)[0]
        else:
            chosen = sorted(paths, key=lambda p: p.name.lower())[0]
        selected.append(chosen)
    return selected




# ============================================================
# COMPUTE COP
# ============================================================

def compute_cop(
    foot_pressure,
    x_coords,
    y_coords
):
    """
    Pressure-weighted COP:

        COP_ML =
            sum(P_i * x_i) / sum(P_i)

        COP_AP =
            sum(P_i * y_i) / sum(P_i)
    """

    weights = np.clip(
        foot_pressure,
        0,
        None
    )

    total = np.sum(
        weights,
        axis=1
    )

    valid = (
        total
        > MIN_COP_PRESSURE_SUM
    )

    cop_x = np.full(
        len(total),
        np.nan
    )

    cop_y = np.full(
        len(total),
        np.nan
    )

    cop_x[valid] = (
        weights[valid]
        @ x_coords
    ) / total[valid]

    cop_y[valid] = (
        weights[valid]
        @ y_coords
    ) / total[valid]

    return (
        cop_x,
        cop_y
    )


def symmetry_index(left, right):
    """
    Absolute symmetry index.

    0% = perfect bilateral symmetry.
    Higher values = greater asymmetry.
    """

    if (
        not np.isfinite(left)
        or not np.isfinite(right)
    ):
        return np.nan

    denominator = 0.5 * (left + right)

    if denominator <= 0:
        return np.nan

    return (
        abs(left - right)
        / denominator
        * 100.0
    )


def calculate_cop_metrics_for_stance(
    foot_pressure,
    start_idx,
    end_idx,
    x_coords,
    y_coords,
    time,
):
    """
    Calculate quantitative COP metrics for one validated stance.
    """

    stance_pressure = foot_pressure[
        start_idx:end_idx + 1
    ]

    stance_time = time[
        start_idx:end_idx + 1
    ]

    cop_x, cop_y = compute_cop(
        stance_pressure,
        x_coords,
        y_coords,
    )

    valid = (
        np.isfinite(cop_x)
        & np.isfinite(cop_y)
    )

    cop_x = cop_x[valid]
    cop_y = cop_y[valid]
    stance_time = stance_time[valid]

    if len(cop_x) < 3:
        return None

    # --------------------------------------------------------
    # AP excursion
    # --------------------------------------------------------

    ap_excursion = (
        np.max(cop_y)
        - np.min(cop_y)
    )

    # --------------------------------------------------------
    # ML excursion
    # --------------------------------------------------------

    ml_excursion = (
        np.max(cop_x)
        - np.min(cop_x)
    )

    # --------------------------------------------------------
    # COP path length
    # --------------------------------------------------------

    dx = np.diff(cop_x)
    dy = np.diff(cop_y)

    segment_lengths = np.sqrt(
        dx**2 + dy**2
    )

    path_length = np.sum(
        segment_lengths
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = (
        stance_time[-1]
        - stance_time[0]
    )

    # --------------------------------------------------------
    # Mean COP velocity
    # --------------------------------------------------------

    if duration > 0:
        mean_velocity = (
            path_length / duration
        )
    else:
        mean_velocity = np.nan

    # --------------------------------------------------------
    # Path efficiency
    #
    # Straight-line displacement / actual path length
    #
    # 1.0 = perfectly straight path
    # lower = more wandering
    # --------------------------------------------------------

    net_displacement = np.sqrt(
        (cop_x[-1] - cop_x[0])**2
        + (cop_y[-1] - cop_y[0])**2
    )

    if path_length > 0:
        path_efficiency = (
            net_displacement
            / path_length
        )
    else:
        path_efficiency = np.nan

    return {
        "ap_excursion_mm":
            float(ap_excursion),

        "ml_excursion_mm":
            float(ml_excursion),

        "cop_path_length_mm":
            float(path_length),

        "stance_duration_s":
            float(duration),

        "mean_cop_velocity_mm_s":
            float(mean_velocity),

        "path_efficiency":
            float(path_efficiency),
    }

def calculate_foot_cop_metrics(
    patient_id,
    condition,
    foot_name,
    foot_pressure,
    stances,
    x_coords,
    y_coords,
    time,
):
    """
    Calculate COP metrics for every validated stance
    of one foot.
    """

    rows = []

    for stance_number, (
        start_idx,
        end_idx,
    ) in enumerate(
        stances,
        start=1,
    ):

        metrics = (
            calculate_cop_metrics_for_stance(
                foot_pressure,
                start_idx,
                end_idx,
                x_coords,
                y_coords,
                time,
            )
        )

        if metrics is None:
            continue

        row = {
            "patient": patient_id,
            "condition":
                condition_label(condition),

            "bws_percent":
                np.nan
                if condition == "ground"
                else condition,

            "foot": foot_name,

            "stance_number":
                stance_number,

            **metrics,
        }

        rows.append(row)

    return rows

# ============================================================
# PLOT ONE FOOT
# ============================================================

def plot_foot_cop(
    ax,
    foot_pressure,
    stances,
    x_coords,
    y_coords,
    title
):

    # Physical sensor positions
    ax.scatter(
        x_coords,
        y_coords,
        s=20,
        alpha=0.35
    )

    valid_stances = 0

    for start_idx, end_idx in stances:

        stance_pressure = (
            foot_pressure[
                start_idx:
                end_idx + 1
            ]
        )

        cop_x, cop_y = compute_cop(
            stance_pressure,
            x_coords,
            y_coords
        )

        valid = (
            np.isfinite(cop_x)
            & np.isfinite(cop_y)
        )

        cop_x = cop_x[valid]
        cop_y = cop_y[valid]

        if len(cop_x) < 3:
            continue

        # COP trajectory
        ax.plot(
            cop_x,
            cop_y,
            linewidth=0.8,
            alpha=0.25
        )

        # Start marker
        ax.scatter(
            cop_x[0],
            cop_y[0],
            s=12,
            color="green",
            alpha=0.65
        )

        # End marker
        ax.scatter(
            cop_x[-1],
            cop_y[-1],
            s=12,
            color="red",
            alpha=0.65
        )

        valid_stances += 1

    ax.set_title(
        f"{title}\n"
        f"{valid_stances} valid stances"
    )

    ax.set_xlabel(
        "ML (mm)"
    )

    ax.set_ylabel(
        "AP (mm)"
    )

    ax.grid(
        alpha=0.2
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def process_trial(
    file_path,
    patient_id,
    condition,
    patient_sizes,
    sensor_layouts,
):

    # --------------------------------------------------------
    # Determine insole size
    # --------------------------------------------------------

    if patient_id not in patient_sizes:
        raise ValueError(
            f"No insole-size information found "
            f"for patient '{patient_id}'"
        )

    insole_size = patient_sizes[patient_id]

    if insole_size not in sensor_layouts:
        raise ValueError(
            f"No sensor layout available "
            f"for size {insole_size}"
        )

    left_x, left_y = sensor_layouts[insole_size]["left"]
    right_x, right_y = sensor_layouts[insole_size]["right"]

    # --------------------------------------------------------
    # Load filtered pressure recording
    # --------------------------------------------------------

    time, pressure, fs = load_recording(file_path)

    print(
        f"  Pressure source: {file_path.name}"
    )

    # --------------------------------------------------------
    # Load corresponding validated event file
    # --------------------------------------------------------

    trial_stem = canonical_trial_stem(file_path)

    event_path = (
        EVENTS_FOLDER
        / f"{trial_stem}_events.npz"
    )

    events = load_gait_events(event_path)

    print(
        f"  Event source: {event_path.name}"
    )

    # --------------------------------------------------------
    # Split pressure into feet
    # --------------------------------------------------------

    left_pressure = pressure[:, :64]
    right_pressure = pressure[:, 64:]

    # --------------------------------------------------------
    # Convert validated stance times to pressure indices
    # --------------------------------------------------------

    left_stances = stance_times_to_indices(
        events["left_stances_times"],
        time,
    )

    right_stances = stance_times_to_indices(
        events["right_stances_times"],
        time,
    )

    # --------------------------------------------------------
# Quantitative COP metrics
# --------------------------------------------------------

    left_metric_rows = (
        calculate_foot_cop_metrics(
            patient_id,
            condition,
            "left",
            left_pressure,
            left_stances,
            left_x,
            left_y,
            time,
        )
    )

    right_metric_rows = (
        calculate_foot_cop_metrics(
            patient_id,
            condition,
            "right",
            right_pressure,
            right_stances,
            right_x,
            right_y,
            time,
        )
    )

    stance_metric_rows = (
        left_metric_rows
        + right_metric_rows
    )

    print(
        f"{patient_id.upper()} "
        f"{condition_label(condition)} "
        f"[size {insole_size}]: "
        f"L={len(left_stances)} validated stances, "
        f"R={len(right_stances)} validated stances"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 10),
        constrained_layout=True,
    )

    plot_foot_cop(
        axes[0],
        left_pressure,
        left_stances,
        left_x,
        left_y,
        "Left COP: green start, red end",
    )

    plot_foot_cop(
        axes[1],
        right_pressure,
        right_stances,
        right_x,
        right_y,
        "Right COP: green start, red end",
    )

    fig.suptitle(
        f"{patient_id.upper()} — "
        f"{condition_label(condition)} — "
        f"Insole size {insole_size}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    patient_folder = (
        OUTPUT_FOLDER / patient_id
    )

    patient_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        patient_folder
        / (
            f"{patient_id}_"
            f"{condition_filename(condition)}_cop.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    return stance_metric_rows


def condition_label(condition):
    return "Ground" if condition == "ground" else f"{condition}% BWS"


def condition_filename(condition):
    return "ground" if condition == "ground" else f"{condition}_bws"

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_sizes = load_patient_sizes(
        PATIENT_INFO_FILE
    )

    sensor_layouts = load_all_sensor_layouts()

    files = sorted(
        INPUT_FOLDER.glob("*_filtered.npz")
    )

    if not files:
        raise RuntimeError(
            f"No filtered NPZ recordings found in:\n"
            f"{INPUT_FOLDER}"
        )

    trials = []
    all_stance_metrics = []

    for file_path in files:

        parsed = parse_trial_name(
            file_path
        )

        if parsed is None:
            print(
                f"Skipping unrecognized filename: "
                f"{file_path.name}"
            )
            continue

        patient_id, condition = parsed

        trials.append(
            (
                patient_id,
                condition,
                file_path,
            )
        )

    trials.sort(
        key=lambda x: (
            x[0],
            isinstance(x[1], str),
            x[1]
            if isinstance(x[1], int)
            else 0,
        )
    )

    print(
        f"\nFound {len(trials)} trials."
    )

    successful = 0
    failed = 0

    for (
        patient_id,
        condition,
        file_path,
    ) in trials:

        try:
            trial_stance_metrics = process_trial(
                file_path,
                patient_id,
                condition,
                patient_sizes,
                sensor_layouts,
            )

            all_stance_metrics.extend(
                trial_stance_metrics
            )

            successful += 1

        except Exception as e:

            failed += 1


            print(
                f"ERROR processing "
                f"{file_path.name}: {e}"
            )


    if all_stance_metrics:

        grouped = {}

        for row in all_stance_metrics:

            key = (
                row["patient"],
                row["condition"],
                row["bws_percent"],
            )

            grouped.setdefault(
                key,
                {
                    "left": [],
                    "right": [],
                }
            )

            grouped[key][
                row["foot"]
            ].append(row)

        trial_rows = []

        metric_names = [
            "ap_excursion_mm",
            "ml_excursion_mm",
            "cop_path_length_mm",
            "mean_cop_velocity_mm_s",
            "path_efficiency",
        ]

        for (
            patient,
            condition,
            bws_percent,
        ), feet in grouped.items():

            trial_row = {
                "patient": patient,
                "condition": condition,
                "bws_percent":
                    bws_percent,
                "n_left_stances":
                    len(feet["left"]),
                "n_right_stances":
                    len(feet["right"]),
            }

            for metric in metric_names:

                left_values = np.asarray(
                    [
                        x[metric]
                        for x in feet["left"]
                        if np.isfinite(x[metric])
                    ],
                    dtype=float,
                )

                right_values = np.asarray(
                    [
                        x[metric]
                        for x in feet["right"]
                        if np.isfinite(x[metric])
                    ],
                    dtype=float,
                )

                left_mean = (
                    float(np.mean(left_values))
                    if len(left_values)
                    else np.nan
                )

                right_mean = (
                    float(np.mean(right_values))
                    if len(right_values)
                    else np.nan
                )

                left_std = (
                    float(np.std(
                        left_values,
                        ddof=1,
                    ))
                    if len(left_values) >= 2
                    else np.nan
                )

                right_std = (
                    float(np.std(
                        right_values,
                        ddof=1,
                    ))
                    if len(right_values) >= 2
                    else np.nan
                )

                trial_row[
                    f"{metric}_left_mean"
                ] = left_mean

                trial_row[
                    f"{metric}_right_mean"
                ] = right_mean

                trial_row[
                    f"{metric}_left_sd"
                ] = left_std

                trial_row[
                    f"{metric}_right_sd"
                ] = right_std

                trial_row[
                    f"{metric}_SI_percent"
                ] = symmetry_index(
                    left_mean,
                    right_mean,
                )

            trial_rows.append(
                trial_row
            )

        update_csv(
            TRIAL_METRICS_CSV,
            trial_rows,
            key_fields=[
                "patient",
                "condition",
            ],
            )

       
        print(
            f"\nTrial-level COP metrics:\n"
            f"{TRIAL_METRICS_CSV}"
        )
        print(
            "\n===================================="
        )
        print("COP PROCESSING FINISHED")
        print("====================================")
        print(f"Successful: {successful}")
        print(f"Failed:     {failed}")
        print(
            f"\nPlots saved to:\n"
            f"{OUTPUT_FOLDER}"
        )

        # ============================================================
    # UPDATE MASTER PER-STANCE CSV
    # ============================================================

    if all_stance_metrics:

        update_csv(
            STANCE_METRICS_CSV,
            all_stance_metrics,
            key_fields=[
                "patient",
                "condition",
                "foot",
                "stance_number",
            ],
        )

        print(
            f"\nPer-stance COP metrics:\n"
            f"{STANCE_METRICS_CSV}"
        )
def update_csv(
    path,
    new_rows,
    key_fields,
):
    """
    Update a master CSV.

    Existing rows with the same key are replaced.
    New rows are appended.
    """

    if not new_rows:
        return

    existing_rows = []

    if path.exists():
        with open(
            path,
            "r",
            newline="",
            encoding="utf-8",
        ) as f:
            existing_rows = list(
                csv.DictReader(f)
            )

    # Identify trials/stances being recalculated
    new_keys = {
        tuple(
            str(row[field])
            for field in key_fields
        )
        for row in new_rows
    }

    # Keep previous rows that were NOT recalculated
    existing_rows = [
        row
        for row in existing_rows
        if tuple(
            str(row[field])
            for field in key_fields
        ) not in new_keys
    ]

    combined_rows = (
        existing_rows
        + new_rows
    )

    fieldnames = list(
        new_rows[0].keys()
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
            combined_rows
        )
        
if __name__ == "__main__":
    main()


