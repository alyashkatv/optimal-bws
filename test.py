from pathlib import Path

import ezc3d
import matplotlib.pyplot as plt
import numpy as np


FOLDER = Path("/home/alya/Desktop/test_insoles")
OUTPUT_FILE = FOLDER / "insole_pressure_comparison.png"


def load_insole_pressure(file_path):
    """Load average left/right insole pressure from a COMETA .insoleX file."""

    c3d = ezc3d.c3d(str(file_path))

    # Analog data shape:
    # (subframes, channels, frames)
    analogs = c3d["data"]["analogs"]

    # Flatten time dimension
    analogs = analogs.transpose(1, 0, 2).reshape(analogs.shape[1], -1)

    labels = [
        label.strip()
        for label in c3d["parameters"]["ANALOG"]["LABELS"]["value"]
    ]

    sample_rate = float(
        c3d["parameters"]["ANALOG"]["RATE"]["value"][0]
    )

    time = np.arange(analogs.shape[1]) / sample_rate

    print(f"\nFile: {file_path.name}")
    print(f"Sampling rate: {sample_rate} Hz")
    print("Available channels:")

    for i, label in enumerate(labels):
        print(f"{i:3d}: {label}")

    # Try to identify whole-foot average pressure channels
    left_idx = None
    right_idx = None

    for i, label in enumerate(labels):
        label_lower = label.lower()

        if (
            "left" in label_lower
            and "insole" in label_lower
            and "average" in label_lower
        ):
            left_idx = i

        if (
            "right" in label_lower
            and "insole" in label_lower
            and "average" in label_lower
        ):
            right_idx = i

    pressures = {}

    if left_idx is not None:
        pressures["Left"] = analogs[left_idx]

    if right_idx is not None:
        pressures["Right"] = analogs[right_idx]

    if not pressures:
        raise RuntimeError(
            f"Could not find average insole pressure channels in {file_path.name}"
        )

    return time, pressures


def main():
    files = sorted(FOLDER.glob("*.insoleX"))

    if len(files) != 2:
        raise RuntimeError(
            f"Expected exactly 2 .insoleX files in {FOLDER}, "
            f"but found {len(files)}:\n"
            + "\n".join(str(f.name) for f in files)
        )

    print("Files found:")
    for file in files:
        print(" ", file.name)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=False,
        constrained_layout=True,
    )

    for ax, file_path in zip(axes, files):
        time, pressures = load_insole_pressure(file_path)

        for side, pressure in pressures.items():
            ax.plot(
                time,
                pressure,
                label=f"{side} insole",
                linewidth=1.0,
            )

        ax.set_title(file_path.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pressure (kPa)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Insole Pressure vs Time", fontsize=14)

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"\nGraph saved to:\n{OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    main()