from pathlib import Path

import ezc3d
import numpy as np


FILE_PATH = Path(
    "/home/alya/Desktop/optimal-bws/raw_data/second_batch/insoles/al_0_bws.insoleX"
)


def inspect_binary_header(file_path, n_bytes=128):
    """Print the first bytes of the file for format inspection."""

    with open(file_path, "rb") as f:
        data = f.read(n_bytes)

    print("\n" + "=" * 80)
    print("RAW FILE HEADER")
    print("=" * 80)

    print(f"First {len(data)} bytes:")

    for offset in range(0, len(data), 16):
        chunk = data[offset:offset + 16]

        hex_part = " ".join(f"{byte:02x}" for byte in chunk)

        ascii_part = "".join(
            chr(byte) if 32 <= byte <= 126 else "."
            for byte in chunk
        )

        print(
            f"{offset:04x}: "
            f"{hex_part:<47} "
            f"{ascii_part}"
        )


def inspect_insole_file(file_path):
    """Inspect one COMETA .insoleX file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    print("\n" + "=" * 80)
    print("FILE INFORMATION")
    print("=" * 80)

    print(f"File: {file_path.name}")
    print(f"Full path: {file_path}")
    print(f"File size: {file_path.stat().st_size} bytes")
    print(
        f"File size: "
        f"{file_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    # ------------------------------------------------------------
    # Try loading the file as C3D
    # ------------------------------------------------------------
    try:
        c3d = ezc3d.c3d(str(file_path))

    except Exception as e:
        print("\n" + "=" * 80)
        print("EZC3D LOAD FAILED")
        print("=" * 80)

        print(
            "The file exists, but ezc3d could not interpret it "
            "as a valid C3D file."
        )

        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {e}")

        inspect_binary_header(file_path)

        return None

    # ------------------------------------------------------------
    # Analog data
    # ------------------------------------------------------------
    analogs_raw = c3d["data"]["analogs"]

    print("\n" + "=" * 80)
    print("ANALOG DATA")
    print("=" * 80)

    print(f"Raw analog shape: {analogs_raw.shape}")

    # Convert:
    # (subframes, channels, frames)
    #
    # into:
    # (channels, samples)
    analogs = analogs_raw.transpose(1, 0, 2).reshape(
        analogs_raw.shape[1],
        -1,
    )

    print(f"Flattened analog shape: {analogs.shape}")
    print(f"Number of channels: {analogs.shape[0]}")
    print(f"Number of samples: {analogs.shape[1]}")

    # ------------------------------------------------------------
    # Analog parameters
    # ------------------------------------------------------------
    analog_parameters = c3d["parameters"]["ANALOG"]

    labels = [
        str(label).strip()
        for label in analog_parameters["LABELS"]["value"]
    ]

    sample_rate = float(
        analog_parameters["RATE"]["value"][0]
    )

    units = []

    if "UNITS" in analog_parameters:
        units = [
            str(unit).strip()
            for unit in analog_parameters["UNITS"]["value"]
        ]

    duration = analogs.shape[1] / sample_rate

    time = np.arange(
        analogs.shape[1],
        dtype=float,
    ) / sample_rate

    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Duration: {duration:.3f} s")

    # ------------------------------------------------------------
    # Print all channels
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHANNELS")
    print("=" * 80)

    for i, label in enumerate(labels):

        unit = units[i] if i < len(units) else ""

        if i >= analogs.shape[0]:
            print(
                f"{i:3d}: {label} "
                f"[WARNING: no corresponding analog signal]"
            )
            continue

        signal = analogs[i]

        print(
            f"{i:3d}: "
            f"{label}"
            f"{f' [{unit}]' if unit else ''}"
        )

        print(
            f"     min={np.min(signal):.6f}, "
            f"max={np.max(signal):.6f}, "
            f"mean={np.mean(signal):.6f}, "
            f"std={np.std(signal):.6f}"
        )

    # ------------------------------------------------------------
    # Print parameter groups
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AVAILABLE C3D PARAMETER GROUPS")
    print("=" * 80)

    for group_name in c3d["parameters"]:
        print(group_name)

    # ------------------------------------------------------------
    # Print C3D header
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("C3D HEADER")
    print("=" * 80)

    for section_name, section in c3d["header"].items():

        print(f"\n[{section_name}]")

        if isinstance(section, dict):
            for key, value in section.items():
                print(f"{key}: {value}")
        else:
            print(section)

    # ------------------------------------------------------------
    # Return useful data
    # ------------------------------------------------------------
    return {
        "c3d": c3d,
        "file_path": file_path,
        "labels": labels,
        "units": units,
        "sample_rate": sample_rate,
        "duration": duration,
        "time": time,
        "analogs": analogs,
    }


def main():

    data = inspect_insole_file(FILE_PATH)

    if data is None:
        print("\nInspection stopped because ezc3d could not read the file.")
        return

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"Loaded: {data['file_path'].name}")
    print(f"Channels: {len(data['labels'])}")
    print(f"Samples: {data['analogs'].shape[1]}")
    print(f"Sampling rate: {data['sample_rate']} Hz")
    print(f"Duration: {data['duration']:.3f} s")


if __name__ == "__main__":
    main()