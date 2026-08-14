from pathlib import Path
import re


FOLDER = Path(
    "/home/alya/Desktop/test_insoles/second_batch"
)

HEADER_BYTES = 256


def inspect_insolex(path):
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)

    print("\n" + "=" * 70)
    print(path.name)
    print("=" * 70)

    print("\nHEX:")
    for offset in range(0, len(raw), 16):
        chunk = raw[offset:offset + 16]

        hex_part = " ".join(
            f"{b:02x}" for b in chunk
        )

        ascii_part = "".join(
            chr(b) if 32 <= b <= 126 else "."
            for b in chunk
        )

        print(
            f"{offset:04x}: "
            f"{hex_part:<47} "
            f"{ascii_part}"
        )

    print("\nASCII strings:")

    strings = re.findall(
        rb"[\x20-\x7e]{3,}",
        raw
    )

    for s in strings:
        print(
            " ",
            s.decode(
                "ascii",
                errors="replace"
            )
        )


def main():
    files = sorted(
        FOLDER.glob("*.insoleX")
    )

    for path in files:
        inspect_insolex(path)


if __name__ == "__main__":
    main()