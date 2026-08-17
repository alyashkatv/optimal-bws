# Optimal BWS Insole Pressure Analysis

Python scripts for processing and analysing plantar-pressure recordings collected with COMETA pressure insoles during body-weight-support (BWS) trials. The workflow converts raw recordings into filtered datasets, detects gait events, and produces whole-foot, regional-loading, and centre-of-pressure analyses.

Clinical recordings, participant metadata, derived datasets, and generated figures are intentionally excluded from Git. This repository contains analysis code and sensor-layout definitions only.

## Analysis workflow

```text
COMETA .insoleX recordings
          |
          v
  quality control and filtering
          |
          v
 filtered .npz datasets
          |
          +--> gait-event detection
          +--> whole-foot loading vs BWS
          +--> regional loading
          +--> centre-of-pressure plots
          +--> pressure visualisations and CSV summaries
```

Each recording is expected to contain 128 pressure channels: 64 sensors for the left insole followed by 64 sensors for the right insole. Pressure values are handled in kPa.

## Repository contents

| Path | Purpose |
| --- | --- |
| `filtering.py` | Reads raw `.insoleX` files, applies a zero-phase Butterworth low-pass filter, and saves analysis-ready `.npz` files. |
| `filter_test.py` | Compares filter cut-off frequencies on representative sensors and creates diagnostic plots. |
| `manual_cleaning.py` | Performs recording-specific trimming, interval removal, and spike correction before filtering. |
| `gait_event/event_detection.py` | Detects initial contact, toe-off, stance, and swing periods using adaptive thresholds and hysteresis. |
| `loading_vs_bws.py` | Calculates trial and participant loading across BWS conditions and exports plots and a CSV summary. |
| `regional_pressure.py` | Divides each insole into heel, midfoot, forefoot, and toe regions and analyses absolute and normalised loading. |
| `cop.py` | Calculates and plots centre-of-pressure trajectories using the insole sensor coordinates. |
| `plot_insoles.py` | Produces whole-foot pressure plots and summaries directly from raw recordings. |
| `plot_filtered.py` | Produces whole-foot pressure plots and summaries from filtered datasets. |
| `size_detect.py` | Inspects recording headers when investigating insole metadata or file structure. |
| `test.py` | Experimental comparison of two recordings through `ezc3d`. |
| `documents/sensorsS.ts` | Left/right sensor coordinates for size-S insoles. |
| `documents/sensorsM.ts` | Left/right sensor coordinates for size-M insoles. |

The anatomical regions in `regional_pressure.py` are geometry-based approximations, not clinically validated anatomical masks.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib
- `ezc3d` (only required by `test.py`)

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib ezc3d
```

## Data organisation

The scripts are currently configured by editing `Path` constants near the top of each file. Many of those constants still reference the former project location:

```text
/home/alya/Desktop/test_insoles
```

Before running an analysis, update its input, output, and document paths to match your local checkout. A typical local layout is:

```text
optimal-bws/
├── raw_data/
│   ├── first_batch/
│   └── second_batch/
├── filtered_data/
│   ├── filtered_first_batch/
│   ├── filtered_second_batch/
│   └── events_first_batch/
├── documents/
│   ├── patient_information_summary.csv
│   ├── sensorsS.ts
│   └── sensorsM.ts
└── ...analysis scripts
```

Raw trial names should follow the convention:

```text
<participant>_<condition>.insoleX
```

Examples include `ab_20_bws.insoleX` and `ab_ground.insoleX`. Processed files may add suffixes such as `_cleaned`, `_cropped`, or `_filtered`.

The participant-information CSV used by the loading, regional, and COP analyses must include at least these columns:

```csv
Patient,Insole size
ab,M
bo,S
```

Supported insole sizes are `S` and `M`. Keep this file local because it may contain clinical or participant information.

## Running the workflow

There is no command-line interface yet. Configure the constants at the top of the relevant script, activate the virtual environment, and run scripts from the repository root.

### 1. Inspect and clean recordings when needed

Use `plot_insoles.py` for an initial visual check. For a recording that needs manual correction, configure the file and affected time intervals in `manual_cleaning.py`, then run:

```bash
python manual_cleaning.py
```

Manual-cleaning settings are recording-specific and should be reviewed before every run.

### 2. Select and apply the filter

Use the diagnostic comparison when evaluating filter settings:

```bash
python filter_test.py
```

Then configure `INPUT_FOLDER`, `OUTPUT_FOLDER`, and the filter parameters in `filtering.py` and run:

```bash
python filtering.py
```

The default implementation uses a fourth-order, 20 Hz Butterworth low-pass filter applied with zero-phase forward/backward filtering. Output NPZ files retain pressure, time, sampling rate, provenance, and filtering metadata.

### 3. Detect gait events

Configure the filtered input and event-output folders in `gait_event/event_detection.py`, then run:

```bash
python gait_event/event_detection.py
```

Detection combines per-sensor activity, adaptive whole-foot thresholds, hysteresis, and minimum contact/swing durations. Review the constants near the top of the script before applying them to a new acquisition protocol or population.

### 4. Run downstream analyses

Run one or more analyses after configuring their paths:

```bash
python plot_filtered.py
python loading_vs_bws.py
python regional_pressure.py
python cop.py
```

These scripts create output directories automatically and write PNG plots and/or CSV summaries. Generated outputs are ignored by Git.

## Filtered NPZ format

Files produced by `filtering.py` include the principal fields below:

| Field | Description |
| --- | --- |
| `time` | Time vector in seconds. |
| `pressure` | Two-dimensional array with shape `(samples, 128)`. |
| `sampling_rate` | Sampling rate in Hz. |
| `original_trial` | Original trial filename. |
| `source_file` | Actual source used for processing. |
| `source_type` | Source classification, such as raw or cleaned. |
| `recording_datetime` | Recording timestamp read from the source. |
| `filter_type`, `filter_order`, `cutoff_hz`, `zero_phase` | Filtering provenance. |
| `sensors_per_insole`, `total_sensors` | Dataset dimensions. |

Event-detection output adds arrays describing contact masks, initial-contact and toe-off indices/times, and stance intervals for each foot.

## Data protection and reproducibility

- Do not commit raw recordings, filtered participant data, metadata, CSV summaries, or generated plots. The project `.gitignore` covers the current data and output locations.
- Use coded participant identifiers in filenames and analysis outputs.
- Confirm that the participant-to-insole-size mapping is correct before calculating force or centre of pressure.
- Record any manual cleaning intervals and parameter changes used for a result.
- Treat thresholds and regional boundaries as analysis assumptions that require validation for the intended study.

## Current limitations

- Paths and analysis settings are hard-coded rather than supplied through command-line arguments or a configuration file.
- `manual_cleaning.py` contains trial-specific corrections and is not a general automatic-cleaning pipeline.
- `gait_event/validation.py` is currently a placeholder.
- The scripts do not yet include an automated test suite or a pinned dependency file.
- Several scripts duplicate the raw `.insoleX` parser; changes to the file format must be kept consistent across them.

