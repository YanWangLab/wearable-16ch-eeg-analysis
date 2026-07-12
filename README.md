# Wearable 16-Channel EEG Analysis Code

This repository contains the code associated with the manuscript:

**“Wearable 16-channel electroencephalography for 8-day continuous monitoring”**

The repository includes:

- a demonstration workflow for EEG segmentation, preprocessing, dataset construction, model training, and model testing;

- the model checkpoint used for the manuscript-related evaluation;

- selected implementations of the analysis and visualization methods used in the study.

The repository may be updated during the manuscript review process. The version corresponding to the published article will be archived upon publication.

## Repository structure

```text
.
├── segment.py
├── preprocessing.py
├── build_dataset.py
├── model.py
├── train.py
├── test.py
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── data/
│   ├── train_data/
│   │   ├── raw/
│   │   │   └── 20260805T193613.csv
│   │   ├── segment_datas/
│   │   │   ├── find_numbers_segment.npy
│   │   │   ├── fn_grades.npy
│   │   │   ├── relax_numbers_segment.npy
│   │   │   └── sleep_data.csv
│   │   ├── processed/
│   │   └── fn_rx_timetable.xlsx
│   │
│   └── test_data/
│       ├── processed_data/
│       │   └── test_data.npy
│       └── result/
│
├── checkpoints/
│   ├── best_model.pth
│   └── train_demo.pth
│
└── additional analysis and visualization/
    ├── python/
    │   ├── plot_functional_connectivity.py
    │   └── plot_scalp_topography.py
    │
    └── matlab/
        ├── eeg_signal_processing_pipeline.m
        ├── motion_artifact_fft_wtc_snr.m
        └── psg_validation_sleep_analysis.m
```

## Main Python scripts

| File               | Description                                                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segment.py`       | Demonstrates the EEG segmentation procedure using the included example recording and timetable file.                                              |
| `preprocessing.py` | Performs EEG filtering, notch filtering, short-time Fourier transform calculation, frequency-feature extraction, and sliding-window construction. |
| `build_dataset.py` | Builds the PyTorch training and validation datasets from the segmented demo EEG data and demo sleep data.                                         |
| `model.py`         | Defines the LSTM-based EEG regression model.                                                                                                      |
| `train.py`         | Demonstrates the model-training procedure and saves the resulting demo checkpoint.                                                                |
| `test.py`          | Runs model inference on the processed demo test data and saves the prediction results.                                                            |

## Demo data

A short segment of EEG data is included to demonstrate the file-segmentation and dataset-construction workflow. The provided scripts focus on illustrating the main processing logic for a single example file.

The repository includes the following demo files:

```text
data/train_data/raw/20260805T193613.csv
data/train_data/fn_rx_timetable.xlsx
data/train_data/segment_datas/sleep_data.csv
data/test_data/processed_data/test_data.npy
```

### Training data

| File or folder                                            | Description                                                                   |
| --------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `data/train_data/raw/20260805T193613.csv`                 | Short example segment of a raw 16-channel EEG recording used by `segment.py`. |
| `data/train_data/fn_rx_timetable.xlsx`                    | Example timetable defining the EEG segmentation intervals.                    |
| `data/train_data/segment_datas/find_numbers_segment.npy`  | Segmented EEG data for the find-number task.                                  |
| `data/train_data/segment_datas/fn_grades.npy`             | Demo labels for the find-number task.                                         |
| `data/train_data/segment_datas/relax_numbers_segment.npy` | Segmented EEG data for the relaxation condition.                              |
| `data/train_data/segment_datas/sleep_data.csv`            | Demo sleep data used in the dataset-construction and training workflow.       |
| `data/train_data/processed/`                              | Output directory for the generated PyTorch training and validation datasets.  |

### Test data

| File or folder                                | Description                                         |
| --------------------------------------------- | --------------------------------------------------- |
| `data/test_data/processed_data/test_data.npy` | Processed demo EEG data used by `test.py`.          |
| `data/test_data/result/`                      | Output directory for model predictions and figures. |

## Raw EEG data format

The raw EEG recording is stored as a CSV file. Each row represents one EEG sample.

The file contains 17 columns:

| Column       | Description                                        |
| ------------ | -------------------------------------------------- |
| Column 1     | Time variable, denoted as `t`.                     |
| Columns 2–17 | Voltage signals recorded from the 16 EEG channels. |

The expected structure is:

```text
t, EEG_channel_1, EEG_channel_2, ..., EEG_channel_16
```

Conceptually, the data matrix has the following form:

```text
t1,  V1_1,  V2_1,  ..., V16_1
t2,  V1_2,  V2_2,  ..., V16_2
t3,  V1_3,  V2_3,  ..., V16_3
...
tn,  V1_n,  V2_n,  ..., V16_n
```

where:

- `t1` to `tn` are the time values;

- `V1` to `V16` are the voltage signals from the 16 EEG channels;

- `n` is the number of recorded samples.

The example file is located at:

```text
data/train_data/raw/20260805T193613.csv
```

Users applying the scripts to their own EEG recordings should retain the same column organization or adapt the data-loading code accordingly.

## Example EEG data

This repository includes a de-identified example recording of real human electroencephalography (EEG) data. The example data are provided solely to demonstrate and evaluate the analysis workflow implemented in this repository. All direct personal identifiers have been removed.

The original data collection was conducted under the ethics approval and informed-consent procedures described in the associated manuscript. The example recording is not intended to represent the complete study dataset or to support independent reproduction of all study-level results.

The example EEG data may be downloaded and used only for evaluating, testing, and reproducing the code workflow provided in this repository. Redistribution, publication, commercial use, or secondary research use of the example data is not permitted without prior written permission from the authors.

## Model checkpoints

Two types of model checkpoints are used in this repository:

| Checkpoint                   | Description                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `checkpoints/best_model.pth` | Model checkpoint used for the manuscript-related evaluation. This is the default checkpoint loaded by `test.py`. |
| `checkpoints/train_demo.pth` | Checkpoint generated by running the demonstration training script `train.py`.                                    |

The default testing workflow uses:

```text
checkpoints/best_model.pth
```

To test the model generated by `train.py`, change the checkpoint path in `test.py` to:

```text
checkpoints/train_demo.pth
```

## Requirements

The code was developed and tested with:

```text
Python 3.9.16
```

Main Python dependencies include:

```text
numpy
pandas
scipy
mne
mne-connectivity
torch
matplotlib
h5py
openpyxl
```

A CUDA-enabled GPU can be used for model training and inference. The Python scripts also select the CPU when CUDA is unavailable.

Install the required packages from the repository root:

```bash
pip install -r requirements.txt
```

## Usage

Run all commands from the repository root.

### 1. Segment the example EEG data

```bash
python segment.py
```

`segment.py` demonstrates the segmentation procedure using one short example EEG file:

```text
data/train_data/raw/20240805T193613.csv
```

and its corresponding timetable:

```text
data/train_data/fn_rx_timetable.xlsx
```

The script is intended to illustrate the segmentation logic for a single example file.

The raw EEG CSV file contains the time variable `t` in the first column, followed by the voltage signals from the 16 EEG channels in the remaining columns.

The segmented outputs are saved under:

```text
data/train_data/segment_datas/
```

Expected outputs are:

```text
data/train_data/segment_datas/find_numbers_segment.npy
data/train_data/segment_datas/fn_grades.npy
data/train_data/segment_datas/relax_numbers_segment.npy
```

### 2. Build the training and validation datasets

```bash
python build_dataset.py
```

The dataset-construction script demonstrates the processing logic using the included demo data:

```text
data/train_data/segment_datas/find_numbers_segment.npy
data/train_data/segment_datas/relax_numbers_segment.npy
data/train_data/segment_datas/fn_grades.npy
data/train_data/segment_datas/sleep_data.csv
```

The generated datasets are saved as:

```text
data/train_data/processed/train_dataset.pt
data/train_data/processed/valid_dataset.pt
```

### 3. Run the demonstration training workflow

```bash
python train.py
```

The training script loads:

```text
data/train_data/processed/train_dataset.pt
data/train_data/processed/valid_dataset.pt
```

The best checkpoint produced by the demonstration training workflow is saved as:

```text
checkpoints/train_demo.pth
```

### 4. Test the model

```bash
python test.py
```

By default, the testing script loads:

```text
checkpoints/best_model.pth
data/test_data/processed_data/test_data.npy
```

The prediction figure and CSV file are saved as:

```text
data/test_data/result/test_result.png
data/test_data/result/test_result.csv
```

To evaluate the checkpoint generated by the demonstration training workflow, update the checkpoint path in `test.py` from:

```python
params = torch.load(r"./checkpoints/best_model.pth")
```

to:

```python
params = torch.load(r"./checkpoints/train_demo.pth")
```

## Additional analysis and visualization

The [`additional analysis and visualization/`](additional%20analysis%20and%20visualization/) directory contains reference implementations of the analysis and visualization procedures used in the manuscript.

These scripts are provided as reference implementations of the analysis and visualization procedures used in the manuscript. The complete study datasets required by these scripts are not included in this review repository.

The default input paths, analysis intervals, channel mappings, and execution switches are defined near the beginning of each script. Users applying these scripts to their own data should update these settings and prepare input files with the expected variables and data structures.

### Python scripts

#### `plot_functional_connectivity.py`

Generates frequency-band functional-connectivity matrices and circular connectivity plots from filtered 16-channel EEG data.

**Default input file**

```text
example_data/eeg_data_filtered.mat
```

**Expected variables**

| Variable             | Required | Description                                                                                |
| -------------------- | -------- | ------------------------------------------------------------------------------------------ |
| `eeg_data_flt`       | Yes      | Two-dimensional filtered EEG array. One dimension must contain the 16 EEG channels.        |
| `data_invalid_array` | No       | Array with the same orientation as `eeg_data_flt`. Nonzero samples are treated as invalid. |
| `fs`                 | No       | Sampling frequency in Hz. A default value is used if this variable is absent.              |

**Python dependencies**

```text
numpy
pandas
matplotlib
h5py
mne
mne-connectivity
```

**Main configuration variables**

```text
DATA_FILE
OUTPUT_FOLDER
START_TIME_SEC
END_TIME_SEC
EPOCH_LENGTH_SEC
CH_NAMES
BANDS
VMIN
VMAX
```

#### `plot_scalp_topography.py`

Generates scalp-topography maps from channel-level EEG band-power values stored in CSV files.

**Default input folder**

```text
example_data/scalp_map_csv/
```

**Expected input files**

```text
*.csv
```

**Expected CSV columns**

| Column           | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `channel`        | EEG channel name compatible with the MNE montage used by the script. |
| `total_power_dB` | Total EEG power in decibels.                                         |
| `delta_power_dB` | Delta-band power in decibels.                                        |
| `theta_power_dB` | Theta-band power in decibels.                                        |
| `alpha_power_dB` | Alpha-band power in decibels.                                        |
| `beta_power_dB`  | Beta-band power in decibels.                                         |

**Python dependencies**

```text
numpy
pandas
matplotlib
mne
```

**Main configuration variables**

```text
INPUT_FOLDER
OUTPUT_FOLDER
SFREQ
COLOR_RANGE_TOTAL_POWER
COLOR_RANGE_BAND_DB
CONTOUR_NUM
COLORMAP
```

### MATLAB scripts

#### `eeg_signal_processing_pipeline.m`

Implements channel reordering, FIR band-pass filtering, data-loss marking, motion-artifact marking, and optional correction of the F3 disconnection period.

**Default input files**

```text
example_data/raw_data.mat
example_data/label.mat
```

**Required variables in `raw_data.mat`**

| Variable          | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `eeg_data_raw`    | Raw EEG samples arranged as samples × channels.         |
| `timePoints`      | Datetime value for each EEG sample.                     |
| `t_list`          | Relative time vector.                                   |
| `sample_period`   | Sampling period.                                        |
| `fs`              | Sampling frequency in Hz.                               |
| `startTime`       | Recording start time.                                   |
| `data_loss_index` | Logical or numeric index identifying data-loss samples. |

**Optional variable in `label.mat`**

| Variable         | Description                                 |
| ---------------- | ------------------------------------------- |
| `timePoints_arr` | Activity label associated with each sample. |

**Required MATLAB toolbox**

```text
Signal Processing Toolbox
```

The toolbox is required for signal-processing functions such as `fir2`, `freqz`, and `filtfilt`.

**Main configuration variables**

```text
RAW_DATA_FILE
LABEL_FILE
OUTPUT_FILE
APPLY_F3_DISCONNECTION_CORRECTION
PLOT_FILTER_RESPONSE
```

The hard-coded F3 disconnection interval is specific to the original recording and should be disabled or updated when processing another dataset.

#### `motion_artifact_fft_wtc_snr.m`

Implements FFT spectrum export, EEG–motion wavelet-coherence visualization, and signal-to-noise-ratio calculation after harmonic-peak removal.

The analysis components can be enabled or disabled using:

```text
RUN_EXPORT_FFT
RUN_WAVELET_COHERENCE
RUN_SNR_CALCULATION
```

**Default FFT and wavelet-coherence input files**

```text
example_data/motion_artifact/ECG_data_*.mat
example_data/motion_artifact/ECG_data_example.mat
```

**Expected variable**

| Variable     | Description                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------- |
| `data_table` | MATLAB table containing a metadata or time column followed by EEG and motion-sensor channels. |

**Optional colormap file**

```text
example_data/motion_artifact/colormap_c.mat
```

**Optional colormap variable**

| Variable | Description                                                                                |
| -------- | ------------------------------------------------------------------------------------------ |
| `c`      | Custom colormap matrix. MATLAB's default colormap is used if this variable is unavailable. |

**Default SNR-analysis input folder**

```text
example_data/motion_artifact/filtered_data/
```

**Supported SNR input formats**

```text
*.xlsx
*.xls
*.csv
```

The first table column is treated as an index or metadata column. The remaining columns should contain numeric EEG signals.

Recommended EEG column names include both the channel and electrode type, for example:

```text
T8_gel
T8_paste
P4_gel
P4_paste
```

**Required MATLAB toolbox**

```text
Signal Processing Toolbox
```

This toolbox is required for functions such as `pwelch` and `hamming`.

**Additional toolbox required for wavelet coherence**

```text
Wavelet Toolbox
```

The Wavelet Toolbox is required when `RUN_WAVELET_COHERENCE` is enabled because the analysis uses `wcoherence`.

#### `psg_validation_sleep_analysis.m`

Implements wireless-EEG preprocessing, whole-night waveform and spectrogram generation, 30-second PSG comparison plots, and sleep-stage agreement analysis.

The individual procedures can be enabled or disabled using the `RUN_*` switches near the beginning of the script.

**Default input files**

```text
example_data/psg_validation/raw_data.mat
example_data/psg_validation/eeg_data_filtered.mat
example_data/psg_validation/eeg_data_hospital.mat
example_data/psg_validation/sleep_stage_labels.xlsx
```

##### `raw_data.mat`

This file is required when:

```text
RUN_PREPROCESS_WIRELESS_SLEEP_EEG = true
```

It must contain:

| Variable        | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `raw_data_list` | Structure array containing the raw wireless EEG recording segments. |

Each element of `raw_data_list` is expected to contain:

```text
eeg_data_raw
fs
sample_period
startTime
timePoints
t_list
data_loss_index
```

##### `eeg_data_filtered.mat`

| Variable             | Description                                     |
| -------------------- | ----------------------------------------------- |
| `eeg_data_flt`       | Filtered wireless EEG data.                     |
| `fs`                 | Wireless EEG sampling frequency.                |
| `timePoints`         | Datetime value for each wireless EEG sample.    |
| `data_invalid_array` | Per-sample or per-channel invalid-data markers. |

##### `eeg_data_hospital.mat`

| Variable            | Description                        |
| ------------------- | ---------------------------------- |
| `eeg_data_hospital` | Hospital PSG EEG data.             |
| `t_list`            | Relative hospital PSG time vector. |
| `fs`                | Hospital PSG sampling frequency.   |

##### `sleep_stage_labels.xlsx`

The spreadsheet must contain at least three columns:

| Column   | Description                       |
| -------- | --------------------------------- |
| Column 1 | Epoch time or epoch index.        |
| Column 2 | Hospital PSG sleep-stage code.    |
| Column 3 | Wireless-system sleep-stage code. |

The sleep-stage columns are expected to use numeric stage codes from `1` to `5`.

**Required MATLAB toolbox**

```text
Signal Processing Toolbox
```

This toolbox is required for functions such as `highpass`, `fir2`, `filtfilt`, `hann`, and `stft`.

**Optional external function**

```text
cbrewer2
```

If `cbrewer2` is unavailable, the script uses MATLAB's built-in colormap instead.

### MATLAB toolbox summary

| Script                             | Signal Processing Toolbox | Wavelet Toolbox                  | Other requirements     |
| ---------------------------------- |:-------------------------:|:--------------------------------:| ---------------------- |
| `eeg_signal_processing_pipeline.m` | Required                  | No                               | None                   |
| `motion_artifact_fft_wtc_snr.m`    | Required                  | Required when using `wcoherence` | None                   |
| `psg_validation_sleep_analysis.m`  | Required                  | No                               | `cbrewer2` is optional |

### Data availability

The file paths and variable structures above describe the expected interfaces of the scripts. The complete study datasets required to reproduce all manuscript analyses are not included in this repository.

Users must supply appropriately formatted input data or obtain authorized access to the relevant study data before running the complete analysis workflows.

## Generated outputs

Depending on the scripts executed, generated files are saved under:

```text
data/train_data/segment_datas/
data/train_data/processed/
checkpoints/
data/test_data/result/
outputs/
```

## Citation

If you use this code, please cite the associated manuscript.

Citation metadata are provided in:

```text
CITATION.cff
```

## License

This repository is released under the MIT License. See [`LICENSE`](https://chatgpt.com/LICENSE) for details.
