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
data/train_data/raw/20260805T193613.csv
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

The `additional analysis and visualization/` directory contains selected implementations of analysis and visualization methods used in the manuscript.

### Python scripts

| File                              | Description                                       |
| --------------------------------- | ------------------------------------------------- |
| `plot_functional_connectivity.py` | Generates functional-connectivity visualizations. |
| `plot_scalp_topography.py`        | Generates scalp-topography visualizations.        |

### MATLAB scripts

| File                               | Description                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| `eeg_signal_processing_pipeline.m` | Implements EEG signal-processing procedures used in the study.                                    |
| `motion_artifact_fft_wtc_snr.m`    | Implements motion-artifact, FFT, wavelet-transform coherence, and signal-to-noise-ratio analyses. |
| `psg_validation_sleep_analysis.m`  | Implements PSG validation and sleep-related analyses.                                             |

These scripts provide implementations of the corresponding methods described in the manuscript.

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
