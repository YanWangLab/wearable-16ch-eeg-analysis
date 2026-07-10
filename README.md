# Wearable 16-Channel EEG Analysis Code

This repository contains demo code for EEG segmentation, preprocessing, feature extraction, model training, testing, and supplementary visualization. The code is provided as the code-availability repository for the associated manuscript.

The full experimental EEG dataset is not publicly released. A small demo dataset is included to demonstrate the complete computational workflow.

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
│   │   ├── segment_datas/
│   │   ├── processed/
│   │   └── fn_rx_timetable.xlsx
│   └── test_data/
│       ├── processed_data/
│       └── result/
│
├── checkpoints/
│
└── additional analysis and visualization/
    ├── python/
    └── matlab/
```

## Main scripts

| File | Description |
|---|---|
| `segment.py` | Segments raw EEG data according to the timetable. |
| `preprocessing.py` | Performs EEG filtering, STFT calculation, frequency-feature extraction, and sliding-window construction. |
| `build_dataset.py` | Builds PyTorch training and validation datasets from segmented EEG data. |
| `model.py` | Defines the LSTM-based EEG regression model. |
| `train.py` | Trains the model and saves the best checkpoint. |
| `test.py` | Runs inference on demo test data and saves the prediction results. |

## Requirements

The code was developed with Python 3.9. A CUDA-enabled GPU is optional; the demo can also run on CPU.

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main Python dependencies include:

```text
numpy
pandas
scipy
mne
torch
matplotlib
h5py
mne-connectivity
openpyxl
```

## Demo data

The demo data are organized under `data/`. The expected files include:

```text
data/train_data/raw/20260805T193613.csv
data/train_data/fn_rx_timetable.xlsx
data/test_data/processed_data/test_data.npy
```

The full original dataset used in the manuscript is not included in this repository because of data-sharing restrictions.

## Usage

Run the pipeline from the repository root.

### 1. Segment EEG data

```bash
python segment.py
```

This creates segmented EEG files under:

```text
data/train_data/segment_datas/
```

### 2. Build train and validation datasets

```bash
python build_dataset.py
```

This creates:

```text
data/train_data/processed/train_dataset.pt
data/train_data/processed/valid_dataset.pt
```

### 3. Train the model

```bash
python train.py
```

The training script saves the demo checkpoint to:

```text
checkpoints/train_demo.pth
```

### 4. Test the model

```bash
python test.py
```

The current test script loads:

```text
checkpoints/best_model.pth
```

and saves the outputs to:

```text
data/test_data/result/test_result.png
data/test_data/result/test_result.csv
```

If using a newly trained checkpoint, update the checkpoint path in `test.py` or copy the trained checkpoint to `checkpoints/best_model.pth`.

## Supplementary analysis and visualization

Additional scripts are provided for visualization and supplementary analysis:

```text
additional analysis and visualization/python/plot_functional_connectivity.py
additional analysis and visualization/python/plot_scalp_topography.py
additional analysis and visualization/matlab/eeg_signal_processing_pipeline.m
additional analysis and visualization/matlab/motion_artifact_fft_wtc_snr.m
additional analysis and visualization/matlab/psg_validation_sleep_analysis.m
```

Run the Python visualization scripts with:

```bash
python "additional analysis and visualization/python/plot_functional_connectivity.py"
python "additional analysis and visualization/python/plot_scalp_topography.py"
```

The MATLAB scripts require MATLAB and the corresponding example input files.

## Outputs

Typical generated outputs include:

```text
data/train_data/segment_datas/
data/train_data/processed/
checkpoints/
data/test_data/result/
outputs/
```

## Code availability statement

The analysis code and demo data required to reproduce the computational workflow are available in this repository under the MIT License. The complete raw EEG dataset is not publicly available because of data-sharing and privacy restrictions. Demo files are provided to verify the code execution pipeline.

## Citation

If you use this code, please cite the associated manuscript. Citation metadata are provided in `CITATION.cff`.

## License

This repository is released under the MIT License. See `LICENSE` for details.
