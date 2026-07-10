# -*- coding: utf-8 -*-
"""
Compute EEG functional connectivity from a filtered MATLAB EEG file and
save coherence matrices and circular connectivity plots.
"""

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle


# =============================================================================
# Paths
# =============================================================================

DATA_FILE = "example_data/eeg_data_filtered.mat"
OUTPUT_FOLDER = "outputs/functional_connectivity"


# =============================================================================
# Segment selection
# =============================================================================

START_TIME_SEC = 15 * 3600 + 33 * 60 + 5
END_TIME_SEC   = 15 * 3600 + 34 * 60 + 5

EPOCH_LENGTH_SEC = 2.0


# =============================================================================
# Channel and connectivity settings
# =============================================================================

CH_NAMES = [
    "O2", "Oz", "O1", "P8", "P4", "Pz", "P3", "P7",
    "T8", "C4", "Cz", "C3", "T7", "F4", "Fz", "F3"
]

CH_PLOT_ORDER = [
    "O2", "Oz", "O1", "P8", "P4", "Pz", "P3", "P7",
    "T8", "C4", "Cz", "C3", "T7", "F4", "Fz", "F3"
]

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "theta_alpha_beta": (4.0, 30.0),
    "delta_theta_alpha_beta": (1.0, 30.0),
}

VMIN = 0.9
VMAX = 1.0
N_LINES = None

MODE = "multitaper"

CONN_COLORMAP = "RdBu_r"
NODE_COLORMAP = "RdBu_r"

DPI = 600
SAVE_SELECTED_TRACE = True


# =============================================================================
# Data loading
# =============================================================================

def read_scalar_from_h5(h5_file, key, default_value=None):
    """Read a scalar variable from a MATLAB v7.3 HDF5 file."""
    if key not in h5_file:
        if default_value is None:
            raise KeyError(f"Variable '{key}' was not found in the MAT file.")
        return default_value

    value = np.array(h5_file[key])
    return float(np.ravel(value)[0])


def orient_to_channels_by_samples(arr, n_channels):
    """Return a 2D EEG array as channels x samples."""
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

    if arr.shape[0] == n_channels:
        return arr

    if arr.shape[1] == n_channels:
        return arr.T

    raise ValueError(
        f"Cannot determine data orientation for shape {arr.shape}. "
        f"One dimension should be n_channels={n_channels}."
    )


def load_filtered_eeg(mat_file):
    """Load filtered EEG data, invalid-sample labels and sampling rate."""
    mat_file = Path(mat_file)

    if not mat_file.is_file():
        raise FileNotFoundError(f"MAT file was not found: {mat_file}")

    with h5py.File(mat_file, "r") as f:
        if "eeg_data_flt" not in f:
            raise KeyError("Variable 'eeg_data_flt' was not found in the MAT file.")

        eeg_data_mv = orient_to_channels_by_samples(
            np.array(f["eeg_data_flt"]),
            n_channels=len(CH_NAMES)
        )

        if "data_invalid_array" in f:
            invalid_array = orient_to_channels_by_samples(
                np.array(f["data_invalid_array"]),
                n_channels=len(CH_NAMES)
            )
        else:
            print("Warning: data_invalid_array was not found. All samples are treated as valid.")
            invalid_array = np.zeros_like(eeg_data_mv)

        fs = read_scalar_from_h5(f, "fs", default_value=125.0)

    eeg_data_mv = eeg_data_mv.astype(float)
    invalid_array = invalid_array.astype(float)

    eeg_data_mv[invalid_array != 0] = 0.0

    print(f"Loaded EEG data: {eeg_data_mv.shape} [channels x samples]")
    print(f"Sampling rate: {fs:.6f} Hz")

    return eeg_data_mv, invalid_array, fs


# =============================================================================
# Epoch creation
# =============================================================================

def create_epochs_from_segment(eeg_data_mv, fs, start_time_sec, end_time_sec, epoch_length_sec):
    """Select a time segment and create fixed-length MNE epochs."""
    start_index = int(round(start_time_sec * fs))
    end_index = int(round(end_time_sec * fs))

    start_index = max(start_index, 0)
    end_index = min(end_index, eeg_data_mv.shape[1])

    if end_index <= start_index:
        raise ValueError("END_TIME_SEC must be greater than START_TIME_SEC.")

    selected_mv = eeg_data_mv[:, start_index:end_index]
    selected_v = selected_mv * 1e-3

    if selected_v.shape[1] < int(round(epoch_length_sec * fs)) * 2:
        raise ValueError(
            "The selected segment is too short. "
            "Use a longer segment or a shorter EPOCH_LENGTH_SEC."
        )

    info = mne.create_info(
        ch_names=CH_NAMES,
        sfreq=fs,
        ch_types=["eeg"] * len(CH_NAMES)
    )

    info.set_montage("standard_1005", match_case=False, on_missing="ignore")

    raw = mne.io.RawArray(selected_v, info, verbose="ERROR")

    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_length_sec,
        overlap=0.0,
        preload=True,
        verbose="ERROR"
    )

    print(f"Selected segment: {start_time_sec:.2f}-{end_time_sec:.2f} s")
    print(f"Created {len(epochs)} epochs, each {epoch_length_sec:.2f} s.")

    return epochs, selected_mv, start_index, end_index


# =============================================================================
# Connectivity calculation
# =============================================================================

def compute_coherence_matrix(epochs, fmin, fmax):
    """Compute a pairwise spectral-coherence matrix."""
    con = spectral_connectivity_epochs(
        epochs,
        method="coh",
        mode=MODE,
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose="ERROR"
    )

    conmat = con.get_data(output="dense")[:, :, 0]
    conmat = np.nan_to_num(conmat, nan=0.0)

    conmat = np.maximum(conmat, conmat.T)
    np.fill_diagonal(conmat, 0.0)

    return conmat


def save_connectivity_matrix(conmat, output_file):
    """Save a labeled connectivity matrix as CSV."""
    df = pd.DataFrame(conmat, index=CH_NAMES, columns=CH_NAMES)
    df.to_csv(output_file)


# =============================================================================
# Plotting
# =============================================================================

def get_node_colors():
    """Generate node colors using the selected node colormap."""
    cmap = mpl.colormaps.get_cmap(NODE_COLORMAP)
    return [cmap(i / len(CH_PLOT_ORDER)) for i in range(len(CH_PLOT_ORDER))]


def plot_connectivity_on_axis(conmat, band_name, ax):
    """Plot one connectivity circle on a polar axis."""
    conmat_plot = np.where(conmat > VMIN, conmat, np.nan)

    node_angles = circular_layout(
        CH_NAMES,
        CH_PLOT_ORDER,
        start_pos=90
    )

    plot_connectivity_circle(
        conmat_plot,
        CH_NAMES,
        n_lines=N_LINES,
        node_angles=node_angles,
        node_width=20,
        node_height=1.0,
        node_colors=get_node_colors(),
        node_edgecolor="white",
        node_linewidth=2.0,
        facecolor="white",
        textcolor="black",
        linewidth=4,
        title=band_name,
        fontsize_title=12,
        fontsize_names=32,
        fontsize_colorbar=10,
        padding=6.0,
        colormap=CONN_COLORMAP,
        colorbar=True,
        colorbar_size=0.3,
        colorbar_pos=(-0.3, 0.3),
        show=False,
        ax=ax,
        vmin=VMIN,
        vmax=VMAX
    )


def plot_individual_connectivity(conmat, band_name, output_file):
    """Save one standalone circular connectivity figure."""
    fig, ax = plt.subplots(
        figsize=(8, 8),
        facecolor="white",
        subplot_kw={"polar": True}
    )

    plot_connectivity_on_axis(conmat, band_name, ax)

    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined_connectivity(conmat_dict, output_file):
    """Save a combined 3 x 2 connectivity figure."""
    fig, axes = plt.subplots(
        3, 2,
        figsize=(12, 18),
        facecolor="white",
        subplot_kw={"polar": True}
    )

    panel_order = [
        ("delta", "Delta Band (1-4 Hz)"),
        ("theta", "Theta Band (4-8 Hz)"),
        ("alpha", "Alpha Band (8-12 Hz)"),
        ("beta", "Beta Band (12-30 Hz)"),
        ("theta_alpha_beta", "Total Band (4-30 Hz)"),
        ("delta_theta_alpha_beta", "Total Band + Delta (1-30 Hz)"),
    ]

    for ax, (key, title) in zip(axes.ravel(), panel_order):
        plot_connectivity_on_axis(conmat_dict[key], title, ax)

    plt.tight_layout()
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_selected_trace(selected_mv, fs, start_index, output_file):
    """Save the selected 16-channel EEG trace for visual inspection."""
    t = (np.arange(selected_mv.shape[1]) + start_index) / fs

    plt.figure(figsize=(12, 8))

    offset_uv = 100
    for i, ch_name in enumerate(CH_NAMES):
        plt.plot(t, selected_mv[i, :] * 1000 + i * offset_uv, label=ch_name, linewidth=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("EEG with channel offset (uV)")
    plt.title("Selected EEG segment")
    plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=8)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_data_mv, invalid_array, fs = load_filtered_eeg(DATA_FILE)

    epochs, selected_mv, start_index, end_index = create_epochs_from_segment(
        eeg_data_mv=eeg_data_mv,
        fs=fs,
        start_time_sec=START_TIME_SEC,
        end_time_sec=END_TIME_SEC,
        epoch_length_sec=EPOCH_LENGTH_SEC
    )

    if SAVE_SELECTED_TRACE:
        trace_file = output_dir / "selected_eeg_trace.png"
        save_selected_trace(selected_mv, fs, start_index, trace_file)
        print(f"Saved selected trace: {trace_file}")

    conmat_dict = {}

    for band_name, (fmin, fmax) in BANDS.items():
        print(f"Computing {band_name}: {fmin:.1f}-{fmax:.1f} Hz")

        conmat = compute_coherence_matrix(epochs, fmin=fmin, fmax=fmax)
        conmat_dict[band_name] = conmat

        csv_file = output_dir / f"connectivity_matrix_{band_name}.csv"
        png_file = output_dir / f"connectivity_circle_{band_name}.png"

        save_connectivity_matrix(conmat, csv_file)
        plot_individual_connectivity(conmat, band_name, png_file)

        print(f"Saved matrix: {csv_file}")
        print(f"Saved figure: {png_file}")

    combined_file = output_dir / "functional_connectivity_combined_3x2.png"
    plot_combined_connectivity(conmat_dict, combined_file)
    print(f"Saved combined figure: {combined_file}")

    print("All connectivity analysis finished.")


if __name__ == "__main__":
    main()
