# -*- coding: utf-8 -*-
"""
Draw EEG scalp topography maps from MATLAB-exported band-power CSV files.
Input CSV files are read from example_data/scalp_map_csv/, and output PNG
figures are saved to outputs/scalp_topography/.
"""

import os
import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================= Paths =======================

INPUT_FOLDER = os.path.join("example_data", "scalp_map_csv")
OUTPUT_FOLDER = os.path.join("outputs", "scalp_topography")


# ======================= Plot parameters =======================

SFREQ = 125  # Used only to create MNE info.

COLOR_RANGE_TOTAL_POWER = (10, 25)
COLOR_RANGE_BAND_DB = (-5, 15)
CONTOUR_NUM = 12
DPI = 600

DRAW_CHANNEL_POSITION = True
COLORMAP = "RdBu_r"


# ======================= CSV column compatibility =======================

COLUMN_ALIASES = {
    "total_power_dB": ["total power (dB)", "total_power_dB", "total power dB"],
    "delta_power_dB": ["delta power (dB)", "delta_power_dB", "delta power dB"],
    "theta_power_dB": ["theta power (dB)", "theta_power_dB", "theta power dB"],
    "alpha_power_dB": ["alpha power (dB)", "alpha_power_dB", "alpha power dB"],
    "beta_power_dB":  ["beta power (dB)",  "beta_power_dB",  "beta power dB"],
}


def find_column(df: pd.DataFrame, canonical_name: str) -> str:
    """Find a matching column name in the CSV file."""
    for col in COLUMN_ALIASES[canonical_name]:
        if col in df.columns:
            return col

    raise KeyError(
        f"Cannot find column for {canonical_name}. "
        f"Tried {COLUMN_ALIASES[canonical_name]}. "
        f"Available columns are {list(df.columns)}"
    )


def create_info_from_channels(ch_names):
    """Create an MNE Info object with the standard 10-05 montage."""
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=SFREQ,
        ch_types="eeg"
    )

    info.set_montage("standard_1005", match_case=False, on_missing="raise")
    return info


def draw_one_map(data, info, title, color_range, output_file):
    """Draw one scalp topography map."""
    fig, ax = plt.subplots(figsize=(6, 6))

    sensors = "k." if DRAW_CHANNEL_POSITION else False

    im, _ = mne.viz.plot_topomap(
        data=np.asarray(data, dtype=float),
        pos=info,
        axes=ax,
        sensors=sensors,
        names=None,
        contours=CONTOUR_NUM,
        res=300,
        size=8,
        cmap=COLORMAP,
        vlim=color_range,
        show=False,
        outlines="head"
    )

    ax.set_title(title, fontsize=14)

    for line in ax.lines:
        line.set_linewidth(3)
        line.set_color("black")

    for patch in ax.patches:
        patch.set_linewidth(3)
        patch.set_edgecolor("black")

    for coll in ax.collections:
        try:
            coll.set_linewidth(2)
        except Exception:
            pass

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Power (dB)")

    fig.savefig(output_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def process_one_csv(csv_file: str):
    """Read one CSV file and draw scalp topography maps."""
    csv_path = Path(csv_file)
    df = pd.read_csv(csv_path)

    if "channel" not in df.columns:
        raise KeyError(f"'channel' column not found in {csv_path}")

    ch_names = df["channel"].astype(str).tolist()
    info = create_info_from_channels(ch_names)

    stem = csv_path.stem.replace(" ", "_").replace("@", "_")
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_col = find_column(df, "total_power_dB")
    draw_one_map(
        data=df[total_col].values,
        info=info,
        title="Total power",
        color_range=COLOR_RANGE_TOTAL_POWER,
        output_file=output_dir / f"{stem}_total_power_dB.png"
    )

    for band_name in ["delta", "theta", "alpha", "beta"]:
        canonical = f"{band_name}_power_dB"
        col = find_column(df, canonical)

        draw_one_map(
            data=df[col].values,
            info=info,
            title=f"{band_name} power",
            color_range=COLOR_RANGE_BAND_DB,
            output_file=output_dir / f"{stem}_{band_name}_power_dB.png"
        )


def main():
    if not os.path.isdir(INPUT_FOLDER):
        raise FileNotFoundError(f"INPUT_FOLDER does not exist: {INPUT_FOLDER}")

    csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in: {INPUT_FOLDER}")

    print(f"Found {len(csv_files)} CSV files.")
    print(f"Output folder: {OUTPUT_FOLDER}")

    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        process_one_csv(csv_file)

    print("All scalp topography maps finished.")


if __name__ == "__main__":
    main()
