"""Load the course datasets.

A dataset is a table of detections, one row per cell per frame, with the columns
    track_id   integer, the same for all detections of one cell
    frame      integer frame number
    t          time in s
    x_um, y_um position in µm
plus a JSON file with its metadata (frame rate, localization noise, description).
Synthetic datasets also have a "truth" table with the true behavioral state of every detection.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset(name, folder=DATA_DIR / "synthetic"):
    """Return (tracks, metadata) for <folder>/<name>.csv and <folder>/<name>.json.
    tracks is a pandas DataFrame; metadata is a dict (frame rate, localization noise, description, parameters)."""
    folder = Path(folder)
    df = pd.read_csv(folder / f"{name}.csv")
    with open(folder / f"{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    return df, meta


def load_truth(name, folder=DATA_DIR / "synthetic"):
    """True states of a synthetic dataset: track_id, frame, state ('run' or 'tumble')."""
    return pd.read_csv(Path(folder) / f"{name}_truth.csv")
