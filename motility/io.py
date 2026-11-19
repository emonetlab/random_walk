"""Load the course datasets.

A dataset is a table of detections, one row per cell per frame, with the columns
    track_id   integer, the same for all detections of one cell
    frame      integer frame number
    t          time in s
    x_um, y_um position in µm
plus metadata (frame rate, localization noise, description, ...) as a dict.

Synthetic datasets (data/synthetic) are CSV + JSON files; they also have a "truth" table with the true behavioral
state of every detection.  Experimental datasets (data/experimental) are MATLAB .mat files as they come out of
tracking software; `load_mat_tracks` converts them to the same table, so every tool of the course applies.
"""
import json
from pathlib import Path

import numpy as np
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


# ------------------------------------------------------------------ MATLAB files

def mat_header(path):
    """The text header of a .mat file (MATLAB version, platform, creation date), or '' if there is none."""
    with open(path, "rb") as f:
        head = f.read(116)
    return head.decode("latin-1").strip() if head.startswith(b"MATLAB") else ""


def read_mat(path):
    """Every variable of a .mat file as a dict {name: value}.

    Numeric arrays become numpy arrays (vectors are 1-d), strings become str, cell arrays become lists, and a struct
    becomes a dict {field: value}, a struct array a list of such dicts.  Classic files (MATLAB -v7 and older) are read
    with scipy.io, -v7.3 files (which are HDF5) with h5py: the result looks the same.
    """
    import h5py

    path = Path(path)
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as h:
            return {k: _from_h5(h, h[k]) for k in h.keys() if not k.startswith("#")}
    import scipy.io as sio
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: _from_scipy(v) for k, v in raw.items() if not k.startswith("__")}


def _from_scipy(v):
    """scipy.io.loadmat value -> plain Python / numpy."""
    if hasattr(v, "_fieldnames"):                        # one struct
        return {f: _from_scipy(getattr(v, f)) for f in v._fieldnames}
    if isinstance(v, np.ndarray) and v.dtype == object:  # struct array or cell array
        return [_from_scipy(e) for e in v.ravel()]
    return v


def _from_h5(h, obj):
    """h5py object of a MATLAB -v7.3 file -> plain Python / numpy."""
    import h5py

    if isinstance(obj, h5py.Group):                      # struct or struct array
        fields = {k: obj[k] for k in obj.keys()}
        if fields and all(isinstance(d, h5py.Dataset) and d.dtype == h5py.ref_dtype for d in fields.values()):
            refs = {k: d[()].ravel() for k, d in fields.items()}
            n = len(next(iter(refs.values())))
            if n > 1:                                    # struct array: one dict per element
                return [{k: _from_h5(h, h[r[i]]) for k, r in refs.items()} for i in range(n)]
        return {k: _from_h5(h, d) for k, d in fields.items()}
    arr = obj[()]
    if obj.attrs.get("MATLAB_class", b"") in (b"char", "char"):
        return "".join(chr(c) for c in np.asarray(arr).ravel())
    if arr.dtype == h5py.ref_dtype:                      # cell array
        return [_from_h5(h, h[r]) for r in arr.ravel()]
    arr = np.asarray(arr).T                              # MATLAB stores arrays column-major
    return np.squeeze(arr) if arr.ndim > 1 and min(arr.shape) == 1 else arr


def _describe(value, indent="   "):
    if isinstance(value, dict):
        yield "struct with fields: " + ", ".join(value)
        for k, v in value.items():
            yield f"{indent}.{k}: " + next(_describe(v, indent + "   "))
    elif isinstance(value, list):
        if value and isinstance(value[0], dict):
            yield f"struct array, {len(value)} elements, fields: " + ", ".join(value[0])
            for k, v in value[0].items():
                yield f"{indent}.{k} of element 1: " + next(_describe(v, indent + "   "))
        else:
            yield f"cell array, {len(value)} elements"
    elif isinstance(value, np.ndarray):
        yield f"{value.dtype} array, shape {value.shape}"
    else:
        yield f"{type(value).__name__}: {str(value)[:60]}"


def describe_mat(path):
    """Print what is inside a .mat file: its header, then every variable with type, shape and (for structs) fields."""
    path = Path(path)
    print(f"{path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    header = mat_header(path)
    if header:
        print(header)
    for name, value in read_mat(path).items():
        lines = list(_describe(value))
        print(f"\n{name}: {lines[0]}")
        for line in lines[1:]:
            print(line)


def mat_files_summary(folder, time_fields=("time", "t")):
    """One row per .mat file in `folder`: size, creation date, variables, and for a struct array of tracks the number
    of tracks, its fields, and the frame rate and movie length measured from its time field.  Returns a DataFrame
    (empty if the folder has no .mat file)."""
    rows = []
    for path in sorted(Path(folder).glob("*.mat")):
        row = {"file": path.name, "MB": round(path.stat().st_size / 1e6, 1),
               "created": mat_header(path).split("Created on: ")[-1] if mat_header(path) else ""}
        data = read_mat(path)
        row["variables"] = ", ".join(data)
        for name, value in data.items():
            if isinstance(value, dict):                  # a struct array with one element comes back as one struct
                value = [value]
            if isinstance(value, list) and value and isinstance(value[0], dict):   # a struct array: the tracks
                row["tracks_variable"], row["n_tracks"] = name, len(value)
                row["fields"] = ", ".join(value[0])
                tf = next((f for f in time_fields if f in value[0]), None)
                if tf is not None:
                    times = [np.atleast_1d(np.asarray(tr[tf], dtype=float)) for tr in value]
                    dts = np.concatenate([np.diff(t) for t in times if t.size > 1])
                    row["fps"] = round(1.0 / float(np.median(dts)), 1)
                    row["movie_s"] = round(float(max(t[-1] for t in times) - min(t[0] for t in times)), 1)
                break
        rows.append(row)
    return pd.DataFrame(rows)


def _longest_run(mask):
    """Length of the longest run of True values."""
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def load_mat_tracks(path, var="tracks", fields=("x", "y", "time"), missing="drop"):
    """Tracks from a .mat file -> (tracks table, metadata), in the format of the course datasets.

    The file must hold a struct array `var` with one element per track and the three `fields` giving, for every
    frame of that track, its x position, y position and time.  Positions are taken to be in µm and times in s
    (a .mat file does not store units; check them against the data, as the hands-on notebook suggests).

    frame = round(t / dt) with dt the median time step of the file, so frame numbers count frames of the whole movie.
    NaN positions (frames in which the tracker lost the cell) are handled according to `missing`:
        "drop"         leave those rows out (the default): `frame` then has gaps inside a track
        "interpolate"  fill them by linear interpolation in time, so every track has consecutive frames
    Leading and trailing NaNs are always cut off.
    The metadata are measured from the data: frame rate, time span, track durations, field of view, missing positions.
    """
    if missing not in ("drop", "interpolate"):
        raise ValueError("missing must be 'drop' or 'interpolate'")
    path = Path(path)
    data = read_mat(path)
    if var not in data:
        raise KeyError(f"{path.name} has no variable {var!r}; it has {list(data)}")
    tracks = data[var]
    if isinstance(tracks, dict):                         # a single track
        tracks = [tracks]
    fx, fy, ft = fields

    times = [np.atleast_1d(np.asarray(tr[ft], dtype=float)) for tr in tracks]
    dts = np.concatenate([np.diff(t) for t in times if t.size > 1])
    dt = float(np.median(dts))

    parts, n_missing, longest_gap, durations = [], 0, 0, []
    for i, (tr, t) in enumerate(zip(tracks, times)):
        x = np.atleast_1d(np.asarray(tr[fx], dtype=float))
        y = np.atleast_1d(np.asarray(tr[fy], dtype=float))
        ok = ~(np.isnan(x) | np.isnan(y))
        if not ok.any():
            continue
        first, last = np.flatnonzero(ok)[[0, -1]]        # cut leading and trailing NaNs
        x, y, t, ok = x[first:last + 1], y[first:last + 1], t[first:last + 1], ok[first:last + 1]
        n_missing += int((~ok).sum())
        longest_gap = max(longest_gap, _longest_run(~ok))
        if missing == "interpolate":
            x, y = np.interp(t, t[ok], x[ok]), np.interp(t, t[ok], y[ok])
        else:
            x, y, t = x[ok], y[ok], t[ok]
        durations.append(t[-1] - t[0])
        parts.append(pd.DataFrame({"track_id": i, "frame": np.rint(t / dt).astype(int),
                                   "t": t, "x_um": x, "y_um": y}))
    df = pd.concat(parts, ignore_index=True)
    durations = np.asarray(durations)

    meta = {
        "file": path.name,
        "description": f"tracks from {path.name}: variable '{var}', fields {fx}, {fy} (µm) and {ft} (s)",
        "n_tracks": int(df["track_id"].nunique()),
        "n_detections": int(len(df)),
        "fps": round(1.0 / dt, 3),
        "dt_s": round(dt, 6),
        "t_start_s": float(df["t"].min()),
        "t_end_s": float(df["t"].max()),
        "track_duration_min_s": float(durations.min()),
        "track_duration_median_s": float(np.median(durations)),
        "track_duration_max_s": float(durations.max()),
        "x_range_um": (float(df["x_um"].min()), float(df["x_um"].max())),
        "y_range_um": (float(df["y_um"].min()), float(df["y_um"].max())),
        "n_missing_positions": n_missing,
        "longest_gap_frames": longest_gap,
        "missing": missing,
    }
    return df, meta
