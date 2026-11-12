"""Trajectory analysis: kinematics, MSD, autocorrelations.

Tracks come as a table with one row per detection and the columns
track_id, frame, t (s), x_um, y_um  (see motility.io).
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs, savgol_filter


# ------------------------------------------------------------------ tracks

def track_arrays(df, cols=("x_um", "y_um")):
    """List of arrays, one per track (sorted by frame), with the given columns."""
    df = df.sort_values(["track_id", "frame"])
    return [g[list(cols)].to_numpy() for _, g in df.groupby("track_id", sort=True)]


# ------------------------------------------------------------------ kinematics

def kinematics(df, fps, window=7, polyorder=2):
    """Velocity, speed, heading and angular velocity for every frame of every track.

    Positions are smoothed with a Savitzky-Golay filter: in a sliding window of `window` frames, fit a polynomial
    of order `polyorder` and take its derivatives at the window center. Tracks shorter than `window` are dropped.
    Angular velocity: omega = (vx*ay - vy*ax) / |v|^2  (rad/s), the rate of change of the heading.
    """
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and larger than polyorder")
    dt = 1.0 / fps
    parts = []
    for tid, g in df.sort_values(["track_id", "frame"]).groupby("track_id", sort=True):
        if len(g) < window:
            continue
        xy = g[["x_um", "y_um"]].to_numpy()
        vel = savgol_filter(xy, window, polyorder, deriv=1, delta=dt, axis=0)
        acc = savgol_filter(xy, window, polyorder, deriv=2, delta=dt, axis=0)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            omega = (vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / speed**2
        parts.append(pd.DataFrame({
            "track_id": tid, "frame": g["frame"].to_numpy(), "t": g["t"].to_numpy(),
            "vx": vel[:, 0], "vy": vel[:, 1], "speed": speed,
            "heading": np.arctan2(vel[:, 1], vel[:, 0]), "omega": omega}))
    return pd.concat(parts, ignore_index=True)


def velocity_noise(sigma_loc, fps, window=None, polyorder=2):
    """Standard deviation of each velocity component caused by localization noise sigma_loc (µm).
    window=None: plain frame-to-frame differences, sqrt(2)*sigma_loc*fps.
    Otherwise: the Savitzky-Golay derivative filter used by `kinematics`."""
    if window is None:
        return np.sqrt(2) * sigma_loc * fps
    c = savgol_coeffs(window, polyorder, deriv=1, delta=1.0 / fps)
    return sigma_loc * np.sqrt(np.sum(c**2))


def track_summary(kin, fps, block_s=1.0):
    """Per-track mean speed with its standard error.
    Speeds in neighboring frames are correlated, so the error comes from the scatter of block means
    (blocks of block_s seconds), not from the scatter of single frames."""
    block = max(1, int(round(block_s * fps)))
    rows = []
    for tid, g in kin.groupby("track_id", sort=True):
        s = g["speed"].to_numpy()
        nb = len(s) // block
        means = s[: nb * block].reshape(nb, block).mean(axis=1) if nb else np.array([])
        sem = means.std(ddof=1) / np.sqrt(nb) if nb >= 2 else np.nan
        rows.append({"track_id": tid, "duration_s": len(s) / fps, "mean_speed": s.mean(),
                     "sem_speed": sem, "n_blocks": nb})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ MSD

def log_lags(max_lag, n=30):
    """Log-spaced integer lags (in frames) from 1 to max_lag."""
    return np.unique(np.logspace(0, np.log10(max_lag), n).astype(int))


def msd(tracks, lags):
    """Time- and ensemble-averaged MSD, < |r(t + lag) - r(t)|^2 >, over all start times t of all tracks.

    tracks: an array (n_tracks, n_frames, 2) or a list of (n_frames_i, 2) arrays of different lengths.
    lags: in frames. Returns NaN for lags longer than every track.
    For long lags, start times are spaced lag/10 frames apart: heavily overlapping windows add almost
    no information, and skipping them keeps this fast."""
    if isinstance(tracks, np.ndarray):
        tracks = [tracks]                                 # one (n, nf, 2) block: all tracks at once
    out = []
    for k in lags:
        s = max(1, k // 10)
        total, count = 0.0, 0
        for tr in tracks:
            n = tr.shape[-2]
            if n <= k:
                continue
            d = tr[..., k::s, :] - tr[..., : n - k : s, :]
            total += np.sum(d**2)
            count += d[..., 0].size
        out.append(total / count if count else np.nan)
    return np.array(out)


def msd_many(simulate, lags, n_batches):
    """MSD averaged over n_batches batches of simulated tracks; simulate() returns one batch (n, n_frames, 2).
    Only one batch is in memory at a time."""
    return sum(msd(simulate(), lags) for _ in range(n_batches)) / n_batches


def run_durations(events, fps):
    """Durations (s) of the complete runs between consecutive tumbles; events (n_tracks, n_frames) bool."""
    return np.concatenate([np.diff(np.flatnonzero(e)) for e in events]) / fps


def local_slope(tau, m):
    """alpha(tau) = d log MSD / d log tau."""
    return np.gradient(np.log(m), np.log(tau))


# ------------------------------------------------------------------ autocorrelation

def autocorrelation(vectors, max_lag):
    """C(k) = < a(t) . a(t+k) >, averaged over all t of all tracks, for k = 0..max_lag frames.

    vectors: list of (n_i, 2) arrays (for example velocities, or unit heading vectors), one per track.
    Returns (C, counts); counts[k] is the number of pairs averaged at lag k."""
    total = np.zeros(max_lag + 1)
    count = np.zeros(max_lag + 1)
    for a in vectors:
        n = len(a)
        for k in range(min(max_lag, n - 1) + 1):
            total[k] += np.sum(a[: n - k] * a[k:])
            count[k] += n - k
    with np.errstate(invalid="ignore"):
        return total / count, count


# ------------------------------------------------------------------ theory

def msd_brownian(tau, D, sigma_loc=0.0):
    """Brownian motion in 2D, plus the offset 4 sigma^2 from localization noise."""
    return 4 * D * tau + 4 * sigma_loc**2


def msd_biased(tau, D, vd):
    """Drift plus diffusion in 2D."""
    return 4 * D * tau + vd**2 * tau**2


def msd_persistent(tau, v, tau_p):
    """Persistent random walk in 2D (constant speed v, velocity correlation decaying as exp(-t/tau_p))."""
    return 2 * v**2 * tau_p**2 * (tau / tau_p - 1 + np.exp(-tau / tau_p))
