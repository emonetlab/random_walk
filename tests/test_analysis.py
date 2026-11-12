"""Checks of the analysis code on trajectories whose answer is known.  Run:  python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest

from motility import analysis as an
from motility import simulate as sim

FPS = 30.0


def as_table(positions):
    """(n_tracks, n_frames, 2) array -> tracks table."""
    n, nf, _ = positions.shape
    return pd.DataFrame({"track_id": np.repeat(np.arange(n), nf), "frame": np.tile(np.arange(nf), n),
                         "t": np.tile(np.arange(nf) / FPS, n),
                         "x_um": positions[..., 0].ravel(), "y_um": positions[..., 1].ravel()})


def test_msd_straight_line_is_ballistic():
    t = np.arange(300) / FPS
    track = np.stack([20 * t, 0 * t], axis=1)
    lags = np.array([1, 10, 100])
    assert np.allclose(an.msd([track], lags), (20 * lags / FPS) ** 2)


def test_msd_list_and_array_agree():
    rng = np.random.default_rng(0)
    r = sim.simulate_brownian(5, 10, FPS, 1.0, rng)
    lags = an.log_lags(100)
    assert np.allclose(an.msd(r, lags), an.msd(list(r), lags))


def test_msd_brownian():
    rng = np.random.default_rng(1)
    r = sim.simulate_brownian(200, 60, FPS, 0.5, rng)
    lags = an.log_lags(180, 10)
    m = an.msd(r, lags)
    assert np.allclose(m, an.msd_brownian(lags / FPS, 0.5), rtol=0.05)


def test_msd_too_long_lag_is_nan():
    assert np.isnan(an.msd([np.zeros((10, 2))], [20])[0])


def test_kinematics_circle():
    """Constant speed v on a circle of radius R: speed = v and omega = v / R."""
    v, R = 20.0, 50.0
    t = np.arange(600) / FPS
    th = v / R * t
    df = as_table(np.stack([R * np.cos(th), R * np.sin(th)], axis=-1)[None])
    kin = an.kinematics(df, FPS, window=7)
    assert np.allclose(kin["speed"], v, rtol=5e-3)        # slightly less exact at the track ends
    assert np.allclose(kin["omega"], v / R, rtol=5e-3)


def test_velocity_noise_matches_simulation():
    rng = np.random.default_rng(2)
    noise = 0.3 * rng.standard_normal((1, 20000, 2))       # a cell that does not move
    kin = an.kinematics(as_table(noise), FPS, window=9)
    assert kin["vx"].std() == pytest.approx(an.velocity_noise(0.3, FPS, window=9), rel=0.05)
    raw = np.diff(noise[0, :, 0]) * FPS
    assert raw.std() == pytest.approx(an.velocity_noise(0.3, FPS), rel=0.05)


def test_autocorrelation_constant_and_lengths():
    a = [np.tile([1.0, 0.0], (50, 1)), np.tile([0.0, 2.0], (5, 1))]
    C, counts = an.autocorrelation(a, 10)
    assert counts[0] == 55 and counts[10] == 40
    assert C[10] == pytest.approx(1.0)


def test_run_tumble_constant_rate_matches_persistent_walk():
    """Instantaneous tumbles at a constant rate: tau_p = 1 / (lambda (1 - <cos phi>) + Dr)."""
    rng = np.random.default_rng(3)
    lam, Dr, v = 1.0, 0.15, 25.0
    r, _, _ = sim.simulate_run_tumble(200, 120, FPS, rng, speed=v, run_rate=lam, Dr=Dr, tumble_time=0, substeps=1)
    tau_p = 1 / (lam * (1 - sim.turn_angle_cos()) + Dr)
    lags = an.log_lags(600, 12)
    assert np.allclose(an.msd(r, lags), an.msd_persistent(lags / FPS, v, tau_p), rtol=0.06)


def test_as_dataset_format_noise_and_truth():
    rng = np.random.default_rng(5)
    pos, states, _ = sim.simulate_run_tumble(50, 20, FPS, rng, **sim.ECOLI)
    tracks, truth = sim.as_dataset(pos, FPS, rng, states=states, sigma_loc=0.2, mean_duration=5.0)
    assert list(tracks.columns) == ["track_id", "frame", "t", "x_um", "y_um"]
    assert tracks["track_id"].nunique() == 50
    assert tracks[["track_id", "frame"]].equals(truth[["track_id", "frame"]])
    lengths = tracks.groupby("track_id").size() / FPS
    assert lengths.min() >= 2.0 and lengths.max() <= 20.0 + 1 / FPS
    # localization noise: frame-to-frame velocity noise of a stuck cell
    stuck = np.zeros((1, 6001, 2))
    noisy, _ = sim.as_dataset(stuck, FPS, rng, sigma_loc=0.2, mean_duration=1e6)
    assert (np.diff(noisy["x_um"]) * FPS).std() == pytest.approx(an.velocity_noise(0.2, FPS), rel=0.05)


def test_tumbles_have_expected_duration():
    rng = np.random.default_rng(4)
    _, tumbling, _ = sim.simulate_run_tumble(100, 60, FPS, rng, tumble_time=0.15)
    frac = tumbling.mean()
    expected = 0.15 / (1.0 + 0.15)                          # mean tumble / (mean run + mean tumble)
    assert frac == pytest.approx(expected, rel=0.15)
