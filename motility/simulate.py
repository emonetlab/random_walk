"""Synthetic trajectories.

Every simulator returns positions in µm sampled at the camera frame rate, with shape (n_tracks, n_frames, 2),
and every track starts at the origin. `as_dataset` then turns these ideal trajectories into a dataset that looks
like microscopy data: finite tracks, localization noise, one row per detection.
"""
import numpy as np
import pandas as pd

# Parameters of an E. coli-like run-and-tumble swimmer (see simulate_run_tumble for their meaning).
ECOLI = dict(speed=25.0, speed_sd=4.0, run_rate=1.0, tumble_time=0.15, tumble_speed=0.2,
             turn_angle=(68.0, 36.0), Dr=0.15)


def heading(theta):
    """Unit vector e(theta) = (cos theta, sin theta)."""
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


def n_frames(duration, fps):
    return int(round(duration * fps)) + 1


def simulate_brownian(n_tracks, duration, fps, D, rng):
    """Brownian motion (a non-swimming cell):  r_{n+1} = r_n + sqrt(2 D dt) * xi_n."""
    nf = n_frames(duration, fps)
    steps = np.sqrt(2 * D / fps) * rng.standard_normal((n_tracks, nf - 1, 2))
    r = np.zeros((n_tracks, nf, 2))
    r[:, 1:] = np.cumsum(steps, axis=1)
    return r


def simulate_biased_brownian(n_tracks, duration, fps, D, vd, rng):
    """Drift plus diffusion:  r_{n+1} = r_n + vd*dt*x_hat + sqrt(2 D dt) * xi_n."""
    nf = n_frames(duration, fps)
    steps = np.sqrt(2 * D / fps) * rng.standard_normal((n_tracks, nf - 1, 2))
    steps[..., 0] += vd / fps
    r = np.zeros((n_tracks, nf, 2))
    r[:, 1:] = np.cumsum(steps, axis=1)
    return r


def turn_angle_cos(turn_angle=(68.0, 36.0), n=200_000, seed=0):
    """<cos phi> for the turn-angle distribution used by simulate_run_tumble (|normal(mean, sd)| in degrees)."""
    rng = np.random.default_rng(seed)
    mean, sd = np.radians(turn_angle)
    return float(np.mean(np.cos(mean + sd * rng.standard_normal(n))))


def simulate_run_tumble(n_tracks, duration, fps, rng, *, speed=25.0, speed_sd=0.0, run_rate=1.0,
                        tumble_time=0.15, tumble_speed=0.2, turn_angle=(68.0, 36.0), Dr=0.15,
                        ou_sigma=0.0, ou_tau=10.0, gradient=0.0, substeps=10):
    """Run-and-tumble swimmers.

    Runs: swim at the cell's speed; the heading diffuses with rotational diffusion Dr (rad^2/s).
    Run -> tumble at rate  run_rate * exp(x - ou_sigma^2/2) * (1 - gradient*cos(theta)):
        x is an Ornstein-Uhlenbeck process (mean 0, std ou_sigma, correlation time ou_tau), which models slow
        fluctuations of the motor's switching (for example in CheY-P). ou_sigma = 0 gives a constant rate.
        gradient > 0 makes runs toward +x last longer (chemotaxis up a gradient along x).
    Tumbles: last on average tumble_time (exponential). The cell slows to tumble_speed * its speed and turns by an
        angle |normal(turn_angle)| degrees with a random sign, spread evenly over the tumble: every tumble turns by
        exactly its drawn angle, whether it is short or long.
        tumble_time = 0 gives instantaneous tumbles (the cell turns and keeps swimming).
    The cell speeds are normal(speed, speed_sd), one value per cell.
    substeps: simulation steps per frame (resolves short tumbles).

    Returns (positions, tumbling, cell_speed): positions (n_tracks, n_frames, 2) in µm;
    tumbling (n_tracks, n_frames) bool, the true state at each frame (for instantaneous tumbles: True in the
    frames during which a tumble happened); cell_speed (n_tracks,) in µm/s.
    """
    nf = n_frames(duration, fps)
    h = 1.0 / (fps * substeps)
    cell_speed = np.maximum(speed + speed_sd * rng.standard_normal(n_tracks), 0.2 * speed)
    theta = rng.uniform(0, 2 * np.pi, n_tracks)
    x = ou_sigma * rng.standard_normal(n_tracks)          # start the OU process in its stationary state
    a = np.exp(-h / ou_tau)
    b = ou_sigma * np.sqrt(1 - a**2)                      # exact OU update over one step
    left = np.zeros(n_tracks)                             # time left in each cell's current tumble (s); 0 = running
    tumbling = np.zeros(n_tracks, dtype=bool)
    omega = np.zeros(n_tracks)                            # turning speed during the current tumble (rad/s)
    pos = np.zeros((n_tracks, 2))
    positions = np.zeros((n_tracks, nf, 2))
    states = np.zeros((n_tracks, nf), dtype=bool)
    turn_mean, turn_sd = np.radians(turn_angle)
    rot = np.sqrt(2 * Dr * h)

    for f in range(1, nf):
        event = np.zeros(n_tracks, dtype=bool)
        for _ in range(substeps):
            if ou_sigma > 0:
                x = a * x + b * rng.standard_normal(n_tracks)
            rate = run_rate * np.exp(x - ou_sigma**2 / 2) * (1 - gradient * np.cos(theta))
            start = (left <= 0) & (rng.random(n_tracks) < rate * h)
            k = int(start.sum())
            if k:
                phi = np.abs(turn_mean + turn_sd * rng.standard_normal(k)) * rng.choice([-1.0, 1.0], k)
                if tumble_time > 0:
                    left[start] = rng.exponential(tumble_time, k)   # this tumble's duration
                    omega[start] = phi / left[start]                # so that the whole tumble turns by phi
                else:
                    theta[start] += phi
                    event |= start
            if tumble_time > 0:
                tumbling = left > 0
                theta += omega * np.minimum(left, h)              # the last step of a tumble turns only what is left
                v = cell_speed * np.where(tumbling, tumble_speed, 1.0)
                left = np.maximum(left - h, 0.0)
            else:
                v = cell_speed
            theta += rot * rng.standard_normal(n_tracks)
            pos += (v * h)[:, None] * heading(theta)
        positions[:, f] = pos
        states[:, f] = tumbling if tumble_time > 0 else event
    return positions, states, cell_speed


def as_dataset(positions, fps, rng, *, states=None, sigma_loc=0.2, mean_duration=10.0, min_duration=2.0,
               field=600.0):
    """Turn ideal simulated trajectories into a dataset that looks like microscopy data.

    positions: (n_tracks, n_frames, 2) from a simulator; states: optional (n_tracks, n_frames) bool, True = tumbling.
    - Each track is cut to a random duration: exponential with mean `mean_duration` s, at least `min_duration` s,
      at most the simulated length. (Real tracks end when the cell leaves the focal plane.)
    - Each track starts at a random place in a field of view of field x field µm, and at a random movie frame.
    - Every position gets independent Gaussian localization noise with standard deviation `sigma_loc` µm.
    Returns (tracks, truth): tracks has the columns track_id, frame, t (s), x_um, y_um; truth has the columns
    track_id, frame, state ('run' or 'tumble'), or is None when `states` is None.
    """
    n_tracks, nf, _ = positions.shape
    durations = np.clip(rng.exponential(mean_duration, n_tracks), min_duration, (nf - 1) / fps)
    lengths = (durations * fps).astype(int) + 1
    offsets = rng.uniform(0, field, (n_tracks, 2))
    first_frames = rng.integers(0, 3000, n_tracks)

    tracks, truth = [], []
    for i in range(n_tracks):
        n = lengths[i]
        xy = positions[i, :n] + offsets[i] + sigma_loc * rng.standard_normal((n, 2))
        frame = first_frames[i] + np.arange(n)
        tracks.append(pd.DataFrame({"track_id": i + 1, "frame": frame, "t": np.round(frame / fps, 4),
                                    "x_um": np.round(xy[:, 0], 3), "y_um": np.round(xy[:, 1], 3)}))
        if states is not None:
            truth.append(pd.DataFrame({"track_id": i + 1, "frame": frame,
                                       "state": np.where(states[i, :n], "tumble", "run")}))
    tracks = pd.concat(tracks, ignore_index=True)
    return tracks, (pd.concat(truth, ignore_index=True) if states is not None else None)
