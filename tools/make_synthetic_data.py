"""Generate the synthetic datasets in data/synthetic/.

Run from the repository folder, with the mcdb5910 environment active:
    python tools/make_synthetic_data.py

Datasets (in the same format as real tracking data, one row per detection, positions in µm):
    brownian           non-swimming cells: Brownian motion only
    run_tumble         E. coli-like run-and-tumble swimmers in a uniform environment
    biased_run_tumble  the same swimmers in an attractant gradient toward +x (a biased random walk)

For each dataset the script writes <name>.csv (tracks), <name>.json (metadata and model parameters) and, for the
swimmers, <name>_truth.csv (the true run/tumble state of every detection).

How the data are made:
 1. Simulate N_TRACKS ideal trajectories of SIM_DURATION s at FPS frames per second (motility.simulate).
 2. Make them look like microscopy data (motility.simulate.as_dataset): cut each track to a random duration
    (exponential, mean MEAN_DURATION s), place it in the field of view, and add localization noise SIGMA_LOC.
Each dataset has its own fixed random seed, so the files are identical every time the script runs.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from motility import simulate as sim  # noqa: E402

SEED = 5910
FPS = 30.0              # frames per second
SIM_DURATION = 40.0     # s, longest possible track
N_TRACKS = 400
MEAN_DURATION = 10.0    # s, mean track duration (cells leave the focal plane)
MIN_DURATION = 2.0      # s, shortest track kept
SIGMA_LOC = 0.2         # µm, localization noise

DATASETS = {
    "brownian": {
        "description": "Non-swimming cells. Brownian motion with diffusion coefficient D (µm²/s).",
        "model": "brownian",
        "parameters": {"D": 0.35},
    },
    "run_tumble": {
        "description": "Run-and-tumble swimmers (E. coli-like) in a uniform environment. Each cell has its own "
                       "swimming speed. Tumbles happen at a constant rate.",
        "model": "run_tumble",
        "parameters": dict(sim.ECOLI),
    },
    "biased_run_tumble": {
        "description": "The same run-and-tumble swimmers in an attractant gradient toward +x: the tumble rate is "
                       "multiplied by (1 - gradient * cos(heading)), so runs toward +x last longer.",
        "model": "run_tumble",
        "parameters": dict(sim.ECOLI, gradient=0.7),
    },
}


def main():
    out = ROOT / "data" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, spec) in enumerate(DATASETS.items()):
        rng = np.random.default_rng([SEED, i])
        if spec["model"] == "brownian":
            positions = sim.simulate_brownian(N_TRACKS, SIM_DURATION, FPS, spec["parameters"]["D"], rng)
            states = None
        else:
            positions, states, _ = sim.simulate_run_tumble(N_TRACKS, SIM_DURATION, FPS, rng, **spec["parameters"])
        tracks, truth = sim.as_dataset(positions, FPS, rng, states=states, sigma_loc=SIGMA_LOC,
                                       mean_duration=MEAN_DURATION, min_duration=MIN_DURATION)

        tracks.to_csv(out / f"{name}.csv", index=False)
        if truth is not None:
            truth.to_csv(out / f"{name}_truth.csv", index=False)
        durations = tracks.groupby("track_id")["t"].agg(lambda t: t.max() - t.min())
        meta = {
            "name": name,
            "description": spec["description"],
            "synthetic": True,
            "fps": FPS,
            "units": {"t": "s", "x_um": "µm", "y_um": "µm"},
            "localization_noise_um": SIGMA_LOC,
            "n_tracks": N_TRACKS,
            "track_duration_s": {"mean": round(float(durations.mean()), 2), "min": round(float(durations.min()), 2),
                                 "max": round(float(durations.max()), 2)},
            "model": spec["model"],
            "model_parameters": spec["parameters"],
            "seed": [SEED, i],
            "generated_by": "tools/make_synthetic_data.py",
        }
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"{name}: {N_TRACKS} tracks, {len(tracks)} detections, mean track {durations.mean():.1f} s")


if __name__ == "__main__":
    main()
