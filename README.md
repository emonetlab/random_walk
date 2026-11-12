# MCDB 5910 · Quantifying bacterial motility: notebooks and code

Hands-on material for the bacterial motility module of MCDB 5910 (Day 1, section 3, "Trajectory analysis"). Starting from tracks of swimming bacteria, the notebooks compute kinematics, find behavioral states (runs and tumbles), measure how long cells remember their direction, and compare the mean squared displacement (MSD) with analytic models.

| Notebook | Part | What you learn | Run time |
|---|---|---|---|
| [`NB1_tracks_to_states`](notebooks/NB1_tracks_to_states.ipynb) | A, B, C | **A** speeds from positions, and how localization noise and smoothing affect them; **B** runs and tumbles in the (speed, angular velocity) state space; **C** velocity and orientation autocorrelations | ~20 s |
| [`NB2_msd`](notebooks/NB2_msd.ipynb) | D | MSD of Brownian motion, run-and-tumble (constant and OU-driven tumble rate) and the biased random walk: equations of motion, analytic results and simulations, then the datasets | ~1 min |
| [`NB3_smoothing_window`](notebooks/NB3_smoothing_window.ipynb) | A | Goes with NB1, part A: how the Savitzky–Golay window changes the velocity of a noisy swimming track (noise against blurred tumbles), and how to choose the window. It also writes the interactive explorer `NB3_smoothing_explorer.html` | ~10 s |
| [`NB_extension`](notebooks/NB_extension.ipynb) | extra | The Lévy walk, and five random walks side by side (random, biased, run-and-tumble, smooth turner, Lévy) with matched parameters | ~1 min |

Start with NB1, then NB2. Every notebook runs from top to bottom without changes.

**Smoothing explorer.** [`notebooks/NB3_smoothing_explorer.html`](notebooks/NB3_smoothing_explorer.html) is an interactive page that opens in any web browser, without Python. Move the window slider to see how smoothing changes the speed of a swimming cell. We use it in class for NB1, part A.

## Run the notebooks

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda) and git. In a terminal:

```bash
git clone <repository address>
cd <repository folder>
conda env create -f environment.yml
conda activate mcdb5910
python tools/install_kernel.py
jupyter lab
```

In JupyterLab, open a notebook from `notebooks/`, choose the kernel **Python (mcdb5910)**, and run all cells. You only need the `conda env create` and `install_kernel.py` steps once.

The notebooks import the course package `motility` from the repository folder, so keep them in `notebooks/` and open them from the cloned repository.

- `environment.yml` creates the environment `mcdb5910` with Python 3.12 and every package the notebooks use.
- `tools/install_kernel.py` registers the environment as a Jupyter kernel. On Windows, use it instead of plain `ipykernel install`: a kernel that starts without `conda activate` cannot find numpy's math library and stops at the first matrix operation (Windows error 0xc06d007f). The script adds the environment folders to the kernel's search path.
- Without conda: `pip install -r requirements.txt` in a Python 3.12 environment.

To check your installation, run `python -m pytest` (tests of the analysis code, ~10 s) and `python tools/run_notebooks.py` (runs every notebook, ~3 min). Both should report no failures.

## What is in this repository

| Path | Contents |
|---|---|
| `notebooks/` | The notebooks. |
| `motility/` | The course Python package, imported by the notebooks. `simulate`: synthetic trajectories (Brownian motion, run-and-tumble) and `as_dataset`, which makes them look like microscopy data. `analysis`: velocity, speed and angular velocity (`kinematics`), velocity noise, per-cell summaries, MSD, autocorrelations, and the analytic MSD formulas. `io`: load a dataset. `plotting`: figure style and the MSD figure. |
| `data/synthetic/` | The synthetic datasets (see below). |
| `tests/` | Tests of the analysis code on trajectories whose answer is known. |
| `tools/` | Scripts: `install_kernel.py` registers the Jupyter kernel; `run_notebooks.py` runs notebooks without a browser; `make_synthetic_data.py` regenerates the datasets. |
| `environment.yml`, `requirements.txt` | The software environment (conda or pip). |

## The datasets

Each dataset has three files in `data/synthetic/`:

- `<name>.csv`: the tracks, one row per detection, with the columns `track_id` (one number per cell), `frame`, `t` (s), `x_um` and `y_um` (position in µm). This is the same format as tracking output from real movies.
- `<name>.json`: metadata: frame rate, localization noise, track durations, and the parameters used to simulate the cells.
- `<name>_truth.csv` (swimmers only): the true state, `run` or `tumble`, of every detection. Real data do not have this; here it lets you check your methods.

| Dataset | Cells | Used in |
|---|---|---|
| `brownian` | non-swimming cells (Brownian motion, D = 0.35 µm²/s) | NB2 |
| `run_tumble` | *E. coli*-like run-and-tumble swimmers: 25 ± 4 µm/s, 1 tumble per second, tumbles of 0.15 s, turn angles 68 ± 36° | NB1, NB2 |
| `biased_run_tumble` | the same swimmers in an attractant gradient toward +x | NB2 |

**How the data are made** (`tools/make_synthetic_data.py`): 400 cells per dataset are simulated for 40 s at 30 frames per second (`motility.simulate`). Each track is then cut to a random duration (mean 10 s), because real cells leave the focal plane, and every position gets a localization error of 0.2 µm (`motility.simulate.as_dataset`). The random seeds are fixed, so the script makes identical files every time.

## For instructors

Run these from the repository folder with the environment active.

| Task | Command |
|---|---|
| Test the analysis code | `python -m pytest` |
| Regenerate the datasets | `python tools/make_synthetic_data.py` |
| Run every notebook top to bottom | `python tools/run_notebooks.py` (add `--save` to keep the outputs in the notebooks) |

Edit the notebooks directly in JupyterLab; they are the source. After editing, run `python tools/run_notebooks.py` to check that every notebook still runs from top to bottom without errors. Running NB3 also rewrites the smoothing explorer, `notebooks/NB3_smoothing_explorer.html`.

Rules that keep the notebooks working for every student:

- Use only packages in `environment.yml`. Students may have older versions than the ones we tested (within the bounds in `environment.yml`), so avoid features added in the newest releases.
- Use relative paths only.
- Fix every random seed, so every student gets the same numbers and figures.
- Keep the run time of a notebook to a few minutes.
