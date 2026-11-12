"""Helper code for the MCDB 5910 bacterial motility module.

- `simulate`: synthetic trajectories (Brownian, run-and-tumble with optional OU-driven switching and gradient bias)
- `analysis`: kinematics (velocity, speed, angular velocity), MSD, autocorrelations, per-track summaries
- `io`: load the course datasets
- `plotting`: figure style and the MSD figure used in the notebooks
"""
from . import analysis, io, plotting, simulate  # noqa: F401
