"""Run notebooks top to bottom without a browser, as "Run all" does in JupyterLab.

Usage (from the random_walk folder, in the mcdb5910 environment):
    python tools/run_notebooks.py                      # every notebook in notebooks/
    python tools/run_notebooks.py notebooks/NB1_tracks_to_states.ipynb
    python tools/run_notebooks.py --save               # also write the outputs back into each notebook

Each notebook runs in its own folder, so relative data paths behave as they do for students.
The script prints the run time of each notebook and any error.
It exits with status 1 if a notebook stops with an error.
"""
import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent


def run(path, kernel, timeout, save):
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(path.parent)}})
    t0 = time.time()
    error = None
    try:
        client.execute()
    except CellExecutionError as exc:
        error = str(exc).strip().splitlines()[-1]
    elapsed = time.time() - t0

    if save:
        nbformat.write(nb, path)
    return elapsed, error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="notebooks to run (default: notebooks/*.ipynb)")
    parser.add_argument("--kernel", default="mcdb5910", help="Jupyter kernel name (default: mcdb5910)")
    parser.add_argument("--timeout", type=int, default=900, help="time limit per cell in seconds (default: 900)")
    parser.add_argument("--save", action="store_true", help="write the outputs back into the notebooks")
    args = parser.parse_args()

    paths = [p.resolve() for p in args.notebooks] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    ok = True
    for path in paths:
        elapsed, error = run(path, args.kernel, args.timeout, args.save)
        ok &= error is None
        print(f"{'OK  ' if error is None else 'FAIL'} {path.relative_to(ROOT)}  ({elapsed:.0f} s)")
        if error:
            print(f"     error: {error}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
