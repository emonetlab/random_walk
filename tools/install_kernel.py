"""Register the active conda environment as the Jupyter kernel "Python (mcdb5910)".

Run it once, from the activated environment:
    conda activate mcdb5910
    python tools/install_kernel.py

Why not only `python -m ipykernel install`? On Windows, numpy in a conda environment loads its math
library (MKL or OpenBLAS) from the environment's Library\\bin folder. A kernel that starts without
`conda activate`, for example from `jupyter lab` in another environment or from tools/run_notebooks.py,
does not have that folder on PATH, and the kernel dies at the first matrix operation. This script adds
the environment folders to the kernel's PATH, as `conda activate` does.
"""
import json
import os
import subprocess
import sys

NAME, DISPLAY = "mcdb5910", "Python (mcdb5910)"

subprocess.run([sys.executable, "-m", "ipykernel", "install", "--user",
                "--name", NAME, "--display-name", DISPLAY], check=True)

from jupyter_client.kernelspec import KernelSpecManager  # noqa: E402

spec_file = os.path.join(KernelSpecManager().get_kernel_spec(NAME).resource_dir, "kernel.json")
with open(spec_file) as f:
    spec = json.load(f)

prefix = sys.prefix
if os.name == "nt":
    dirs = [prefix] + [os.path.join(prefix, *p) for p in
                       (("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"), ("Library", "bin"), ("Scripts",), ("bin",))]
else:
    dirs = [os.path.join(prefix, "bin")]
spec["env"] = {"PATH": os.pathsep.join(dirs + ["${PATH}"]), "CONDA_PREFIX": prefix}

with open(spec_file, "w") as f:
    json.dump(spec, f, indent=1)
print(f"Kernel '{DISPLAY}' uses {sys.executable}; environment folders added to its PATH.")
