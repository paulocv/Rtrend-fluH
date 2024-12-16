"""
A simple script to save the relevant jupyter notebooks into a zip file
that can be commited to the repository.

The program chooses among an "include" and an "exclude" list of Notebooks

- Output file: ibes-usa/py_src/jupyter_notebooks.zip
"""
from glob import glob
import os
from pathlib import Path
import zipfile

# --- Select here the notebooks to include and exclude
include_notebooks = [
    *glob("jupyter_notebooks/*.ipynb"),  # Include all notebooks in the dedicated directory
    # "jupyter_notebooks/prepare_cell_data.ipynb",
]

exclude_notebooks = [
    "jupyter_notebooks/anything_fore.ipynb",
    "jupyter_notebooks/reporting_delay_rehearsals.ipynb",
    "jupyter_notebooks/reproduction_number_rehearsals.ipynb",
]


# ------------------
use_notebooks = [n for n in include_notebooks if n not in exclude_notebooks]

print(f"Notebooks to save: {use_notebooks}")

pass
# Output file path
output_file = Path('jupyter_notebooks.zip')
# output_file.parent.mkdir(parents=True, exist_ok=True)

# Create a zip file
with zipfile.ZipFile(output_file, 'w') as zip_file:
    for notebook in use_notebooks:
        try:
            zip_file.write(notebook, os.path.basename(notebook))
        except FileNotFoundError:
            print(f"[ERROR]   Notebook not found: {notebook}. Skipped.")
        else:
            print(f"[SUCCESS] Saved: {notebook}")

print(f"Notebooks saved to {output_file}")
