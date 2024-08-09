"""
Fetch the latest FluSight hospitalization data from the CDC Github repository.

CDC Repo: https://raw.githubusercontent.com/cdcepi/Flusight-forecast-data

Usage
-----
> python fetch_cdc_truth_file.py

No arguments required, the file will already be saved in the correct directory (which is created if not existing).
"""

import os
import sys

from colorama import Fore, Back, Style
import requests

from rtrend_tools.cdc_params import FLU_TRUTH_URL
from rtrend_tools.scripting import FLU_TRUTH_DATA_FILE
from rtrend_tools.scripting import fetch_file_online_raise

TIMEOUT = 20  # Time in seconds to wait for the file request.


def main():

    if "-f" in sys.argv:  # Force-Run the fetch without prompt
        print("-f option passed, I will fetch and overwrite the previous truth data file without prompting.")

    else:  # Prompts for backup of previous file before fetching.

        # --- Preamble: make sure that the user wants to overwrite current file
        answer = input("You're about to download updated hospitalization data from CDC github.\n"
                       "Did you backup the previous file? (y/n)")

        if answer.lower() != "y":
            print("Then go do it...")
            return

        answer = input("Ok, are you sure you want to download new data and overwrite existing file then? (y/n)")

        if answer.lower() != "y":
            print("Quitting now...")
            return

    # --- Download commands
    print(f"Downloading from... \n'{FLU_TRUTH_URL}'")
    r = fetch_file_online_raise(FLU_TRUTH_URL, timeout=TIMEOUT)

    # Check status, raise error if unsuccessful
    r.raise_for_status()

    # Otherwise, exports the resulting file.
    print(f"Content successfully fetched, now saving into...\n{FLU_TRUTH_DATA_FILE}")
    os.makedirs(os.path.dirname(FLU_TRUTH_DATA_FILE), exist_ok=True)
    with open(FLU_TRUTH_DATA_FILE, "w") as fp:
        fp.write(r.text)

    # # # -()- Simple solution: use curl. This requires that it is installed, though.
    # os.system(f"curl {URL} -o \"{FLU_TRUTH_DATA_FILE}\"")

    print("Completed!")


if __name__ == "__main__":
    main()
