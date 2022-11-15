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

import requests

URL = "https://raw.githubusercontent.com/cdcepi/Flusight-forecast-data/master/data-truth/" \
      "truth-Incident%20Hospitalizations.csv"
OUT_FNAME = 'hosp_data/truth-Incident Hospitalizations.csv'
TIMEOUT = 20  # Time in seconds to wait for the file request.


def main():

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
    print(f"Downloading from... \n'{URL}'")

    try:
        # FILE REQUEST COMMAND
        r: requests.Response = requests.get(URL, timeout=TIMEOUT)

    except requests.exceptions.ConnectionError as e:
        sys.stderr.write("\n\n#############################\nTHERE WAS A CONNECTION ERROR"
                         "\n#############################\n\n")
        print(e.args[0])
        print("\nThe file was NOT downloaded.")
        return

    # Check status, raise error if unsuccessful
    r.raise_for_status()

    # Otherwise, exports the resulting file.
    print(f"Content successfully fetched, now saving into...\n{OUT_FNAME}")
    os.makedirs(os.path.dirname(OUT_FNAME), exist_ok=True)
    with open(OUT_FNAME, "w") as fp:
        fp.write(r.text)

    # # # -()- Simple solution: use curl. Requires that it is installed, though.
    # os.system(f"curl {URL} -o \"{OUT_FNAME}\"")

    print("Completed!")


if __name__ == "__main__":
    main()
