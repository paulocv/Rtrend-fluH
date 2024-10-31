import filecmp
import io
import json
import os
import pandas as pd
import subprocess
import sys

from colorama import Fore, Style
from datetime import datetime, timedelta

from rtrend_tools.cdc_params import FLU_TRUTH_URL
# from rtrend_tools.data_io import load_cdc_truth_data
from rtrend_tools.scripting import NEW_SECTION, ENV_NAME, ENV_FLEX_PATH, prompt_yn, conda_env_exists, \
    MCMC_COMPILE_SCRIPT, FLU_TRUTH_DATA_BKPDIR, fetch_file_online_raise, FLU_TRUTH_DATA_FILE


def main():

    git_pull_all_changes()
    check_and_compare_environment()
    # recompile_mcmc_code()  # Not used now, troublesome: PATH variable not set, and it requires GSL.
    bkp_and_fetch_truth_data()


def git_pull_all_changes():
    """This discards local changes and pull updates from the main remote branch."""
    print(NEW_SECTION)
    print("UPDATING THE LOCAL REPOSITORY\n")

    if not prompt_yn("Do you want to pull project changes from github?"):
        print("Ok. Mind that the files may be outdated.")
        return

    # --- Optionally reset local changes.
    answer = prompt_yn("Would you like to reset local changes before pulling from the remote repository? You may need "
                       "this if local changes were made in the past but not committed.")

    if answer:  # Git reset everything
        print("Ok, resetting now...")
        os.system("git restore .")
        os.system("git submodule foreach 'git reset'")
    else:
        print("Ok. If you get conflicts from git in the next step, please restart this script and select 'Y' for the "
              "previous question.")

    # --- Pull into main branch
    print()
    print("Now let's pull changes from remote to local repo....")

    try:
        subprocess.run("git pull --recurse-submodules origin main".split(), check=True)
    except subprocess.CalledProcessError:
        print(Fore.RED + "\nThere was an error with the project updates. \nQuitting now..." + Style.RESET_ALL)
        sys.exit(1)

    print(Fore.GREEN + "\nAll set with the project updates!" + Style.RESET_ALL)


def check_and_compare_environment():
    """
    Will check the status of the project's conda environment.
    """
    num_problems = 0

    print(NEW_SECTION)
    print("CHECKING FOR ENVIRONMENT UPDATES\n")

    # --- Check existence
    if not conda_env_exists(ENV_NAME):
        print(Fore.RED + f"Hey, I did not find a conda environment called {ENV_NAME}. Please run 'python setup.py' and "
              f"then run this script again." + Style.RESET_ALL)
        print("Quitting now...")
        sys.exit(1)

    # --- Check that it's active
    out = subprocess.run("conda info --json".split(), capture_output=True)
    conda_info = json.loads(out.stdout)
    active_env = conda_info["active_prefix_name"]

    if active_env != ENV_NAME:
        print(Fore.RED +f"Hey, currently active conda environment ({active_env}) is not the expected one ({ENV_NAME}).")
        print("Please run 'source activate_env.sh' first, then rerun this script." + Style.RESET_ALL)
        print("Quitting now...")
        sys.exit(1)

    print(f"Environment {ENV_NAME} is active.")

    # --- Compare with latest environment file
    try:
        out = subprocess.run(f"conda compare --json {ENV_FLEX_PATH}".split(), capture_output=True)
    except subprocess.CalledProcessError:
        sys.exit(1)

    compare_msg = json.loads(out.stdout)[0]

    # --- Update env if package changes were detected
    if "Success" in compare_msg:  # Only way I could find to check the results of the comparison...
        print("Environment is up-to-date with the yml file. No updates required.")
    else:
        print("Looks like there are changes to be done in the environment. Let update the packages with conda.")
        print("THIS CAN TAKE SOME TIME, sit down and relax...")

        code = os.system(f"conda env update --file {ENV_FLEX_PATH} --prune")

        if code:
            print(Fore.YELLOW + "Problem while running the conda update.\n"
                                f"Error code = {code}" + Style.RESET_ALL)
            num_problems += 1

    # --- Final feedback about problems in the script
    if num_problems:
        print(Fore.YELLOW + "\nThere were problems." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + "\nAll set with the virtual environment!" + Style.RESET_ALL)


def recompile_mcmc_code():
    """
    Call the compiler script for the MCMC R(t) estimation.
    """
    print(NEW_SECTION)
    print("RECOMPILING C CODE FOR MCMC\n")

    try:
        subprocess.run(f"sh {MCMC_COMPILE_SCRIPT}".split(), check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


def bkp_and_fetch_truth_data():
    """"""
    print(NEW_SECTION)
    print("FETCHING THE LATEST TRUTH DATA FROM CDC\n")

    # --- Fetch latest from CDC and keep in memory
    response = fetch_file_online_raise(FLU_TRUTH_URL, timeout=20)
    print("Fetch complete.")

    # Perform checks after download
    if response is None:
        print(Fore.RED + "There was a problem fetching the file. Quitting now..." + Style.RESET_ALL)
        sys.exit(1)
    response.raise_for_status()

    # --- Parse the file content as a dataframe, then perform checks.
    print("Parsing file into a dataframe")
    bts = io.BytesIO(response.content)
    raw_df = pd.read_csv(bts, header=0, parse_dates=["date"])  # Just parse.
    # cdc_bunch = load_cdc_truth_data(io.BytesIO(response.content))  # Read and interpret file.

    # print(raw_df["date"].max())
    last_time_label = raw_df["date"].max()
    now = datetime.today()

    # Simple check for date: if difference is greater than one week, it's probably outdated
    if abs(now - last_time_label) > timedelta(weeks=1):
        print(Fore.YELLOW + "Warning: the day of the latest data from CDC is more than one week old. "
                            "This probably means that they did not update the data yet...\n"
                            "Please check the online file at:" + Style.RESET_ALL)
        print(FLU_TRUTH_URL)

    # --- Find the latest bkp file
    print("Now backing up the previous file.")
    _backup_truth_file_if_needed(last_time_label)

    # --- Saves the content of the new fetched file as the latest one
    os.makedirs(os.path.dirname(FLU_TRUTH_DATA_FILE), exist_ok=True)
    with open(FLU_TRUTH_DATA_FILE, "w") as fp:
        fp.write(response.text)
    print(f"The new data from CDC was saved as {FLU_TRUTH_DATA_FILE}")

    print(Fore.GREEN + "\nAll set with the truth data file!" + Style.RESET_ALL)


def _backup_truth_file_if_needed(last_time_label):

    # Nothing to do if there is no current (latest) truth data file.
    if not os.path.exists(FLU_TRUTH_DATA_FILE):
        print("No previous truth file. Proceeding to saving the new one.")
        return

    # Backup file format, using the last date as a time stamp
    bkp_fmt = os.path.join(FLU_TRUTH_DATA_BKPDIR, f"{last_time_label.date()}" +
                           "_truth-Incident Hospitalizations_{:02d}.csv")
    # Find it
    bkp_fname = None  # Last (if existing) backup file
    for i in range(100):
        new_bkp_fname = bkp_fmt.format(i + 1)  # First bkp file path that does not exist yet.
        if not os.path.exists(new_bkp_fname):
            break
        bkp_fname = new_bkp_fname

    else:
        print(Fore.RED + "Exhausted loop for finding the latest backup file. There may be an error in the code." +
              Style.RESET_ALL)
        print("Quitting now...")
        sys.exit(1)

    # --- Do backup file if different from the previous one
    if bkp_fname is not None and filecmp.cmp(bkp_fname, FLU_TRUTH_DATA_FILE):  # There is already an identical backup
        print(f"An identical truth file was already saved as '{bkp_fname}'.\nBackup is not needed.")
        return

    # If this point is reached, backup the file
    os.makedirs(os.path.dirname(new_bkp_fname), exist_ok=True)
    os.rename(FLU_TRUTH_DATA_FILE, new_bkp_fname)  # DOES THE BACKUP
    print(f"Previous truth file was saved as '{new_bkp_fname}'.")


if __name__ == "__main__":
    main()
