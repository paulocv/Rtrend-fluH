import json
import os
import subprocess
import sys

from rtrend_tools.scripting import NEW_SECTION, ENV_NAME, ENV_FLEX_PATH, MCMC_BIN, prompt_yn, conda_env_exists


def main():
    """
    - v Pull changes
    - v Check and compare environment
    - Recompile MCMC
    - BKP truth data
    - Fetch new truth data from CDC

    """
    # git_pull_all_changes()
    check_and_compare_environment()


def git_pull_all_changes():
    """This discards local changes and pull updates from the main remote branch."""
    print(NEW_SECTION)
    print("UPDATING THE LOCAL REPOSITORY\n")

    # --- Optionally reset local changes.
    answer = prompt_yn("Would you like to reset local changes before pulling from the remote repository? You may need "
                       "this if local changes were made in the past but not committed.")

    if answer:  # Git reset everything
        print("Ok, resetting now...")
        os.system("git restore .")
        os.system("git submodules foreach 'git reset'")
    else:
        print("Ok. If you get conflicts from git in the next step, please restart this script and select 'Y' for the "
              "previous question.")

    # --- Pull into main branch
    print()
    print("Now let's pull changes from remote to local repo....")

    try:
        subprocess.run("git pull --recurse-submodules origin main".split(), check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)

    print("\nApparently, all set here.")


def check_and_compare_environment():
    """
    Will check the status of the project's conda environment.
    """
    print(NEW_SECTION)
    print("CHECKING FOR ENVIRONMENT UPDATES\n")

    # --- Check existence
    if not conda_env_exists(ENV_NAME):
        print(f"Hey, I did not find a conda environment called {ENV_NAME}. Please run 'python setup.py' and then "
              f"run this script again.")
        print("Quitting now...")
        sys.exit(1)

    # --- Check that it's active
    out = subprocess.run("conda info --json".split(), capture_output=True)
    conda_info = json.loads(out.stdout)
    active_env = conda_info["active_prefix_name"]

    if active_env != ENV_NAME:
        print(f"Hey, currently active conda environment ({active_env}) is not the expected one ({ENV_NAME}).")
        print("Please run 'source activate_env.sh' first, then rerun this script.")
        print("Quitting now...")
        sys.exit(1)

    print(f"Environment {ENV_NAME} is active.")

    # --- Compare with latest environment file
    try:
        out = subprocess.run(f"conda compare --json {ENV_FLEX_PATH}".split(), capture_output=True)
    except subprocess.CalledProcessError:
        sys.exit(1)

    compare_msg = json.loads(out.stdout)[0]

    if "Success" in compare_msg:  # Only way I could find to check the results of the comparison...
        print("Environment is up-to-date with the yml file. No updates required.")
    else:
        print("Looks like there are changes to be done in the environment. Let update the packages with conda.")
        print("THIS CAN TAKE SOME TIME, sit down and relax...")

        os.system(f"conda env update --file {ENV_FLEX_PATH} --prune")


if __name__ == "__main__":
    main()
