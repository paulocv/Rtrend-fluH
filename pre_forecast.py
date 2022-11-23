
import os
import subprocess
import sys

from rtrend_tools.scripting import NEW_SECTION, ENV_NAME, ENV_FLEX_PATH, MCMC_BIN, prompt_yn, conda_env_exists


def main():
    """
    - v Pull changes
    - Check and compare environment
    - Recompile MCMC
    - BKP truth data
    - Fetch new truth data from CDC

    """
    git_pull_all_changes()
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
    """Will check the status of the project's conda environment. Update it if needed, then activate it with source."""
    print(NEW_SECTION)
    print("CHECKING FOR ENVIRONMENT UPDATES\n")

    # --- Check existence
    if not conda_env_exists(ENV_NAME):
        print(f"Hey, I did not find a conda environment called {ENV_NAME}. Please run 'python setup.py' and then "
              f"run this script again.")
        print("Quitting now...")
        sys.exit(1)

    # --- Check for differences





if __name__ == "__main__":
    main()
