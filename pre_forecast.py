
import os
import subprocess
import sys

from rtrend_tools.scripting import NEW_SECTION, ENV_NAME, ENV_FLEX_PATH, MCMC_BIN, prompt_yn


def main():
    """
    - Pull changes
    - Check and compare environment
    - Recompile MCMC
    - BKP truth data
    - Fetch new truth data from CDC

    """
    git_pull_all_changes()


def git_pull_all_changes():
    """This discards local changes and pull updates from the main remote branch."""
    print(NEW_SECTION)
    print("UPDATING THE LOCAL REPOSITORY\n")

    # --- Optionally reset local changes.
    answer = prompt_yn("Would you like to reset local changes before pulling from the remote repository? You may need "
                       "this if local changes were made in the past but not committed.")

    if answer:  # Git reset everything
        print("Ok, resetting now...")
        os.system("git restore")
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

    # try:
    #     subprocess.run("git pull origin main".split(), check=True)
    # except subprocess.CalledProcessError:
    #     sys.exit(1)


if __name__ == "__main__":
    main()
