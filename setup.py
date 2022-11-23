"""
Performs steps to set up and install all the tools required to run the forecast. This includes:
* Locally log in to a GitHub account that has access to the private Rtrend repository.
"""

import json
import os
import subprocess
# import tempfile

from rtrend_tools.scripting import NEW_SECTION, ENV_NAME, ENV_FLEX_PATH, MCMC_BIN, prompt_yn


def main():
    # ------------------------------------------------------------------------------------------------------------------
    print(NEW_SECTION)
    print("CONFIGURING THE PYTHON ENVIRONMENT\n")

    # Check if the evironment exists
    proc = subprocess.run("conda env list --json".split(), capture_output=True)
    env_list = json.loads(proc.stdout)["envs"]
    env_exists = ENV_NAME in [os.path.basename(env) for env in env_list]

    # --- Take action
    if env_exists:
        print(f"You already seem to have an environment called '{ENV_NAME}', so I'll skip this step.")
    else:  # Prompt to install.
        print(f"Let's create a python environment called '{ENV_NAME}'.")
        proceed = prompt_yn("This will run 'conda create', and python packages will be downloaded. Proceed?")
        if not proceed:
            print("Ok, the environment was *not* created yet. Quitting now...")
            return

        print("Ok. This can take some time, don't worry.")
        print(NEW_SECTION)

        # Create conda environment
        try:
            proc = subprocess.run(f"conda env create -f {ENV_FLEX_PATH}".split(), check=True)
        except subprocess.CalledProcessError:
            print(NEW_SECTION)
            print("\nOops, looks like there was an error while creating the environment.")
            print("Sugggestions:")
            print(f"- Make sure you are running this script from the root directory of this project (Rtrend-fluH)")
            print(f"- Run 'conda env create -f {ENV_FLEX_PATH}'. Then run setup.py again.")
            print(f"- If that fails, check if {ENV_FLEX_PATH} exists. It could have been renamed or moved...")
            print(f"- If that fails, create the environment manually. Required packages are listed in the docs.")
            print("\nQuitting now...")
            return

    # Check the evironment again
    proc = subprocess.run("conda env list --json".split(), capture_output=True)
    env_list = json.loads(proc.stdout)["envs"]
    env_exists = ENV_NAME in [os.path.basename(env) for env in env_list]

    if not env_exists:
        print("Oops, I could not find the environment. :(")
        print("If you aborted the installation, just run setup.py again. Otherwise, please try to find the problem and"
              " try running setup again.")
        print("Quitting now...")
        return

    # ------------------------------------------------------------------------------------------------------------------
    # from colorama import Fore, Back, Style
    import glob
    print(NEW_SECTION)
    print("GIVING EXECUTION PERMISSION TO OTHER SCRIPTS\n")
    # print(Fore.CYAN + "Ps: now we can use colors, yay!!" + Style.RESET_ALL)

    execs_list = list()
    execs_list += glob.glob("*.sh")  # Include sh scripts
    execs_list.append(MCMC_BIN)

    for ex in execs_list:
        cmd = f"chmod +x '{ex}'"
        print(cmd)
        os.system(cmd)  # Mmmmh, there's os.chmod, but it's numeric only, and Unix-only anyway.


def prompt_yn(msg):
    answer = input(msg + " (y/n)")
    while True:
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            answer = input("Please answer \"y\" or \"n\"")


if __name__ == "__main__":
    main()


# ---------------- OLD PROCEDURES
    # # --- Check installation of gh (GitHub CLI) package.
    # print("Checking if GitHub CLI is installed...")
    # cmd = "conda list gh --json".split()
    # proc = subprocess.run(cmd, capture_output=True)
    # result_dict = json.loads(proc.stdout)
    #
    # # --- Take action: install GitHub CLI if needed.
    # if len(result_dict) == 0:  # gh not installed.
    #     # Prompt to install.
    #     if not prompt_yn("It seems that GitHub CLI (gh) is not installed in your system. "
    #                      "Should we do this now via conda?"):
    #         print("Okay, please do it manually or use an alternative method. Quitting now...")
    #         return
    #
    #     print("Calling conda to install gh...")
    #
    #     try:
    #         subprocess.run("conda install -c conda-forge gh".split(), check=True)
    #     except subprocess.CalledProcessError:
    #         print("-----------------------------------------")
    #         print("Oops, looks like the installation of gh was not successful. Please try the alternative method, "
    #               "then run 'python setup.py' again.")
    #         print("Quitting now...")
    #         return
    #
    # code = os.system("gh auth login")
    # ......
