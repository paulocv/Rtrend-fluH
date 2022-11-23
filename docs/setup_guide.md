# Setup guide for Rtrend FluH model

This document guides trough the process to setup the Rtrend FluH on your computer, including its dependencies. 
This process is tested for on MacOS. 

Steps 1 and 2 require some installation and manual configuration. Once you reach Step 3, the repo will be downloaded, so it should be a matter of running the right scripts in the right order.

## Prerequisites

### Pre 1: Conda package manager (python & etc)
Conda is a package and environment management system. It was originally created for python, but currently spans other useful software, like git and Github CLI. This means that we can install and manage all dependencies of the Rtrend project with Conda.

Visit the [Conda installation page](https://docs.conda.io/projects/conda/en/stable/user-guide/install/) and follow the instructions to install. (If the link is broken, just Google for conda.)

* Note: you can choose to install the lightweight version of Conda to save space. The setup will create an environment with the necessary dependencies.

### Pre 2: C compiler
The MCMC procedure for R(t) estimation is written in C. Although we provide a precompiled executable in the project, you may need to recompile if it doesn't work in your system. 

The script will try to compile with `gcc`, but you can use the one available in your system to compile [rtrend_tools/rt_mcmc/Rt.c](../rtrend_tools/rt_mcmc/Rt.c) into [main_mcmc_rt](../main_mcmc_rt).


## Step-by-step setup

### Step 1: Configure git locally to access the CEPH GitHub account

This is needed to work locally on the repository (clone the repository, pull updates). If you already configured git locally (i.e., logged in to the CEPH account or another one that has access to this repository), you can skip this step.

**Recommended method: GitHub CLI**

GitHub CLI is a tool to use GitHub from the command line. Both git and GitHub CLI can be installed using Conda.

* 1.1: Install git and GitHub CLI. Open a terminal window and type:
    ```
    conda install -c conda-forge -c anaconda gh git
    ```
    Wait for the data to be collected, type "y" to confirm and wait for the packages to be installed.

* 1.2: Run the GitHub CLI to authenticate into CEPH and cache the credentials
    This step will run a command-line application. For more information, check [Caching your GitHub credentials in Git](https://docs.github.com/en/get-started/getting-started-with-git/caching-your-github-credentials-in-git). 

    Start the application:
    ```
    git auth login
    ```

    Then follow the instructions. The CEPH username is: CEPH-Lab (case-sensitive)
    * When prompted for your preferred protocol for Git operations, select HTTPS.
    * When asked if you would like to authenticate to Git with your GitHub credentials, enter Y.

    
### Step 2: Clone the Rtrend repository (with submodules)

* 2.1: Within a terminal window, go to any directory (cd) where you would like to have the Rtrend project folder installed. Example:
    
    ```
    cd path/to/where_i_want/
    ```
    In the above example, the Rtrend project will be created in a subfolder of the directory named `where_i_want`.

* 2.2: Clone the repository with its submodules. Type:
    ```
    git clone  --recurse-submodules https://github.com/paulocv/Rtrend-fluH
    ```
    This will download the contents of this repository into the `Rtrend-fluH` folder. Conveniently, it will also configure it as a git repository, connecting it to the remote (GitHub repo) and downloading its submodules. For this reason, cloning with git is preferred over just downloading the repository from GitHub as a .zip.

* 2.3: Enter the project directory. Type:
    ```
    cd Rtrend-fluH 
    ```


### Step 3: Run the `setup.py` script to complete the installation

Ok, there's still a lot to do yet, but from now on the scripts should do most of the hard work.


