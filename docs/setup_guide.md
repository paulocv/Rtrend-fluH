# Setup guide for Rtrend FluH model

This document guides through the process to setup the Rtrend FluH on your computer, including its dependencies. 
This process was tested for MacOS with Apple Silicon.

Steps 1 and 2 require some installation and manual configuration. Once you reach Step 3, the repo will be downloaded, so it should be a matter of running the right scripts in the right order.

## Prerequisites

### Pre 1: Conda package manager (python & etc)
Conda is a package and environment management system. It was originally created for python, but currently supports other useful software, like git and GitHub CLI. This means that we can install and manage all dependencies of the Rtrend project with Conda.

Visit the [Conda installation page](https://docs.conda.io/projects/conda/en/stable/user-guide/install/) and follow the instructions to install. (If the link is broken, just Google for conda.)

* Note: you can choose to install the lightweight version of Conda to save space. The setup will create an environment with the necessary dependencies.

### Pre 2: C compiler
The MCMC procedure for R(t) estimation is written in C. Although we provide a precompiled executable in the project, you may need to recompile if it doesn't work in your system. 

The script will try to compile with `gcc`, but you can use the one available in your system to compile [rtrend_tools/rt_mcmc/Rt.c](../rtrend_tools/rt_mcmc/Rt.c) into [main_mcmc_rt](../main_mcmc_rt).

### Pre 3: Gnu Scientific Library (GSL)
This library is required by the MCMC code. It can be obtained from Conda ([see here](https://anaconda.org/conda-forge/gsl)) or [check the GSL page](https://www.gnu.org/software/gsl/) for installation.


## Step-by-step setup

### Step 1: login to git in your computer

This step is optional and user-dependent. You can achieve it by [installing GitHub CLI](https://github.com/cli/cli#installation) and [authenticating](https://cli.github.com/manual/gh_auth_login).
    
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

There's still a lot that must be done, but now the scripts should take care of the heavy work.

In the same terminal, type: 
```
python setup.py
```

This script exerts two main tasks:

* Create a Conda virtual environment;
* Give execution permission to scripts and executables.

A virtual environment is a safe way to use Python libraries in an isolated space, where changes to these libraries and environment variables do not affect the base scope. 

The script will prompt to create a virtual environment using Conda. Unless you plan to do this manually, type "y" to proceed. Conda may take anywhere from a few seconds to minutes to initialize, then it will prompt you before downloading Python packages. Once again, enter "y" and wait for the process.

If there is a problem with the environment creation, the `setup.py` script will continue, but the required python libraries may not be installed yet.

Finally, the script gives execution permission to some scripts that will be used later on. No action is required in this step.
