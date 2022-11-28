# How to use - Rtrend FluH Forecast

Before using, it is recommended that you read and execute the steps in the [Setup guide](setup_guide.md).

## Stage 1: activate the virtual environment

This stage is short, but it has to be executed every time a new terminal window is created. It does not need to be executed before every call to `run_forecast_states.py` in the same terminal window.

Open a terminal, then type and run:

````
source ./activate_env.sh
````

Check that now your command line starts with `(rtrend_forecast)` instead of `(base)`, indicating that the environment was successfully activated.

> Note: if the environment fails to load:
> - Make sure that `setup.py` was executed successfully, following the instructions in the [Setup guide](./setup_guide.md).
> 


## Stage 2: run the pre-forecast routine

This stage, if successful, only needs to be executed once at each new week of forecast. It performs the following tasks:

- Update the local project by pulling changes from the remote repository.
- Check for updates in the virtual environment and perform them, if any.
- Downloads the latest truth data from CDC (the past number of hospitalizations).

All these tasks are performed by the `pre_forecast.py` script. In the terminal (with the `rtrend_forecast` environment active), run:

````
python pre_forecast.py
````

And follow the instructions on screen and below.

> Note: your computer must be connected to the internet.

### Step 2.1: Updating the local repository

> If you are running the forecast again after having done this in past weeks, it is likely that the code and parameters have changed. This step will pull the latest code from Github. Mind that locally made changes will be overwritten.

For the next two prompts, type "y" to accept the pull operation and "y" again to discard possible local changes (which may cause a conflict).

### Step 2.2: Checking and updating the virtual environment

> Some new libraries may have been included since the last changes. This step will install these libraries into your local Conda environment.

This step usually requires no interaction. However, if updates are needed, the script will call `conda env update`, which can take anywhere from a few seconds to a few minutes and use considerable RAM. It can also prompt you to confirm (type "y") the new packages.

### Step 2.3: Updating the truth data from CDC

> Every few days, the CDC updates its GitHub file that contains the past data that is used into the forecast. At some point between Saturday and Monday, they include data from the last week (up to Saturday), which is the one that Rtrend needs to produce up-to-date forecast.

This step requires no interaction. The script will fetch the data from CDC GitHub repo, make a backup from the last file (if needed) then overwrite it with the new data. A warning will be shown if the new data is more than one week old, which probably means that the latest week was not uploaded yet by the CDC.


## Stage 3: run and tweak the forecast

Once everything is up-to-date, the forecast can be executed. The steps of this stage will possibly be performed multiple times, depending on the feedback about the produced forecast.

### Step 3.1: Run the forecast code

With the previous stages successfully performed, in a terminal with the `rtrend_forecast` environment active, type:

````
python run_forecast_states.py
````

A series of messages will be printed on the screen, possibly with some warnings. Unless the code stops with an error traceback, the forecast should be performed for all states.

### Step 3.2: Visually inspect the forecasts

### Step 3.3: Tweak parameters (if needed)




