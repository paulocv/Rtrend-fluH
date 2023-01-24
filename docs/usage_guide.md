# How to use - Rtrend FluH Forecast

Before using by the first time, it is recommended that you read and execute the steps in the [Setup guide](setup_guide.md).

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

This step usually requires no interaction. However, if updates are needed, the script will call `conda env update`, which can take anywhere from a few seconds to a few minutes. It can also prompt you to confirm (type "y") the new packages.

### Step 2.3: Updating the truth data from CDC

> Every few days, the CDC updates its GitHub file that contains the past data that is used into the forecast. At some point between Saturday and Monday, they include data from the last week (up to Saturday), which is the one that Rtrend needs to produce up-to-date forecast.

This step requires no interaction. The script will fetch the data from CDC GitHub repo, make a backup from the last file (if needed) then overwrite it with the new data. A warning will be shown if the new data is more than one week old, which probably means that the latest week was not uploaded yet by the CDC.

If all steps above succeed, your code repository should be ready to run the forecast. Otherwise, in case of errors/warnings, follow the instructions from the terminal.

## Stage 3: check past week forecasts

> In this step, we check how we did in the past weeks, given the updated truth data from CDC.

### Step 3.1: Run the code to produce past forecast plots

You will call a script that plots a previous forecast file against new data. In the same terminal with the `rtrend_forecast` environment active, replace the desired date in the following command:

```
python check_forecast_file.py forecast_out/YYYY-MM-DD-CEPH-Rtrend_fluH.csv 
```

Where `YYYY`, `MM` and `DD` should be the year, month and day of the last week's forecast. Autocompletion (pressing 'tab' after `forecast_out/`) may help you with finding the existing files. Usually, you will use _last Monday_ for the past week forecast, the previous Monday for past two weeks forecast, etc.

If the code runs smoothly, you can safely ignore the warning about the target end date not matching the expected one for CDC, as this is a past submission deadline.

> Note: if you can't find the desired forecast file, check that you have updated your local files by Stage 2 running again (press 'y' when prompted).

### Step 3.2: Check the forecast plots

If the past step succeeded, open the project in a finder/explorer window and look for the file `ct_states_reconstructed.pdf` inside the `tmp_figs/` directory. You should see, for each state, the forecast (quantiles and median) overlapping with the latest truth data from CDC, such as in the following figure:

![past forecast example](figs/past_forecast_example.png)

## Stage 4: run and tune the forecast

 the forecast can be executed. The steps of this stage will possibly be performed multiple times, depending on the feedback about the produced forecast.

### Step 4.1: Run the forecast code

In the same terminal with the `rtrend_forecast` environment active, type:

````
python run_forecast_states.py
````

A series of messages will be printed on the screen, possibly with some warnings. If the code ends with a "Plots done!" message or similar, the forecast probably completed without errors. Otherwise, an error traceback message should indicate at which point there was an error.

### Step 4.2: Visually inspect the forecasts

### Step 4.3: Tweak parameters (if needed)

## Stage 5: check and submit





