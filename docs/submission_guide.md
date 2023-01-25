# Forecast submission guide – CEPH Lab

This guide covers the instructions to submit a forecast file to an online repository for the current week. It covers submission for the following challenges:

* Flusight forecast (Flu hospitalizations)
* COVID-19 forecast (short-term quantitative COVID-19 forecasts)

Note: To make changes to the above fork repositories, you must be logged in to the CEPH-Lab GitHub account or another one that is invited as a collaborator. 

> **Note: a few concepts that can help...**
> 
> You will modify our own fork of the CDC main GitHub repository. To better understand:
> 
> * An _online repository_ is a collection of files and folders that belong to a project. It supports version control by keeping track of changes made to it. 
>    * The CDC and the Reich Lab manage repositories for the Flusight and COVID-19 forecast challenges. Only they can directly edit these repositories.
> 
> * A _fork repository_ (or simply _fork_) is a copy of another repository that you manage. Forks let you make changes to a project without affecting the original repository.
>   * We have our own forks from the repositories of each challenge. You will only make changes to one of our forks, then submit a request for these changes to be incorporated into the original repository. This is done by the so-called _pull requests_. 

 ## Stage 1 – Add the forecast file to our fork

First, go to the fork repository corresponding to the forecast that you want to submit:

* Flusight forecast repository (CEPH fork): [https://github.com/paulocv/Flusight-forecast-data](https://github.com/paulocv/Flusight-forecast-data)

* COVID-19 forecast repository (CEPH fork): [https://github.com/paulocv/covid19-forecast-hub](https://github.com/paulocv/covid19-forecast-hub)

Before any changes, it is "polite" (but not required) to update our fork with external changes. Look for the "Sync fork" dropdown, then click on "Update branch". 

Next, look for one of the following directories and click:

* `data-forecasts` for the Flu challenge;
* `data-processed` for the COVID challenge.

Then search for the `CEPH-Rtrend_[covid, fluH]` folder and click on it. This is the folder you are allowed to modify.

In the top right corner, click on the "Add file" dropdown and select "Upload files". 

Add the latest forecast .csv file either by drag-dropping or by browsing through "choose your files". The file will be in the corresponding project folder, within the `forecast_out` subdirectory. You will look for a file with this format:

```
YYYY-MM-DD-CEPH-Rtrend_[covid, fluH].csv
```

Where YYYY-MM-DD is the current forecast date (either today or the nearest Monday), and the suffix is the corresponding challenge. 

Once you add the file, you can rename the commit by modifying the "Add files via upload" to something more meaningful, like "CEPH YYYY-MM-DD" (current date). Click on "Commit changes" to apply and upload the file. To be sure that it was uploaded, you can navigate again to `[data-forecasts` or `data-processed]`/`CEPH-Rtrend_[...]` and check that it is there.

## Stage 2 – Send the forecast to the original repository via pull request

Once the file was uploaded, look for the "Contribute" dropdown in the main page of the fork, then click on the "Open pull request" button.

If you submitted the file to the right folder, there should be no conflicts (you'll see a green "Able to merge" message), otherwise undo changes and try again. In the title of the pull request, keep the name as it is or change it to something meaningful like "CEPH YYYY-MM-DD" (just make sure that the title has the team name, "CEPH", for the COVID challenge). 

Go down and click on "Create pull request". This will submit the request, and GitHub will begin to run validation checks for the data formatting rules.

The checks should take about one minute to run, during which you will see a yellow text. If the checks run with no errors, you should see all messages in green, and the submission will be completed – no further actions needed. Yay!

If some checks fail, you will see a message in red. Don't panic. Click on the message to open the log, which is a long text with many outputs from the checks that were run. You want to specifically look for the error messages to see what went wrong; this will give you a clue about the error found in the data. You probably need to go back a few steps in the forecast or edit the data file to fix the error. See the next stage for some help.


## Stage 3 – [If there was an error in the submission]

If the validation of the data fails on GitHub, the submission is not accepted. You need to inspect and fix the prolems (check the common errors below).

Once you think you have the forecast .csv file in good shape, you will repeat the *Stage 1* of this guide as it is. However, **you won't need to repeat Stage 2**. This is because the pull request that you made before is still there; it's simply updated when you make changes.

To check the progress of the new validation, click on the link after the message "forked from [link...]" to access the official repository. Then click on the "Pull requests" tab and scroll until you find the latest CEPH pull request. Click on it and wait for the validations to run again. Repeat Stage 3 if the validation returns with errors again.


### Common errors and their solutions

I will try to update this list as I run through more error situations. 

#### "Target is not 'n' days ahead of forecast date" (or a similar message)

This can happen for the COVID challenge, as the truth data is not very consistent with the date of the last registered hospitalizations. As of the writing, I do not have an automated way to deal with this. 

_Solution_: first, inspect the forecast .csv file in the column `target`. The first entries should contain "0 day ahead inc hosp" (for COVID). Otherwise, you need to change the export range to compensate the difference.  

In the computer, open the `rtrend_tools/data_io.py` file and look for the line that contains: 

```
 [...]    = target_fmt.format(i_day    [...]
```

Add the number of days to compensate for the difference in the file. For example, if the file had "-1 day ahead inc hosp", change it to:

```
 = target_fmt.format(i_day + 1)
```

Then run the forecast again, from [Stage 4 of the usage guide](./usage_guide.md/#Stage-4:-run-and-tune-the-current-forecast).
 (Do not worry with the forecast dates, though. These are correct, it's a shift in the exported file.)