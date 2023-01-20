# Parameter guide – Rtrend

All tunable parameters of the model are written directly in the `run_forecast_states.py` 
script. Open the file and scroll down until you find the `main()` function:

```python
def main():
    ...
```

You will find the parameters in the PARAMETERS section, which goes through many lines (down to the EXECUTION section).
Some parameters (like `week_pres` and `week_roi_start`) are in the main scope of the `main()` function, while others
(such as `shape`, `rate`, `r1_start`) will be inside dictionaries (`interp_params`, `synth_params`, etc).

## List of the most important parameters and their meaning

### Ramp parameters

We currently use the so-called "dynamic ramp" to synthesize (project) R(t) values. These parameters control the bending 
of the projected reproduction number over time. These are the parameters that we most often need to tune.

For each input R value (estimated from the past time series), an output (projected) R linear ramp is created for the 
forecast period.

* `r1_start` = If the input R is 1, this is the value of the projected R at the _start_ of the ramp.
* `r2_start` = If the input R is 2, this is the value of the projected R at the _start_ of the ramp.
* `r1_end`   = If the input R is 1, this is the value of the projected R at the _end_ of the ramp.
* `r2_end`   = If the input R is 2, this is the value of the projected R at the _end_ of the ramp.

We may also use the so-called "static ramp", which is a particular case of the "dynamic ramp". If this is the case, 
then the ramp is given by coefficients that multiply the input (past) R value:

* `k_start` = Coefficient that multiplies the input R to obtain the projected R at the _start_ of the ramp.
* `k_end` = Coefficient that multiplies the input R to obtain the projected R at the _end_ of the ramp.

Note: `k_start` and `k_end` will only take effect if the "static ramp" method is activated.

### Ramp saturation

We can break (or "saturate") the ramp by making R(t) constant after a given day in the forecast period. This is controlled by 
the following parameter:

* `i_saturate` = number of **days** into the future from which the ramp saturates. This number can be positive
(ex. 7 represents one week after today) or negative (ex. -7 represents one week before the end of the forecast). 
 If set to -1, no saturation occurs.


### Quantile width of the past R(t) ensemble

We take a subset of the past R(t) time series estimated from MCMC. This subset is expressed as an inter-quantile range:

* `q_low` = Lowest quantile to be considered in the subset.
* `q_hig` = Highest quantile to be put into the subset.

### R(t) averaging period 

After past R(t) is calculated by MCMC, we take the average of the sorted ensemble for _n_ days into the past:  

* `ndays_past` = number of days into the past (since present) to be considered for the average R ensemble.

Higher values will capture the trend during a longer past period. Shorter values (like 7 or 14 days) are preferred 
if the trend in data has changed abruptly.

### Past _Region of Interest_ (ROI) for the forecast pipeline

Before the MCMC estimation takes place, we crop the entire time series (from CDC) into just a Region of Interest (ROI).

* `week_pres` = Index of the week regarded as "present". The forecast will predict for weeks starting after this one.
* `week_roi_start` = Index of the first week of the region of interest. Must be earlier than `week_pres`.

The above parameters can be informed in multiple formats:

* Negative integer _-n_: represents the n-th week since the last one in the truth data. For example, 
 `week_pres` = -1 takes the last week as present (**use this for an actual forecast**), `week_pres` = -2 assumes the
week before the last, and so on.
* Timestamp: exact date as a pandas Timestamp object. Example: ```pd.Timestamp("2022-08-29")``` (yyyy-mm-dd)

We usually set `week_pres` to -1 and `week_roi_start` to week_pres - 5, which starts the ROI 5 weeks before week_pres.




