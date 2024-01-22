# Rtrend Forecasting Method – Influenza Hospitalization

![image](rt_flu_logo.png "Rtrend Flu Logo")

This is the source code for the Rtrend method for Influenza hospitalizations forecast, designed for the [CDC Flusight initiative](https://www.cdc.gov/flu/weekly/flusight/how-flu-forecasting.htm) by the CEPH Lab at Indiana University and MOBS Lab at Northeastern University.

Team members:
- Paulo C. Ventura
- Maria Litvinova
- Allisandra G. Kummer
- Alessandro Vespignani
- Marco Ajelli 

## Setup and usage 

| Important: for the 2023/2024 season, an entirely refurbished version of the forecast code was developed. The guides shown below refer to the previous season and are outdated. Documentation for the new version is being written. |
| :-------- |

A complete guide is available in the following docs: 

- [Setup Guide](./docs/setup_guide.md) – Follow this once to install.
- [Usage Guide](./docs/usage_guide.md) – Follow this everytime you need to do a forecast.
- [Parameter Guide](./docs/parameter_guide.md) – Meaning of parameters, use this to tune the model. 
- [Submission Guide](./docs/submission_guide.md) – Follow this to submit the forecast to the main repository (instructions for CEPH Lab).


## Methodology in brief

The truth data with daily resolution (Influenza confirmed hospitalizations) is fetched from [healthdata.gov](https://healthdata.gov/). The data from each location (except USA) is processed independently through a forecast pipeline.

### Preprocessing

The past truth time series is subject to a lowpass filtering to remove short-term noise. Noise patterns, given by the difference between the raw and filtered past time series), are fitted to a Gaussian model with zero mean.

### R(t) estimation via MCMC

A Monte Carlo Markov Chain (MCMC) approach with Poisson likelihood is used to estimate the near-past reproduction number R(t) trajectories from the past data, using a gamma-distributed generation time with parameters known from the literature [1].  

### Future R(t) synthesis

The ensemble of past R(t) trajectories is used as a base to construct the forecasted R(t) trajectories. This is done by calculating the time-average R for each past trajectory within a selected interquantile range, then projecting this value through the forecast period. The projection is modulated by a time-dependent ramp, which represents estimated changes to the reproduction number trend.  

### Time series C(t) reconstruction

Each of the forecasted R(t) trajectories is combined with the generation time and the past incidence data into a renewal equation. The results of this equation are forecasted trajectories of the target incidence. Extra uncertainty is added by adding synthetic noise with the noise parameters fitted during the preprocessing stage.

### Postprocessing

The trajectories of each location are aggregated from daily resolution to weekly values. The final forecast quantiles are taken from this ensemble of trajectories.

For the USA forecast, the trajectories of all states are weekly sorted from lowest to highest incidence and then aggregated. The quantile forecasts are calculated from these aggregated trajectories. 

## References

[1] Ajelli, Marco, et al. "The role of different social contexts in shaping influenza transmission during the 2009 pandemic." Scientific reports 4.1 (2014): 1-7.
