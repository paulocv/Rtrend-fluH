# Rtrend Forecasting Method – Influenza Hospitalization

![image](rt_flu_logo.png "Rtrend Flu Logo")

This is the source code for the Rtrend method for Influenza hospitalizations forecast, designed for the [CDC Flusight challenge](https://www.cdc.gov/flu/weekly/flusight/how-flu-forecasting.htm) by the CEPH team at Indiana University and MOBS Lab at Northeastern University.

Team members:
- Marco Ajelli 
- Paulo C. Ventura
- Maria Litvinova
- Allisandra G. Kummer
- Alessandro Vespignani

## Setup and usage 

A complete guide is available in the following docs: 

- [Setup Guide](./docs/setup_guide.md) – Follow this once to install.
- [Usage Guide](./docs/usage_guide.md) – Follow this everytime you need to do a forecast.
- [Parameter Guide](./docs/parameter_guide.md) – Meaning of parameters, use this to tune the model. 
- [Submission Guide](./docs/submission_guide.md) – Follow this to submit the forecast to the main repository (instructions for CEPH Lab).


## Implementation details

[_IN CONSTRUCTION_]

### Preprocessing

### Interpolation

* The day of the weekly report (by default, every Saturday) is assumed to represent the **7 days prior to the report**. Example:
	* Report date: 2022-11-12
	* First day covered: 2022-11-05
	* Last day covered: 2022-11-11

* The `day_pres` time label represents "today", meaning the first day to be forecast. It is assumed as the first day in which data is unknown.

### R(t) MCMC estimation

### Future R(t) synthesis

### Time series C(t) reconstruction
