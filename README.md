# Rtrend Forecasting Method


## Implementation details

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