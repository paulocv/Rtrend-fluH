"""
Data-based statistics of a given season (ROI) for Flu.

NOTE: This module uses functions from the rtrend_forecast 1.0 package, under development.
Functions may be broken after a while.
It also requires that the package is accessible trhough PATH or PYTHONPATH.

"""

import argparse
import os
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from rtrend_forecast.rt_estimation import run_mcmc_rt_c
# Initial 0.0 Rtrend version functions
from rtrend_tools.data_io import load_cdc_truth_data
from rtrend_tools.forecast_structs import CDCDataBunch
from rtrend_tools.utils import map_parallel_or_sequential

# Rtrend library 1.0 functions
from rtrend_forecast.preprocessing import apply_lowpass_filter_pdseries
from rtrend_forecast.structs import RtData, ConstGammaTg, TgBase

DEFAULT_TRUTH = "hosp_data/daily_truth-Incident Hospitalizations.csv"
DEFAULT_POP = "population_data/locations.csv"

DEFAULT_OUTPUT_DIR = "season_outputs/"
DEFAULT_SCALAR_FNAME = "scalar_states.csv"


# -------------------------------
# Input parameters
# -------------------------------
class MainParams:

    def __init__(self):

        self.roi_start = pd.Timestamp("2022-07-25")
        self.roi_end = pd.Timestamp("2023-03-15")

        self.use_states = None  # ["California", "Indiana", "Colorado"]  # None  # Include-only list
        self.dont_use_states = ["US", "Virgin Islands"]  # Exclusion list

        # --- Preprocessing params
        self.cutoff = 0.05

        # --- Generation Time parameters
        self.tg_shape = 10.
        self.tg_rate = 10. / 3.
        self.tg_max = 30

        # --- MCMC parameters
        self.mcmc_dir = f"season_mcmc/cutoff{self.cutoff:0.3f}/".replace(".", "p")
        self.force_run_mcmc = False  # If true, rerun MCMC for all states

        # --- ETC
        self.ncpus = 5
        self.export_files = True
        self.show_plots = False
        self.export_plots = True


def main():

    params = MainParams()

    # Load and define stuff
    args = parse_args()
    data = load_data(args, params)
    state_dict = split_and_preprocess_all(args, data, params)
    tg = make_tg_object(params)

    # Calculate stuff
    mcmc_dict: MCMCDict = run_mcmc_for_all(state_dict, args, data, params, tg)
    scalars_df: pd.DataFrame = calc_all_scalar_metrics(params, data, state_dict, mcmc_dict)

    wrap_and_export_results(scalars_df, params, args)
    # make_plots(scalar_dfs, params)


#

# ===============================================================================================
# ===============================================================================================

#


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--truth-file", "-t", default=DEFAULT_TRUTH)
    parser.add_argument("--pop-file", "--population-file", "-p", default=DEFAULT_POP)
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


class DataBunch:
    """Groups data from external sources"""
    hosp: CDCDataBunch
    pop_df: pd.DataFrame


class StateData:
    """Preprocessed and processed data for each state"""
    name: str

    raw_sr: pd.Series  # Series of incident hospitalizations
    prep_sr: pd.Series  # Preprocessed series, used for R(t) estimation
    cum_sr: pd.Series  # Cumulative hospitalizations within ROI


StateDict = OrderedDict[str, StateData]  # Collection of StateData keyed by states
MCMCDict = OrderedDict[str, RtData]   # Collection of past R(t) data keyed by states


class OutputBunch:
    pass

# ----------------------------------------
# MAIN PROCEDURAL FUNCTIONS
# ----------------------------------------


# noinspection PyUnusedLocal
def load_data(args, params):
    print("Loading data from sources...")
    data = DataBunch()

    # Truth file
    data.hosp = load_cdc_truth_data(args.truth_file)
    # Population demographics file
    data.pop_df = pd.read_csv(args.pop_file, index_col=2)  # Col 2 = location_name

    return data


def split_and_preprocess_all(args, data: DataBunch, params: MainParams):
    """
    - Gather all required state ROIs
    - Produce filtered and cumulative incidence series

    Returns
    -------
    state_dict : OrderedDict[StateData]
    """
    print("Splitting and preprocessing data...")

    state_dict: OrderedDict[StateData] = OrderedDict()

    # Define states to run (and not run)
    use_states = (list(data.hosp.loc_names)
                  if params.use_states is None
                  else params.use_states)

    if params.dont_use_states is not None:
        for sn in params.dont_use_states:
            if sn in use_states:
                use_states.remove(sn)

    # ----------
    # Main loop
    for sn in use_states:
        state = StateData()
        state.name = sn

        # Do all the data selection: state, column, ROI
        state.raw_sr = data.hosp.xs_state(sn)["value"].loc[params.roi_start:params.roi_end]

        # Preprocess with a lowpass filter
        # apply_lowpass_filter_pdseries
        state.prep_sr = apply_lowpass_filter_pdseries(state.raw_sr, cutoff=params.cutoff)

        # Cumulative incidence series from RAW
        state.cum_sr = state.raw_sr.cumsum()

        state_dict[sn] = state

    return state_dict


def make_tg_object(params: MainParams):
    return ConstGammaTg(params.tg_shape, params.tg_rate, tmax=params.tg_max)


def run_mcmc_for_all(
        state_dict: OrderedDict[StateData], args, data: DataBunch, params: MainParams, tg: TgBase,
):
    """"""
    print("--------------------")
    print("MCMC stage")
    print("--------------------")
    num_sims = len(state_dict)

    def task(inputs):
        i_sn, state = inputs
        state: StateData
        sn = state.name

        # Prepare custom prefixes and paths
        sim_name = sn.replace(" ", "_")  # Replace spaces on state names to use as path
        out_fname = os.path.join(params.mcmc_dir, "outputs", f"{sim_name}_rt.out")
        # ^ ^ WARNING: change this if run_mcmc_rt_c changes

        # Main decision between rerunning or just loading
        if params.force_run_mcmc:
            print(f"RUNNING MCMC for {sn} ({i_sn + 1} of {num_sims})")

            result: RtData = run_mcmc_rt_c(state.prep_sr.to_numpy(), tg.get_param_arrays_bysize(state.prep_sr.shape[0]),
                                           in_prefix=os.path.join(params.mcmc_dir, "inputs/"),
                                           out_prefix=os.path.join(params.mcmc_dir, "outputs/"),
                                           log_prefix=os.path.join(params.mcmc_dir, "logs/"), sim_name=sim_name)

        else:  # Try to load from file.
            print(f"LOADING MCMC for {sn} ({i_sn + 1} of {num_sims})")
            try:
                result = RtData(np.loadtxt(out_fname))
            except FileNotFoundError:
                print(f"MCMC File for {sn} ({out_fname}) not found. Skipping...")
                result = None

        result.df.columns = state.prep_sr.index  # Attribute dates to columns
        return result

    content = list(enumerate(state_dict.values()))

    # --- RUN COMMAND
    mcmc_list: list[RtData] = map_parallel_or_sequential(task, content, ncpus=params.ncpus)
    mcmc_dict = OrderedDict(zip(state_dict.keys(), mcmc_list))

    print("=======================================")

    return mcmc_dict


#

# ===============================================================================================
# ===============================================================================================

#

def calc_all_scalar_metrics(
        params: MainParams, data: DataBunch, state_dict: StateDict, mcmc_dict: MCMCDict):
    """"""
    print("Calculating scalar metrics...")
    metrics = defaultdict(list)

    def task(input_tup):
        # Collect basic info
        sn, state = input_tup
        mcmc = mcmc_dict[sn]
        pop_size = data.pop_df.loc[sn, "population"]

        state: StateData

        # --- Calculate metrics
        # Relative and absolute attack rates
        atk_rate = int(state.raw_sr.sum())

        # *Very* Simple maximum incidence: from raw data
        raw_max_inc_date = state.raw_sr.idxmax()
        raw_max_inc = state.raw_sr.loc[raw_max_inc_date]

        # Simple maximum incidence (just the max of the preprocessed series)
        max_inc_date = state.prep_sr.idxmax()
        max_inc = state.prep_sr.loc[max_inc_date]

        # Trigger stuff (e.g. beginning of the outbreak)
        outb_start = calc_trigger_stuff(state.raw_sr, pop_size)

        # Peak (by finding R = 1 or by simple maximum incidence)
        if mcmc is not None:
            peak_date, peak_inc, peak_method = calc_criterious_peak(mcmc.df.median(axis=0), state.prep_sr)
            peak_cum_inc = int(state.cum_sr.loc[peak_date])
            rel_peak_cum_inc = peak_cum_inc / pop_size
        else:
            peak_date, peak_inc, peak_method = None, None, None
            peak_cum_inc, rel_peak_cum_inc = None, None

        # --- Append results
        # Please append None explicitly if a metric can't be calculated
        metrics["population"].append(pop_size)
        metrics["sum_hosp"].append(atk_rate)
        metrics["rel_attack_rate"].append(atk_rate / pop_size)
        metrics["peak_date"].append(peak_date)
        metrics["peak_inc"].append(peak_inc)
        metrics["peak_cum_inc"].append(peak_cum_inc)
        metrics["rel_peak_cum_inc"].append(rel_peak_cum_inc)
        metrics["peak_method"].append(peak_method)
        metrics["outb_start_date"].append(outb_start)
        metrics["max_inc_date"].append(max_inc_date)
        metrics["max_inc"].append(max_inc)
        metrics["raw_max_inc_date"].append(raw_max_inc_date)
        metrics["raw_max_inc"].append(raw_max_inc)

    content = list(state_dict.items())

    # CAN't RUN IN PARALLEL: shared memory `metrics`
    for arg in content:
        task(arg)

    df = pd.DataFrame(metrics, index=state_dict.keys())
    df.index.name = "state_name"
    return df


def calc_one_crossings(sr: pd.Series):
    """Calculates all dates (index values) in which the series crosses R = 1."""
    ft = (sr - 1.0)  # f(t) = sr(t) - 1.0
    mask = (ft.iloc[1:].values * ft.iloc[:-1].values) < 0.  # Find changes in signal of f(t) = R(t) - 1
    # Takes the day FOLLOWING a zero-crossing
    crossing_dates = ft.index[1:][mask]

    return crossing_dates


def calc_one_crossings_toplot(sr: pd.Series):
    crossing_dates = calc_one_crossings(sr)
    xpts = np.repeat(crossing_dates.to_numpy()[np.newaxis, :], 2, axis=0)
    ypts = np.repeat([[0.5, ], [2.0, ]], crossing_dates.shape[0], axis=1)

    return xpts, ypts


def calc_trigger_stuff(sr: pd.Series, pop_size, start_thres=50.E-6):
    """
    start_thres : float
        Value of the cumulative series that's considered as the "start" of the outbreak.
        The value is directly applied to the relative series (i.e., divided by pop_size).
    """
    rel_csum = sr.cumsum() / pop_size

    # Start trigger detection
    mask: pd.Series = rel_csum > start_thres
    start_date = mask.idxmax() if mask.any() else mask.index[-1]  # If not found, take the LAST DATE!

    return start_date


def calc_criterious_peak(rt_series: pd.Series, inc_series: pd.Series, max_inc_date=None):
    """Calculates the peak of incidence using an elaborate criterion.
    The method first tries to find the R = 1 crossing that has the largest
    associated incidence. If this fails, simply takes the date of the largest
    incidence.

    Ideally, rt_series must be the median of the MCMC estimations, but this
    can be modified to use more data from the MCMC R(t) ensemble.
    """
    # Simple maximum of the incidence
    max_inc_date = inc_series.idxmax() if max_inc_date is None else max_inc_date
    max_inc = inc_series.loc[max_inc_date]

    # One-crossings of the R(t) median
    one_dates = calc_one_crossings(rt_series)

    # Criterion application
    if one_dates.shape[0] == 0:  # No crossings by the median - take maximum incidence
        chosen = max_inc_date
        method = "max_inc_nocross"

    else:  # At least one crossing - take that with the largest incidence
        chosen = inc_series.loc[one_dates].idxmax()  # Choose the one with the largest incidence

        if inc_series.loc[chosen] / max_inc < 0.75:  # Incidence too small compared with actual peak
            chosen, method = max_inc_date, "max_inc_smallcross"
        else:  # Accepts the choice made by the default method
            method = "r1_max_crossing"

    # Signature: chosen date, incidence at chosen, maximum incidence, method of choice
    return chosen, inc_series.loc[chosen], method


def wrap_and_export_results(
        scalars_df: pd.DataFrame, params: MainParams, args):

    # ----------
    # Scalar metrics
    # scalar_df = pd.concat(scalar_dfs, axis=1)

    # To view scalar metrics on screen
    print(scalars_df)

    if params.export_files:
        os.makedirs(args.output_dir, exist_ok=True)

        # Scalar metrics file
        scalars_df.to_csv(
            os.path.join(args.output_dir, "scalar_metrics.csv")
        )


def make_plots():
    pass


if __name__ == "__main__":
    main()
