"""
Load CDC data, run forecast for each state, export results and reports.

For the first version, this script was made in a hurry.

loc == state

ROI = region of interest: ideally, comprises the current outbreak since its "beginning".
ndays_past = Used only for the R(t) synthesis: number of days in the past to consider.

pres = present = Day of the last CDC report.
"""
import copy
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
# from pathos.multiprocessing import ProcessPool
from scipy.interpolate import UnivariateSpline

import py_modules.interpolate as interp
import py_modules.synthesis as synth
import py_modules.visualization as vis
import py_modules.data_io as dio

from py_modules.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES, NUM_STATES, NUM_WEEKS_FORE
from py_modules.utils import map_parallel_or_sequential
from toolbox.file_tools import make_folder
from toolbox.plot_tools import make_axes_seq


WEEKLEN = interp.WEEKLEN
MAX_PIPE_STEPS = 100  # Max. number of overall steps in the forecast pipeline to declare failure
PIPE_STAGE = ("interp", "r_estim", "r_synth", "c_reconst")  # Sequence of stages of the forecast

MIN_CT_SAMPLES = 100  # Have at least this number of forecast c(t) curves for "approval"


def main():
    # PARAMETERS
    # --------------------
    # --- Preprocess params
    ct_cap_fac = 8  # Multiply the historical maximum by this factor to obtain the ct_cap (maximum)

    # ROI DEFINITION

    # # -()- PAST OUTBREAK - Peak
    # week_pres = pd.Timestamp("2022-04-30")
    # week_roi_start = week_pres - pd.Timedelta(6, "W")

    # # -()- PAST OUTBREAK - beginning
    # week_pres = pd.Timestamp("2022-03-01")
    # week_roi_start = week_pres - pd.Timedelta(6, "W")

    # -()- Current season
    week_pres = -1  # Last week
    week_roi_start = week_pres -5  # pd.Timestamp("2022-08-29")

    # --- Forecast params
    # General
    nweeks_fore = NUM_WEEKS_FORE  # Number of weeks to forecast
    nsamples_us = 1000  # Number of C(t) samples to take from each state for building the US forecast.
    # # First used values
    # tg_params = dict(
    #     shape=17.,
    #     rate=7.39,
    #     tmax=30,  # int
    # )

    tg_params = dict(
        shape=10.,
        rate=10./3.,
        tmax=30,  # int
    )

    # W->D Interpolation
    interp_params = dict(
        rel_smooth_fac=0.01,
        spline_degree=3,
    )

    # MCMC calculation of R(t)
    mcmc_params = dict(
        in_prefix="mcmc_inputs/states_in/",
        out_prefix="mcmc_outputs/states_out/",
        log_prefix="mcmc_logs/states_log/",
        nsim=20000,  # Number of simulations (including 10,000 burn-in period)
        seed=10,
        sigma=0.05,
    )

    # R(t) Synthesis
    synth_params = dict(
        # Ramp params
        q_low=0.40,    # Sorted ensemble: low quantile
        q_hig=0.60,    # Sorted ensemble: high quantile
        ndays_past=21,  # Number of days (backwards) to consider in synthesis.
        r_max=1.8,     # Clamp up to these R values (for the average)
        k_start=0.95,   # Ramp method: starting coefficient
        k_end=0.85,      # Ramp method: ending coefficient

        # Rtop ramp extra params
        rmean_top=1.40,  # (Obs: Starting value, but it is decreased at each pass, including the first)
        max_width=0.15,

        # Random normal params
        seed=10,
        center=1.0,
        sigma=0.05,
        num_samples=1000  # Use this instead of the MCMC ensemble size.
    )

    # C(t) reconstruction
    recons_params = dict(seed=2)

    cdc_data_fname = "hosp_data/truth-Incident Hospitalizations.csv"
    cdc_out_fname = "forecast_out/latest.csv"

    # --- Program parameters
    do_export = True
    do_show_plots = False
    write_synth_names = True
    ncpus = 5  # If 1, runs sequential. Otherwise, calls pathos.

    # --- DICTIONARY OF EXCEPTIONS - For states that require different parameters than the default.

    exception_dict = dict()

    # SELECT ONLY THESE STATES
    # use_states = ["California", "Texas", "Michigan", "Indiana", "Wyoming", "New York", "Colorado", "Maryland"]  #
    use_states = None

    # PRELIMINARIES
    # -------------
    if not do_export:
        print(10 * "#" + " EXPORT SWITCH IS OFF " + 10 * "#")

    print("Importing and preprocessing CDC data...")
    cdc = load_cdc_data(cdc_data_fname)
    preproc_dict, ct_cap_dict = preprocess_cdc_data(cdc, week_pres, week_roi_start, ct_cap_fac=ct_cap_fac)
    print("\n#################\n")

    # FORECAST EXECUTION
    # ------------------

    def task(inputs):
        i_s, state_name = inputs

        # Select states for testing purposes (#filter)
        if use_states and state_name not in use_states:
            return None

        if state_name in ["US"]:  # Exclusion list
            return None

        print(f"-------------- ({i_s+1} of {cdc.num_locs}) Starting {state_name}...")

        # ----------------- FORECAST

        state_series = preproc_dict[state_name]
        except_params = exception_dict.get(state_name, None)

        # FORECAST COMMAND
        fc: ForecastOutput = \
            forecast_state(state_name, state_series, nweeks_fore, except_params, tg_params, interp_params, mcmc_params,
                           synth_params, recons_params, ct_cap=ct_cap_dict[state_name])

        if fc is None:
            print(f"   [{state_name}] Returned empty, skipped.")
            return None
        print(f"   [{state_name}] Forecast done with {fc.num_ct_samples} c(t) samples.")

        # POSTPROCESS COMMAND
        return postprocess_state_forecast(fc, state_name, i_s, cdc, nweeks_fore, nsamples_us)

    # RUN LINE - PARALLEL OR SEQUENCE
    result_list = map_parallel_or_sequential(task, enumerate(cdc.loc_names), ncpus=ncpus)

    # # Check the execution time of some chunk
    # print(f"Execution time = {sum(post.xt for post in result_list if post is not None)}s")

    # --- Construct the total US time series
    print("-------------- Starting for US ")
    us_post = postprocess_us(result_list, cdc, nweeks_fore, nsamples_us)
    print(f"\t[US] Done ({nsamples_us} c(t) samples).")

    # VISUALIZATION and EXPORT
    # --------------------------

    if do_export:
        print(f"__________ \n Exporting to '{cdc_out_fname}'...")
        dio.export_forecast_cdc(cdc_out_fname, result_list, us_post, cdc, nweeks_fore)

    print("\n___________________\nNOW PLOTTING\n")
    make_plot_tables(result_list, cdc, preproc_dict, nweeks_fore, us_post, write_synth_names,
                     ncpus=1)
    print("Plots done!")
    if do_show_plots:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# STRUCTS AND CLASSES
# ----------------------------------------------------------------------------------------------------------------------


class CDCDataBunch:

    def __init__(self):
        self.df: pd.DataFrame = None  # DataFrame with CDC data as read from the file.

        self.num_locs = None  # Number of unique locations
        self.loc_names = None  # List of unique location names
        self.loc_ids = None  # List of location ids (as strings)
        self.to_loc_id = None  # Conversion dictionary from location name to location id
        self.to_loc_name = None  # Conversion dictionary from location id to location name

        self.data_time_labels = None  # Array of dates present in the whole dataset.

        # Preprocessing data (not set during loading)
        self.day_roi_start: pd.Timestamp = None
        self.day_pres: pd.Timestamp = None

    def xs_state(self, state_name):
        return self.df.xs(self.to_loc_id[state_name], level="location")


class ForecastExecutionData:
    """
    Data for the handling of the forecast pipeline for a single state.
    Handled inside the function forecast_state().
    """

    def __init__(self):
        # Input parameters and data
        self.state_name = None
        self.state_series: pd.Series = None
        self.nweeks_fore = None
        self.except_params: dict = None
        self.tg_params: dict = None
        self.interp_params: dict = None
        self.mcmc_params: dict = None
        self.synth_params: dict = None
        self.recons_params: dict = None
        self.ct_cap = None

        # Pipeline handling
        self.stage = None   # Determines the forecast stage to be run in the next pipeline step. Written at debriefing.
        self.method = None  # Determines the method to use in current stage. Written at briefing.
        self.notes = None   # Debriefing notes to the briefing of the next stage. Written at debriefing.
        self.stage_log = list()

    def log_current_pipe_step(self):
        """Writes current stage and method into the stage_log"""
        self.stage_log.append((self.stage, self.method))


class ForecastOutput:
    """Contains data from forecasting process in one state: from interpolation to weekly reaggregation."""

    def __init__(self):
        self.day_0: pd.Timestamp = None  # First day to have report (7 days before the first stamp in ROI)
        self.day_pres: pd.Timestamp = None  # Day of the current report.
        self.ndays_roi = None
        self.ndays_fore = None

        # MCMC (and etc)
        self.tg_dist: np.ndarray = None

        # Interpolation results
        self.t_daily: np.ndarray = None
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        # self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.ct_past: np.ndarray = None
        self.float_data_daily: np.ndarray = None  # Float data from the spline of the cumulative sum
        self.daily_spline: UnivariateSpline = None

        # MCMC
        self.rtm: synth.McmcRtEnsemble = None

        # Synthesis
        self.synth_name = None
        self.rt_fore2d: np.ndarray = None
        self.num_ct_samples = None

        # Reconstruction
        self.ct_fore2d: np.ndarray = None
        self.ct_fore2d_weekly: np.ndarray = None
        # self.weekly_quantiles: np.ndarray = None


class ForecastPost:
    """
    Contains post-processed data from the forecasting process in one state: the quantiles and other lightweight
    arrays that are sent back to the main scope.
    """
    def __init__(self):
        self.state_name = None
        self.synth_name = None

        # Forecasted data
        self.quantile_seq = None  # Just CDC default sequence of quantiles
        self.num_quantiles = None
        self.weekly_quantiles: np.ndarray = None  # Forecasted quantiles | a[i_q, i_week]
        self.samples_to_us: np.ndarray = None  # ct_fore_weekly samples to sum up the US series. | a[i_sample, i_week]

        # Interpolation
        self.t_daily: np.ndarray = None
        self.ct_past: np.ndarray = None
        self.float_data_daily: np.ndarray = None
        self.daily_spline: UnivariateSpline = None

        # MCMC past R(t) stats
        self.rt_past_mean: np.ndarray = None
        self.rt_past_median: np.ndarray = None
        self.rt_past_loquant: np.ndarray = None
        self.rt_past_hiquant: np.ndarray = None

        # Synthesis forecast R(t) stats
        self.rt_fore_median: np.ndarray = None
        self.rt_fore_loquant: np.ndarray = None
        self.rt_fore_hiquant: np.ndarray = None

        # Time labels
        self.day_0: pd.Timestamp = None     # First day of ROI.
        self.day_pres: pd.Timestamp = None  # Day time stamp of present
        self.day_fore: pd.Timestamp = None  # Last day forecasted
        self.past_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the ROI
        self.fore_daily_tlabels: pd.DatetimeIndex = None  # Daily time stamps for the forecast region
        self.fore_time_labels: pd.DatetimeIndex = None  # Weekly time stamps for forecast region

        # Misc
        self.xt = None  # Just some counter for execution time


class USForecastPost:
    """Special post process object for the US time series."""

    def __init__(self):
        self.weekly_quantiles: np.ndarray = None  # Forecasted quantiles | a[i_q, i_week]
        self.num_quantiles = None

        self.day_0: pd.Timestamp = None     # First day of ROI.
        self.day_pres: pd.Timestamp = None  # Day time stamp of present
        self.day_fore: pd.Timestamp = None  # Last day forecasted
        self.fore_time_labels: pd.DatetimeIndex = None


# ----------------------------------------------------------------------------------------------------------------------
# FORECAST PIPELINE FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------


def callback_interpolation(exd: ForecastExecutionData, fc: ForecastOutput):

    # Briefing
    # ---------------------------

    exd.method = "STD"

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------
    fc.day_0 = exd.state_series.index[0] - pd.Timedelta(WEEKLEN, "D")  # First accounted day (7 DAYS BEFORE 1st REPORT)
    fc.day_pres = exd.state_series.index[-1]
    t_weekly = (exd.state_series.index - fc.day_0).days.array  # Weekly number of days as integers.
    fc.t_daily, fc.ct_past, fc.daily_spline, fc.float_data_daily = \
        interp.weekly_to_daily_spline(t_weekly, exd.state_series, return_complete=True, **exd.interp_params)

    fc.past_daily_tlabels = fc.day_0 + pd.TimedeltaIndex(fc.t_daily, "D")

    fc.ndays_roi = fc.t_daily.shape[0]
    fc.ndays_fore = WEEKLEN * exd.nweeks_fore

    # Debriefing
    # ---------------------------

    exd.stage, exd.notes = "r_estim", "NONE"  # Approve with default method


def callback_r_estimation(exd: ForecastExecutionData, fc: ForecastOutput):

    # Briefing
    # ---------------------------

    # BRIEFING COMES HERE
    exd.method = "STD"

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------

    # --- Make Tg distribution
    fc.tg_dist = synth.make_tg_gamma_vector(**exd.tg_params)
    sn = exd.state_name.replace(" ", "")  # Eliminates spaces from state names to use in file names.

    # --- Prepare directories and file names
    make_folder(os.path.dirname(exd.mcmc_params["in_prefix"]))
    make_folder(os.path.dirname(exd.mcmc_params["out_prefix"]))
    make_folder(os.path.dirname(exd.mcmc_params["log_prefix"]))
    mcmc_tseries_fname = exd.mcmc_params["in_prefix"] + sn + "_tseries.in"
    mcmc_tgdist_fname = exd.mcmc_params["in_prefix"] + sn + "_tgdist.in"
    mcmc_rtout_fname = exd.mcmc_params["out_prefix"] + sn + "_rt.out"
    mcmc_log_fname = exd.mcmc_params["log_prefix"] + sn + "_err.log"

    # --- Prepare inputs and save to files.
    # Cases (tseries) file
    impcases_array = np.zeros(fc.ndays_roi, dtype=int)
    impcases_array[0] = 1  # Add an imported case
    np.savetxt(mcmc_tseries_fname, np.array([fc.t_daily, fc.ct_past, impcases_array]).T, fmt="%d")
    # Tg file (tgdist)
    mcmc_tg_array = np.repeat([[exd.tg_params["shape"], exd.tg_params["rate"]]], fc.ndays_roi, axis=0)
    np.savetxt(mcmc_tgdist_fname, mcmc_tg_array)

    # --- Call MCMC code
    errcode = os.system(f"./main_mcmc_rt {exd.mcmc_params['seed']} {exd.mcmc_params['nsim']} {exd.mcmc_params['sigma']}"
                        f" {mcmc_tgdist_fname} {mcmc_tseries_fname} > {mcmc_rtout_fname} 2> {mcmc_log_fname}")

    # Debriefing
    # ---------------------------
    if errcode:
        sys.stderr.write(f"\n[ERROR {exd.state_name}] MCMC ERROR CODE = {errcode}. || "
                         f"Please check \"{mcmc_log_fname}\".\n")
        exd.stage = "ERROR"  # Send error signal to pipeline
        return

    # --- Load outputs from MCMC (parameters are for stats calculations over the R(t) ensemble)
    fc.rtm = synth.McmcRtEnsemble(np.loadtxt(mcmc_rtout_fname), rolling_width=WEEKLEN, quantile_q=0.025)

    exd.stage, exd.notes = "r_synth", "NONE"  # Approve with default method


def callback_r_synthesis(exd: ForecastExecutionData, fc: ForecastOutput):

    # Briefing
    # ---------------------------
    # Exception states (these would have exceedingly explosive outbreaks with the default ramp).
    special_states_01 = ["Alabama", "Illinois", "Louisiana", "Maryland", "North Carolina", "South Carolina"]

    # exd.method = "STD"

    # PALEATIVE: add cases to the latest days, preventing absolute zeros.
    if fc.ct_past[-3:].max() == 0:
        fc.ct_past[-2] = 1

    # Cumulative hospitalizations inside ROI
    sum_roi = exd.state_series.sum()

    # --- Method decision regarding the cumulative hospitalizations in the ROI
    # OBS: TRY TO KEEP A SINGLE IF/ELIF CHAIN for better clarity of the choice!

    if sum_roi <= 50 or exd.notes == "use_rnd_normal":  # Too few cases for ramping, or rnd_normal previously asked.
        synth_method = synth.random_normal_synth
        fc.synth_name = exd.method = "rnd_normal"

        if exd.state_name == "Virgin Islands":  # Chose params for more conservative estimates.
            exd.synth_params["center"] = 0.8
            exd.synth_params["sigma"] = 0.02

    elif exd.notes == "NONE" and sum_roi > 1200:        # --- Large numbers: increase width of the sample
        # exd.synth_params["q_low"] = 0.30
        # exd.synth_params["q_hig"] = 0.70
        synth_method = synth.sorted_bent_mean_ensemble
        fc.synth_name = exd.method = "static_ramp_highc"

    elif exd.notes == "NONE":  # DEFAULT CONDITION, does not fall in any of the previous
        # # -()- Static ramp
        # synth_method = synth.sorted_bent_mean_ensemble
        # fc.synth_name = exd.method = "static_ramp"

        # -()- Dynamic ramp
        synth_method = synth.sorted_dynamic_ramp
        fc.synth_name = exd.method = "dynamic_ramp"

    elif exd.notes == "use_static_ramp_rtop":
        exd.synth_params["rmean_top"] -= 0.05  # Has an initial value. Decreases at each pass.
        # exd.synth_params["max_width"] = 0.35
        exd.synth_params["k_start"] = 1.0
        exd.synth_params["k_end"] = 0.85
        synth_method = synth.sorted_bent_ensemble_valbased
        fc.synth_name = exd.method = f"static_ramp_rtop_{exd.synth_params['rmean_top']:0.2f}"

    else:
        sys.stderr.write(f"\nHey, abnormal condition found in r_synthesis briefing. (exd.method = {exd.method})\n")
        exd.stage = "ERROR"
        return

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------

    # Apply the method
    fc.rt_fore2d = synth_method(fc.rtm, fc.ct_past, fc.ndays_fore, **exd.synth_params)

    # if exd.state_name in special_states_01:
    # # if fc.synth_name == "rnd_normal_special":
    #     # Apply extra ramp to the normal samples
    #     fc.rt_fore2d = synth.apply_ramp_to(fc.rt_fore2d, 1.0, 0.98, fc.ndays_fore)

    if fc.rt_fore2d.shape[0] == 0:  # Problem: no R value satisfied the filter criterion
        # Action: repeat the synth right away
        fc.synth_name = exd.method = "rnd_normal"
        exd.log_current_pipe_step()

        fc.rt_fore2d = synth.random_normal_synth(fc.rtm, fc.ct_past, fc.ndays_fore, **exd.synth_params)

    # Debriefing
    # ---------------------------

    # if exd.state_name in special_states_01:
    #     exd.stage, exd.notes = "c_reconst", "no_filter"  # Approve, but signal that state is mustn't be ct-filtered
    # else:
    exd.stage, exd.notes = "c_reconst", "NONE"  # Approve with default method


def callback_c_reconstruction(exd: ForecastExecutionData, fc: ForecastOutput):

    # Briefing
    # ----------------------------
    can_apply_ct_cap = (exd.notes != "no_filter")
    exd.method = "STD"

    exd.log_current_pipe_step()

    # Execution
    # ----------------------------
    fc.ct_fore2d = synth.reconstruct_ct(fc.ct_past, fc.rt_fore2d, fc.tg_dist,
                                        tg_max=exd.tg_params["tmax"], **exd.recons_params)
    fc.ct_fore2d_weekly = interp.daily_to_weekly(fc.ct_fore2d)

    # Debriefing
    # ----------------------------

    # --- Filter series that exceeded ct_cap
    if exd.ct_cap is not None and can_apply_ct_cap:
        mask = np.max(fc.ct_fore2d_weekly, axis=1) <= exd.ct_cap  # Mask: True for reasonable entries
        num_remaining = mask.sum()

        if num_remaining < MIN_CT_SAMPLES:  # Too few cases after filtering
            # exd.stage, exd.method = "r_synth", "rnd_normal"
            # exd.stage, exd.notes = "r_synth", "use_rnd_normal"
            exd.stage, exd.notes = "r_synth", "use_static_ramp_rtop"
            return

        # Otherwise, applies the filter
        fc.ct_fore2d_weekly = fc.ct_fore2d_weekly[mask, :]

    # Attribute number of samples
    fc.num_ct_samples = fc.ct_fore2d_weekly.shape[0]

    # If this point was reached, results are satisfactory
    exd.stage = "DONE"  # Approve the forecast

# ----------------------------------------------------------------------------------------------------------------------
# MAIN FORECAST ROUTINE
# ----------------------------------------------------------------------------------------------------------------------


def forecast_state(state_name: str, state_series: pd.Series, nweeks_fore, except_params: dict,
                   tg_params, interp_params, mcmc_params, synth_params, recons_params, ct_cap=None):
    """
    """
    # PREAMBLE
    # ------------------------------------------------------------------------------------------------------------------
    # TODO: replace except params

    synth_params = copy.deepcopy(synth_params)  # Deepcopy to protect original values.

    exd = ForecastExecutionData()
    fc = ForecastOutput()

    # Populate execution data (exd). Params are deep-copied to protect original master dictionary.
    exd.state_name = state_name
    exd.state_series = state_series
    exd.nweeks_fore = nweeks_fore
    exd.except_params = except_params
    exd.tg_params = copy.deepcopy(tg_params)
    exd.interp_params = copy.deepcopy(interp_params)
    exd.mcmc_params = copy.deepcopy(mcmc_params)
    exd.synth_params = copy.deepcopy(synth_params)
    exd.recons_params = copy.deepcopy(recons_params)
    exd.ct_cap = ct_cap

    # PIPELINE LOOP OF STAGES
    # ------------------------------------------------------------------------------------------------------------------

    # Initialize the
    exd.stage, exd.method, exd.notes = PIPE_STAGE[0], "STD", "NONE"

    for i_pipe_step in range(MAX_PIPE_STEPS):

        if exd.stage == "interp":  # STAGE 0: C(t) weekly to daily interpolation
            callback_interpolation(exd, fc)

        elif exd.stage == "r_estim":  # STAGE 1: R(t) MCMC estimation
            callback_r_estimation(exd, fc)

        elif exd.stage == "r_synth":  # STAGE 2: Future R(t) synthesis
            callback_r_synthesis(exd, fc)

        elif exd.stage == "c_reconst":  # STAGE 3: Future C(t) reconstruction
            callback_c_reconstruction(exd, fc)

        elif exd.stage == "DONE":  # Successful completion of all stages
            break

        elif exd.stage == "ERROR":  # A problem occurred in previous pipe step.
            """HANDLE FAILURE HERE"""
            # Current policy for pipeline errors: skip state.
            sys.stderr.write(f"\n\t[{state_name} will be skipped.]\n")
            return None

        else:
            raise ValueError(f"[ERROR {state_name}] Hey, exd.method = \"{exd.method}\" unrecognized.")

    else:
        sys.stderr.write(f"\n[ERROR {state_name}] Max number of pipeline steps reached. Results will be broken.\n"
                         f"Log of stages:\n")
        sys.stderr.write(exd.stage_log.__str__())
        sys.stderr.write('\n')

    # print(f"WATCH [{state_name}] LOG")
    # print(exd.stage_log)

    return fc


def calc_ct_fore_quantiles(ct_fore2d, quantiles_seq=None):
    if quantiles_seq is None:
        quantiles_seq = CDC_QUANTILES_SEQ

    result = np.empty((quantiles_seq.shape[0], ct_fore2d.shape[1]), dtype=float)

    for i_q, q in enumerate(quantiles_seq):  # OBS: q can be an array, so there may be a vectorized alternative.
        np.quantile(ct_fore2d, q, axis=0, out=result[i_q, :])

    return result


def load_cdc_data(fname):

    cdc = CDCDataBunch()

    # Import
    cdc.df = pd.read_csv(fname, index_col=(0, 1), parse_dates=["date"])
    
    # Extract unique names and their ids
    cdc.loc_names = np.sort(cdc.df["location_name"].unique())  # Alphabetic order
    cdc.num_locs = cdc.loc_names.shape[0]
    cdc.loc_ids = cdc.df.index.levels[1].unique()

    # Make location id / name conversion
    cdc.to_loc_name = dict()  # Converts location id into name
    cdc.to_loc_id = dict()  # Converts location name into id
    for l_id in cdc.loc_ids:
        name = cdc.df.xs(l_id, level=1).iloc[0]["location_name"]  # Get the name from the first id occurrence
        cdc.to_loc_name[l_id] = name
        cdc.to_loc_id[name] = l_id

    cdc.data_time_labels = cdc.df.index.levels[0].unique().sort_values()

    return cdc


def preprocess_cdc_data(cdc: CDCDataBunch, week_pres, week_roi_start, ct_cap_fac=3):
    """
    Splits dataset by state, crop ROI.
    FOR NOW, returns only a dict with the cropped ROI (with time stamps as index).
    """

    # Argument handling
    # -----------------
    if isinstance(week_pres, int):
        week_pres = cdc.data_time_labels[week_pres]

    if isinstance(week_roi_start, int):
        week_roi_start = cdc.data_time_labels[week_roi_start]

    if week_roi_start >= week_pres:
        raise ValueError(f"Hey, ROI start {week_roi_start.date()} must be prior to present week {week_pres.date()}.")

    print(f"Past ROI: from {week_roi_start.date()} to {week_pres.date()}")
    cdc.day_roi_start = week_roi_start
    cdc.day_pres = week_pres

    # Preprocessing
    # -------------
    res_dict = dict()
    ct_cap_dict = dict()

    for state_name in cdc.loc_names:
        state_df = cdc.df.xs(cdc.to_loc_id[state_name], level="location")

        # Set cap based on maximum historical data (whole tseries, not just ROI).
        ct_cap_dict[state_name] = ct_cap_fac * state_df["value"].max()

        roi_df = state_df.loc[week_roi_start:week_pres]["value"]

        # Check for problems
        if roi_df.shape[0] == 0:
            sys.stderr.write(f"\nHEY, empty dataset for state {state_name} in ROI.\n")

        res_dict[state_name] = roi_df

    return res_dict, ct_cap_dict


def postprocess_state_forecast(fc: ForecastOutput, state_name, i_s, cdc: CDCDataBunch, nweeks_fore,
                               nsamples_us):
    """
    Prepare data for final aggregation and for plot reports.
    Returns a ForecastPost object, which is lightweight and can be returned from the forecast loop to the main scope.
    """

    post = ForecastPost()
    post.state_name = state_name
    post.synth_name = fc.synth_name

    # Forecast data statistics
    post.quantile_seq = CDC_QUANTILES_SEQ
    post.num_quantiles = NUM_QUANTILES
    post.weekly_quantiles = calc_ct_fore_quantiles(fc.ct_fore2d_weekly)
    post.num_ct_samples = fc.ct_fore2d_weekly.shape[0]

    # Interpolation
    post.daily_spline = fc.daily_spline
    post.t_daily = fc.t_daily
    post.ct_past = fc.ct_past
    post.float_data_daily = fc.float_data_daily

    # Stats of the past R(t) from MCMC
    post.rt_past_mean = fc.rtm.get_avg()
    post.rt_past_median = fc.rtm.get_median()
    post.rt_past_loquant, post.rt_past_hiquant = fc.rtm.get_quantiles()

    # Stats of the R(t) synthesis
    post.rt_fore_median = np.median(fc.rt_fore2d, axis=0)
    post.rt_fore_loquant = np.quantile(fc.rt_fore2d, 0.025, axis=0)
    post.rt_fore_hiquant = np.quantile(fc.rt_fore2d, 0.975, axis=0)

    # Datetime-like array of week day stamps
    post.day_0, post.day_pres = fc.day_0, fc.day_pres
    post.day_fore = fc.day_pres + pd.Timedelta(nweeks_fore, "W")
    post.past_daily_tlabels = fc.past_daily_tlabels
    post.fore_daily_tlabels = pd.date_range(fc.day_pres, periods=WEEKLEN*nweeks_fore, freq="D")
    post.fore_time_labels = \
        pd.DatetimeIndex((fc.day_pres + pd.Timedelta((i+1) * WEEKLEN, "D") for i in range(nweeks_fore)))
    
    # Contribution of the state to the US time series.
    post.samples_to_us = synth.sample_series_with_replacement(fc.ct_fore2d_weekly, nsamples_us, seed=10 + i_s)

    return post


def postprocess_us(fc_list: list[ForecastPost], cdc: CDCDataBunch, nweeks_fore, nsamples_us):

    post = USForecastPost()

    # Cut ROI and determine key days based on the overall CDC data, so it's independent of any state.
    post.day_0 = cdc.day_roi_start
    post.day_pres = cdc.day_pres
    post.day_fore = post.day_pres + pd.Timedelta(nweeks_fore, "W")
    post.fore_time_labels = \
        pd.DatetimeIndex((post.day_pres + pd.Timedelta((i+1) * WEEKLEN, "D") for i in range(nweeks_fore)))

    # Perform the sum of samples from each state
    ct_fore2d_weekly = sum(post_.samples_to_us for post_ in fc_list if post_ is not None)
    post.weekly_quantiles = calc_ct_fore_quantiles(ct_fore2d_weekly)
    post.num_quantiles = post.weekly_quantiles.shape[0]

    return post


def make_plot_tables(post_list, cdc: CDCDataBunch, preproc_dict, nweeks_fore, us,
                     write_synth_names=True, ncols=3, nrows=3, ncpus=1):

    # --- Checks
    if len(post_list) != len(cdc.loc_names):
        warnings.warn(f"Hey, list of results (= {len(post_list)}) and of state names (= {len(cdc.loc_names)} do "
                      f"not match.\nI expected that uncalculated states had None in post_list. Results may be"
                      f" inconsistent.)\n")

    # PLOTS PREAMBLE
    # -------------------------------------------------------------------
    # Filter states/locs without forecast (ignored)
    content_zip = [(post, state_name) for post, state_name in zip(post_list, cdc.loc_names) if post is not None]
    num_filt_items = len(content_zip)

    # Prepare the subdivision of plots into pages
    axes_per_page = ncols * nrows  # Maximum number of plots per page
    npages_states = (num_filt_items - 1) // axes_per_page + 1  # Number of pages for the states plots
    naxes_last = (num_filt_items - 1) % axes_per_page + 1  # Number of plots on the last page

    # Group items by pages
    content_chunks = [content_zip[axes_per_page * i: axes_per_page * (i+1)] for i in range(npages_states-1)]
    content_chunks.append(content_zip[-naxes_last:])

    # Generic plot function for a whole single page
    def plot_page_generic(chunk_seq, plot_func, page_width=9., ax_height=3., plot_kwargs=None):
        """
        Signature of the callable:
            plot_func(ax, item_from_chunk, i_ax, **plot_kwargs)
        """
        n_plots = len(chunk_seq)
        plot_kwargs = dict() if plot_kwargs is None else plot_kwargs

        fig, axes = make_axes_seq(n_plots, ncols, total_width=page_width, ax_height=ax_height)
        plot_outs = list()

        for i_ax, (ax, item) in enumerate(zip(axes, chunk_seq)):
            plot_outs.append(plot_func(ax, item, i_ax, **plot_kwargs))

        return fig, axes, plot_outs

    # PLOTS - FORECAST C(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_forecast_ct(ax: plt.Axes, item, i_ax):
        post, state_name = item  # Unpack content tuple

        # Contents
        ct_past: pd.Series = preproc_dict[state_name]
        last_val = ct_past.iloc[-1]
        factual_ct = cdc.xs_state(state_name).loc[post.day_0:post.day_fore]["value"]
        ct_color = "C1" if post.day_pres < cdc.data_time_labels[-1] else "C0"  # C1 if there's a future week available

        # Plot function
        vis.plot_ct_past_and_fore(ax, post.fore_time_labels, post.weekly_quantiles, factual_ct, state_name, i_ax,
                                  post.synth_name if write_synth_names else None, post.num_quantiles, ct_color,
                                  (post.day_pres, last_val))

        print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")

    def task(chunk):
        return plot_page_generic(chunk, plot_table_forecast_ct)

    print("Plotting forecasts...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Separate plot for US
    fig_us = None
    if us is not None:
        fig_us, ax = plt.subplots()

        # Write about an incomplete list of states (which may be intentional)
        if num_filt_items < NUM_STATES:
            ax.text(0.2, 0.5, "WARNING: not all states were used.", transform=ax.transAxes, color="r")

        ct_past_us: pd.Series = preproc_dict["US"]
        _last_val = ct_past_us.iloc[-1]
        _factual_ct = cdc.xs_state("US").loc[us.day_0:us.day_fore]["value"]
        _ct_color = "C1" if us.day_pres < cdc.data_time_labels[-1] else "C0"  # C1 if there's a future week available

        vis.plot_ct_past_and_fore(ax, us.fore_time_labels, us.weekly_quantiles, _factual_ct, "US", None,
                                  None, us.num_quantiles, _ct_color, (us.day_pres, _last_val))
        print(f"   [US] Forecast plotted ()")

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/ct_states.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)

        if fig_us is not None:
            fig_us.tight_layout()
            pdf_pages.savefig(fig_us)

    # PLOTS - INTERPOLATION OF PAST C(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_interpolation(ax: plt.Axes, item, i_ax):
        post, state_name = item  # Unpack content tuple

        ct_past: pd.Series = preproc_dict[state_name]
        ax.plot(*interp.weekly_to_daily_uniform(ct_past.index, ct_past.values), label="Unif. interp.")
        ax.plot(post.t_daily, post.float_data_daily, "-", label="Float interp.")
        ax.plot(post.t_daily, post.ct_past, "o", label="Int interp.", ms=1)
        ax.text(0.05, 0.9, f"{i_ax+1}) {state_name}", transform=ax.transAxes)

        print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")

    def task(chunk):
        return plot_page_generic(chunk, plot_table_interpolation)

    print("\nPlotting interpolations...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/interp_states.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)

    # PLOTS - PAST AND SYNTHESISED R(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_rt(ax: plt.Axes, item, i_ax):
        post, state_name = item  # Unpack content tuple
        # ct_past: pd.Series = preproc_dict[state_name]

        # Estimated R(t) from past
        ax.plot(post.past_daily_tlabels, post.rt_past_median, color="C0")
        ax.plot(post.past_daily_tlabels, post.rt_past_mean, color="C0", ls="--")
        ax.fill_between(post.past_daily_tlabels, post.rt_past_loquant, post.rt_past_hiquant, color="C0", alpha=0.25)

        # Synthesised R(t)
        ax.plot(post.fore_daily_tlabels, post.rt_fore_median, color="blue")
        ax.fill_between(post.fore_daily_tlabels, post.rt_fore_loquant, post.rt_fore_hiquant, color="blue", alpha=0.15)

        # Plot annotations
        ax.text(0.05, 0.9, f"{i_ax+1}) {state_name}", transform=ax.transAxes)
        if write_synth_names:
            ax.text(0.05, 0.8, post.synth_name, transform=ax.transAxes)

        # Axes configuration
        vis.rotate_ax_labels(ax)
        ax.set_ylim(0.0, 3.1)
        print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")

    def task(chunk):
        return plot_page_generic(chunk, plot_table_rt)

    print("\nPlotting R(t)...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/rt_states.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------------------------------
# CODE GRAVEYARD
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------- PLOT PREVIOUS METHOD - SEQUENTIAL PLOTS

    # # Plot preamble
    # # -------------
    # ct_fig, ct_axes = make_axes_seq(num_filt_items, max_cols=6, total_width=15)  # Forecast
    # ip_fig, ip_axes = make_axes_seq(num_filt_items, max_cols=6, total_width=15)  # Interpolation
    #
    # # MAIN PLOT LOOP - CT
    # # --------------
    # # for i_ax, (post, state_name) in enumerate(zip(filt_post_list, filt_state_names)):
    # for i_ax, (post, state_name) in enumerate(content_zip):
    #
    #     # C(t) PAST AND FORECAST TIME SERIES (layered quantiles)
    #     # ------------------------------------------------------
    #     ax: plt.Axes = ct_axes[i_ax]
    #
    #     ct_past: pd.Series = preproc_dict[state_name]
    #     last_val = ct_past.iloc[-1]
    #     factual_ct = cdc.xs_state(state_name).loc[post.day_0:post.day_fore]["value"]
    #     ct_color = "C1" if post.day_pres < cdc.data_time_labels[-1] else "C0"  # C1 if there's a future week available
    #
    #     vis.plot_ct_past_and_fore(ax, post.fore_time_labels, post.weekly_quantiles, factual_ct, state_name,
    #                               post.synth_name if write_synth_names else None, post.num_quantiles, ct_color,
    #                               (post.day_pres, last_val))
    #
    #     # Interpolation
    #     # --------------------------------------
    #     ax = ip_axes[i_ax]
    #
    #     ax.plot(*interp.weekly_to_daily_uniform(ct_past.index, ct_past.values), label="Unif. interp.")
    #     ax.plot(post.t_daily, post.float_data_daily, "-", label="Float interp.")
    #     ax.plot(post.t_daily, post.ct_past, "o", label="Int interp.", ms=1)
    #     ax.text(0.05, 0.9, state_name, transform=ax.transAxes)
    #
    #     ip_fig.tight_layout()
    #
    #     print(f"Plot {state_name} done ({i_ax + 1} of {num_filt_items})")
    #
    # # Separated plots for US series
    # # -----------------------------
    # if us is not None:
    #     fig_us, ax = plt.subplots()
    #
    #     # Write about an incomplete list of states (which may be intentional)
    #     if num_filt_items < NUM_STATES:
    #         ax.text(0.2, 0.5, "WARNING: not all states were used.", transform=ax.transAxes, color="r")
    #
    #     ct_past: pd.Series = preproc_dict["US"]
    #     last_val = ct_past.iloc[-1]
    #     factual_ct = cdc.xs_state("US").loc[us.day_0:us.day_fore]["value"]
    #     ct_color = "C1" if us.day_pres < cdc.data_time_labels[-1] else "C0"  # C1 if there's a future week available
    #
    #     vis.plot_ct_past_and_fore(ax, us.fore_time_labels, us.weekly_quantiles, factual_ct, "US",
    #                               None, us.num_quantiles, ct_color, (us.day_pres, last_val))
    #
    #     fig_us.tight_layout()
    #     fig_us.savefig("tmp_figs/ct_us.pdf")
    #
    # # FINAL REMARKS
    # # -------------
    # ip_axes[0].legend()
    #
    # ct_fig.tight_layout()
    #
    # ct_fig.savefig("tmp_figs/ct_states.png")
    # ct_fig.savefig("tmp_figs/ct_states.pdf")
    #
    # ip_fig.savefig("tmp_figs/interp_states.pdf")


# --------------------

    # # -()- Filter by maximum R value in the whole time series.
    # if r_max is not None:
    #     # Find the first iteration to exceed r_max in any time point, then filters
    #     i_exceed = min(np.searchsorted(filt[:, i_t], r_max) for i_t in range(ndays_past))
    #     filt = filt[:i_exceed]
    #
    #     if i_exceed == 0:
    #       warnings.warn(f"\nHey, all time series had at least one value above r_max = {r_max}. Result will be empty.")
