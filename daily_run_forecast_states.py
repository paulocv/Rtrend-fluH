"""
Load CDC Flu hospitalization data, run forecast for each state, export results and reports.
RUNS FOR DAILY DATA.

loc == state

ndays_past = Used only for the R(t) synthesis: number of days in the past to consider.

pres = present = Day of the last CDC report.

Author: Paulo Cesar Ventura (https://github.com/paulocv, https://paulocv.netlify.app)
On behalf of: CEPH Lab @ Indiana University (https://github.com/CEPH-Lab)
"""
import copy
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from colorama import Fore, Style
# from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages
# from pathos.multiprocessing import ProcessPool
from scipy.interpolate import UnivariateSpline

import rtrend_tools.data_io as dio
import rtrend_tools.interpolate as interp
import rtrend_tools.preprocessing as preproc
import rtrend_tools.synthesis as synth
import rtrend_tools.visualization as vis

from rtrend_tools.cdc_params import CDC_QUANTILES_SEQ, NUM_QUANTILES, NUM_STATES, NUM_WEEKS_FORE, EPWK_REF_START, \
    get_epiweek_unique_id, get_epiweek_label_from_id, get_next_weekday_after, WEEKDAY_FC_DAY, WEEKDAY_TGT
from rtrend_tools.forecast_structs import CDCDataBunch, ForecastExecutionData, CovHospForecastOutput, ForecastPost, \
    USForecastPost
from rtrend_tools.utils import map_parallel_or_sequential
from toolbox.file_tools import make_folder


WEEKLEN = interp.WEEKLEN
MAX_PIPE_STEPS = 100  # Max. number of overall steps in the forecast pipeline to declare failure
PIPE_STAGE = ("preproc", "r_estim", "r_synth", "c_reconst")  # Sequence of stages of the forecast

MIN_CT_SAMPLES = 100  # Have at least this number of forecast c(t) curves for "approval"


def main():
    # PARAMETERS
    # --------------------
    # --- Preprocess params
    ct_cap_fac = 8  # Multiply the historical maximum by this factor to obtain the ct_cap (maximum)

    # ROI DEFINITION
    # -------------------//
    # For this version, ROI is defined in weeks but then converted to days.

    # -()- Current season
    week_last = -2
    week_roi_start = week_last - 10  # pd.Timestamp("2022-08-29")

    pre_roi_len = 15  # Length of the preprocessing ROI in weeks. Reduced to avoid off-season estimation.

    # -------------------//

    # --- Forecast params
    # General
    nweeks_fore = NUM_WEEKS_FORE   # Number of weeks ahead to forecast. Add one to account for daily data delay.
    export_range = np.arange(nweeks_fore)  # Weeks (0 = day_pres) to export to file.
    nsamples_us = 1000  # Number of C(t) samples to take from each state for building the US forecast.

    tg_params = dict(  # Mean 3.0 IQR 1.4 to 5.1
        shape=10.,
        rate=10./3.,
        tmax=30,  # int
    )

    # --- Preprocessing: denoise and noise analysis
    preproc_params = dict(
        method="lowpass",  # polyfit, lowpass, rollav, none
        noise_type="none",  # normal, weekday_normal, none
        use_denoised=True,  # Whether to use the denoised data for the forecast
        use_pre_roi=True,  # Use the preprocessing ROI for the noise estimation.

        # ---- Denoising
        # Polyfit params
        poly_degree=3,

        # Lowpass params
        cutoff=0.15,
        filt_order=2,

        # Rolling average params
        rollav_window=0,  # 4 * WEEKLEN,  # Number of weeks to take rolling average of. Use 0 or None to deactivate.

        # ----- Noise synthesis (for reconstruction phase)
        noise_coef=0.0,  # Multiply estimated noise by this factor for the synthesis
        noise_seed=10,
    )

    # MCMC calculation of R(t)
    mcmc_params = dict(
        in_prefix="mcmc_inputs/states_inchosp_in/",
        out_prefix="mcmc_outputs/states_inchosp_out/",
        log_prefix="mcmc_logs/states_inchosp_log/",
        nsim=20000,  # Number of simulations (including 10,000 burn-in period)
        seed=10,
        sigma=0.05,
    )

    # R(t) Synthesis
    synth_params = dict(
        # Ramp params
        q_low=0.30,      # Static ramp: low quantile
        q_hig=0.70,      # Static ramp: high quantile
        ndays_past=14,   # Number of days (backwards) to consider in synthesis.
        r_max=1.8,       # Clamp up to these R values (for the average)
        k_start=0.95,    # Ramp method: starting coefficient
        k_end=0.85,      # Ramp method: ending coefficient
        i_saturate=-1,   # -2 * WEEKLEN,  # Ramp method: saturate ramp at this number of days. Use -1 to deactivate.

        # # Dynamic ramp
        # r1_start=1.0,
        # r2_start=1.3,
        # r1_end=0.8,   # R_new Value to which R = 1 is converted
        # r2_end=1.0,    # R_new Value to which R = 2 is converted

        # Dynamic ramp  # FLAT (still dynamic though)!!
        r1_start=0.95,
        r2_start=1.1,
        r1_end=0.95,  # R_new Value to which R = 1 is converted
        r2_end=1.1,  # R_new Value to which R = 2 is converted

        # Rtop ramp extra params
        rmean_top=1.40,  # (Obs: Starting value, but it is decreased at each pass, including the first)
        max_width=0.15,

        # Random normal params
        seed=10,
        center=1.015,
        sigma=0.010,
        num_samples=500,  # Use this instead of the MCMC ensemble size.
    )

    # C(t) reconstruction
    recons_params = dict(seed=2)

    # --- Misc
    post_weekly_coefs = np.ones(nweeks_fore, dtype=float)  # Postprocess weekly multiplicative coefficient
    # post_weekly_coefs[0] = 0.6  # Thanksgiving underreporting effect

    # --- Program parameters
    cdc_data_fname = "hosp_data/daily_truth-Incident Hospitalizations.csv"
    # cdc_data_fname = "hosp_data/past_daily_BKP/daily_truth-Incident Hospitalizations_2023-02-20.csv"
    # print("##############using old file!!!! #############!!!!!!!!")
    cdc_out_fname = "forecast_out/daily_latest.csv"
    do_export = True
    do_show_plots = False
    write_synth_names = True
    ncpus = 6  # If 1, runs sequential. Otherwise, calls pathos.

    # --- DICTIONARY OF EXCEPTIONS - For states that require different parameters than the default.

    exception_dict = dict()

    # SELECT ONLY THESE STATES
    # use_states = ["California", "Texas", "Michigan", "Indiana", "Wyoming", "New York", "Colorado", "Maryland"]  #
    # use_states = ["California", "Texas", "Indiana", "Wyoming"]  #
    use_states = None

    # PRELIMINARIES
    # ------------------------------------------------------------------------------------------------------------------
    if not do_export:
        print(10 * "#" + " EXPORT SWITCH IS OFF " + 10 * "#")

    print("Importing and preprocessing CDC data...")
    cdc: CDCDataBunch = dio.load_cdc_truth_data(cdc_data_fname)
    roi_dict, pre_roi_dict, ct_cap_dict = preprocess_cdc_data(
        cdc, week_last, week_roi_start, pre_roi_len, ct_cap_fac=ct_cap_fac)
    print("\n#################\n")

    # FORECAST EXECUTION
    # ------------------------------------------------------------------------------------------------------------------

    def task(inputs):
        i_s, state_name = inputs

        # Select states for testing purposes (#filter)
        if use_states and (state_name not in use_states):  # Inclusion-only list
            return None
        if state_name in ["US", "Virgin Islands"]:  # Exclusion list
            print(f"############# EXCLUDED {state_name}")
            return None

        print(f"-------------- ({i_s+1} of {cdc.num_locs}) Starting {state_name}...")

        # ----------------- FORECAST
        state_series = roi_dict[state_name]
        preproc_series = pre_roi_dict[state_name]
        except_params = exception_dict.get(state_name, None)

        # FORECAST COMMAND
        fc: CovHospForecastOutput = \
            forecast_state(state_name, state_series, preproc_series, nweeks_fore, except_params, tg_params,
                           preproc_params, mcmc_params, synth_params, recons_params, ct_cap=ct_cap_dict[state_name])

        if fc is None:
            print(f"   [{state_name}] Returned empty, skipped.")
            return None
        print(f"   [{state_name}] Forecast done with {fc.num_ct_samples} c(t) samples.")

        # POSTPROCESS COMMAND
        return postprocess_state_forecast(fc, state_name, i_s, cdc, nweeks_fore, nsamples_us,
                                          post_coefs=post_weekly_coefs)

    # RUN LINE - PARALLEL OR SEQUENCE
    result_list = map_parallel_or_sequential(task, enumerate(cdc.loc_names), ncpus=ncpus)

    # --- Construct the total US time series
    print("-------------- Starting for US ")
    us_post = postprocess_us(result_list, cdc, nweeks_fore, nsamples_us, post_coefs=post_weekly_coefs)
    print(f"\t[US] Done ({nsamples_us} c(t) samples).")
    #
    # VISUALIZATION and EXPORT
    # --------------------------
    if do_export:
        print(f"__________ \n Exporting to '{cdc_out_fname}'...")
        # dio.export_forecast_cov_hosp(cdc_out_fname, result_list, us_post, cdc, nweeks_fore, add_days=0,
        #                              export_range=export_range)
        dio.export_forecast_flu(cdc_out_fname, result_list, us_post, cdc, nweeks_fore, export_range=export_range)

    print("\n___________________\nNOW PLOTTING\n")
    make_plot_tables(result_list, cdc, roi_dict, nweeks_fore, us_post, write_synth_names,
                     ncpus=1)
    print("Plots done!")
    if do_show_plots:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------------------------------------------------

def preprocess_cdc_data(cdc: CDCDataBunch, week_last, week_roi_start, pre_roi_len, ct_cap_fac=3):
    """
    Splits dataset by state, crop ROI.
    """

    # Argument handling
    # -----------------
    if isinstance(week_last, int):
        idx_pres = 7 * week_last + (week_last < 0) * 6  # Convert to days for both positive and negative indexes
        week_last = cdc.data_time_labels[idx_pres]

    if isinstance(week_roi_start, int):
        idx_rs = 7 * week_roi_start + (week_roi_start < 0) * 6  # Convert to days for both positive and negative indexes
        week_roi_start = cdc.data_time_labels[idx_rs + 1]  # Add one to get a multiple of 7 days in ROI.

    if isinstance(pre_roi_len, int):
        pre_roi_len = pd.Timedelta(pre_roi_len, "w")  # Convert int to the number of weeks

    if week_roi_start >= week_last:
        raise ValueError(f"Hey, ROI start {week_roi_start.date()} must be prior to present week {week_last.date()}.")

    # Rounds the ROI borders to the EpiWeek they belong to
    week_last = get_epiweek_label_from_id(get_epiweek_unique_id(week_last))
    week_roi_start = get_epiweek_label_from_id(get_epiweek_unique_id(week_roi_start))

    # Check if rounded dates are present in data:
    if week_last not in cdc.data_time_labels:
        raise ValueError(Fore.RED +
                         f"Hey, the epiweek given by the (rounded) label {week_last.date()} is incomplete or "
                         f"missing in the truth file. It may be that the truth file contains data for the current "
                         f"week; in this case, just use week_last = -2." + Style.RESET_ALL)

    cdc.day_roi_start = week_roi_start
    cdc.day_pres = week_last + pd.Timedelta(1, "D")  # Adds one to make "pres" the first day to forecast
    print(f"Past ROI (rounded to nearest epiweek): from {week_roi_start.date()} to {week_last.date()}")
    print(f"Day present = {cdc.day_pres.date()}")

    # Preprocessing
    # -------------
    roi_dict = dict()
    pre_roi_dict = dict()
    ct_cap_dict = dict()

    for state_name in cdc.loc_names:
        state_df = cdc.df.xs(cdc.to_loc_id[state_name], level="location")

        # Set cap based on maximum historical data (whole tseries, not just ROI).
        ct_cap_dict[state_name] = ct_cap_fac * state_df["value"].max()

        # Crop regular ROI
        roi_df = state_df.loc[week_roi_start:week_last]["value"]

        # Crop preprocessing ROI
        pre_roi_df = state_df.loc[week_last - pre_roi_len + pd.Timedelta(1, "d"):week_last]["value"]

        # Check for problems
        if roi_df.shape[0] == 0 or pre_roi_df.shape[0] == 0:
            sys.stderr.write(f"\nHEY, empty dataset for state {state_name} in ROI.\n")

        roi_dict[state_name] = roi_df
        pre_roi_dict[state_name] = pre_roi_df

    return roi_dict, pre_roi_dict, ct_cap_dict


# ----------------------------------------------------------------------------------------------------------------------
# FORECAST PIPELINE FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------


def callback_preprocessing(exd: ForecastExecutionData, fc: CovHospForecastOutput):

    # Briefing
    # ---------------------------

    if exd.notes == "NONE":
        exd.method = "STD"
        denoise = preproc.DENOISE_CALLABLE[exd.preproc_params["method"]]
        fc.noise_obj = preproc.NOISE_CLASS[exd.preproc_params["noise_type"]](**exd.preproc_params)
    else:
        sys.stderr.write(f"\nHey, abnormal condition found in preproc briefing. (exd.notes = {exd.notes})\n")
        exd.stage = "ERROR"
        return

    exd.log_current_pipe_step()

    # Execution
    # ---------------------------
    # Preamble - store time-related quantities
    fc.day_0 = exd.state_series.index[0]  # First accounted day (7 DAYS BEFORE 1st REPORT)
    fc.day_pres = exd.state_series.index[-1] + pd.Timedelta(1, "D")  # One past last day. First forecasted day.
    fc.ndays_roi = (fc.day_pres - fc.day_0).days
    fc.ndays_fore = WEEKLEN * exd.nweeks_fore
    fc.fore_daily_tlabels = pd.date_range(fc.day_pres, periods=WEEKLEN * exd.nweeks_fore, freq="D")
    fc.t_daily = np.arange(0, exd.state_series.shape[0])  # Int array of days for
    fc.past_daily_tlabels = fc.day_0 + pd.TimedeltaIndex(fc.t_daily, "D")

    if exd.preproc_params["method"] != "none":  # Apply denoising

        # Noise extraction and estimation
        roi = preproc.choose_roi_for_denoise(exd)
        fc.float_data_daily = denoise(exd, fc, **exd.preproc_params)
        fc.noise_obj.fit(roi, fc.float_data_daily)

        # Convert to numpy array if needed
        if isinstance(fc.float_data_daily, pd.Series):
            fc.float_data_daily = fc.float_data_daily.to_numpy()

        # Return to normal ROI, if pre-ROI was used
        if exd.preproc_params.get("use_pre_roi", "False"):
            fc.float_data_daily = fc.float_data_daily[-fc.ndays_roi:]

        # Select between raw and denoised data for the forecast pipeline
        if exd.preproc_params["use_denoised"]:
            fc.ct_past = fc.float_data_daily.round().astype(int)  # Round processed values
        else:
            fc.ct_past = exd.state_series.to_numpy()

    else:
        fc.float_data_daily = exd.state_series.to_numpy()
        fc.ct_past = exd.state_series.to_numpy()

    # Post operations
    # fc.ct_past = fc.float_data_daily.round().astype(int)  # Round smoothened values

    # Debriefing
    # ---------------------------
    if np.min(fc.ct_past) < 0:  # Negative data from denoising procedure
        exd.stage, exd.notes = "ERROR", "negative"
        sys.stderr.write(f"[{exd.state_name}] Negative value produced on denoising stage. Try another method.\n")

    else:
        exd.stage, exd.notes = "r_estim", "NONE"  # Approve with default method


def callback_r_estimation(exd: ForecastExecutionData, fc: CovHospForecastOutput):

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
    for key in ["in_prefix", "out_prefix", "log_prefix"]:
        os.makedirs(os.path.dirname(exd.mcmc_params[key]), exist_ok=True)

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


def callback_r_synthesis(exd: ForecastExecutionData, fc: CovHospForecastOutput):
    # Briefing
    # ---------------------------

    # PALEATIVE: add cases to the latest days, preventing absolute zeros.
    if fc.ct_past[-4:].max() <= 1:
        fc.ct_past[-2] += 1

    # Cumulative hospitalizations inside ROI
    sum_roi = exd.state_series.sum()

    # --- Method decision regarding the cumulative hospitalizations in the ROI
    # OBS: TRY TO KEEP A SINGLE IF/ELIF CHAIN for better clarity of the choice!

    # TODO: NORMALS FOR EVERYONE!!!!!! 2023-03-21
    exd.notes = "use_rnd_normal"

    # TODO Puerto Rico has a surge - 2023-04-03
    if exd.state_name in ["Puerto Rico"]:
        exd.notes = "NONE"


    if sum_roi <= 50 or exd.notes == "use_rnd_normal":  # Too few cases for ramping, or rnd_normal previously asked.

        if exd.state_name == "Virgin Islands":  # Chose params for more conservative estimates.
            exd.synth_params["center"] = 0.8
            exd.synth_params["sigma"] = 0.02

        # Random normal
        synth_method = synth.random_normal_synth
        fc.synth_name = exd.method = f"rnd_normal_{exd.synth_params['center']:0.2f}"

    elif exd.notes == "NONE":  # DEFAULT CONDITION, does not fall into any of the previous
        # # -()- Static ramp
        # synth_method = synth.sorted_bent_mean_ensemble
        # fc.synth_name = exd.method = "static_ramp"

        # -()- Dynamic ramp
        synth_method = synth.sorted_dynamic_ramp
        fc.synth_name = exd.method = "dynamic_ramp" + (exd.synth_params["i_saturate"] != -1) * "_sat"  # saturation

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

    if fc.rt_fore2d.shape[0] == 0:  # Problem: no R value satisfied the filter criterion
        # Action: repeat the synth right away
        fc.synth_name = exd.method = "rnd_normal_after"
        exd.log_current_pipe_step()

        fc.rt_fore2d = synth.random_normal_synth(fc.rtm, fc.ct_past, fc.ndays_fore, **exd.synth_params)

    # Debriefing
    # ---------------------------

    # if exd.state_name in special_states_01:
    #     exd.stage, exd.notes = "c_reconst", "no_filter"  # Approve, but signal that state is mustn't be ct-filtered
    # else:
    exd.stage, exd.notes = "c_reconst", "NONE"  # Approve with default method


def callback_c_reconstruction(exd: ForecastExecutionData, fc: CovHospForecastOutput):

    # Briefing
    # ----------------------------
    can_apply_ct_cap = (exd.notes != "no_filter")
    exd.method = "STD"

    exd.log_current_pipe_step()

    # Execution
    # ----------------------------
    fc.ct_fore2d = synth.reconstruct_ct(fc.ct_past, fc.rt_fore2d, fc.tg_dist,
                                        tg_max=exd.tg_params["tmax"], **exd.recons_params)

    # Incorporate noise uncertainty
    fc.ct_fore2d = fc.noise_obj.generate(fc.ct_fore2d, fc.fore_daily_tlabels)

    # # Aggregate from daily to weekly
    # fc.ct_fore2d_weekly = interp.daily_to_weekly(fc.ct_fore2d, dtype=float)  # Aggregate from day 0
    fc.ct_fore2d_weekly, fc.fore_weekly_time_labels = \
        interp.daily_to_epiweeks(fc.ct_fore2d, fc.fore_daily_tlabels, dtype=float)

    # --- Filter series that exceeded ct_cap
    if exd.ct_cap is not None and can_apply_ct_cap:
        mask = np.max(fc.ct_fore2d, axis=1) <= exd.ct_cap  # Mask: True for reasonable entries
        num_remaining = mask.sum()

        if num_remaining < MIN_CT_SAMPLES:  # Too few cases after filtering
            exd.stage, exd.notes = "r_synth", "use_static_ramp_rtop"  # Request another synthesis
            return

        # Otherwise, applies the filter
        fc.ct_fore2d = fc.ct_fore2d[mask, :]

    # Attribute number of samples
    fc.num_ct_samples = fc.ct_fore2d.shape[0]

    # If this point was reached, results are satisfactory
    exd.stage = "DONE"  # Approve the forecast


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FORECAST ROUTINE
# ----------------------------------------------------------------------------------------------------------------------


def forecast_state(state_name: str, state_series: pd.Series, preproc_series: pd.Series, nweeks_fore,
                   except_params, tg_params, preproc_params, mcmc_params, synth_params, recons_params, ct_cap=None):
    """
    """
    # PREAMBLE
    # ------------------------------------------------------------------------------------------------------------------
    synth_params = copy.deepcopy(synth_params)  # Deepcopy to protect original values.

    exd = ForecastExecutionData()
    fc = CovHospForecastOutput()

    # Populate execution data (exd). Params are deep-copied to protect original master dictionary.
    exd.state_name = state_name
    exd.state_series = state_series
    exd.preproc_series = preproc_series
    exd.nweeks_fore = nweeks_fore
    exd.except_params = except_params
    exd.tg_params = copy.deepcopy(tg_params)
    exd.preproc_params = copy.deepcopy(preproc_params)
    exd.mcmc_params = copy.deepcopy(mcmc_params)
    exd.synth_params = copy.deepcopy(synth_params)
    exd.recons_params = copy.deepcopy(recons_params)
    exd.ct_cap = ct_cap

    # PIPELINE LOOP OF STAGES
    # ------------------------------------------------------------------------------------------------------------------

    # Initialize the
    exd.stage, exd.method, exd.notes = PIPE_STAGE[0], "STD", "NONE"

    for i_pipe_step in range(MAX_PIPE_STEPS):

        if exd.stage == "preproc":  # STAGE 0: C(t) weekly to daily interpolation
            callback_preprocessing(exd, fc)

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

    # print(f"[{state_name}] LOG")
    # print(exd.stage_log)

    return fc


# ----------------------------------------------------------------------------------------------------------------------
# POSTPROCESSING
# ----------------------------------------------------------------------------------------------------------------------


def postprocess_state_forecast(fc: CovHospForecastOutput, state_name, i_s, cdc: CDCDataBunch, nweeks_fore,
                               nsamples_us, post_coefs: np.ndarray = None):
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
    post.weekly_quantiles = synth.calc_tseries_ensemble_quantiles(fc.ct_fore2d_weekly)
    if post_coefs is not None:  # Apply post-coefficient for each week, if informed
        post.weekly_quantiles *= post_coefs
    post.num_ct_samples = fc.ct_fore2d_weekly.shape[0]

    # Interpolation
    # post.daily_spline = fc.daily_spline
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
    post.fore_daily_tlabels = fc.fore_daily_tlabels
    post.fore_time_labels = \
        fc.fore_weekly_time_labels
    # pd.DatetimeIndex((fc.day_pres + pd.Timedelta((i + 1) * WEEKLEN, "D") for i in range(nweeks_fore)))

    # Contribution of the state to the US time series.
    post.samples_to_us = synth.sample_series_with_replacement(fc.ct_fore2d_weekly, nsamples_us, seed=10 + i_s)

    return post


def postprocess_us(fc_list: list[ForecastPost], cdc: CDCDataBunch, nweeks_fore, nsamples_us, post_coefs=None):
    post = USForecastPost()

    # Cut ROI and determine key days based on the overall CDC data, so it's independent of any state.
    post.day_0 = cdc.day_roi_start
    post.day_pres = cdc.day_pres
    post.fore_time_labels = \
        get_next_weekday_after(post.day_pres, WEEKDAY_TGT) + pd.TimedeltaIndex(np.arange(nweeks_fore), unit="w")
    # pd.date_range(get_next_weekday_after(post.day_pres, WEEKDAY_TGT), periods=nweeks_fore, freq="W") # DON't USE!
    # pd.DatetimeIndex((post.day_pres + pd.Timedelta((i + 1) * WEEKLEN, "D") for i in range(nweeks_fore)))
    post.day_fore = post.fore_time_labels[-1]  # post.day_pres + pd.Timedelta(nweeks_fore, "W")

    # Perform the sum of samples from each state

    # # -()- AVG-SORT Sort the ensembles of each state to better estimate the quantiles.
    # sorted_samples = synth.sort_ensembles_by_average([post_.samples_to_us for post_ in fc_list if post_ is not None])

    # -()- T-SORT Sorts the ensemble independently at each time point.
    sorted_samples = synth.sort_ensembles_at_each_point([post_.samples_to_us for post_ in fc_list if post_ is not None])

    # # # -()- ORIGINAL
    # sorted_samples = [post_.samples_to_us for post_ in fc_list if post_ is not None]

    ct_fore2d_weekly = sum(sorted_samples)

    # Calculate quantiles from the sum
    post.weekly_quantiles = synth.calc_tseries_ensemble_quantiles(ct_fore2d_weekly)
    if post_coefs is not None:  # Apply post-coefficient for each week, if informed
        post.weekly_quantiles *= post_coefs
    post.num_quantiles = post.weekly_quantiles.shape[0]

    return post


def make_plot_tables(post_list, cdc: CDCDataBunch, preproc_dict, nweeks_fore, us,
                     write_synth_names=True, ncols=3, nrows=3, ncpus=1):

    os.makedirs("tmp_figs", exist_ok=True)

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

    # PLOTS - FORECAST C(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_forecast_ct(ax: plt.Axes, item, i_ax):
        post, state_name = item  # Unpack content tuple
        post: ForecastPost

        # Contents
        factual_ct = cdc.xs_state(state_name).loc[post.day_0:post.day_fore]["value"]
        weekly = factual_ct.groupby(get_epiweek_unique_id).sum()
        factual_ct = pd.Series(
            weekly.values, index=weekly.index.map(get_epiweek_label_from_id))

        last_val = factual_ct.loc[post.day_pres - pd.Timedelta((post.day_pres.weekday() - WEEKDAY_TGT) % WEEKLEN, "d")]
        # last_val = state_series.iloc[-1] * 7  # TODO TEMPORARY, CHANGE TO ACTUAL AGG VALUE
        ct_color = "C1" if post.day_pres < cdc.data_time_labels[-1] else "C0"  # C1 if there's a future week available

        # Plot function
        vis.plot_ct_past_and_fore(ax, post.fore_time_labels, post.weekly_quantiles, factual_ct, post.quantile_seq,
                                  state_name, i_ax, post.synth_name if write_synth_names else None, post.num_quantiles,
                                  ct_color, (post.day_pres, last_val), plot_trend=False, bkg_color="#E8F8FF")
        # EXTRA: plot multiplied preprocessed series
        # ax.plot(post.past_daily_tlabels, WEEKLEN * post.float_data_daily, "--")
        ax.plot(post.past_daily_tlabels, WEEKLEN * post.ct_past, "--")

        ax.set_xlim(post.day_0 + pd.Timedelta("1w"), post.day_fore + pd.Timedelta("1d"))

        print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")

    def task(chunk):
        return vis.plot_page_generic(chunk, plot_table_forecast_ct, ncols=ncols)

    print("Plotting forecasts...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Separate plot for US
    fig_us = None
    if us is not None:
        fig_us, ax = plt.subplots()

        # Write about an incomplete list of states (which may be intentional)
        if num_filt_items < NUM_STATES:
            ax.text(0.05, 0.1, "WARNING: not all states were used.", transform=ax.transAxes, color="r")

        ct_past_us: pd.Series = preproc_dict["US"]
        # _last_val = ct_past_us.iloc[-1]
        # _factual_ct = cdc.xs_state("US").loc[us.day_0:us.day_fore]["value"]
        _factual_ct = cdc.xs_state("US").loc[us.day_0:us.day_fore]["value"]
        weekly = _factual_ct.groupby(get_epiweek_unique_id).sum()
        _factual_ct = pd.Series(
            weekly.values, index=weekly.index.map(get_epiweek_label_from_id))
        _last_val = _factual_ct.loc[us.day_pres - pd.Timedelta((us.day_pres.weekday() - WEEKDAY_TGT) % WEEKLEN, "d")]
        _ct_color = "C1" if us.day_pres < cdc.data_time_labels[-1] else "C0"  # C1 if there's a future week available

        vis.plot_ct_past_and_fore(ax, us.fore_time_labels, us.weekly_quantiles, _factual_ct, CDC_QUANTILES_SEQ,
                                  "US", None, None, us.num_quantiles, _ct_color, (us.day_pres - pd.Timedelta("1d"),
                                                                                  _last_val),
                                  plot_trend=False, bkg_color="#E8F8FF")
        print(f"   [US] Forecast plotted ()")

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/daily_ct_states.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)

        if fig_us is not None:
            fig_us.tight_layout()
            pdf_pages.savefig(fig_us)

    # PLOTS - PREPROCESSING OF PAST C(t)
    # ------------------------------------------------------------------------------------------------------------------

    def plot_table_preproc(ax: plt.Axes, item, i_ax):
        post, state_name = item  # Unpack content tuple

        ct_past: pd.Series = preproc_dict[state_name]
        ax.plot(post.t_daily, ct_past, "--", label="Truth data")
        ax.plot(post.t_daily, post.float_data_daily, "-", label="Float interp.")
        ax.plot(post.t_daily, post.ct_past, "o", label="Int interp.", ms=2)
        ax.text(0.05, 0.9, f"{i_ax+1}) {state_name}", transform=ax.transAxes)

        ax.set_facecolor("#E8F8FF")
        print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")

    def task(chunk):
        return vis.plot_page_generic(chunk, plot_table_preproc, ncols=ncols)

    print("\nPlotting preprocessing...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/daily_preproc_state.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)

    # #
    # # PLOTS - INTERPOLATION OF PAST C(t)
    # # ------------------------------------------------------------------------------------------------------------------
    #
    # def plot_table_interpolation(ax: plt.Axes, item, i_ax):
    #     post, state_name = item  # Unpack content tuple
    #
    #     ct_past: pd.Series = preproc_dict[state_name]
    #     ax.plot(*interp.weekly_to_daily_uniform(ct_past.index, ct_past.values), label="Unif. interp.")
    #     ax.plot(post.t_daily, post.float_data_daily, "-", label="Float interp.")
    #     ax.plot(post.t_daily, post.ct_past, "o", label="Int interp.", ms=1)
    #     ax.text(0.05, 0.9, f"{i_ax+1}) {state_name}", transform=ax.transAxes)
    #
    #     ax.set_facecolor("#E8F8FF")
    #
    #     print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")
    #
    # def task(chunk):
    #     return vis.plot_page_generic(chunk, plot_table_interpolation, ncols=ncols)
    #
    # print("\nPlotting interpolations...")
    # results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    # print()
    #
    # # --- Plot postprocess and export
    # with PdfPages("tmp_figs/daily_interp_states.pdf") as pdf_pages:
    #     for fig, axes, plot_outs in results:
    #         fig.tight_layout()
    #         pdf_pages.savefig(fig)
    #
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
        ax.plot([post.past_daily_tlabels[0], post.fore_daily_tlabels[-1]], [1, 1], "k--", alpha=0.25)

        # Axes configuration
        vis.rotate_ax_labels(ax)
        ax.set_ylim(0.0, 3.1)
        ax.set_facecolor("#E8F8FF")
        print(f"  [{state_name}] ({i_ax+1} of {num_filt_items})  |", end="")

    def task(chunk):
        return vis.plot_page_generic(chunk, plot_table_rt, ncols=ncols)

    print("\nPlotting R(t)...")
    results = map_parallel_or_sequential(task, content_chunks, ncpus=ncpus)  # Returns: [(fig, axes, plot_outs)]
    print()

    # --- Plot postprocess and export
    with PdfPages("tmp_figs/daily_rt_states.pdf") as pdf_pages:
        for fig, axes, plot_outs in results:
            fig.tight_layout()
            pdf_pages.savefig(fig)


if __name__ == "__main__":
    main()
