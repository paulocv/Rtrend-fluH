"""
Performs a training loop for the parameter selection maching of
Rtrend forecasts.

This implements the "Par. select. with preprocessing" pipeline, which
also performs preprocessing within each forecasting loop.

"""

import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rtrend_forecast.reporting import (
    ExecTimeTracker,
    get_rtrend_logger,
    get_main_exectime_tracker,
)
from rtrend_forecast.structs import RtData, get_tg_object
from rtrend_forecast.preprocessing import apply_lowpass_filter_pdseries
from rtrend_forecast.utils import map_parallel_or_sequential
from rtrend_interface.forecast_operators import WEEKLEN, ParSelTrainOperator
from rtrend_interface.parsel_utils import load_population_data, make_date_state_index, load_truth_cdc_simple, make_mcmc_prefixes, \
    make_file_friendly_name, make_rt_out_fname, make_filtered_fname

_CALL_TIME = datetime.datetime.now()


# DEFAULT PARAMETERS
# -------------
DEFAULT_PARAMS = dict(  # Parameters that go in the main Params class
    # Field names in the preprocessed data file
    filt_col_name="filtered",
    filtround_col_name="filtered_round",

    export=True,

    # Parallel (DON'T USE BOTH - Subprocesses can't have children)
    ncpus_dates=4,
    ncpus_states=1,
)

DEFAULT_PARAMS_GENERAL = dict(  # Parameters from the `general` dict
    pop_data_path="population_data/locations.csv",
    forecast_days_ahead=2,  # Forecast evaluation ahead of the day_pres
)

DEFAULT_PARAMS_RT_ESTIMATION = dict(  # Parameters from the `general` dict
    use_tmp=True
)

# Helpful objects from the Rtrend library
# LOCAL_XTT = ExecTimeTracker(category=__name__)
GLOBAL_XTT = get_main_exectime_tracker()
_LOGGER = get_rtrend_logger().getChild(__name__)


#


# -------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------

def main():

    args: CLArgs = parse_args()
    params: Params = import_params(args)
    data = Data()

    import_simulation_data(params, data)

    # run_training(params, data)  # THIS WILL RUN THE SEASON MULTIPLE TIMES
    run_forecasts_once(params, data)  # RUNS once FOR ONE "SEASON", ALL LOCATIONS

    # Manually reporting exec time
    print("-------------------\nEXECUTION TIMES")
    # for key, val in LOCAL_XXT.get_dict().items():
    for key, val in GLOBAL_XTT.get_dict().items():
        print(key.ljust(25, ".") + f" {val:0.4f}s")
#


#

# -------------------------------------------------------------------
# PROGRAM STRUCTURES
# -------------------------------------------------------------------


class CLArgs:
    """Command line arguments."""
    input_file: Path
    # output_dir: Path  # Not required here
    export: bool


class Params:
    """All parameters. Include hardcoded script params, those read
    from inputs, and overriden by command line flags.
    """
    input_dict: dict  # Contains all inputs read from the file.
    call_time: datetime.datetime

    general: dict
    preprocessing: dict
    rt_estimation: dict
    rt_synthesis: dict
    inc_reconstruction: dict
    postprocessing: dict

    # Program parameters
    filt_col_name: str
    filtround_col_name: str
    ncpus_dates: int
    ncpus_states: int
    export: bool


class Data:
    """Input, intermediate and output data.
    Input data is reserved for large structures. Smaller ones that
    can be manually set should be reserved as parameters.
    """
    pop_df: pd.DataFrame  # Data frame with population demographic data
    all_state_names: list  # Names of all available locations to forecast
    state_name_to_id: dict  # From `location_name` to `location`

    day_pres_seq: pd.DatetimeIndex  # Sequence of present days to use
    use_state_names: None  # Names of locations that shall be used.
    date_state_idx: pd.MultiIndex  # Combined index for dates and states
    truth_df_sr: pd.Series  # sr.loc[day_pres] = truth_df
    # summary_metrics_df: pd.DataFrame  # Overall summary metrics


#


#


# -------------------------------------------------------------------
# PROGRAM PROCEDURES
# -------------------------------------------------------------------

def parse_args():
    """Interprets and stores the command line arguments."""

    parser = argparse.ArgumentParser(
        # usage="[[COMMAND LINE SIGNATURE (autogenerated)[]",
        description="Promotes the preprocessing and R(t) estimation "
                    "for the Rtrend FluH model. Stores results into "
                    "buffer files that can be used in the future.",
        # epilog="[[TEXT DISPLAYED AFTER ARGUMENTS DESCRIPTION]]",
    )

    # --- Positional paths
    parser.add_argument(
        "input_file", type=Path,
        help="Path to the file with input parameters."
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Path to the output directory, where all "
             "output files are stored."
    )

    #
    # --- Optional flags - SET TO NONE to have no effect
    parser.add_argument(
        "--export", action=argparse.BooleanOptionalAction,
        help="Whether the outputs (results of the training procedure) "
             "should be exported to files.",
        default=None,
    )

    return parser.parse_args()  # When all arguments are defined here
    # return parser.parse_known_args()  # If there are extra arguments


def import_params(args: CLArgs):
    """Imports the main YAML parameter file."""

    # --- Read file
    with open(args.input_file, "r") as fp:
        input_dict = yaml.load(fp, yaml.Loader)

    # --- Populate the Params in priority order
    # Script Default < Input File < Command Line Arguments
    params = Params()
    params.call_time = _CALL_TIME  # Datetime of program call

    params.__dict__.update(DEFAULT_PARAMS)
    params.__dict__.update(input_dict)
    params.__dict__.update(
        {key: val for key, val in args.__dict__.items()
         if val is not None})

    # params.input_dict = input_dict  # Use this to keep the input_dict

    # --- Inner parameters overriding
    for key, val in DEFAULT_PARAMS_GENERAL.items():
        if key not in params.general:
            params.general[key] = val

    for key, val in DEFAULT_PARAMS_RT_ESTIMATION.items():
        if key not in params.rt_estimation:
            params.rt_estimation[key] = val

    # You can rename parameters here.

    if not params.export:
        _LOGGER.warn(
            " --- EXPORT SWITCH IS OFF --- No outputs will be produced.")

    return params


def import_simulation_data(params: Params, data: Data):
    # --------------------
    # IMPORT DEMOGRAPHIC DATA
    # --------------------
    try:
        pop_fname = params.general["pop_data_path"]
    except KeyError:
        raise KeyError(
            "Hey, parameter 'general.pop_data_path' is required "
            "(must inform the path to the population file).")

    data.__dict__.update(load_population_data(pop_fname))

    # --------------------
    # BUILD ITERABLES
    # --------------------
    # Sequence of `day_pres` and `state_names` values
    # As well as a product of the two (`date_state_idx`)
    data.day_pres_seq, data.use_state_names, data.date_state_idx = (
        make_date_state_index(params.general, data.all_state_names))

    # ------------------------------------
    # IMPORT ALL REQUIRED TRUTH DATA FILES
    # ------------------------------------
    data.truth_df_sr = pd.Series(
        [load_truth_for_day_pres(params, day_pres)
         for day_pres in data.day_pres_seq],
        index=data.day_pres_seq
    )


#


#


#

#


# -------------------------------------------------------------------
# AUXILIARY PROCEDURES
# -------------------------------------------------------------------

#


def load_truth_for_day_pres(params: Params, day_pres):
    """Loads a truth file of a given date, using a given directory
    and format string for the file name.
    """
    # Extracts the date of the file for that given day_pres
    date_str = (
        (day_pres + pd.Timedelta(params.general["forecast_days_ahead"], "d"))
        .date()
        .isoformat()
    )

    # Create the file name
    truth_fname = os.path.join(
        params.general["hosp_data_dir"],
        params.general["hosp_data_fmt"].format(date=date_str))

    # Load the data as a dataframe
    truth_df = load_truth_cdc_simple(truth_fname)  # About 5MB.
    _LOGGER.debug(f"Truth file {truth_fname} loaded for day_pres = {day_pres.date().isoformat()}.")

    return truth_df


# -------------------------------------------------------------------
# MAIN TRAINING ROUTINES
# -------------------------------------------------------------------


#

# @LOCAL_XTT.track()
@GLOBAL_XTT.track()
def run_forecasts_once(params: Params, data: Data, WHAT_ELSE=None):
    """Runs the forecast with the
    given parameters, for all locations and selected dates.
    """

    # --------------------
    # Prepare the loop
    # --------------------

    # Unpacks structs to pass into subprocesses
    # day_pres_seq = data.day_pres_seq
    use_state_names = data.use_state_names
    state_name_to_id = data.state_name_to_id
    pop_df = data.pop_df

    # ---

    def date_task(date_inputs):
        """Forecast for one date, all states."""
        # i_date, day_pres = date_inputs
        i_date, (day_pres, truth_df) = date_inputs

        day_pres: pd.Timestamp
        day_pres_str = day_pres.date().isoformat()

        # # -()- Load from file
        # (Now it's preloaded)
        # truth_df = load_truth_for_day_pres(params, day_pres)

        def state_task(state_inputs):
            """Forecast for one location and date."""
            i_state, state_name = state_inputs
            _sn = make_file_friendly_name(state_name)
            _LOGGER.debug(f"Called state_task for day_pres={day_pres} and state_name={state_name}")

            # Preparation work
            # ----------------------------

            # Retrieve state truth (full series)
            state_series = truth_df.xs(
                state_name_to_id[state_name], level="location")["value"]

            # Make Generation time
            tg = get_tg_object(params.general["tg_model"], **params.general)

            # Create the forecast operator
            # ----------------------------
            fop = ParSelTrainOperator(
                # --- Child class arguments
                day_pres, params.general["nperiods_main_roi"],  # Child class args
                population=pop_df.loc[state_name, "population"],
                # --- Base class arguments
                incid_series=state_series,  # Base class args
                params=params.__dict__,
                nperiods_fore=WEEKLEN * params.general["nperiods_fore"],
                name=f"{_sn}_{day_pres_str}",
                tg_past=tg,
            )
            _LOGGER.debug(f"Created forecast operator {fop.name}")

            # Run the forecast operations
            # --------------------------
            fop.run()

            # TEST STUFF HERE
            print(f"Watch {fop.name}\n"
                  # f"{fop.inc.fore_aggr_df}\n"
                  f"{fop.fore_quantiles}\n"
                  )

            # ----
            fop.dump_heavy_stuff()

            return fop

        # --------------------------------- End of state_task ---

        # --- Run for states
        return map_parallel_or_sequential(
            state_task, enumerate(use_state_names), params.ncpus_states
        )

    # --- Run for dates
    date_state_fop_list = map_parallel_or_sequential(
        date_task,
        enumerate(zip(data.day_pres_seq, data.truth_df_sr)),
        params.ncpus_dates
    )


    # # todo test outcomes of one forecast ---------------------------
    # fop_df = pd.DataFrame(
    #     date_state_fop_list,
    #     index=day_pres_seq,
    #     columns=use_state_names,
    # )
    #
    # # fop: ParSelTrainOperator = date_state_fop_list[1][0]  # From nested list
    # probe_state = "California"  # "Florida" # "Wyoming" # "Illinois"
    # fop: ParSelTrainOperator = fop_df.iloc[1][probe_state]
    #
    # print("----- TEST ONE FORECAST -----")
    # print(fop.name)
    #
    # latest_truth_df = load_truth_for_day_pres(params, day_pres_seq[-1])
    # latest_incid_sr = (
    #     latest_truth_df.xs(state_name_to_id[probe_state], level="location")["value"])
    #
    # from matplotlib import pyplot as plt
    # from rtrend_tools.visualization import plot_precalc_quantiles_as_layers
    #
    # fig, ax = plt.subplots()
    # ax: plt.Axes
    #
    # # fop.logger.error(f"WATCH {fop.inc.past_aggr_sr}")
    # # fop.logger.error(f"WATCH {fop.time.past_aggr_idx}")
    #
    # # # WEEKLY
    # # ax.plot(fop.inc.past_aggr_sr)
    # # ax.plot(fop.fore_quantiles.loc[0.5], "--")
    # # plot_precalc_quantiles_as_layers(
    # #     ax, fop.fore_quantiles.to_numpy(), fop.time.fore_aggr_idx)
    #
    # # DAILY
    #
    # ax.plot(fop.inc.past_gran_sr)  # Filtered past daily
    # print(latest_incid_sr[fop.time.pg0:fop.time.fa1])
    # ax.plot(latest_incid_sr[fop.time.pg0:fop.time.fg1])  # Truth series
    # plot_precalc_quantiles_as_layers(
    #     ax, fop.daily_quantiles.to_numpy(), fop.time.fore_gran_idx
    # )
    # ax.plot(fop.daily_quantiles.loc[0.5], "--", color="gray")
    #
    # ax.set_title(fop.name)
    #
    # plt.show()


# ----------------
# FUTURE FUNCTIONS Z– These will be moved to the Rtrend package
import numba as nb


# def _calc_rt_next_tmp(last_r_vals, nsamples, last_ct, _rng):
#     """HARDCODED FUNCTION TO TEST MULTIPLE METHODS OF R(T) SYNTHESIS"""
#     # --- INCIDENCE EXPONENTIAL DRIFT
#     # TMP: define here drift parameters
#     sigma = 0.00  # Width of the random walk normal steps
#
#     # alpha = 2.E-5  # California. Weight of the current incidence to deplete susceptibles.
#     alpha = 6.E-5  # Illinois. Weight of the current incidence to deplete susceptibles.
#     # alpha = 10.E-4  # Wyoming. Weight of the current incidence to deplete susceptibles.
#
#     bias = 0.000
#
#     return last_r_vals * np.exp(
#         sigma * _rng.normal(scale=sigma, size=nsamples)  # RW
#         - alpha * last_ct + bias
#     )


# @nb.njit
# def step_reconstruction(
#         i_t_fore, ct_past, rt_fore_2d, tg_past, tg_fore, tg_max,
#         ct_fore_2d, nsamples, seed
# ):
#     """One hardcoded step of the reconstruction (renewal eq.)
#     """
#     np.random.seed(seed + i_t_fore)  # Seeds the numba or numpy generator
#
#     # Main loop over R(t) samples
#     for i_sample in range(nsamples):
#         rt_fore = rt_fore_2d[:, i_sample]
#         ct_fore = ct_fore_2d[:, i_sample]
#
#         lamb = 0.  # Sum of generated cases from past cases
#         r_curr = rt_fore[i_t_fore]
#
#         # Future series chunk
#         for st in range(1, min(i_t_fore + 1, tg_max)):
#             lamb += r_curr * tg_fore[st] * ct_fore[i_t_fore - st]
#
#         # Past series chunk
#         for st in range(i_t_fore + 1, tg_max):
#             lamb += r_curr * tg_past[st] * ct_past[-(st - i_t_fore)]
#
#         # Poisson number
#         ct_fore[i_t_fore] = np.random.poisson(lamb)


if __name__ == "__main__":
    main()


#


# =============================================================================
# ==================  OLD CODE VAULT  =========================================
# =============================================================================


# # -()- HARDCODED DYNAMIC RAMP
# r1, r2 = [params.rt_synthesis[k] for k in ["r1_start", "r2_start"]]
# r_start = (r2 - r1) * mean_vals + 2 * r1 - r2
# r1, r2 = [params.rt_synthesis[k] for k in ["r1_end", "r2_end"]]
# r_end = (r2 - r1) * mean_vals + 2 * r1 - r2
#
# rt_fore = RtData(np.linspace(r_start, r_end, ndays_fore).T)
#
# # -//-