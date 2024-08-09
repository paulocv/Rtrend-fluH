"""
Performs a training loop for the parameter selection maching of
Rtrend forecasts.

This implements the "Par. select. with preprocessing" pipeline, which
also performs preprocessing within each forecasting loop.

"""

import argparse
import datetime
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rtrend_forecast.postprocessing import aggregate_in_periods
from rtrend_forecast.reporting import (
    ExecTimeTracker,
    get_rtrend_logger,
    get_main_exectime_tracker,
    SUCCESS,
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
    ncpus_dates=1,
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

    run_training(params, data)  # THIS WILL RUN THE SEASON MULTIPLE TIMES
    # run_forecasts_once(params, data)  # RUNS once FOR ONE "SEASON", ALL LOCATIONS

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
    golden_truth_daily_sr: pd.Series  # sr.loc[day_pres] = incidence. Most up-to-date truth.
    golden_truth_weekly_sr: pd.Series  # sr.loc[day_pres] = incidence. Most up-to-date truth.
    # summary_metrics_df: pd.DataFrame  # Overall summary metrics

    fop_sr: pd.Series  # Series of forecast operators, keyed by state
    fop_sr_valid_mask: pd.Series

    # Scoring and exporting
    weekly_horizon_array: np.ndarray  # Requested horizons as int array (in weeks)
    daily_horizon_array: np.ndarray  # Daily horizons as int array (in days)
    weekly_horizon_deltas: pd.TimedeltaIndex  # Horizons as time deltas
    daily_horizon_deltas: pd.TimedeltaIndex  # Daily horizons as time
    q_alphas: np.ndarray

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
    # Incidence data *as of* each forecast date
    data.truth_df_sr = pd.Series(
        [load_truth_for_day_pres(params, day_pres)
         for day_pres in data.day_pres_seq],
        index=data.day_pres_seq
    )

    # Golden truth data, containing data for all forecasts
    data.golden_truth_daily_sr, data.golden_truth_weekly_sr = (
        load_and_aggregate_golden_truth(params)
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


def load_and_aggregate_golden_truth(params: Params):
    # Load the (expectedly daily) truth file
    daily_sr = load_truth_cdc_simple(params.general["golden_data_path"])["value"]
    _LOGGER.debug(
        f"Golden truth file {params.general['golden_data_path']} loaded.")

    # Aggregate into weekly data for each location
    vals = list()
    keys = daily_sr.index.get_level_values("location").unique()
    for loc_id in keys:
        # Selects data from that location and aggregates
        vals.append(aggregate_in_periods(
            daily_sr.xs(loc_id, level="location"),
            pd.Timedelta("1w"),
            params.postprocessing["aggr_ref_tlabel"],
            full_only=True,
        ))

    weekly_sr = pd.concat(vals, keys=keys).reorder_levels([1, 0])

    return daily_sr, weekly_sr


# ---


# ---


# -------------------------------------------------------------------
# MAIN TRAINING ROUTINES
# -------------------------------------------------------------------


# ----


# ---

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
            try:
                fop.run()
            
            except Exception:
                fop.logger.critical("Created an unhandled exception.")
                raise
            
            finally:
                fop.dump_heavy_stuff()

            return fop

            # --------------------------------- End of state_task ---

        # Run for states
        # ----------------------------
        return map_parallel_or_sequential(
            state_task, enumerate(use_state_names), params.ncpus_states
        )

        # TODO: BUILD USA HERE?!????

        # --------------------------------- End of date_task ---

    # Run for dates
    # ----------------------------
    date_state_fop_list = map_parallel_or_sequential(
        date_task,
        enumerate(zip(data.day_pres_seq, data.truth_df_sr)),
        params.ncpus_dates
    )

    # Post-forecast collection
    # ---------------------------------------------
    # --- Build one series of fop objects (with 2-level multiindex)
    data.fop_sr = (
        pd.Series(sum(date_state_fop_list, []),
                  index=data.date_state_idx))

    # --- Store and check valid outputs
    data.fop_sr_valid_mask = data.fop_sr.map(
        lambda fop: fop.success
    )

    #
    # # todo test outcomes of one forecast ---------------------------
    # oanfoainf o ------------------ APAGAR LATER ===============
    # fop_df = pd.DataFrame(
    #     date_state_fop_list,
    #     index=data.day_pres_seq,
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
    # latest_truth_df = load_truth_for_day_pres(params, data.day_pres_seq[-1])
    # latest_incid_sr = (
    #     latest_truth_df.xs(state_name_to_id[probe_state], level="location")["value"])
    # fop.daily_quantiles = fop.make_quantiles_for("gran") # DAILY (optional)
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


@GLOBAL_XTT.track()
def score_forecasts_once(params: Params, data: Data):
    """
    Scores prerun forecasts with given parameters, for all locations
    and selected dates.
    """
    # -----------------------------------
    # Preamble operations
    # -----------------------------------
    _LOGGER.debug("Begin scoring forecasts.")
    # Filter invalid
    fop_sr = data.fop_sr.loc[data.fop_sr_valid_mask]

    # Unpack values
    q_alphas = data.q_alphas
    golden_truth_daily_sr = data.golden_truth_daily_sr
    state_name_to_id = data.state_name_to_id

    daily_horizon_array = data.daily_horizon_array
    weekly_horizon_array = data.weekly_horizon_array
    daily_horizon_deltas = data.daily_horizon_deltas
    weekly_horizon_deltas = data.weekly_horizon_deltas

    # -----------------------------------
    # Define the scoring routine
    # -----------------------------------

    def score_one_forecast(
            fop: ParSelTrainOperator,
            day_pres: pd.Timestamp,
            state_name: str,
    ):
        """"""
        # Cross-section the golden truth for given state
        truth_daily = golden_truth_daily_sr.xs(
            state_name_to_id[state_name], level="location"
        )
        truth_weekly = data.golden_truth_weekly_sr.xs(
            data.state_name_to_id[state_name], level="location"
        )

        # Initialize the local scores dataframe
        daily_d = OrderedDict()
        weekly_d = OrderedDict()

        # --- TEST: WIS - DAILY
        scores = fop.score_forecast(
            which="gran",
            method="wis",
            truth_sr=truth_daily,
            time_labels=day_pres + daily_horizon_deltas,
            alphas=q_alphas,
        )
        daily_d["wis"], daily_d["wis_sharp"], daily_d["wis_calib"] = scores

        # --- TEST: WIS - WEEKLY
        scores = fop.score_forecast(
            which="aggr",
            method="wis",
            truth_sr=truth_weekly,
            time_labels=day_pres + weekly_horizon_deltas,
            alphas=q_alphas
        )
        weekly_d["wis"], weekly_d["wis_sharp"], weekly_d["wis_calib"] = scores

        # Construct the single forecast dataframes
        weekly_df = pd.DataFrame(weekly_d, index=weekly_horizon_array)
        weekly_df.index.name = "horizon_w"

        daily_df = pd.DataFrame(daily_d, index=daily_horizon_array)
        daily_df.index.name = "horizon_d"

        return daily_df, weekly_df

    # -----------------------------------
    # Run all scores
    # -----------------------------------
    results = map_parallel_or_sequential(
        lambda item: score_one_forecast(item[1], item[0][0], item[0][1]),
        contents=fop_sr.items(),
        ncpus=1,
    )

    # TODO: USA

    # ----------------------------------------------
    # Construct report and aggregate for final score
    # ----------------------------------------------
    # --- Weekly full report
    weekly_scores_df = pd.concat(
        [res[1] for res in results],
        keys=fop_sr.index
    )

    # --- Daily full report
    daily_scores_df = pd.concat(
        [res[0] for res in results],
        keys=fop_sr.index
    )

    print(weekly_scores_df)
    print(daily_scores_df)
    # for res in results:
    #     print(res[1])

    _LOGGER.log(SUCCESS, "Scoring completed.")


    #
    #     # --- TEST - WEEKLY
    #     scores = fop.score_forecast(
    #         which="aggr",
    #         method="wis",
    #         truth_sr=truth_weekly,
    #         time_labels=day_pres + weekly_horizon_deltas,
    #         alphas=alphas,
    #     )


    # # fop: ParSelTrainOperator = data.fop_sr.iloc[0]
    # # day_pres, state_name = data.fop_sr.index[0]
    #
    # truth_daily = data.golden_truth_daily_sr.xs(
    #     data.state_name_to_id[state_name], level="location"
    # )
    # truth_weekly = data.golden_truth_weekly_sr.xs(
    #     data.state_name_to_id[state_name], level="location"
    # )



# ----


# ----


def run_training(params: Params, data: Data, WHAT_ELSE=None):
    """"""
    #
    # ----------------------------------------------------------------
    # PREAMBLE
    # ----------------------------------------------------------------
    #
    # --- Calculate the time deltas for the required daily horizons
    # -()- Weekly
    data.weekly_horizon_array = np.array(params.general["export_horizons"])
    data.weekly_horizon_deltas = (
        pd.TimedeltaIndex(data.weekly_horizon_array, unit="w"))
    # -()- Daily
    data.daily_horizon_array = b = np.array([
        np.arange(WEEKLEN * (h - 1), WEEKLEN * h)
        for h in data.weekly_horizon_array]).flatten()
    data.daily_horizon_deltas = pd.TimedeltaIndex(b, unit="d")

    # --- Calculate the alpha values for the required IQRs
    quantiles = np.array(params.postprocessing["inc_quantiles"])
    # TODO: warn if quantiles list is malformed!!!!!
    num_quantiles = quantiles.shape[0]
    data.q_alphas = 2 * quantiles[: (num_quantiles + 1) // 2]



    # for muitas vezes (training loop):
    run_forecasts_once(params, data, WHAT_ELSE=WHAT_ELSE)
    score_forecasts_once(params, data)



# -- -- - - - OAO APAGAR LATER

    # # # # TODO TESTING THE SCORING
    # fop: ParSelTrainOperator = data.fop_sr.iloc[0]
    # day_pres, state_name = data.fop_sr.index[0]
    #
    # from rtrend_forecast.scoring import covered_score
    #
    # # WHAT DO I NEED
    # # - Array of truth observations (daily or weekly)
    # # - Dictionary-like of quantile forecasts
    # # -
    #
    # truth_daily = data.golden_truth_daily_sr.xs(
    #     data.state_name_to_id[state_name], level="location"
    # )
    # truth_weekly = data.golden_truth_weekly_sr.xs(
    #     data.state_name_to_id[state_name], level="location"
    # )
    #
    # # # --- TEST - DAILY
    # # fop.score_forecast(
    # #     which="gran",
    # #     truth_sr=truth_daily,
    # #     time_labels=day_pres + daily_horizon_deltas,
    # #     alphas=alphas,
    # # )
    #
    # # --- TEST - WEEKLY
    # scores = fop.score_forecast(
    #     which="aggr",
    #     method="wis",
    #     truth_sr=truth_weekly,
    #     time_labels=day_pres + weekly_horizon_deltas,
    #     alphas=alphas,
    # )
    #
    # # covered_score()


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