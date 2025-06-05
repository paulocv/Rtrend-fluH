"""
Performs a training loop for the parameter selection maching of
Rtrend forecasts.

This implements the "Par. select. with preprocessing" pipeline, which
also performs preprocessing within each forecasting loop.

"""

import argparse
import datetime
import os
import sys
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
    SUCCESS, config_rtrend_logger,
)
from rtrend_forecast.structs import RtData, get_tg_object, IncidenceData, TimeData
from rtrend_forecast.preprocessing import apply_lowpass_filter_pdseries
from rtrend_forecast.utils import map_parallel_or_sequential
from utils.forecast_operators_experimental import WEEKLEN, ParSelTrainOperator
from utils.parsel_utils import load_population_data, make_date_state_index, load_truth_cdc_simple, make_mcmc_prefixes, \
    make_file_friendly_name, make_rt_out_fname, make_filtered_fname

_CALL_TIME = datetime.datetime.now()


# DEFAULT PARAMETERS
# -------------
DEFAULT_PARAMS = dict(  # Parameters that go in the main Params class
    # Field names in the preprocessed data file
    filt_col_name="filtered",
    filtround_col_name="filtered_round",
    us_location_name="US",

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
    loglevel: int


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
    us_location_name: str
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
    truth_df_sr: pd.Series  # sr.loc[day_pres] = truth_df . Truth AS OF each day.
    golden_truth_daily_sr: pd.Series  # sr.loc[day_pres] = incidence. Most up-to-date truth.
    golden_truth_weekly_sr: pd.Series  # sr.loc[day_pres] = incidence. Most up-to-date truth.
    # summary_metrics_df: pd.DataFrame  # Overall summary metrics

    # fop_sr: pd.Series  # Series of forecast operators, keyed by state
    # fop_sr_valid_mask: pd.Series

    # Scoring and exporting
    weekly_horizon_array: np.ndarray  # Requested horizons as int array (in weeks)
    daily_horizon_array: np.ndarray  # Daily horizons as int array (in days)
    weekly_horizon_deltas: pd.TimedeltaIndex  # Horizons as time deltas
    daily_horizon_deltas: pd.TimedeltaIndex  # Daily horizons as time
    q_alphas: np.ndarray


class TrainIterationData:
    """A data bunch for storing data from a specific iteration."""
    fop_sr: pd.Series  # Signature: sr[(date_str, state_str)] = Forecast operator or USA operator
    fop_sr_valid_mask: pd.Series
    daily_scores_df: pd.DataFrame
    weekly_scores_df: pd.DataFrame

    def __init__(self, **kwargs):
        """"""
        # --- Smart init: create elements if they are informed as constructor keyword args
        for name in [
                "fop_sr", "fop_sr_valid_mask", "daily_scores_df", "weekly_scores_df"
        ]:
            if name in kwargs:
                setattr(self, name, kwargs[name])


class USAForecast:
    """A simplified forecast operator which holds data from the aggregation
    of multiple locations.
    """
    # raw_incid_sr: pd.Series  # Complete series of incidence
    fore_quantiles: pd.DataFrame  # Aggregated period forecasted quantiles
    inc: IncidenceData  #
    is_aggr: bool = False
    gran_dt = pd.Timedelta("1d")
    aggr_dt = pd.Timedelta("1w")
    aggr_nperiods = 7  # Number of gran periods that an aggregated period has
    logger: None
    state_name: str = "US"

    def __init__(
            self,
            raw_incid_sr: pd.Series,
            nperiods_fore: int,
            day_pres: pd.Timestamp,
            name: str,
    ):
        self.nperiods_fore = nperiods_fore
        self.name = name

        # --- Composite structs
        self.inc = IncidenceData()
        self.inc.is_aggr = self.is_aggr
        self.inc.raw_sr = raw_incid_sr

        # if tg_past is None:
        #     self.tg_past = TgBase()
        #     raise NotImplementedError("Hey, by now you must inform a "
        #                               "Tg object at construction.")
        # else:
        #     self.tg_past = tg_past
        #
        # self.tg_fore = tg_fore  # `None` is acceptable
        #
        # self.rt_past: RtData = None
        # self.rt_fore: RtData = None
        #
        # self.noise_obj_dict: dict[AbstractNoise] = dict()

        # --- Infrastructure related to time stamps
        self.time = TimeData()
        self.time.is_aggr = self.is_aggr
        self.time.pres = day_pres

        if self.is_aggr:
            self.inc.past_aggr_sr = self.inc.raw_sr
            self.time.pa1 = self.inc.raw_sr.index.max()

        # --- Other infrastructure
        self.logger = _LOGGER.getChild(self.name)
        self.success = True

        # --- Manually calculate number of periods in granular time.
        # Useful for categorical forecast
        if self.is_aggr:
            self.nperiods_fore_gran = (
                    self.nperiods_fore * self.aggr_nperiods)
        else:
            self.nperiods_fore_gran = self.nperiods_fore

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

    parser.add_argument(
        "--loglevel", type=int,
        help="Poject logger level. Set according to python's "
             "`logging` library documentation."
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

    if args.loglevel:
        config_rtrend_logger(args.loglevel)

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
def run_forecasts_once(
        params: Params, data: Data, WHAT_ELSE=None
) -> TrainIterationData:
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
                state_name=_sn
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
        state_fop_list: list[ParSelTrainOperator] = (
            map_parallel_or_sequential(
                state_task, enumerate(use_state_names), params.ncpus_states
            )
        )

        # --- Build aggregate forecast over all locations
        usa_fop = aggregate_all_states_forecast(
            params, state_fop_list, state_name_to_id,
            day_pres, truth_df
        )
        state_fop_list.append(usa_fop)

        # if len(state_fop_list) == 0: return state_fop_list
        # us_series = truth_df.xs(
        #     state_name_to_id[params.us_location_name], level="location")["value"]
        # usa_fop = USAForecast(us_series)
        # usa_fop.fore_quantiles = state_fop_list[0].fore_quantiles
        # for fop in state_fop_list[1:]:
        #     usa_fop.fore_quantiles += fop.fore_quantiles

        return state_fop_list

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
    iter_data = TrainIterationData()

    # --- Build a single series of fop objects (with 2-level multiindex)
    # For all dates and states.
    flat_fop_list = sum(date_state_fop_list, [])
    # Agnostically builds the index instead of using `data.date_state_idx`
    flat_index = pd.MultiIndex.from_tuples(
        [(fop.time.pres.date().isoformat(), fop.state_name) for fop in flat_fop_list])
    iter_data.fop_sr = pd.Series(flat_fop_list, index=flat_index)

    # --- Store and check valid outputs
    iter_data.fop_sr_valid_mask = iter_data.fop_sr.map(
        lambda fop: fop.success
    )

    return iter_data


def aggregate_all_states_forecast(
        params: Params,
        state_fop_list: list[ParSelTrainOperator],
        state_name_to_id,
        day_pres: pd.Timestamp,
        truth_df: pd.DataFrame,
) -> USAForecast:
    """ Combines forecasts from all states to produce a USA-level forecast.
    If not all states have produced forecasts, this should not be
    interpreted as USA-level.
    """
    day_pres_str = day_pres.date().isoformat()
    # Build a series of fop objects for easier handling
    fop_sr = pd.Series(
        state_fop_list,
        index=[fop.name for fop in state_fop_list],
    )

    # --- Select USA-level truth
    truth_sr = (
        truth_df
        .xs(state_name_to_id[params.us_location_name], level="location")["value"]
        .sort_index()
    )
    # --- Create USA forecast object, special
    usa_fop = USAForecast(
        truth_sr,
        nperiods_fore=WEEKLEN * params.general["nperiods_fore"],
        day_pres=day_pres,
        name=f"US_{day_pres_str}",
    )

    # --- Filter invalid forecasts
    fop_sr_valid_mask = fop_sr.map(lambda fop: fop.success)
    fop_sr: pd.Series = fop_sr.loc[fop_sr_valid_mask]
    if fop_sr.shape[0] == 0:  # Valid dataset is empty
        raise RuntimeError(
            "No forecasts were produced with the `success` flag set to"
            " `True`."
        )

    if not fop_sr_valid_mask.all():
        diff = (~fop_sr_valid_mask).sum()
        _LOGGER.warning(
            f"{diff} out of {fop_sr_valid_mask.shape[0]} "
            f"states did not produce successful "
            f"forecasts. The USA ensemble will be built without these "
            f"states."
        )

    # ------------------
    # Quantile forecasts
    # ------------------
    usa_fop.fore_quantiles = fop_sr.iloc[0].fore_quantiles.copy()
    for fop in fop_sr.iloc[1:]:
        if fop.fore_quantiles is not None:
            usa_fop.fore_quantiles += fop.fore_quantiles

    # # -----------------------------------
    # # Categorical – Rate change forecasts
    # # -----------------------------------
    #
    # # Take samples from all locations
    # # -------------------------------
    # rng = np.random.default_rng(20)  # Take a temporary generator
    # df_list = list()
    #
    # for state_name, fop in fop_sr.items():
    #     fop: FluSight2024ForecastOperator
    #     nsamples_state, nsteps = fop.inc.fore_aggr_df.shape
    #
    #     # --- Take samples
    #     i_samples = rng.integers(
    #         0, nsamples_state, size=params.general["nsamples_to_us"]
    #     )
    #     df = fop.inc.fore_aggr_df.loc[i_samples, :]  # Take sample trajectories
    #     # --- Sort at each date
    #     df: pd.DataFrame = df.apply(lambda x: x.sort_values().values, axis=0)
    #     df.reset_index(inplace=True, drop=True)
    #
    #     df_list.append(df)
    #
    # multistate_df = pd.concat(
    #     df_list, keys=list(fop_sr.keys()),
    #     names=["location", "i_sample"],
    # )
    #
    # # Aggregate spatially
    # # -------------------
    # usa_fop.inc.fore_aggr_df = multistate_df.groupby("i_sample").sum()
    # last_observed_date = usa_fop.inc.raw_sr.index.max() + usa_fop.gran_dt
    # # last_observed_date = usa_fop.inc.raw_sr.index.max()  # TODO: change here too
    #
    # # Calculate the categorical rate change forecasts
    # # -----------------------------------------------
    # df = calc_rate_change_categorical_flusight(
    #     usa_fop,
    #     count_rates=data.count_rates.loc["US"],
    #     dates=data.dates,
    #     # day_pres=data.day_pres,
    #     last_observed_date=last_observed_date,
    #     thresholds=params.postprocessing.get("categorical_thresholds", None)
    # )
    # # Prepare the df to be "appended" to the other states df
    # df.columns.name = "horizon"
    # df.index = pd.MultiIndex.from_product(
    #     [["US"], df.index], names=["location_name", "rate_change_id"]
    # )
    # # df.index.name = "rate_change_id"
    # df.name = "US"
    #
    # # data.rate_fore_df = pd.concat([data.rate_fore_df, df])
    # data.rate_fore_df = pd.concat(
    #     [data.rate_fore_df, df])
    #
    # usa_fop.logger.log(
    #     15, "USA quantile and rate change aggregations concluded.")

    return usa_fop


@GLOBAL_XTT.track()
def score_forecasts_once(params: Params, data: Data, iter_data: TrainIterationData):
    """
    Scores prerun forecasts with given parameters, for all locations
    and selected dates.
    """
    # -----------------------------------
    # Preamble operations
    # -----------------------------------
    _LOGGER.info("Begin scoring of forecasts.")
    # Filter invalid
    fop_sr = iter_data.fop_sr.loc[iter_data.fop_sr_valid_mask]

    # Unpack values
    q_alphas = data.q_alphas
    golden_truth_daily_sr = data.golden_truth_daily_sr
    golden_truth_weekly_sr = data.golden_truth_weekly_sr
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
        truth_weekly = golden_truth_weekly_sr.xs(
            state_name_to_id[state_name], level="location"
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

        # --- p-score (predicted interval that captures the observation)
        weekly_d["p"] = fop.score_forecast(
            which="aggr",
            method="p",
            truth_sr=truth_weekly,
            time_labels=day_pres + weekly_horizon_deltas,
            alphas=q_alphas
        )

        # Construct the single forecast dataframes
        weekly_df = pd.DataFrame(weekly_d, index=weekly_horizon_array)
        weekly_df.index.name = "horizon_w"

        daily_df = pd.DataFrame(daily_d, index=daily_horizon_array)
        daily_df.index.name = "horizon_d"

        return daily_df, weekly_df

    # -----------------------------------
    # Run all scores
    # -----------------------------------
    # TODO FIX
    results = map_parallel_or_sequential(
        lambda item: score_one_forecast(item[1], item[0][0], item[0][1]),
        contents=fop_sr.items(),
        ncpus=1,
    )

    # TODO: USA

    # ----------------------------------------------
    # Construct report and aggregate for final score
    # ----------------------------------------------
    # --- Daily full report
    iter_data.daily_scores_df = pd.concat(
        [res[0] for res in results],
        keys=fop_sr.index
    )

    # --- Weekly full report
    iter_data.weekly_scores_df = pd.concat(
        [res[1] for res in results],
        keys=fop_sr.index
    )


    # print(daily_scores_df)
    # print(weekly_scores_df)
    # # for res in results:
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

    #
    # ----------------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------------
    #

    # for muitas vezes (training loop):

    iter_data = run_forecasts_once(params, data, WHAT_ELSE=WHAT_ELSE)

    score_forecasts_once(params, data, iter_data)

    # Now you can work with `iter_data` to postprocess and evaluate this iteration.



    pass


# -- -- - - - OAO APAGAR LATER

    # # TODO Test the forecasts
    # fop: ParSelTrainOperator = iter_data.fop_sr.iloc[0]
    # print(fop.fore_quantiles)

    # # TODO test the consolidated code for scores
    print(iter_data.weekly_scores_df)

    # # # # TODO TESTING THE SCORING
    # fop: ParSelTrainOperator = data.fop_sr.iloc[0]
    # day_pres, state_name = data.fop_sr.index[0]
    #
    # from rtrend_forecast.scoring import covered_score

    # WHAT DO I NEED
    # - Array of truth observations (daily or weekly)
    # - Dictionary-like of quantile forecasts
    # -
    #
    # truth_daily = data.golden_truth_daily_sr.xs(
    #     data.state_name_to_id[state_name], level="location"
    # )
    # truth_weekly = data.golden_truth_weekly_sr.xs(
    #     data.state_name_to_id[state_name], level="location"
    # )

    # # --- TEST - DAILY
    # fop.score_forecast(
    #     which="gran",
    #     truth_sr=truth_daily,
    #     time_labels=day_pres + daily_horizon_deltas,
    #     alphas=alphas,
    # )

    # # --- TEST - WEEKLY
    # scores = fop.score_forecast(
    #     which="aggr",
    #     method="wis",
    #     truth_sr=truth_weekly,
    #     time_labels=day_pres + data.weekly_horizon_deltas,
    #     alphas=data.q_alphas,
    # )

    # covered_score()


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