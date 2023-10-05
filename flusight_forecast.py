"""
Runs the Rtrend forecasting method for the FluSight CDC initiative.
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

from rtrend_forecast.reporting import (
    get_rtrend_logger,
    get_main_exectime_tracker,
)
from rtrend_forecast.structs import RtData, get_tg_object
from rtrend_forecast.utils import map_parallel_or_sequential
from rtrend_interface.flusight_tools import (
    FluSightDates,
    calc_rate_change_categorical_flusight,
    calc_horizon_from_dates,
    rate_change_i_to_name,
    calc_target_date_from_horizon,
)
from rtrend_interface.forecast_operators import (
    WEEKLEN,
    ParSelTrainOperator,
    FluSightForecastOperator,
)
from rtrend_forecast.visualization import (
    rotate_ax_labels,
    make_axes_seq,
    plot_fore_quantiles_as_area,
)
from rtrend_interface.parsel_utils import (
    load_population_data,
    make_date_state_index,
    load_truth_cdc_simple,
    make_file_friendly_name,
    make_state_index,
)
from rtrend_interface.truth_data_structs import FluDailyTruthData

_CALL_TIME = datetime.datetime.now()


# DEFAULT PARAMETERS
# -------------
DEFAULT_PARAMS = dict(  # Parameters that go in the main Params class
    # Field names in the preprocessed data file
    filt_col_name="filtered",
    filtround_col_name="filtered_round",

    ncpus=5,
)

DEFAULT_PARAMS_GENERAL = dict(  # Parameters from the `general` dict
    pop_data_path="population_data/locations.csv",
    hosp_data_path="hosp_data/truth_daily_latest.csv",
    forecast_days_ahead=2,  # Forecast evaluation ahead of the day_pres
)

DEFAULT_PARAMS_RT_ESTIMATION = dict(  # Parameters from the `general` dict
    use_tmp=True
)

# Helpful objects from the Rtrend library
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
    run_forecasts_once(params, data)
    # calculate_rate_change_forecasts(params, data)
    calculate_for_usa(params, data)
    postprocess_all(params, data)
    export_all(params, data)

    report_execution_times()


#


#


# --------------------------------------------------------------------
# AUXILIARY STRUCTURES
# --------------------------------------------------------------------

class USAForecast:
    """Structures for the USA-level forecast, aggregated from.

    NOTE: this struct is taylored for the FluSight challenge. As time
    allows, a more general "forecast aggregate" class could be made.
    """
    raw_incid_sr: pd.Series  # Complete series of incidence
    fore_quantiles: pd.DataFrame  # Aggregated period forecasted quantiles


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
    call_time: datetime.datetime  # Program call time
    now: pd.Timestamp  # Time regarded as "now" for FluSight submission

    general: dict
    preprocessing: dict
    rt_estimation: dict
    rt_synthesis: dict
    inc_reconstruction: dict
    postprocessing: dict

    # Program parameters
    filt_col_name: str
    filtround_col_name: str
    ncpus: int
    export: bool


class Data:
    """Input, intermediate and output data.
    Input data is reserved for large structures. Smaller ones that
    can be manually set should be reserved as parameters.
    """
    dates: FluSightDates
    day_pres: pd.Timestamp

    pop_df: pd.DataFrame  # Data frame with population demographic data
    count_rates: pd.DataFrame  # Count thresholds for rate forecast
    truth: FluDailyTruthData  # Object that holds daily truth data
    all_state_names: list  # Names of all available locations to forecast
    state_name_to_id: dict  # From `location_name` to `location`

    use_state_names: None  # Names of locations that shall be used.
    # summary_metrics_df: pd.DataFrame  # Overall summary metrics

    fop_sr: pd.Series  # Series of forecast operators, keyed by state
    fop_sr_valid_mask: pd.Series
    usa: USAForecast  # Results for the USA level

    rate_fore_df: pd.DataFrame   # a.loc[(state_name, rate_change_id), horizon] = probability
    #  ^ ^ Rate change forecasts.

    flusight_df: pd.DataFrame  # Exportable dataframe for FluSight
    main_quantile_df: pd.DataFrame  # Forecast quantiles for plotting


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
        "--now", type=pd.Timestamp,
        help="Timestamp to be considered as \"now\". Defaults to "
             "pandas.Timestamp.now().",
        default=pd.Timestamp.now("US/Eastern"),
    )

    parser.add_argument(
        "--export", action=argparse.BooleanOptionalAction,
        help="Whether the outputs (results of the preprocessing and "
             "R(t) estimation) should be exported to files.",
        default=True,
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

    return params


@GLOBAL_XTT.track()
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
    # IMPORT TRUTH DATA for the given date
    # --------------------
    data.truth = FluDailyTruthData(
        params.general["hosp_data_path"],
        all_state_names=data.all_state_names,
        state_name_to_id=data.state_name_to_id,
        pop_data_path=params.general["pop_data_path"],
    )


def report_execution_times():
    print("-------------------\nEXECUTION TIMES")
    # for key, val in LOCAL_XXT.get_dict().items():
    for key, val in GLOBAL_XTT.get_dict().items():
        print(key.ljust(25, ".") + f" {val:0.4f}s")

#


#


#

#


# -------------------------------------------------------------------
# AUXILIARY PROCEDURES
# -------------------------------------------------------------------

#


# -------------------------------------------------------------------
# MAIN TRAINING ROUTINES
# -------------------------------------------------------------------


#

@GLOBAL_XTT.track()
def run_forecasts_once(params: Params, data: Data):
    """Runs the forecast with the
    given parameters, for all locations and selected dates.
    """

    # ----------------------
    # Prepare the state loop
    # ----------------------
    # --- Create sequence of selected state names
    data.use_state_names = make_state_index(
        params.general, data.all_state_names)

    # --- Truth data
    # truth_df = load_truth_cdc_simple(params.general["hosp_data_path"])
    truth_obj = data.truth  # Object that handles truth data
    truth_df = data.truth.raw_data

    # --- Defines relevant dates from the truth files
    data.dates = FluSightDates(
        params.now,
        due_example=params.general.get("due_example", None),
        ref_example=params.general.get("ref_example", None),
    )
    date_index = (truth_df.index.get_level_values("date").unique()
                  .sort_values())
    data.day_pres = day_pres = date_index[params.general["day_pres_id"]] + pd.Timedelta("1d")
    #  ^  ^ Adds one day so that -1 is the first without data
    day_pres_str = day_pres.date().isoformat()

    # --- Unpacks structs to pass into subprocesses
    use_state_names = data.use_state_names
    state_name_to_id = data.state_name_to_id
    pop_df = data.pop_df

    # --------------------------------
    # Define and run the forecast task
    # --------------------------------

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
        # state_series = truth_obj.xs_state(state_name)

        # Make Generation time
        tg = get_tg_object(params.general["tg_model"], **params.general)

        # Create the forecast operator
        # ----------------------------
        fop = FluSightForecastOperator(
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

        # ----

        return fop

    # --- Run for states
    fop_list = map_parallel_or_sequential(
        state_task, enumerate(use_state_names), params.ncpus
    )

    # --- Store and check valid outputs
    data.fop_sr = (
        pd.Series(fop_list, index=use_state_names))

    data.fop_sr_valid_mask = data.fop_sr.map(
        lambda fop: fop.success
    )

    #

    # ----------------------------------------------
    # Calculate the categorical rate change forecast
    # ----------------------------------------------
    fop_sr = data.fop_sr.loc[data.fop_sr_valid_mask]  # Filter

    results = list()
    for _state_name, _fop in fop_sr.items():

        # Calculate rate changes
        df = calc_rate_change_categorical_flusight(
            _fop,
            count_rates=data.count_rates.loc[_state_name],
            dates=data.dates,
            day_pres=data.day_pres,
        )

        results.append(df)

    data.rate_fore_df = pd.concat(
        results,
        keys=fop_sr.keys(),
        names=["location_name", "rate_change_id"]
    )
    data.rate_fore_df.columns.name = "horizon"

    _LOGGER.log(15, "Rate change forecast concluded.")


    # # TODO DEBUG: why so big numbers??!??/
    # #  ----------------------------------------------------------------
    # _LOGGER.error("DEBUGGING STATES")
    #
    # rly_bad_states_sr = data.fop_sr.loc[
    #     data.fop_sr.map(
    #         lambda fop: fop.fore_quantiles.max().max() > 3.0E3
    #     )
    # ]
    #
    # _LOGGER.error(f"Bad states: {rly_bad_states_sr.index}")
    # num_bad_states = rly_bad_states_sr.shape[0]
    # ndays_past = params.rt_estimation["roi_len"]
    #
    # if num_bad_states == 0:  # No problems!!
    #     return
    #
    # import matplotlib.pyplot as plt
    #
    # fig, axes = make_axes_seq(num_bad_states)
    #
    # for i_ax, fop in enumerate(rly_bad_states_sr):
    #     fop: FluSightForecastOperator
    #     ax: plt.Axes = axes[i_ax]
    #
    #     # -()- Plot R(t) past
    #     ax.plot(fop.time.past_gran_idx[-ndays_past:], fop.extra["rt_past_median"])
    #     ax.fill_between(
    #
    #     )
    #     # -()- Plot R(t) fore
    #     ax.plot(fop.time.fore_gran_idx[-ndays_past:], fop.extra["rt_fore_median"])
    #     # ax.plot(fop.fore_quantiles.loc[0.5])
    #
    #     # # -()- Plot past data
    #     # ax.plot(fop.inc.past_gran_sr)
    #     # ax.plot(fop.extra["past_scaled_int_sr"])
    #
    #     rotate_ax_labels(ax)
    #
    # fig.tight_layout()
    # plt.show()


    # # # ------------- TODO TESTS
    # #  # TODO: CHANGE THE WAY THAT'S AGGREGATED. MUST BE BY REFERENCE DATES
    # #  # TODO CONTINUE (Not here) JUST GO TO AGGREGATE USA DATA
    # # # fop: FluSightForecastOperator = data.fop_sr.iloc[0]
    #
    # usa_q_df = data.fop_sr.iloc[0].fore_quantiles.copy()
    # for fop in data.fop_sr.iloc[1:]:
    #     if fop.fore_quantiles is not None:
    #         usa_q_df += fop.fore_quantiles
    #
    # # ---
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    #
    # # # US
    # # ax.plot(truth_df.xs(state_name_to_id["US"], level="location")["value"])
    # # ax.plot(usa_q_df.loc[0.5])
    #
    # # A given state
    # state_name = "Wyoming"
    # fop: FluSightForecastOperator = data.fop_sr.loc["California"]
    #
    # ax.plot(truth_obj.xs_state_weekly(
    #     "California", params.postprocessing["aggr_ref_tlabel"]))
    # ax.plot(fop.fore_quantiles.loc[0.5], "--")
    #
    #
    # rotate_ax_labels(ax)
    # ax.set_xlim(pd.Timestamp("2023-03-30"), pd.Timestamp("2023-09-01"))
    #
    # plt.show()


@GLOBAL_XTT.track()
def calculate_for_usa(params: Params, data: Data):

    data.usa = USAForecast()
    data.usa.raw_incid_sr = data.truth.xs_state("US")

    # --- Filter invalid forecasts
    fop_sr = data.fop_sr.loc[data.fop_sr_valid_mask]
    if fop_sr.shape[0] == 0:  # Valid dataset is empty
        _LOGGER.error(
            "No forecasts were produced with the `success` flag set to"
            " `True`. The program will quit.")
        sys.exit(1)

    if fop_sr.shape[0] < data.fop_sr.shape[0]:
        diff = data.fop_sr.shape[0] - fop_sr.shape[0]
        _LOGGER.warning(
            f"{diff} out of {data.fop_sr.shape[0]} "
            f"states did not produce successful "
            f"forecasts. The USA ensemble will be built without these "
            f"states."
        )

    # Quantile forecasts
    # ------------------
    data.usa.fore_quantiles = fop_sr.iloc[0].fore_quantiles.copy()
    for fop in fop_sr.iloc[1:]:
        if fop.fore_quantiles is not None:
            data.usa.fore_quantiles += fop.fore_quantiles

    # Categorical – Rate change forecasts
    # -----------------------------------
    # TODO
    # TODO: for the USA (needs some extra info in the US, including
    #    the SUM OF AGGREGATED *TRAJECTORIES*

    # #  ------------- TODO TEST
    # print(data.usa.fore_quantiles)
    # import matplotlib.pyplot as plt
    #
    # fop: FluSightForecastOperator = fop_sr.iloc[0]
    #
    # fig, ax = plt.subplots()
    #
    # ax.plot(data.usa.raw_incid_sr)
    # plot_fore_quantiles_as_area(ax, data.usa.fore_quantiles)
    #
    # ax.set_xlim(
    #     fop.time.pg0 - pd.Timedelta("2w"),
    #     fop.time.fa1 + pd.Timedelta("1w"),
    # )
    # rotate_ax_labels(ax)
    # plt.show()


@GLOBAL_XTT.track()
def postprocess_all(params: Params, data: Data):
    """"""

    # Filter invalid
    fop_sr = data.fop_sr.loc[data.fop_sr_valid_mask]

    # -------------------------------------------
    # Provide some status feedback
    # -------------------------------------------
    print("\n=======================================\n")
    print(f"- Now date ................ {data.dates.now} ")
    print(f"- Next submission due date  {data.dates.due} ")
    print(f"- Reference date .......... {data.dates.ref} ")
    print(f"- Forecast day_present .... {data.day_pres}")

    # -------------------------------------------
    # FluSight export format – QUANTILE forecasts
    # -------------------------------------------

    # Loop to list the exportable dataframes of each state
    def build_flusight_quantile_df(
            state_name, fop):
        """Make the quantile forecast for one location, which may
        include the US.
        #DEVNOTE: `fop` should also be compatible with USAForecast
        """

        fop: FluSightForecastOperator | USAForecast
        d = OrderedDict()  # Dict that's used to construct the dataframe

        size = fop.fore_quantiles.size  # Total number of items

        # Stack state data
        # -----------------
        stack = fop.fore_quantiles.stack()

        d["output_type_id"] = stack.index.get_level_values(0)

        # Forecast index weeks by Sunday, but now
        #   FluSight indexes by Saturday. So we need to subtract 1 day.
        #  v  v
        d["target_end_date"] = stack.index.get_level_values(1) - pd.Timedelta("1d")

        # Calculate horizon number
        d["horizon"] = (
            # (d["target_end_date"] - data.dates.ref) // pd.Timedelta("1w")
            calc_horizon_from_dates(d["target_end_date"], data.dates.ref)
        ).astype(int)

        # Generate other arrays
        # ---------------------
        d["reference_date"] = np.repeat(data.dates.ref.date(), size)  # TODO REPLACE
        d["location"] = np.repeat(data.state_name_to_id[state_name], size)
        d["output_type"] = np.repeat("quantile", size)
        d["target"] = np.repeat("wk inc flu hosp", size)

        d["value"] = stack.values

        return pd.DataFrame(d)

    # --- Runs the function for all states and for US
    concat_df_list = list()
    for _state_name, _fop in fop_sr.items():
        concat_df_list.append(
            build_flusight_quantile_df(_state_name, _fop)
        )

    # USA
    concat_df_list.append(
        build_flusight_quantile_df("US", data.usa)
    )

    # Concatenate into a single dataframe for the quantile forecast
    flusight_quantiles_df = pd.concat(concat_df_list)

    # ------------------------------------------------------------
    # FluSight export format – RATE CHANGE (CATEGORICAL) forecasts
    # ------------------------------------------------------------

    # Initialize the FluSight dataframe as a stack from the other
    df = pd.DataFrame(data.rate_fore_df.stack(), columns=["value"])

    # Build other columns
    df["location"] = df.index.get_level_values("location_name").map(
        lambda name: data.state_name_to_id[name]
    )
    df["target"] = np.repeat("wk flu hosp rate change", df.shape[0])

    # Date-related
    df["reference_date"] = np.repeat(data.dates.ref.date(), df.shape[0])
    df["horizon"] = df.index.get_level_values("horizon")
    df["target_end_date"] = calc_target_date_from_horizon(
        df["horizon"], data.dates.ref)

    df["output_type"] = np.repeat("pmf", df.shape[0])
    df["output_type_id"] = df.index.get_level_values("rate_change_id").map(
        lambda i: rate_change_i_to_name[i],
    )

    flusight_rate_df = df

    # ------------------------------------------------------------
    # Overall FluSight export data postprocessing
    # ------------------------------------------------------------

    # Concatenate into a single dataframe
    df = pd.concat([flusight_quantiles_df, flusight_rate_df])

    # Reorder columns to a better sequence
    df = df[
        ["reference_date", "target", "horizon", "target_end_date",
         "location", "output_type", "output_type_id", "value"]
    ]

    # Select only desired horizons to export
    use_horizons = params.general.get("export_horizons", None)
    if use_horizons:
        data.flusight_df = df.loc[df["horizon"].isin(use_horizons)]
    else:
        data.flusight_df = df

    print(data.flusight_df)

    # ------------------------------------------------------------
    # Other forecast reports
    # ------------------------------------------------------------

    # Quantile forecast data
    # ----------------------
    q_dfs = [fop.fore_quantiles for _, fop in fop_sr.items()]
    keys = [state_name for state_name, _ in fop_sr.items()]
    data.main_quantile_df = pd.concat(
        q_dfs, keys=keys, names=["location_name", "quantile"])

    # R(t) statistics
    # ---------------
    # TODO: collect (tip: make a dataframe out of a subset of the extra dict, then concat)
    r_test_srs = [fop.extra["rt_past_median"] for _, fop in fop_sr.items()]


@GLOBAL_XTT.track()
def export_all(params: Params, data: Data):
    """"""

    if not params.export:
        _LOGGER.warning(
            "`params.export` is False. No outputs will be exported to "
            "files."
        )
        return

    # ------------------------------------------------------------------
    # FluSight forecast file
    # ------------------------------------------------------------------
    data.flusight_df.to_csv("TEST_flusight.csv", index=False)  # TODO change file name

    # ------------------------------------------------------------------
    # Other forecast report files
    # ------------------------------------------------------------------
    data.main_quantile_df.to_csv("TEST_q_df.csv")

    # - Quantiles
    # - Categorical
    # - R(t) quantiles
    # - Preprocessed data


if __name__ == "__main__":
    main()


#


# =============================================================================
# ==================  OLD CODE VAULT  =========================================
# =============================================================================

