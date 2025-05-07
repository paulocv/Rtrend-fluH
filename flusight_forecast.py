"""
Runs the Rtrend forecasting method for the FluSight CDC initiative.
Taylored for the 2024-2025 FluSight forecast.
"""

import argparse
import datetime
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rtrend_forecast.reporting import (
    get_rtrend_logger,
    get_main_exectime_tracker, SUCCESS,
)
from rtrend_forecast.structs import RtData, get_tg_object, IncidenceData, TimeData
from rtrend_forecast.utils import map_parallel_or_sequential
from utils.flusight_tools import (
    FluSightDates,
    calc_rate_change_categorical_flusight,
    calc_horizon_from_dates,
    rate_change_i_to_name,
    calc_target_date_from_horizon,
)
from utils.forecast_operators_flusight_2024 import (
    # WEEKLEN,
    FluSight2024ForecastOperator,
)

from rtrend_forecast.visualization import (
    rotate_ax_labels,
    make_axes_seq,
    plot_fore_quantiles_as_area,
)
from utils.parsel_utils import (
    load_population_data,
    make_date_state_index,
    load_truth_cdc_simple,
    make_file_friendly_name,
    make_state_index,
    prepare_dict_for_yaml_export,
)
from utils.truth_data_structs import FluDailyTruthData, FluWeeklyTruthData

_CALL_TIME = datetime.datetime.now()


# DEFAULT PARAMETERS
# -------------
DEFAULT_PARAMS = dict(  # Parameters that go in the main Params class
    # Field names in the preprocessed data file
    filt_col_name="filtered",
    filtround_col_name="filtered_round",

    # OBS: default paths are defined in `parse_args()`

    # ncpus=5,
)

DEFAULT_PARAMS_GENERAL = dict(  # Parameters from the `general` dict
    pop_data_path="population_data/locations.csv",
    hosp_data_path="hosp_data/truth_latest.csv",
    forecast_days_ahead=2,  # Forecast evaluation ahead of the day_pres
    day_pres_id=-1,
    nsamples_to_us=1000,  # Number of trajectories
)

DEFAULT_PARAMS_RT_ESTIMATION = dict(  # Parameters from the `general` dict
    use_tmp=True
)

DEFAULT_PARAMS_POSTPROCESSING = dict(  # Defaults for `postprocessing`
    use_aggr_level="aggr",
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
    # raw_incid_sr: pd.Series  # Complete series of incidence
    fore_quantiles: pd.DataFrame  # Aggregated period forecasted quantiles
    inc: IncidenceData  #
    is_aggr: bool = True
    gran_dt = pd.Timedelta("1d")
    aggr_dt = pd.Timedelta("1w")
    logger: None
    state_name: str = "US"

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

    general:            dict
    preprocessing:      dict
    rt_estimation:      dict
    rt_synthesis:       dict
    inc_reconstruction: dict
    postprocessing:     dict

    # Program parameters
    filt_col_name: str
    filtround_col_name: str
    ncpus: int
    nsamples_to_us: int
    export: bool

    # Paths
    input_file: Path
    output_dir: Path
    flusight_output_file: Path


class Data:
    """Input, intermediate and output data.
    Input data is reserved for large structures. Smaller ones that
    can be manually set should be reserved as parameters.
    """
    dates: FluSightDates
    day_pres: pd.Timestamp
    day_last_available_in_truth: pd.Timestamp

    pop_df: pd.DataFrame  # Data frame with population demographic data
    count_rates: pd.DataFrame  # Count thresholds for rate forecast
    truth: FluWeeklyTruthData  # Object that holds daily truth data
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

    # General forecast reports
    main_quantile_df: pd.DataFrame  # Forecast quantiles for plotting
    main_rt_past_df: pd.DataFrame
    main_rt_fore_df: pd.DataFrame
    main_preproc_df: pd.DataFrame  # Some preprocessed data


#


#


# -------------------------------------------------------------------
# PROGRAM PROCEDURES
# -------------------------------------------------------------------

def parse_args(argv=None):
    """Interprets and stores the command line arguments."""

    parser = argparse.ArgumentParser(
        # usage="[[COMMAND LINE SIGNATURE (autogenerated)[]",
        description="Executes a forecast of influenza hospitalizations "
                    "for the FluSight CDC initiative.",
        # epilog="[[TEXT DISPLAYED AFTER ARGUMENTS DESCRIPTION]]",
    )

    # --- Positional paths
    # [[there are none]]

    #
    # Optional flags - SET TO NONE to have no effect
    # ----------------------------------------------------
    # --- Parameters file
    default = "inputs/flusight_params.yaml"
    parser.add_argument(
        "--input-file", "--params-file", "-i", type=Path,
        help=f"Path to the parameters file. Defaults to {default}",
        default=default,
    )

    # --- Regular outputs folder
    default = "outputs/latest/"
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        help="Path to the output directory, where general forecast "
             "results are exported. This is different from the FluSight"
             " output directory (where the FluSight formatted file "
             f"goes)! Defaults to \"{default}\"",
        default=default,
    )

    # --- Flusight outputs file path
    default = "forecast_out/latest.csv"
    parser.add_argument(
        "--flusight-output-file", "-f", type=Path,
        help="Path to the FluSight-formatted output file. This file "
             "is submittable to the FluSight forecast repository. "
             f"Defaults to \"{default}\".",
        default=default,
    )

    parser.add_argument(
        "--ncpus",
        help="Number of independent concurrent processes to run.",
        default=5, type=int,
    )

    parser.add_argument(
        "--now", type=pd.Timestamp,
        help="Timestamp to be considered as \"now\". Defaults to "
             "pandas.Timestamp.now().",
        default=pd.Timestamp.now(),  #"US/Eastern"),
    )

    parser.add_argument(
        "--export", action=argparse.BooleanOptionalAction,
        help="Whether the outputs (results of the preprocessing and "
             "R(t) estimation) should be exported to files.",
        default=True,
    )

    return parser.parse_args(argv)  # When all arguments are defined here
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

    for key, val in DEFAULT_PARAMS_POSTPROCESSING.items():
        if key not in params.postprocessing:
            params.postprocessing[key] = val

    # You can rename parameters here.
    if not params.export:
        _LOGGER.warn(
            " --- EXPORT SWITCH IS OFF --- No outputs will be produced.")

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
    data.truth = FluWeeklyTruthData(
        params.general["hosp_data_path"],
        all_state_names=data.all_state_names,
        state_name_to_id=data.state_name_to_id,
        pop_data_path=params.general["pop_data_path"],
    )

    # --- Checks regarding the available dates in the data
    data.day_last_available_in_truth = data.truth.get_date_index().max()

    # Check if all states have the most up-to-date data
    for state_name in data.truth.all_state_names:
        state_xs = data.truth.xs_state(state_name)
        state_max_date = state_xs.index.max()

        if state_max_date.date() < data.day_last_available_in_truth.date():
            # Warn about the state being "outdated"
            msg = ()
            _LOGGER.warn(
                f"State {state_name} seems to lack the latest "
                f"weeks in the data. Overall last week = "
                f"{data.day_last_available_in_truth.date().isoformat()}. "
                f"Last day for {state_name} = {state_max_date.date().isoformat()}"
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

def apply_forecast_exceptions(
        fop: FluSight2024ForecastOperator,
):
    """Handle exceptional conditions by changing forecast parameters."""

    # ALL STATES
    if fop.state_name == "Alabama":
        fop.raw_incid_sr[-1] += 30
        fop.sp["drift_coef"] *= 10.0 / 0.40
        fop.pp["denoise_cutoff"] *= 1.4
        fop.ep["scale_ref_inc"] *= 0.20
        fop.sp["initial_bias"] += -0.12

    if fop.state_name == "Alaska":
        fop.sp["drift_coef"] *= 8. / 0.08
        fop.ep["scale_ref_inc"] *= 0.11
        fop.sp["initial_bias"] -= 0.11

    if fop.state_name == "Arizona":
        # fop.raw_incid_sr.iloc[-1] *= 1.3
        fop.sp["drift_coef"] *= 25.0 / 0.25
        fop.ep["scale_ref_inc"] *= 0.10
        fop.sp["initial_bias"] += -0.04

    if fop.state_name == "Arkansas":
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["drift_coef"] *= 14.0 / 0.20
        fop.sp["initial_bias"] += 0.02
        fop.ep["scale_ref_inc"] *= 0.20

    if fop.state_name == "California":
        fop.sp["drift_coef"] *= 4.5 / 0.25
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["initial_bias"] += -0.04
        fop.ep["scale_ref_inc"] *= 0.25

    if fop.state_name == "Colorado":
        fop.sp["drift_coef"] *= 4.0 / 0.25
        fop.ep["scale_ref_inc"] *= 0.25
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Connecticut":
        fop.sp["drift_coef"] *= 5.0 / 0.25
        fop.raw_incid_sr[-1] += 1
        fop.sp["initial_bias"] += -0.09
        fop.ep["scale_ref_inc"] *= 0.25

    if fop.state_name == "Delaware":
        fop.sp["drift_coef"] *= 25.0 / 0.15
        # fop.pp["denoise_cutoff"] *= 0.75
        # fop.sp["initial_bias"] += -0.09
        fop.ep["scale_ref_inc"] *= 0.15

    if fop.state_name == "District of Columbia":
        # fop.sp["initial_bias"] += -0.05
        fop.pp["denoise_cutoff"] *= 1.2
        fop.sp["drift_coef"] *= 20.0 / 0.4
        fop.ep["scale_ref_inc"] *= 0.4
        fop.raw_incid_sr[-1:] += 1

    if fop.state_name == "Florida":
        # fop.raw_incid_sr[-2:] += 30
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["drift_coef"] *= 5.0 / 0.30
        fop.ep["scale_ref_inc"] *= 0.30
        fop.sp["initial_bias"] -= 0.03

    if fop.state_name == "Georgia":
        fop.sp["initial_bias"] += -0.10
        # fop.raw_incid_sr[-4:] += 5
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["drift_coef"] *= 7.0 / 0.25
        fop.ep["scale_ref_inc"] *= 0.25

    if fop.state_name == "Hawaii":
        fop.sp["drift_coef"] *= 5.5 / 0.5
        fop.ep["scale_ref_inc"] *= 0.5
        # fop.sp["initial_bias"] -= 0.05

    if fop.state_name == "Idaho":
        # fop.raw_incid_sr[-1] += 10
        fop.sp["drift_coef"] *= 13.0 / 0.20
        fop.ep["scale_ref_inc"] *= 0.20
        fop.sp["initial_bias"] -= 0.05

    if fop.state_name == "Illinois":
        fop.pp["denoise_cutoff"] *= 1.2
        fop.sp["initial_bias"] += -0.05
        fop.sp["drift_coef"] *= 6.5 / 0.28
        fop.ep["scale_ref_inc"] *= 0.28

    if fop.state_name == "Indiana":
        fop.ep["scale_ref_inc"] *= 0.4
        fop.sp["drift_coef"] *= 8.0 / 0.4
        fop.sp["initial_bias"] += -0.05

    if fop.state_name == "Iowa":
        fop.sp["initial_bias"] += -0.05
        fop.sp["drift_coef"] *= 24.0 / 0.2
        fop.ep["scale_ref_inc"] *= 0.2

    if fop.state_name == "Kansas":
        # fop.raw_incid_sr[-1] -= 20
        fop.sp["drift_coef"] *= 10.0 / 0.25
        fop.pp["denoise_cutoff"] *= 1.4
        fop.ep["scale_ref_inc"] *= 0.25
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Kentucky":
        fop.sp["drift_coef"] *= 6.0 / 0.40
        fop.pp["denoise_cutoff"] *= 1.3
        fop.ep["scale_ref_inc"] *= 0.40
        fop.sp["initial_bias"] -= 0.04

    if fop.state_name == "Louisiana":
        # fop.raw_incid_sr[-2] += 3
        fop.sp["drift_coef"] *= 5.0 / 0.4
        fop.ep["scale_ref_inc"] *= 0.4
        # fop.pp["denoise_cutoff"] = 0.50
        fop.sp["initial_bias"] += -0.05

    if fop.state_name == "Maine":
        fop.sp["drift_coef"] *= 15 / 0.40
        fop.ep["scale_ref_inc"] *= 0.40
        fop.sp["initial_bias"] += -0.02
        # fop.sp["synth_method"] = "rnd_normal"
        # fop.raw_incid_sr[-3:] += 1
        # fop.sp["sigma"] += 0.05

    if fop.state_name == "Maryland":
        fop.sp["drift_coef"] *= 2.9 / 0.2
        fop.ep["scale_ref_inc"] *= 0.2
        fop.pp["denoise_cutoff"] *= 1.3
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Massachusetts":
        fop.sp["drift_coef"] *= 14.0 / 0.20
        fop.pp["denoise_cutoff"] *= 1.2
        fop.ep["scale_ref_inc"] *= 0.20
        fop.sp["initial_bias"] += -0.12

    if fop.state_name == "Michigan":
        # fop.raw_incid_sr[-1] += 20
        fop.pp["denoise_cutoff"] *= 1.5
        fop.sp["drift_coef"] *= 7.0 / 0.4
        fop.sp["initial_bias"] -= 0.05
        # fop.pp["denoise_cutoff"] = 0.4
        fop.ep["scale_ref_inc"] *= 0.4

    if fop.state_name == "Minnesota":
        fop.sp["drift_coef"] *= 9. / 0.15
        # fop.pp["denoise_cutoff"] *= 1.4
        fop.ep["scale_ref_inc"] *= 0.15
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Mississippi":
        fop.sp["initial_bias"] += 0.10
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["drift_coef"] *= 24.0 / 0.20
        fop.ep["scale_ref_inc"] *= 0.20

    if fop.state_name == "Missouri":
        fop.pp["denoise_cutoff"] *= 1.2
        fop.sp["drift_coef"] *= 8.0 / 0.35
        fop.ep["scale_ref_inc"] *= 0.35
        fop.sp["initial_bias"] += -0.03

    if fop.state_name == "Montana":
        # fop.raw_incid_sr[-3:] += 1
        # fop.sp["synth_method"] = "rnd_normal"
        fop.sp["initial_bias"] -= 0.02
        fop.sp["drift_coef"] *= 9.0 / 0.60
        fop.ep["scale_ref_inc"] *= 0.60

    if fop.state_name == "Nebraska":
        fop.sp["drift_coef"] *= 28. / 0.2
        fop.ep["scale_ref_inc"] *= 0.2
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Nevada":
        fop.sp["drift_coef"] *= 8.0 / 0.24
        # fop.pp["denoise_cutoff"] = 0.4
        fop.ep["scale_ref_inc"] *= 0.24
        fop.sp["initial_bias"] += -0.025

    if fop.state_name == "New Hampshire":
        fop.sp["drift_coef"] *= 22.0 / 0.15
        fop.ep["scale_ref_inc"] *= 0.15
        fop.sp["initial_bias"] += -0.13

    if fop.state_name == "New Jersey":
        fop.sp["drift_coef"] *= 10.0 / 0.15
        # fop.raw_incid_sr[-1] += 4
        fop.sp["initial_bias"] += -0.12
        fop.ep["scale_ref_inc"] *= 0.15

    if fop.state_name == "New Mexico":
        fop.pp["denoise_cutoff"] *= 1.2
        fop.sp["drift_coef"] *= 4.0 / 0.45
        fop.ep["scale_ref_inc"] *= 0.45
        fop.sp["initial_bias"] += -0.05

    if fop.state_name == "New York":
        fop.sp["drift_coef"] *= 4.0 / 0.35
        fop.sp["initial_bias"] += -0.05
        fop.ep["scale_ref_inc"] *= 0.35

    if fop.state_name == "North Carolina":
        fop.sp["drift_coef"] *= 10.0 / 0.30
        fop.ep["scale_ref_inc"] *= 0.30
        fop.pp["denoise_cutoff"] *= 1.5
        # fop.sp["initial_bias"] += -0.10

    if fop.state_name == "North Dakota":
        # fop.raw_incid_sr[-3:] += 1
        fop.sp["drift_coef"] *= 10.0 / 0.20
        fop.ep["scale_ref_inc"] *= 0.20
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["initial_bias"] += -0.06
        # fop.sp["synth_method"] = "rnd_normal"
        # fop.sp["center"] = 1.1
        # fop.sp["sigma"] = 0.15

    if fop.state_name == "Ohio":
        fop.sp["drift_coef"] *= 8.5 / 0.30
        fop.ep["scale_ref_inc"] *= 0.30
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Oklahoma":
        fop.sp["drift_coef"] *= 14.0 / 0.24
        fop.pp["denoise_cutoff"] *= 1.5
        fop.ep["scale_ref_inc"] *= 0.24
        fop.sp["initial_bias"] -= 0.05

    if fop.state_name == "Oregon":
        fop.sp["drift_coef"] *= 8.0 / 0.25
        fop.ep["scale_ref_inc"] *= 0.25
        fop.sp["initial_bias"] -= 0.08

    if fop.state_name == "Pennsylvania":
        # fop.raw_incid_sr[-5:] -= 10
        fop.sp["drift_coef"] *= 3.0 / 0.4
        # fop.pp["denoise_cutoff"] -= 0.05
        fop.ep["scale_ref_inc"] *= 0.4
        fop.sp["initial_bias"] += -0.06

    if fop.state_name == "Rhode Island":
        # fop.raw_incid_sr[-4:] += 1
        fop.sp["drift_coef"] *= 17.0 / 0.10
        fop.ep["scale_ref_inc"] *= 0.10
        fop.sp["initial_bias"] -= 0.08

    if fop.state_name == "South Carolina":
        fop.raw_incid_sr[-2:] += 5
        fop.sp["drift_coef"] *= 9.0 / 0.2
        fop.ep["scale_ref_inc"] *= 0.2
        fop.pp["denoise_cutoff"] *= 1.4
        fop.sp["initial_bias"] += -0.05

    if fop.state_name == "South Dakota":
        fop.raw_incid_sr[-2:] += 2
        fop.sp["drift_coef"] *= 16.0 / 0.10
        fop.ep["scale_ref_inc"] *= 0.10
        # fop.sp["initial_bias"] += -0.10
        # fop.sp["synth_method"] = "rnd_normal"
        # fop.sp["center"] = 1.01
        # fop.sp["sigma"] = 0.20

    if fop.state_name == "Tennessee":
        fop.sp["drift_coef"] *= 7.5 / 0.25
        fop.ep["scale_ref_inc"] *= 0.25
        fop.sp["initial_bias"] += -0.07

    if fop.state_name == "Texas":
        fop.sp["drift_coef"] *= 8.0 / 0.2
        fop.pp["denoise_cutoff"] *= 1.4
        fop.ep["scale_ref_inc"] *= 0.2
        fop.sp["initial_bias"] += -0.04

    if fop.state_name == "Utah":
        # fop.raw_incid_sr[-1:] += -15
        # fop.pp["denoise_cutoff"] *= 1.2
        fop.sp["drift_coef"] *= 12.0 / 0.20
        fop.ep["scale_ref_inc"] *= 0.20
        fop.sp["initial_bias"] += -0.08

    if fop.state_name == "Vermont":
        fop.sp["drift_coef"] *= 7.0 / 0.25
        fop.pp["denoise_cutoff"] *= 1.3
        fop.sp["initial_bias"] += -0.03
        fop.ep["scale_ref_inc"] *= 0.25

    if fop.state_name == "Virginia":
        # fop.sp["bias"] = 0.005
        fop.pp["denoise_cutoff"] *= 1.3
        fop.sp["initial_bias"] += -0.05
        fop.sp["drift_coef"] *= 8.0 / 0.25
        fop.ep["scale_ref_inc"] *= 0.25

    if fop.state_name == "Washington":
        # fop.raw_incid_sr.iloc[-1] *= 0.7  # Yep...
        fop.sp["initial_bias"] += -0.05
        fop.pp["denoise_cutoff"] += 0.1
        fop.sp["drift_coef"] *= 11.0 / 0.3
        fop.ep["scale_ref_inc"] *= 0.3

    if fop.state_name == "West Virginia":
        fop.sp["drift_coef"] *= 15.0 / 0.30
        fop.ep["scale_ref_inc"] *= 0.30
        fop.pp["denoise_cutoff"] *= 1.5
        fop.sp["initial_bias"] += -0.05

    if fop.state_name == "Wisconsin":
        fop.sp["drift_coef"] *= 6.0 / 0.30
        fop.ep["scale_ref_inc"] *= 0.30
        fop.sp["initial_bias"] += -0.10

    if fop.state_name == "Wyoming":
        fop.sp["drift_coef"] *= 14.0 / 0.40
        fop.ep["scale_ref_inc"] *= 0.40
        fop.raw_incid_sr.iloc[-1] += 5.0
        # fop.sp["initial_bias"] -= 0.08

    if fop.state_name == "Puerto Rico":
        fop.sp["initial_bias"] += 0.05
        fop.pp["denoise_cutoff"] *= 1.2
        fop.sp["drift_coef"] *= 3.0 / 0.20
        fop.ep["scale_ref_inc"] *= 0.20


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
        params.general, data.truth.all_state_names)

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
    data.day_pres = day_pres = date_index[params.general["day_pres_id"]]# + pd.Timedelta("1d")
    #  ^  ^ For aggregated (weekly) input, day_pres is the day of the last report considered
    day_pres_str = day_pres.date().isoformat()

    # --- Unpacks structs to pass into subprocesses
    use_state_names = data.use_state_names
    state_name_to_id = data.state_name_to_id
    pop_df = data.pop_df

    pass

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
        fop = FluSight2024ForecastOperator(
            # --- Child class arguments
            day_pres, params.general["nperiods_main_roi"],  # Child class args
            population=pop_df.loc[state_name, "population"],
            state_name=state_name,
            # --- Base class arguments
            incid_series=state_series,  # Base class args
            params=params.__dict__,
            nperiods_fore=params.general["nperiods_fore"],
            name=f"{_sn}_{day_pres_str}",
            tg_past=tg,
        )
        _LOGGER.debug(f"Created forecast operator {fop.name}")

        # Exceptional forecast conditions
        # -------------------------------
        apply_forecast_exceptions(fop)
        # ===============================

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

    _LOGGER.log(SUCCESS, "Forecasts concluded for all states.")

    #

    # ----------------------------------------------
    # Calculate the categorical rate change forecast
    # ----------------------------------------------
    fop_sr = data.fop_sr.loc[data.fop_sr_valid_mask]  # Filter

    results = list()
    for _state_name, _fop in fop_sr.items():
        _fop: FluSight2024ForecastOperator
        # last_observed_date = _fop.raw_incid_sr.index.max()
        last_observed_date = _fop.raw_incid_sr.index.max() + _fop.gran_dt
        # last_observed_date = _fop.time.pg1  #  TODO: rethink what should be the date here.

        # Calculate rate changes
        df = calc_rate_change_categorical_flusight(
            _fop,
            count_rates=data.count_rates.loc[_state_name],
            dates=data.dates,
            # day_pres=data.day_pres,
            last_observed_date=last_observed_date,
            thresholds=_fop.op.get("categorical_thresholds", None)
        )

        results.append(df)

    data.rate_fore_df = pd.concat(
        results,
        keys=fop_sr.keys(),
        names=["location_name", "rate_change_id"]
    )
    data.rate_fore_df.columns.name = "horizon"

    _LOGGER.log(15, "Rate change forecast concluded.")


@GLOBAL_XTT.track()
def calculate_for_usa(params: Params, data: Data):

    data.usa = USAForecast()
    data.usa.inc = IncidenceData()
    data.usa.inc.raw_sr = data.truth.xs_state("US")
    # TODO : Best thing is to create all the infrastructure for USAForecast too, like `time`, `inc`, the ROI truth series, etc...
    data.usa.time = TimeData()
    if data.usa.is_aggr:  # TODO: This is a quick fix for the USA categorical. Please crop the ROI.
        data.usa.inc.past_aggr_sr = data.truth.xs_state("US")#.loc[:last_observed_date]  # TODO do this
        data.usa.time.pa1 = data.usa.inc.raw_sr.index.max()
    data.usa.logger = _LOGGER.getChild(f"US_{data.day_pres.date()}")
    # data.usa.raw_incid_sr = data.truth.xs_state("US")

    # Filter invalid forecasts
    # ------------------------
    fop_sr: pd.Series = data.fop_sr.loc[data.fop_sr_valid_mask]
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

    # ------------------
    # Quantile forecasts
    # ------------------
    data.usa.fore_quantiles = fop_sr.iloc[0].fore_quantiles.copy()
    for fop in fop_sr.iloc[1:]:
        if fop.fore_quantiles is not None:
            data.usa.fore_quantiles += fop.fore_quantiles

    # -----------------------------------
    # Categorical – Rate change forecasts
    # -----------------------------------

    # Take samples from all locations
    # -------------------------------
    rng = np.random.default_rng(20)  # Take a temporary generator
    df_list = list()

    for state_name, fop in fop_sr.items():
        fop: FluSight2024ForecastOperator
        nsamples_state, nsteps = fop.inc.fore_aggr_df.shape

        # --- Take samples
        i_samples = rng.integers(
            0, nsamples_state, size=params.general["nsamples_to_us"]
        )
        df = fop.inc.fore_aggr_df.loc[i_samples, :]  # Take sample trajectories
        # --- Sort at each date
        df: pd.DataFrame = df.apply(lambda x: x.sort_values().values, axis=0)
        df.reset_index(inplace=True, drop=True)

        df_list.append(df)

    multistate_df = pd.concat(
        df_list, keys=list(fop_sr.keys()),
        names=["location", "i_sample"],
    )

    # Aggregate spatially
    # -------------------
    data.usa.inc.fore_aggr_df = multistate_df.groupby("i_sample").sum()
    last_observed_date = data.usa.inc.raw_sr.index.max() + data.usa.gran_dt
    # last_observed_date = data.usa.inc.raw_sr.index.max()  # TODO: change here too

    # Calculate the categorical rate change forecasts
    # -----------------------------------------------
    df = calc_rate_change_categorical_flusight(
        data.usa,
        count_rates=data.count_rates.loc["US"],
        dates=data.dates,
        # day_pres=data.day_pres,
        last_observed_date=last_observed_date,
        thresholds=params.postprocessing.get("categorical_thresholds", None)
    )
    # Prepare the df to be "appended" to the other states df
    df.columns.name = "horizon"
    df.index = pd.MultiIndex.from_product(
        [["US"], df.index], names=["location_name", "rate_change_id"]
    )
    # df.index.name = "rate_change_id"
    df.name = "US"

    # data.rate_fore_df = pd.concat([data.rate_fore_df, df])
    data.rate_fore_df = pd.concat(
        [data.rate_fore_df, df])

    data.usa.logger.log(
        15, "USA quantile and rate change aggregations concluded.")


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
    print(f"- Last day in truth data .. {data.day_last_available_in_truth}")
    print(f"- Forecast day_present .... {data.day_pres}")
    print()

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

        fop: FluSight2024ForecastOperator | USAForecast
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
        d["reference_date"] = np.repeat(data.dates.ref.date(), size)
        d["location"] = np.repeat(data.state_name_to_id[state_name], size)
        d["output_type"] = np.repeat("quantile", size)
        d["target"] = np.repeat("wk inc flu hosp", size)

        d["value"] = stack.values.round()

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
    df = pd.concat(concat_df_list)

    # --- Select horizons
    use_horizons = params.general.get("export_horizons_quantiles", None)
    if use_horizons:
        flusight_quantiles_df = df.loc[df["horizon"].isin(use_horizons)]
    else:
        flusight_quantiles_df = df

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

    # --- Select horizons
    use_horizons = params.general.get("export_horizons_categorical", None)
    if use_horizons:
        df = df.loc[df["horizon"].isin(use_horizons)]

    flusight_rate_df = df

    # ------------------------------------------------------------
    # Overall FluSight export data postprocessing
    # ------------------------------------------------------------

    # Concatenate into a single dataframe
    df = pd.concat(
        [flusight_quantiles_df, flusight_rate_df],
        ignore_index=True
    )

    # Reorder columns to a better sequence
    df = df[
        ["reference_date", "target", "horizon", "target_end_date",
         "location", "output_type", "output_type_id", "value"]
    ]

    data.flusight_df = df

    print("Flusight exportable dataframe:")
    print(data.flusight_df)

    # ------------------------------------------------------------
    # Other forecast reports
    # ------------------------------------------------------------

    # Quantile forecast data
    # ----------------------
    q_dfs = [fop.fore_quantiles for _, fop in fop_sr.items()]
    keys = [state_name for state_name, _ in fop_sr.items()]

    q_dfs.append(data.usa.fore_quantiles)
    keys.append("US")

    data.main_quantile_df = pd.concat(
        q_dfs, keys=keys, names=["location_name", "quantile"])

    # R(t) statistics
    # ---------------
    rt_past_dfs = list()
    rt_fore_dfs = list()
    keys = list()
    for state_name, fop in fop_sr.items():
        fop: FluSight2024ForecastOperator
        # --- Select R(t) stats series from extras
        df_past = pd.DataFrame(
            {key: value for key, value in fop.extra.items()
             if "rt_past" in key}
        )
        df_fore = pd.DataFrame(
            {key: value for key, value in fop.extra.items()
             if "rt_fore" in key}
        )

        rt_past_dfs.append(df_past)
        rt_fore_dfs.append(df_fore)
        keys.append(state_name)

    data.main_rt_past_df = pd.concat(
        rt_past_dfs, keys=keys, names=["location_name", "date"],
    )
    data.main_rt_fore_df = pd.concat(
        rt_fore_dfs, keys=keys, names=["location_name", "date"],
    )

    # Filtered series
    # ---------------
    preproc_dfs = list()
    keys = list()
    for state_name, fop in fop_sr.items():
        fop: FluSight2024ForecastOperator

        state_preproc_df = pd.DataFrame(
            {"past_denoised": fop.extra["past_denoised_sr"]}
        )

        preproc_dfs.append(state_preproc_df)
        keys.append(state_name)

    data.main_preproc_df = pd.concat(
        preproc_dfs, keys=keys, names=["location_name", "date"],
    )


def export_metadata(path, params: Params, data: Data):
    """Creates and exports a yaml file with forecasting metadata."""
    # Convert some data types to be yaml-exportable
    out_dict = prepare_dict_for_yaml_export(params.__dict__, inplace=False)

    # Add more details
    out_dict["num_states"] = len(data.use_state_names)
    out_dict["num_all_states"] = len(data.all_state_names)
    out_dict["num_valid_forecasts"] = int(data.fop_sr_valid_mask.sum())

    # Export synthesis method used for each location
    out_dict["synth_method_used"] = {
        fop.state_name: fop.sp["synth_method"] for fop in data.fop_sr
    }

    out_dict["ref_date"] = data.dates.ref.date()
    out_dict["due_date"] = data.dates.due.isoformat()

    # Export
    with open(path, "w") as fp:
        fp.write(yaml.dump(out_dict))


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
    params.flusight_output_file.parent.mkdir(parents=True, exist_ok=True)
    data.flusight_df.to_csv(params.flusight_output_file, index=False)

    # ------------------------------------------------------------------
    # Other forecast report files
    # ------------------------------------------------------------------
    params.output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters file – Just copy as it is
    shutil.copyfile(
        params.input_file,
        params.output_dir.joinpath("parameters_bkp.yaml")
    )

    # Metadata
    path = params.output_dir.joinpath("metadata.yaml")
    export_metadata(path, params, data)

    # Quantile forecasts
    path = params.output_dir.joinpath("quantile_forecasts.csv")
    data.main_quantile_df.to_csv(path)

    # Reproduction number: past and forecast
    path = params.output_dir.joinpath("rt_past.csv")
    data.main_rt_past_df.to_csv(path)

    path = params.output_dir.joinpath("rt_fore.csv")
    data.main_rt_fore_df.to_csv(path)

    # Preprocessed data
    path = params.output_dir.joinpath("preproc.csv")
    data.main_preproc_df.to_csv(path)

    # ----- Suggested files to export --------
    # TODO implement more
    # - [x] Metadata, summary, as you please
    #     - now, ref_date, call_time, etc...
    # - [x] Quantiles
    # - [ ]  Categorical
    # - [x]  R(t) quantiles
    # - [x] Preprocessed data


if __name__ == "__main__":
    main()
