"""Concentrates operations that are performed after a forecast
iteration was chosen as final.

Devlist:
- [ ] Checks formatting of the flusight file.
- [ ] (Optional) copy-renames the flusight file for submission.
- [ ] Backups the full forecast outputs folder.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from rtrend_forecast.reporting import get_rtrend_logger, SUCCESS
from utils.flusight_tools import (
    FluSight2023Fore,
    FluSightGeneralOutputs,
    FluSightDates,
)
from utils.truth_data_structs import FluDailyTruthData, FluWeeklyTruthData

_LOGGER = get_rtrend_logger().getChild(__name__)


DEFAULT_PARAMS = {
    "flusight_file": "forecast_out/latest.csv",
    "fore_outputs_dir": "outputs/latest/",
    "truth_file": "hosp_data/truth_latest.csv",
    "rename": True,
}


def main():

    args = parse_args()
    params = build_params(args)
    data = import_data(params)

    backup_files(params, data)


class Params:
    """"""
    flusight_file: Path
    fore_outputs_dir: Path
    truth_file: Path

    now: pd.Timestamp

    rename: bool


class Data:
    """"""
    gfobj: FluSightGeneralOutputs  # General forecasts
    fsobj: FluSight2023Fore  # Flusight file data
    truth: FluDailyTruthData  # Truth file data
    dates: FluSightDates  # Reference, now and deadline dates


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flusight-file",
        help="Path to the FluSight forecast output file to check.",
        default=DEFAULT_PARAMS["flusight_file"],
        type=Path,
    )

    parser.add_argument(
        "--fore-outputs-dir",
        help="Path to the complete outputs directory, containing "
             "forecast metadata, quantile forecasts and R(t).",
        default=DEFAULT_PARAMS["fore_outputs_dir"],
        type=Path,
    )

    parser.add_argument(
        "-t", "--truth-file",
        help="(Optional) Alternative path to the truth data file. "
             f"Default is {DEFAULT_PARAMS['truth_file']}.",
        default=DEFAULT_PARAMS['truth_file'],
        type=Path,
    )

    parser.add_argument(
        "-r", "--rename",
        help="If specified, backups the FluSight forecast file and the "
             "complete output folder using the reference date.",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PARAMS["rename"],
    )

    parser.add_argument(
        "--now", type=pd.Timestamp,
        help="Timestamp to be considered as \"now\". Defaults to "
             "pandas.Timestamp.now().",
        default=pd.Timestamp.now(),  #"US/Eastern"),
    )

    return parser.parse_args()


def build_params(args) -> Params:

    params = Params()

    valid_args_dict = {key: val for key, val in args.__dict__.items()
                       if val is not None}
    params.__dict__.update(
        valid_args_dict
    )

    return params


def import_data(params) -> Data:
    data = Data()

    # Import forecast and truth data
    data.fsobj = FluSight2023Fore(params.flusight_file)
    data.gfobj = FluSightGeneralOutputs(params.fore_outputs_dir)
    data.truth = FluWeeklyTruthData(params.truth_file)

    # Construct object that holds reference, deadline and now dates
    data.dates = FluSightDates(params.now, )  # TODO: get from meta?

    return data


# ==========================


# ==========================


# ==========================

def backup_files(params: Params, data: Data):
    """"""
    if not params.rename:
        return

    _LOGGER.info("Backing up forecast files...")

    # --- Copy the FluSight forecast file
    new_fname = params.flusight_file.parent.joinpath(
        f"{data.dates.ref.date()}-CEPH-Rtrend_fluH.csv"
    )
    shutil.copy2(
        src=params.flusight_file,
        dst=new_fname,
    )
    _LOGGER.info(
        f"Copied FluSight file to {new_fname.name}."
    )

    # --- Copy the general outputs
    new_dir = params.fore_outputs_dir.parent.joinpath(
        f"flusight_{data.dates.ref.date()}"
    )
    shutil.copytree(
        src=params.fore_outputs_dir,
        dst=new_dir,
        dirs_exist_ok=True,
    )
    _LOGGER.info(
        f"Copied general outputs to {new_dir.name}."
    )


if __name__ == '__main__':
    main()
