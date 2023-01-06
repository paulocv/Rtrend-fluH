"""
Perform checks and plot data from a forecast csv file.
The file is expected to be in the CDC format.

"""
import os

import colorama
import datetime as dtime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from colorama import Fore, Back, Style
from datetime import datetime, timedelta

import rtrend_tools.data_io as dio
import rtrend_tools.visualization as vis

from rtrend_tools.cdc_params import WEEKDAY_TGT, NUM_WEEKS_FORE, REF_DEADLINE, CDC_QUANTILES_SEQ, NUM_QUANTILES, \
    get_next_flu_deadline_from
from rtrend_tools.forecast_structs import CDCDataBunch
from toolbox.plot_tools import make_axes_seq


# colorama.just_fix_windows_console()  # Requires colorama 0.4.6. the legacy colorama.init() is unsafe.


def main():

    # Hospitalization truth data
    nweeks_past = 6  # Defines the ROI to plot past data

    # ------------------------------------------------------------------------------------------------------------------

    cdc, fore_df = import_data()

    if fore_df is None:
        print("Quitting the program...")
        return

    # WATCH
    print(fore_df)

    check_forecast_data(cdc, fore_df)
    make_plots_for_all(cdc, fore_df, nweeks_past=nweeks_past)
    print_final_remarks()


def import_data(hosp_fname="hosp_data/truth-Incident Hospitalizations.csv"):
    # --- Read forecast file name from data
    try:
        fore_fname = sys.argv[1]
    except IndexError:
        sys.stderr.write("\nHey, missing 1 argument.\nUsage: python check_forecast_file.py [forecast_file_path]\n")
        return None, None

    # --- Read each file
    print("Reading files...")
    cdc: CDCDataBunch = dio.load_cdc_truth_data(hosp_fname)
    fore_df: pd.DataFrame = dio.load_forecast_cdc(fore_fname)

    return cdc, fore_df


def check_forecast_data(cdc, fore_df):
    """
    Performs data consistency checks, assuming that the file will be uploaded as this week's forecast.

    """
    print("\nPerforming data checks")
    print("----------------------")

    # --- Date labels retrieval
    # Get dates in file
    dates = fore_df["target_end_date"].unique()
    first_target_day = pd.to_datetime(dates.min()).date()

    # Get datetime stamps for today (now), next deadline and next forecast target date.
    now = datetime.today()  # - timedelta(weeks=1)
    next_deadline = get_next_flu_deadline_from(now)
    expected_target_day = next_deadline.date() + timedelta(days=(WEEKDAY_TGT - next_deadline.weekday()) % 7)
    #   ^ Get the next Saturday (WEEKDAY_TGT) after next deadline.

    print(f"Now:                      {now}")
    print(f"Next submission deadline: {next_deadline}")

    # --- Checks number of time labels
    if dates.shape[0] != NUM_WEEKS_FORE:
        _data_warn(f"There are {dates.shape[0]} different date labels in 'target_end_date' column, "
                   f"but {NUM_WEEKS_FORE} were expected.")

    # --- Checks correctness of target based on next submission deadline
    if first_target_day - expected_target_day != timedelta(0):
        _data_warn(f"The first target end date in file ({first_target_day}) does not match the expected day for CDC "
                   f"({expected_target_day}), based on the next submission deadline.")

    # ---


def _data_warn(text, end="\n"):
    """Prints a formatted warning message"""
    sys.stderr.write(Fore.YELLOW + "[Data warning] ")
    sys.stderr.write(text)
    sys.stderr.write(Style.RESET_ALL)
    sys.stderr.write(end)
    _data_warn.num_warns += 1


_data_warn.num_warns = 0  # Static counter for the number of warnings raised.


def _reconstruct_quantiles(df: pd.DataFrame, state_name, locid):

    weekly_quantiles = np.full((NUM_QUANTILES, NUM_WEEKS_FORE), np.nan, dtype=float)

    for i_q, q in enumerate(CDC_QUANTILES_SEQ):  # Collect against CDC quantiles. If not found, raise warning.

        xs = df.loc[df["quantile"] == q]

        if xs.shape[0] == 0:  # Quantile was not found.
            _data_warn(f"[{state_name}, {locid}] Required quantile q = {q} not found in data.")
            continue  # Leave the quantile unfilled

        weekly_quantiles[i_q, :] = xs["value"]

    return weekly_quantiles


def make_plots_for_all(cdc: CDCDataBunch, fore_df: pd.DataFrame, nweeks_past=6,
                       plot_fname="tmp_figs/ct_states_reconstructed.pdf"):
    """ """

    print("\nRecreating plots for the data")
    print("-----------------------------")

    # Features from forecast data
    locid_gp = fore_df.groupby("location")
    fore_dates = np.sort(fore_df["target_end_date"].unique())
    num_locs = len(locid_gp)

    # Features from past data
    week_roi_start = cdc.data_time_labels[-nweeks_past]
    # -()- From CDC upated truth data
    # week_pres = cdc.data_time_labels[-1]
    # -()- From our forecast file
    week_pres = fore_dates[0]

    # TMP VARS without loop
    # fig, ax = plt.subplots()
    fig, axes = make_axes_seq(num_locs, max_cols=3)

    # Sequential plotting loop
    for i_ax, (locid, df) in enumerate(locid_gp):
        ax = axes[i_ax]
        state_name = cdc.to_loc_name[locid]
        weekly_quantiles = _reconstruct_quantiles(df, state_name, locid)

        factual_ct = cdc.xs_state(state_name).loc[week_roi_start:]["value"]
        last_day = week_pres - pd.Timedelta(1, "w")
        last_val = factual_ct.loc[last_day]

        # Plot function
        vis.plot_ct_past_and_fore(ax, fore_dates, weekly_quantiles, factual_ct, CDC_QUANTILES_SEQ, state_name, i_ax=0,
                                  num_quantiles=NUM_QUANTILES, insert_point=(last_day, last_val)
                                  )

        print(f"  [{state_name}] ({i_ax + 1} of {num_locs})  |", end="")
        sys.stdout.flush()

    # Final operations, export the plot
    fig.tight_layout()
    print()
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)
    fig.savefig(plot_fname)
    print(f"Plots done. Check file '{plot_fname}'.")

    # # Prompt to open the file
    # input()
    # print("Hello")
    # print("World")

    return fig, axes


def print_final_remarks():
    """Report the number of warnings, if any."""
    print()
    if _data_warn.num_warns == 0:
        sys.stderr.write(Fore.GREEN + "No warnings raised, the data is in good shape!" + Style.RESET_ALL + '\n')
    else:
        sys.stderr.write(Fore.YELLOW + f"There were {_data_warn.num_warns} warnings, please inspect the data file."
                         + Style.RESET_ALL + '\n')


if __name__ == "__main__":
    main()
