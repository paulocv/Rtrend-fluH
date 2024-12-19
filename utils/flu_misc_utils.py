import glob
import re
from pathlib import Path

import pandas as pd


def load_entire_flu_dataset(
        location_index_col="location_name",  # Previously: "location"
        horizon_col="horizon",  # Previously: "lag"
):
    """Auxiliary routine for loading FluSight truth data for multiple
    weeks. Mostly used in jupyter notebooks. Returns a single dataframe
    with all the data, as well as the "locations.csv" data frame.
    """
    # --- Load a set of target data files (from Flu), weekly,
    # =============================================

    locations_fpath = Path("aux_data/locations.csv")
    # truth_file_fmt = "hosp_data/season_2023-2024/truth_weekly_latest_{date}.csv"
    truth_file_fmt = "hosp_data/flusight_target_2023-2024/target-hospital-admissions_{date}.csv"

    truth_file_glob = truth_file_fmt.format(date="*")
    truth_file_re = truth_file_fmt.format(date="(\d{4}-\d{2}-\d{2})")

    # ---=====
    # Look up all files matching pattern, and the dates
    date_pattern = r"truth_weekly_latest_(\d{4}-\d{2}-\d{2}).csv"

    truth_fpath_dict = dict()
    for file_str in glob.glob(truth_file_glob):
        match = re.search(truth_file_re, file_str)
        if match:
            date_str = match.group(1)
            truth_fpath_dict[date_str] = Path(file_str)
            print(f"{date_str}: {pd.Timestamp(date_str).day_name()}")

    # Sort, since glob does not
    truth_fpath_dict = dict(sorted(truth_fpath_dict.items()))

    # ====
    # Load all files
    print("Loading all truth data:")
    truth_big_df = pd.concat(
        [pd.read_csv(fpath, parse_dates=["date"]) for fpath in truth_fpath_dict.values()],
        keys=[pd.Timestamp(date_str) for date_str in truth_fpath_dict.keys()],
        names=["report_date"],
        axis=0,
    )

    truth_big_df = (
        truth_big_df
        .reset_index()
        .set_index([location_index_col, "report_date", "date"], drop=True)
        # .drop(["Unnamed: 0", "level_1"], axis=1)
        .drop(["X", "level_1"], axis=1)
        # .rename({"value": "counts", "date": "reference_date"})  # Epinowcast columns
    )
    truth_big_df[horizon_col] = truth_big_df.index.get_level_values("report_date") - truth_big_df.index.get_level_values("date")

    # Locations file
    locations_df = pd.read_csv(locations_fpath)

    return truth_big_df, locations_df
