"""Utility tools specific to the FluSight forecast."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.forecast_operators_flusight_2024 import FluSight2024ForecastOperator
from utils.forecast_operators_flusight_2023 import FluSight2023ForecastOperator
from utils.parsel_utils import get_next_weekly_timestamp


# ----------------------------------------------------------------------
# FLUSIGHT CLASSES
# ----------------------------------------------------------------------

rate_change_i_to_name = {
    -2: "large_decrease",
    -1: "decrease",
    0: "stable",
    1: "increase",
    2: "large_increase",
}


class FluSightDates:
    """Stores dates related to the FluSight 2023-2024 season foreacst.

    Attributes
    ----------
    now : pd.Timestamp
        Date and time considered as "now" for the forecast. It is used
        to determine the (next) due date and reference date.
    due : pd.Timestamp
        Next forecast submission due date, based on `now`. It takes the
        next weekly stamp based on `due_example`. E.g., if `due_example`
         is "2023-11-15  23:00:00" (11pm of a Wednesday), then `due` is
         set to the next Wednesday 11pm after `now`.
    ref : pd.Timestamp
        Current reference date for forecasts, based on `due`. It takes
        the next weekly stamp based on `ref_example`. E.g.,
        if `ref_example` is "2023-11-04" (00am of a Saturday),
        then `ref` is set to the next Saturday 00am after `due`.
    """
    now: pd.Timestamp
    due: pd.Timestamp
    ref: pd.Timestamp

    def __init__(
            self,
            now: pd.Timestamp,
            due_example=None,
            ref_example=None,
    ):
        """"""
        if due_example is None:
            due_example = pd.Timestamp("2023-11-15  23:00:00")
        if ref_example is None:
            ref_example = pd.Timestamp("2023-11-04")

        self._due_example = pd.Timestamp(due_example)
        self._ref_example = pd.Timestamp(ref_example)

        self.set_now(now)

    def set_now(self, now: pd.Timestamp):
        """Recalculate other dates based on a given `now` date."""
        # due_example = pd.Timestamp("2023-11-15  23:00:00")
        # ref_example = pd.Timestamp("2023-11-04")

        self.now = pd.Timestamp(now)

        self.due = (
            get_next_weekly_timestamp(self._due_example, now=now)
        )

        self.ref = (
            get_next_weekly_timestamp(self._ref_example, now=self.due)
        )


class FluSightGeneralOutputs:
    """Class that loads and collects outputs from a
    `flusight_forecast.py` run. These results are not formatted for
    FluSight submission.
    """

    q_df: pd.DataFrame

    def __init__(self, forecast_dir: Path):
        self.forecast_dir = Path(forecast_dir)

        # ---
        self.meta_dict = self.import_metadata(
            self.forecast_dir.joinpath("metadata.yaml")
        )
        self.q_df = self.import_quantile_forecasts(
            self.forecast_dir.joinpath("quantile_forecasts.csv")
        )
        self.rt_past_df = self.import_rt_stats_file(
            self.forecast_dir.joinpath("rt_past.csv")
        )
        self.rt_fore_df = self.import_rt_stats_file(
            self.forecast_dir.joinpath("rt_fore.csv")
        )

    @staticmethod
    def import_quantile_forecasts(fname: Path):
        """Imports the quantile-based forecast file (general outputs).
        """
        df = pd.read_csv(fname, index_col=[0, 1])
        df.columns = pd.DatetimeIndex(df.columns)  # Parse dates
        return df

    @staticmethod
    def import_rt_stats_file(fname: Path):
        return pd.read_csv(fname, index_col=[0, 1])

    @staticmethod
    def import_metadata(fname: Path):
        d = None
        try:
            with open(fname, "r") as fp:
                d = yaml.load(fp, yaml.Loader)
        except FileNotFoundError:
            msg = ("Hey, forecast outputs has no metadata file: "
                   f"{fname.name}.\nself.meta_dict was set to None.")
            # warnings.warn(msg)  # Why not working?
            print(msg)
        finally:
            return d


class FluSight2023Fore:
    """Data bunch to hold raw and preprocessed data from a FluSight
    forecast hubverse file for the 2023-2024 season
    (single reference date).
    """
    raw_df: pd.DataFrame
    quantiles_df: pd.DataFrame  # df.loc[(loc_id, date), q_val] = q_forecast
    ratechange_df: pd.DataFrame  # df.loc[(loc_id, date), category] = probab

    ref_date: pd.Timestamp

    def __init__(self, fname):
        """Load a FluSight file and preprocess the data."""
        # --- Read file. Keep the default index.
        self.raw_df = pd.read_csv(fname, parse_dates=[0, 3])

        # --- Extract reference date and check if it's unique
        ref_dates = self.raw_df["reference_date"].unique()
        if ref_dates.shape[0] != 1:
            raise ValueError(
                "Hey, the forecast file can only have one reference date,"
                f" but {ref_dates.shape[0] != 1} unique values were found."
            )
        self.ref_date = pd.Timestamp(ref_dates[0])

        # Extract the forecast quantiles
        # ------------------------------
        # Select data
        df: pd.DataFrame = self.raw_df.loc[
            self.raw_df["target"] == "wk inc flu hosp"]

        # Remove redundant labels and make an index with the remaining ones
        df = df.drop(["reference_date", "output_type", "target", "horizon"], axis=1)
        df = df.set_index(["location", "target_end_date", "output_type_id"])

        # Make a table of quantile forecasts
        df = df["value"].unstack(level="output_type_id")
        df.columns.name = "quantile"

        self.quantiles_df = df

        # Extract the categorical rate change forecast
        # --------------------------------------------
        df: pd.DataFrame = self.raw_df.loc[
            self.raw_df["target"] == "wk flu hosp rate change"]

        # Remove redundant labels
        df = df.drop(["reference_date", "output_type", "target", "horizon"], axis=1)
        df = df.set_index(["location", "target_end_date", "output_type_id"])

        # Make a table of quantile forecasts
        df = df["value"].unstack(level="output_type_id")
        df = df.reindex(rate_change_i_to_name.values(), axis=1)
        df.columns.name = "category"

        self.ratechange_df = df


def calc_horizon_from_dates(date_seq, ref_date):
    return (date_seq - ref_date) // pd.Timedelta("1w")


def calc_target_date_from_horizon(horizon_seq, ref_date):
    return ref_date + horizon_seq * pd.Timedelta("1w")


def calc_rate_change_categorical_flusight(
        fop: [FluSight2024ForecastOperator, FluSight2023ForecastOperator],
        count_rates: pd.Series,
        dates: FluSightDates,
        # day_pres: pd.Timestamp,
        last_observed_date: pd.Timestamp,
        use_horizons=None,
        stable_abs_counts=10,  # Absolute counts under which always stable
):
    """Calculate the rate change categorical forecasts for FluSight.

    NOTE: this method may be generalized to the base class if demand
     is identified.


    Technical readme: https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-output#rate-trend-forecast-specifications
    """
    fop.logger.debug("Calculating rate change forecasts.")

    if use_horizons is None:
        # use_horizons = [-1, 0, 1, 2, 3]  # Default FluSight 2023/2024
        use_horizons = [0, 1, 2, 3]  # Default FluSight 2023/2024  # TODO: switch to this and test!!

    # ----------------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------------
    # --- Expands the threshold series to the negative side
    neg_count_rates = -count_rates.iloc[::-1]
    neg_count_rates.index = neg_count_rates.index.map(lambda x: -x)
    count_rates = pd.concat([neg_count_rates, count_rates])

    # ----------------------------------------------------------------
    # Calculate the baseline week incidence (either past or nowcast)
    # ----------------------------------------------------------------
    # --- Edges of baseline from granular data

    # -()- Original: baseline = 2w before reference
    baseline_gran_end = dates.ref - pd.Timedelta("2w")
    baseline_gran_start = baseline_gran_end - pd.Timedelta("1w")
    # last_observed_date = day_pres - pd.Timedelta("1d")  # Last observed

    # # -()- FluSight Update 2023-11-03  TODO: switch to this and test
    # baseline_gran_end = dates.ref - pd.Timedelta("1w")
    # baseline_gran_start = baseline_gran_end - pd.Timedelta("1w")
    # # last_observed_date = day_pres - pd.Timedelta("1d")  # Last observed

    # -\\-

    if baseline_gran_end < last_observed_date:
        fop.logger.debug(
            "Rate change: the baseline is made of observed"
            " data (case 0)")
        # The baseline period is entirely observed
        # --- Observed chunk only
        baseline = (
            int(fop.inc.raw_sr.loc[baseline_gran_start:baseline_gran_end].sum())
        )  # INTEGER

    elif baseline_gran_start <= last_observed_date <= baseline_gran_end:
        fop.logger.debug("Rate change: the baseline is a mix between "
                         "observed and nowcast data (case 1)")
        # The baseline period contains both known and nowcast data
        # THIS IS THE EXPECTED CASE for FluSight 2023/2024

        # --- Nowcasted chunk
        # WE ASSUME that the first forecasted week is the needed nowcast
        baseline = fop.inc.fore_aggr_df.iloc[:, 0]  # First column

        # --- Observed chunk
        baseline += (
            int(fop.inc.raw_sr.loc[baseline_gran_start:baseline_gran_end].sum())
        )

    else:  # last_observed_date < baseline_gran_start
        fop.logger.debug(
            "Rate change: the baseline is entirely nowcasted (case 0)")
        # The baseline period is solely nowcasted
        # --- Nowcast chunk only
        # WE ASSUME that the first forecasted week is the needed nowcast
        baseline = fop.inc.fore_aggr_df.iloc[:, 0]  # First column

    # ----------------------------------------------------------------
    # Calculate the count changes with respect to the baseline
    # ----------------------------------------------------------------

    # Subtract all trajectories from baseline, mathing rows (samples)
    count_changes_df = fop.inc.fore_aggr_df.sub(baseline, axis=0).round().astype(int)

    # Change column names to horizon number
    # Subtract one day because Rtrend index by one-past, and FluSight indexes by the day
    count_changes_df.columns = calc_horizon_from_dates(
        count_changes_df.columns - pd.Timedelta("1d"), ref_date=dates.ref,
    )
    count_changes_df.columns.name = "horizon"

    # ------------------------------------------------------------------
    # Calculate rate change bins for each horizon
    # ------------------------------------------------------------------

    thresholds = {  # Signature: d[i_horizon] = (large_r, stable_r)
        -1: (2.0, 1.0),
        0: (3.0, 1.0),
        1: (4.0, 2.0),
        2: (5.0, 2.5),
        3: (5.0, 2.5),
    }
    # Array with category ids
    bins = np.array(list(rate_change_i_to_name.keys()), dtype=int)
    bins.sort()

    results_dict = dict()  # d[horizon] = bin_counts_series

    for horizon in use_horizons:
        # --- Checks if horizon was calculated and select it
        if horizon not in count_changes_df:
            fop.logger.warn(
                f"Horizon {horizon} required for rate change forecast "
                f"was not present in data.")
            # TODO : fillna and continue
            continue

        if horizon not in thresholds:
            fop.logger.debug(
                f"Horizon {horizon} skipped: rate changes undefined.")
            continue  # Does not calculate for undefined horizons

        count_changes_sr = count_changes_df[horizon]

        # Makes mask data frames for values...:
        cc_array = count_changes_sr.to_numpy()[:, np.newaxis]  # Values
        cr_array = count_rates.to_numpy()  # Thresholds

        #   - "greater than" (gt) each threshold
        gt_mask = pd.DataFrame(cc_array > cr_array, columns=count_rates.index)

        #   - "greater than or equal (ge) each threshold
        ge_mask = pd.DataFrame(cc_array >= cr_array, columns=count_rates.index)

        #   - Less than 10 in absolute counts
        stable_mask = count_changes_sr.abs() < stable_abs_counts

        # Classify by horizon
        def calc_for_horizon(large_r, stable_r):
            """
            large_r = Threshold betweeen large and normal increase/decrease
            stable_r = Threshold between stable and increase/decrease
            stable_abs = Absolute counts under which it's always stable
            """
            # Initialize
            # ----------
            # d = dict()  # Used to create the result dataframe
            # Result: res[i_sample] = rate_change_id (from -2  to 2)
            # Starts with -99, which indicates "unassigned value"
            res = pd.Series(
                np.repeat(-99, count_changes_sr.shape[0]),
                index=count_changes_sr.index
            )

            # Calculate all categories
            # ------------------------

            # Applies rate-based criteria
            res[~gt_mask[-large_r]] = -2  # Large decrease
            res[gt_mask[-large_r] & ~gt_mask[-stable_r]] = -1  # Decrease
            res[gt_mask[-stable_r] & ~ge_mask[stable_r]] = 0  # Stable
            res[ge_mask[stable_r] & ~ge_mask[large_r]] = 1  # Increase
            res[ge_mask[large_r]] = 2  # Large increase

            # Applies absolute counts criterion
            res[stable_mask] = 0  # Stable

            # Counts each category
            # --------------------
            # Initialize counts as 0, to ensure that all bins appear
            bin_counts = pd.Series(np.zeros_like(bins), index=bins)
            bin_counts = bin_counts.add(
                res.value_counts(),   # HERE the count is performed
                fill_value=0).astype(int)

            # TODO HERE: CHECK IF THERE WERE MISSED VALUES (silently ignored for now)

            return bin_counts / bin_counts.sum()

        # Run based on the horizon-specific thresholds
        results_dict[horizon] = (
            calc_for_horizon(*thresholds[horizon])
        )

    # ------------------------------------------------------------------
    # Combines all horizons into a dataframe
    # ------------------------------------------------------------------
    df = pd.DataFrame(results_dict)

    return df
