"""This module contains helper classes to handle truth data."""


import pandas as pd

from rtrend_forecast.postprocessing import (
    aggregate_in_periods,
    get_period_unique_id,
    get_aggr_label_from_id,
)
from utils.parsel_utils import (
    load_truth_cdc_simple,
    load_population_data
)


class FluWeeklyTruthData:
    """Load influenza hospitalization data in FluSight (hubverse) format.

    Minimum Expected Columns:
    - date       (day of the reported incidence)
    - location  (FIPS code or US)
    - location_name  (Name of location)
    - value  (incidence of hospitalizations)
    """

    raw_data: pd.DataFrame
    aggr_ref_date: pd.Timestamp  # Reference date for weekly aggregation
    _date_index: pd.DatetimeIndex  # Sorted index of (weekly) time stamps available

    def __init__(
            self, truth_data,
            all_state_names=None,
            state_name_to_id=None,
            pop_data_path="aux_data/locations.csv",
    ):
        """
        Parameters
        ----------
        truth_data : Union(pd.DataFrame, str, Path)
            Raw truth data – either as a file name/path or as a loaded
            Pandas data frame.
        all_state_names : list
            List with all names of states. If not informed, it's
            extracted from the truth file.
        state_name_to_id : dict
            Dictionary keyed by state names and containing their
            respective FIPS id. If not informed, it is calculated from
            the population data file (in `pop_data_path`).
        pop_data_path : Union(str, Path)
            If the `state_name_to_id` arguments are
            not provided, loads the population metadata from this path.
        """
        # Load raw truth data
        if isinstance(truth_data, pd.DataFrame):
            self.raw_data = truth_data
        else:  # Import it!
            # self.raw_data = load_truth_cdc_simple(truth_data)
            self.raw_data = pd.read_csv(
                truth_data,
                index_col=("date", "location"),
                parse_dates=["date"])

        # Builds auxiliary structs
        # -------------------------------------------
        # --- Conversion dict from state name to FIPS id
        if state_name_to_id is None:
            pop_dict = load_population_data(pop_data_path)
            state_name_to_id = pop_dict["state_name_to_id"]
            all_state_names = pop_dict["all_state_names"]

        # --- List of names of all states
        if all_state_names is None:
            all_state_names = self.raw_data["location_name"].unique()

        self.all_state_names: list = all_state_names
        self.state_name_to_id: dict = state_name_to_id
        self.state_id_to_name = {
            val: key for key, val in state_name_to_id.items()
        }

        # ...

    def xs_state(self, state_name):
        """Gets the truth data for a single state, given by its name."""
        return self.raw_data.xs(
            self.state_name_to_id[state_name],
            level="location")["value"]

    def get_date_index(self):
        """Returns a sorted sequence of all dates present in the dataset.
        """
        if not hasattr(self, "_date_index"):
            self._date_index = (
                self.raw_data.index.get_level_values(
                    "date").unique().sort_values()
            )

        return self._date_index


class FluDailyTruthData:
    """"""

    raw_data: pd.DataFrame
    _weekly_raw_data: pd.DataFrame
    aggr_ref_date: pd.Timestamp  # Reference date for weekly aggregation

    def __init__(
            self, truth_data,
            all_state_names=None,
            state_name_to_id=None,
            pop_data_path="aux_data/locations.csv",
    ):
        """
        Parameters
        ----------
        truth_data : Union(pd.DataFrame, str, Path)
            Raw truth data – either as a file name/path or as a loaded
            Pandas data frame.
        all_state_names : list
            List with all names of states. If not informed, it's
            extracted from the truth file.
        state_name_to_id : dict
            Dictionary keyed by state names and containing their
            respective FIPS id. If not informed, it is calculated from
            the population data file (in `pop_data_path`).
        pop_data_path : Union(str, Path)
            If the `state_name_to_id` arguments are
            not provided, loads the population metadata from this path.
        """
        # Load raw truth data
        if isinstance(truth_data, pd.DataFrame):
            self.raw_data = truth_data
        else:  # Import it!
            self.raw_data = load_truth_cdc_simple(truth_data)

        # Builds auxiliary structs
        # -------------------------------------------
        # --- Conversion dict from state name to FIPS id
        if state_name_to_id is None:
            pop_dict = load_population_data(pop_data_path)
            state_name_to_id = pop_dict["state_name_to_id"]
            all_state_names = pop_dict["all_state_names"]

        # --- List of names of all states
        if all_state_names is None:
            all_state_names = self.raw_data["location_name"].unique()

        self.all_state_names: list = all_state_names
        self.state_name_to_id: dict = state_name_to_id
        self.state_id_to_name = {
            val: key for key, val in state_name_to_id.items()
        }

        # ...

    def xs_state(self, state_name):
        """Gets the truth data for a single state, given by its name."""
        return self.raw_data.xs(
            self.state_name_to_id[state_name],
            level="location")["value"]

    def xs_state_weekly(self, state_name, ref_date=None):
        """Gets the weekly aggregated truth data for a single state,
        given by its name.
        """
        if ref_date is None:
            if not hasattr(self, "aggr_ref_date"):
                raise AttributeError(
                    "Hey, to get a weekly aggregated xs, you first need "
                    "to specify the reference date for aggregation "
                    "(as `ref_date`).")
            ref_date = self.aggr_ref_date

        weekly_truth = self.get_weekly_aggregated(ref_date)

        return weekly_truth.xs(
            self.state_name_to_id[state_name],
            level="location")["value"]

    def get_weekly_aggregated(self, ref_date):
        """Smart getter to calculate and return an aggregated version
        of the raw data.
        """
        if (not hasattr(self, "_weekly_raw_data")  # aggr not defined
            or not hasattr(self, "aggr_ref_date")  # date not defined
            or ref_date != self.aggr_ref_date  # Ref date changed
        ):
            # Do recalculate the aggregation

            # Reference date manipulation
            # ---------------------------
            self.aggr_ref_date = pd.Timestamp(ref_date)
            # SUBTRACT A DAY to comply with Rtrend one-past convention
            start_date = self.aggr_ref_date  # - pd.Timedelta("1d")
            week_dt = pd.Timedelta("1w")
            weeklen = 7

            # Aggregation
            # -----------
            grouped = self.raw_data.groupby(
                lambda idx: (
                    get_period_unique_id(idx[0], week_dt, start_date),
                    idx[1])
            )
            # Sum and take number of points in each group
            group_sum = grouped.sum()

            # Postprocessing
            # --------------

            # # OPTIONAL: remove incomplete weeks
            # group_sizes = grouped.size()
            # group_sum = group_sum.loc[group_sizes == weeklen]

            # Convert the index back to (date, state_id) format
            new_index = pd.MultiIndex.from_tuples(
                group_sum.index.map(
                    lambda idx: (get_aggr_label_from_id(
                        idx[0], week_dt, start_date),
                                 idx[1])
                ),
                names=self.raw_data.index.names)

            group_sum.index = new_index

            # Store and return
            self._weekly_raw_data = group_sum
            return group_sum

        else:  # No recalculation needed
            return self._weekly_raw_data

