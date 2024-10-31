"""
General utilities that are common to multiple pipelines of the parameter
selection project.
"""
import os
from pathlib import Path

import pandas as pd


# -------------------------------------------------------------------
# DATA IO
# -------------------------------------------------------------------

def load_truth_cdc_simple(fname):
    """Load truth file as a data frame indexed by date
    and location id. That's all.
    """
    return pd.read_csv(fname, index_col=(0, 1), parse_dates=["date"])


def load_population_data(
        fname, drop_count_rates=False, count_rate_key="count_rate"):
    """Import population demographic data and create some basic helper
    objects.

    Results are returned as a dict, which can be used to update the
    `data` object in parsel_ programs.
    """

    out = dict()
    out["pop_df"] = pd.read_csv(fname, index_col=2)
    out["all_state_names"] = list(out["pop_df"].index)
    out["state_name_to_id"] = {  # From state name to state ID as str
        idx: sr["location"] for idx, sr in out["pop_df"].iterrows()
    }

    # --- Rate-count tresholds
    names = [name for name in out["pop_df"] if count_rate_key in name]

    if names:  # Any rate counts found
        out["count_rates"] = out["pop_df"][names].copy()

        # Rename columns by float values (rate counts)
        out["count_rates"].columns = out["count_rates"].columns.map(
            lambda name: float(
                name[len(count_rate_key):].replace("p", ".")
            )
        )

        # Remove (optionally) from the main dataframe
        if drop_count_rates:
            out["pop_df"] = out["pop_df"].drop(columns=names)

    else:
        out["count_rates"] = pd.DataFrame()  # Empty df

    return out


# -------------------------------------------------------------------
# STANDARDIZED FILE NAMES AND PATHS
# -------------------------------------------------------------------

def make_mcmc_prefixes(
        date_str, state_str, main_dir,
):
    """Creates the path prefixes for the MCMC R(t) estimation using
    the C engine, in a standard format for the parsel project.
    """
    # Prepare the templates
    path_fmt = os.path.join(
        main_dir, "mcmc_{}", date_str, state_str)

    # Prepare the actual paths
    return dict(
        in_prefix=path_fmt.format("inputs"),
        out_prefix=path_fmt.format("outputs"),
        log_prefix=path_fmt.format("logs"),
    )


def make_rt_out_fname(date_str, state_str, main_dir):
    """ Creates the MCMC output R(t) file name.
    """
    out_prefix = make_mcmc_prefixes(
        date_str, state_str, main_dir)["out_prefix"]
    return f"{out_prefix}_rt.out"


def make_filtered_fname(
        date_str, state_str, main_dir):
    """Creates the pathname for the filtered data file, given a
    state name and a day_pres date.
    """
    return os.path.join(
        main_dir, "preproc_outputs", date_str, state_str + "_preproc.csv")


def make_file_friendly_name(name: str, space_rep="-", point_rep="_"):
    """Create a string that can be used in file paths from a base string.
    """
    return name.replace(" ", space_rep).replace(".", point_rep)


# -------------------------------------------------------------------
# MANIPULATION OF INDEXES
# -------------------------------------------------------------------


def make_state_index(general_params: dict, loc_names: list):
    """Make a pandas.Index with names of states that were selected
    to be directly forecasted.
    """
    # Sequence of state names – (with excluded ones)
    if general_params["use_states"] is None:
        general_params["use_states"] = loc_names

    # Select states by inclusion and exclusion lists
    states = list(filter(
        lambda s: (s in general_params["use_states"]
                   and s not in general_params["dont_use_states"]),
        loc_names
    ))

    return pd.Index(states)


def make_date_state_index(general_params: dict, loc_names: list):
    # Sequence of present days
    day_pres_seq = pd.date_range(
        general_params["date_first"],
        general_params["date_last"],
        freq=f"{general_params['date_step']}w-SUN",
    )

    # Sequence (pd.Index) of state names selected to be forecasted
    states = make_state_index(general_params, loc_names)

    # Multiindex of day_pres and state_name
    date_state_idx = pd.MultiIndex.from_product(
        (day_pres_seq, states),
        names=("day_pres", "state_name")
    )

    return day_pres_seq, states, date_state_idx


def prepare_dict_for_yaml_export(d: dict, inplace=True):
    """Converts some data types within a dictionary into other objects
    that can be read in a file (e.g. strings).
    Operates recursively through contained dictionaries.
    If `inplace` is True (default), changes are made inplace for
    all dictionaries (including inner ones).
    """
    if not inplace:
        d = d.copy()

    for key, val in d.items():

        # Recurse through inner dictionary
        if isinstance(val, dict):
            d[key] = prepare_dict_for_yaml_export(val, inplace=inplace)

        # pathlib.Path into its string
        if isinstance(val, Path):
            d[key] = str(val.expanduser())

        # Pandas timestamp to its isoformat
        if isinstance(val, pd.Timestamp):
            # d[key] = str(val)
            d[key] = val.to_pydatetime()  # Reloadable and human-readable

    return d


# ----------------------------------------------------------------------
# DATE AND TIME TOOLS
# ----------------------------------------------------------------------


def get_next_weekly_timestamp(ref_timestamp, now=None):
    """Returns """
    if now is None:
        now = pd.Timestamp.now()

    delta = (now - ref_timestamp) % pd.Timedelta("1w")
    #  ^ ^ Time past the last deadline
    return now - delta + pd.Timedelta("1w")
