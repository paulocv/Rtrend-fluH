"""
Integrated Bayesian framework with renewal processes to forecast
epidemic time series.

Utilities module, first iteration after the proof of concept.
"""
import math

import numba as nb
import numpy as np
import pandas as pd
import scipy
import scipy.stats

from utils.numba_utils import float_to_str, float_to_str_exp, wrap_numba_error


@wrap_numba_error
@nb.njit
def _renewal_model_drift_exp(
    n_steps_sim: int,
    ct_given_array: np.ndarray,
    rt_current: float,
    tg_pmf_array: np.ndarray,  # EXPECTED: a[i_s] = P(tg = i_s + 1), i_s = 0, ..., tg_max - 2
    #   ^ ^ Notice: s = i_s + 1, s = 1, 2, ..., tg_max - 1
    rt_bias: float,
    drift_coef: float,
):
    """"""
    # given = before simulation time
    # sim   = during simulation time

    # Steps
    # --- Create future time series with nan
    # --- Concatenate past and future time series
    n_steps_given = ct_given_array.shape[0]
    tg_max = tg_pmf_array.shape[0] + 1
    ct_array = np.concatenate((ct_given_array, np.full(n_steps_sim, np.nan)))
    rt_sim_array = np.full(n_steps_sim, np.nan)

    # --- Perform some checks
    if n_steps_given < 1:
        err_msg = "Array `ct_given_array` must have at least 1 element."
        return ct_array[:], rt_sim_array, err_msg

    if n_steps_given < tg_max - 1:
        err_msg = (
            f"Array `ct_given_array` (size = {n_steps_given}) must be at least "
            f"the length of `tg_pmf_array` ({tg_max - 1}).")
        return ct_array[:], rt_sim_array, err_msg

    # # --- Initialize ct_current and rt_current
    # ct_last = ct_past_array[-1]  # Day previous to the iteration

    # --- Loop over time steps
    for i_t_sim in range(n_steps_sim):

        # --- Total infectiousness for current step (convolution of past incidence with generation time)
        tot_infec = 0.
        for i_s in range(tg_max - 1):  # i_s is the index at the tg_pmf_array
            s = i_s + 1  # s = 1, 2, ..., tg_max - 1
            tot_infec += ct_array[n_steps_given + i_t_sim - s] * tg_pmf_array[i_s]

        # --- Update number of infections (for current step)
        ct_current = tot_infec * rt_current

        # --- Update reproduction number (for next step)
        rt_next = rt_current * np.exp(rt_bias - drift_coef * ct_current)

        # --- Store results
        ct_array[n_steps_given + i_t_sim] = ct_current
        rt_sim_array[i_t_sim] = rt_current
        # rt_array .....

        # --- Update
        rt_current = rt_next

    # return ct_array[n_steps_given:], rt_sim_array, ""  # Return only the simulated part
    return ct_array, rt_sim_array, ""  # Given + simulated (concat)


@nb.njit
def _aggregate_in_periods(
    array: np.ndarray,
    period_len: int = 7,
    backwards: bool = True,
):
    """"""
    # If backwards, excess points at the start are ignored.
    # If not backwards, excess points at the end are ignored.

    # -()- For loop method (25% faster than reshape on tests)
    num_aggregated_points = array.shape[0] // period_len
    num_granular_points = num_aggregated_points * period_len
    if backwards:
        array = array[-num_granular_points:]
    else:
        array = array[:num_granular_points]

    out = np.zeros(num_aggregated_points)
    for i in range(num_aggregated_points):
        out[i] = array[i * period_len : (i + 1) * period_len].sum()

    return out


# Deterministic model that converts raw infection incidence (aggregated)
# into hospitalization incidence, by applying a reported hospitalization
# rate and a delay distribution as a PMF.
# In our formulation, the result of this model is used as the average
# of the negative binomial that samples the actual number of hospitalizations.

@wrap_numba_error
@nb.jit
def _infections_to_hospitalizations_model(
    ct_array: np.ndarray,
    hd_pmf_array: np.ndarray,
    i_t0: int,
    hosp_frac: float,
):
    """"""
    # NOTE: The result array will have length given by
    #   len(ct_array) - i_t0

    len_ct_array = ct_array.shape[0]
    hd_max = hd_pmf_array.shape[0] + 1

    # --- Check that i_t0 is not larger than the input array size
    if i_t0 >= len_ct_array:
        err_msg = (
            f"Parameter `i_t0` ({i_t0}) must be smaller than the length"
            f" of `ct_array` ({len_ct_array}).")
        return np.empty(0, dtype=float), err_msg

    # --- Check that i_t0 skips at least the maximum delay value
    if i_t0 < hd_max - 1:
        err_msg = (
            f"Parameter `i_t0` ({i_t0}) must be at least "
            f"the length of `hd_pmf_array` ({hd_max - 1}).")
        return np.empty(0, dtype=float), err_msg

    # ===============

    # --- Loop over time steps
    ht_array = np.full(len_ct_array - i_t0, np.nan, dtype=float)
    for i_t in range(i_t0, len_ct_array):
        h_t = 0.
        for i_s in range(hd_max - 1):  # i_s is the index at the hd_pmf_array
            s = i_s + 1  # s = 1, 2, ..., hd_max - 1
            h_t += ct_array[i_t - s] * hd_pmf_array[i_s]
            # tot_infec += ct_array[n_steps_given + i_t_sim - s] * tg_pmf_array[i_s]

        h_t *= hosp_frac
        ht_array[i_t - i_t0] = h_t

    return ht_array, ""


# Negative binomial likelihood
def _calculate_nb_loglikelihood(

):
    """"""


def main():
    # Test parameters
    # ===============
    # Generation time (TG)
    tg_max = 12
    tg_gamma_shape = 10.
    tg_gamma_rate = 3.333333

    # Hospitalization Delay (HD)
    hd_gamma_shape = 25.
    hd_gamma_rate = 5.0
    hd_max = 10

    hosp_frac = 0.1

    # R(t) and c(t) model
    n_sim_steps = 28
    rt_bias = 0.2
    rt_current = 1.0
    drift_coef = 0.0001
    day_fore = pd.Timestamp("2024-12-29")  # First day that needs forecasting. Sunday

    # Build the generation time distribution
    # ==================
    # CONVENTION: tg_pmf(s) = probab. of tg being of `s` days = CDF(s+1) - CDF(s)
    s_array = np.arange(1, tg_max + 1)  # As of in Rtrend library, tg_max itself is also excluded.
    tg_gamma = scipy.stats.gamma(a=tg_gamma_shape, scale=1. / tg_gamma_rate)  # The continuous-time gamma that creates the TG PMF
    tg_pmf_array = np.diff(tg_gamma.cdf(s_array))   # This excludes the 0 day AND tg_max day
    tg_pmf_array /= tg_pmf_array[:].sum()

    # Build the infection-to-hospitalization delay distribution
    # ==================
    # CONVENTION: hd_pmf(s) = probab. of hd being of `s` days = CDF(s+1) - CDF(s)
    s_array = np.arange(1, hd_max + 1)
    hd_gamma = scipy.stats.gamma(a=hd_gamma_shape, scale=1. / hd_gamma_rate)
    hd_pmf_array = np.diff(hd_gamma.cdf(s_array))
    hd_pmf_array /= hd_pmf_array[:].sum()

    # Build a synthetic empirical time series
    # ================
    # TODO

    #

    # ==================================================================

    #

    # Performance test
    # ===================
    from scipy.stats.qmc import LatinHypercube
    import time
    rng = np.random.default_rng(0)
    num_inner_samples = 1000  # Two-layer loop
    num_outer_samples = 100
    param_ranges = dict(
        rt_bias=(-0.2, 0.2),
        rt_current=(0.5, 2.0),
        ct_current=(10, 1000),
    )
    lhs_01_array = LatinHypercube(d=len(param_ranges)).random(n=num_inner_samples)
    param_inner_samples = {
        key: lhs_01_array[:, i] * (v[1] - v[0]) + v[0]
        for i, (key, v) in enumerate(param_ranges.items())
    }

    ct_given_array = np.empty(tg_max, dtype=float)

    # Run once to compile
    ct_given_array[:] = 100  # param_inner_samples["ct_current"][0]  # TEST
    rt_bias = param_inner_samples["rt_bias"][0]
    rt_current = param_inner_samples["rt_current"][0]

    # noinspection PyTupleAssignmentBalance
    ct_array, rt_sim_array = _renewal_model_drift_exp(
        n_steps_sim=n_sim_steps, ct_given_array=ct_given_array, rt_current=rt_current,
        tg_pmf_array=tg_pmf_array, rt_bias=rt_bias,
        drift_coef=drift_coef
    )
    #  ^ ^ len(ct_array) = len(ct_given_array) + n_steps_sim = tg_max + n_steps_sim

    ht_array = _infections_to_hospitalizations_model(
        ct_array, hd_pmf_array, i_t0=hd_max + 1, hosp_frac=hosp_frac
    )

    ht_aggr_array = _aggregate_in_periods(ht_array, period_len=7, backwards=True)

    print(f"Running performance test... inner = {num_inner_samples} | outer = {num_outer_samples}")
    mean_list, std_list = list(), list()
    xt_inner_array = np.zeros(num_inner_samples, dtype=float)  # Execution time
    total_xt0 = time.time()
    for _ in range(num_outer_samples):

        for i_inner in range(num_inner_samples):
            xt0 = time.time()

            ct_given_array[:] = param_inner_samples["ct_current"][i_inner]
            rt_bias = param_inner_samples["rt_bias"][i_inner]
            rt_current = param_inner_samples["rt_current"][i_inner]

            # noinspection PyTupleAssignmentBalance
            ct_array, rt_sim_array = _renewal_model_drift_exp(
                n_steps_sim=n_sim_steps, ct_given_array=ct_given_array, rt_current=rt_current,
                tg_pmf_array=tg_pmf_array, rt_bias=rt_bias,
                drift_coef=drift_coef
            )
            #  ^ ^ len(ct_array) = len(ct_given_array) + n_steps_sim = tg_max + n_steps_sim

            ht_array = _infections_to_hospitalizations_model(
                ct_array, hd_pmf_array, i_t0=hd_max + 1, hosp_frac=hosp_frac
            )

            ht_aggr_array = _aggregate_in_periods(ht_array, period_len=7, backwards=True)
            xt1 = time.time()
            xt_inner_array[i_inner] = xt1 - xt0

        mean_list.append(xt_inner_array.mean())
        std_list.append(xt_inner_array.std())

    total_xt1 = time.time()

    print(f"Total time: {total_xt1 - total_xt0:.3f} sec")
    mean_array = np.array(mean_list)
    print(f"Average mean time: {1E6 * mean_array.mean():.6f} μs")
    print(f"Std of mean time : {1E6 * mean_array.std():.6f} μs")


    pass


if __name__ == '__main__':
    main()
