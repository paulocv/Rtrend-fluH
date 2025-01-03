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
    tg_max = tg_pmf_array.shape[0] + 2
    ct_array = np.concatenate((ct_given_array, np.full(n_steps_sim, np.nan)))
    rt_sim_array = np.full(n_steps_sim, np.nan)

    # --- Perform some checks
    if n_steps_given < 1:
        err_msg = "Array `ct_past_array` must have at least 1 element."
        return ct_array[n_steps_given:], rt_sim_array, err_msg

    # # --- Initialize ct_current and rt_current
    # ct_last = ct_past_array[-1]  # Day previous to the iteration

    # --- Loop over time steps
    for i_t_sim in range(n_steps_sim):

        # --- Total infectiousness for current step (convolution of past incidence with generation time)
        tot_infec = 0.
        for i_s in range(tg_max - 2):  # i_s is the index at the tg_pmf_array
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

    return ct_array[n_steps_given:], rt_sim_array, ""


@nb.njit
def aggregate_in_periods_numba(
    array: np.ndarray,
    period_len: int = 7,
    backwards: bool = True,
):
    """"""
    # If backwards, excess points at the start are ignored.
    # If not backwards, excess points at the end are ignored.

    # # -()- Reshape method
    # # Remove excess points
    # num_granular_points = (array.shape[0] // period_len) * period_len
    # if backwards:
    #     array = array[-num_granular_points:]
    # else:
    #     array = array[:num_granular_points]
    #
    # return array.reshape(-1, period_len).sum(axis=1)

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






def main():
    # Test parameters
    # ===============
    tg_max = 12
    tg_gamma_shape = 10.
    tg_gamma_rate = 3.333333
    rt_bias = 0.2
    rt_current = 1.0
    drift_coef = 0.01
    day_fore = pd.Timestamp("2024-12-29")  # First day that needs forecasting. Sunday

    # Build the generation time distribution
    # ==================
    # CONVENTION: tg_pmf(s) = probab. of tg being of `s` days = CDF(s+1) - CDF(s)
    s_array = np.arange(1, tg_max + 1)  # As of in Rtrend library, tg_max itself is also excluded.
    tg_gamma = scipy.stats.gamma(a=tg_gamma_shape, scale=1. / tg_gamma_rate)  # The continuous-time gamma that creates the TG PMF
    tg_pmf_array = np.diff(tg_gamma.cdf(s_array))   # This excludes the 0 day AND tg_max day
    tg_pmf_array /= tg_pmf_array[:].sum()

    #  ^ ^ Shape: tg_pmf_array[i_s] = P(tg = i_s + 1), i_s = 0, ..., tg_max - 2
    #        Thus s = i_s + 1  ranges from 1 to tg_max - 1

    # Build a synthetic past time series
    # ===========
    ct_given_array = np.repeat(10, tg_max + 1)

    res = _renewal_model_drift_exp(
        n_steps_sim=10, ct_given_array=ct_given_array, rt_current=1.,
        tg_pmf_array=tg_pmf_array, rt_bias=rt_bias,
        drift_coef=drift_coef
    )
    print(res)

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

    mean_list, std_list = list(), list()
    xt_inner_array = np.zeros(num_inner_samples, dtype=float)
    ct_given_array = np.empty(tg_max, dtype=float)

    # Run once to compile
    ct_given_array[:] = param_inner_samples["ct_current"][0]
    rt_bias = param_inner_samples["rt_bias"][0]
    rt_current = param_inner_samples["rt_current"][0]

    ct_sim_array, rt_sim_array = _renewal_model_drift_exp(
        n_steps_sim=28, ct_given_array=ct_given_array, rt_current=rt_current,
        tg_pmf_array=tg_pmf_array, rt_bias=rt_bias,
        drift_coef=drift_coef
    )
    ct_sim_aggr_array = aggregate_in_periods_numba(ct_sim_array, period_len=7, backwards=True)

    print(f"Running performance test... {num_inner_samples} | {num_outer_samples}")
    total_xt0 = time.time()
    for _ in range(num_outer_samples):

        for i_inner in range(num_inner_samples):
            ct_given_array[:] = param_inner_samples["ct_current"][i_inner]
            rt_bias = param_inner_samples["rt_bias"][i_inner]
            rt_current = param_inner_samples["rt_current"][i_inner]

            ct_sim_array, rt_sim_array = _renewal_model_drift_exp(
                n_steps_sim=28, ct_given_array=ct_given_array, rt_current=rt_current,
                tg_pmf_array=tg_pmf_array, rt_bias=rt_bias,
                drift_coef=drift_coef
            )

            xt0 = time.time()
            ct_sim_aggr_array = aggregate_in_periods_numba(ct_sim_array, period_len=7, backwards=True)
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
