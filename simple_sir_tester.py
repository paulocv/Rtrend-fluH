"""
Performs a simple SIR model with delays and reporting rates in postprocess. Meant for proof of concept.

Notes
------------
i_step = represents an epidemic iteration. i_step = 0 is the first iteration/update.
exd.i_t = represents a time stamp. i_t = 0 represents moment zero, before any update.
"""

import argparse
from collections import OrderedDict
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toolbox.file_tools import write_config_string

DEFAULT_RND_SEED = None
RNG: np.random.Generator = None
WEEKLEN = 7


def main():

    # INPUTS
    # ------------------------------------------------------------------------------------------------------------------
    r0 = 1.2  # Basic reproduction number
    tg = 2.3  # Generation time in days
    do_plot = False
    do_export = True

    param = Params(
        # SIR Model
        beta=r0 / tg,
        mu=1. / tg,
        pop_size=int(10E6),
        max_steps=2000,
        i0=30,
        dt=1.0,  # NOT USED YET. Implementing will require some decisions.

        # Postprocessing
        hosp_rate=0.0002,   # Fraction of infections that are hospitalized.
        hosp_fixed_delay=2,  # Constant delay time in days, to be added to all individuals.
        hosp_avg_exdelay=5.,  # Parameter for the variable part of the delay.
        hosp_max_delay=30,   # (1 + ) Upper bound for hospitalization delays. Shapes the arrays. Actual max is n - 1.
    )

    # PREAMBLE
    # ----------------------------------------------
    args = parse_args()
    global RNG
    RNG = np.random.default_rng(seed=args.seed)

    # SIR MODEL SIMULATION
    # ------------------------------------------------------------------------------------------------------------------

    # Initialization
    # --------------
    exd = SIRdata(max_steps=param.max_steps + param.hosp_max_delay)
    exd.reset_data(param.pop_size, param.i0)  # Sets initial values (current and next step)

    # Step 0 register
    exd.register_data()

    # Execution
    # ---------

    for i_step in range(param.max_steps):  # i_step is the index of the EPIDEMIC ITERATION. exd.i_t is the time stamp.

        # Periodic report
        if i_step % 10 == 0:
            # Periodic report
            print(f"step {i_step}:", end=" ")
            print(exd.new_i, end=", ")

            print()

        # MODEL ITERATION
        calc_new_infections(exd, param)
        calc_new_removals(exd, param)
        exd.update_next_counters()

        # Consolidate
        exd.consolidate_counts()
        exd.i_t += 1

        # Register
        exd.register_data()

        # Absorbing state detection
        if exd.num_i == 0:
            break

    else:  # Max days reached
        print(f"WARNING: max_steps = {param.max_steps} was reached.")

    exd.register_fill_steps(param.hosp_max_delay)  # Register remaining days (space allocated for delayed hosp.)

    # POSTPROCESSING
    # ------------------------------------------------------------------------------------------------------------------

    psd = PostData(exd.i_t, param.hosp_max_delay)
    calc_hosp_tseries(exd, psd, param)

    # PLOT AND EXPORT
    # ------------------------------------------------------------------------------------------------------------------

    if do_export:
        # Create output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        fname_fmt = os.path.join(args.output_dir, args.prefix) + "_{}"

        #
        export_header_file(fname_fmt.format("params.in"), param)
        export_tseries_file(fname_fmt.format("tseries.csv"), exd, psd, param)

    if do_plot:
        fig, ax = plt.subplots()

        ax.plot(exd.incid_tseries[:exd.i_t] / param.pop_size)
        # ax.plot(exd.i_tseries[:exd.i_t] / param.pop_size)
        # ax.plot(psd.hosp_tseries_nodelay / param.pop_size)
        ax.plot(psd.hosp_tseries / param.pop_size)

        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# --- CLASSES, STRUCTS, BUNCHES
# ----------------------------------------------------------------------------------------------------------------------

class Params:
    """Parameters for the SIR model *and* its postprocessing."""

    def __init__(self, beta, mu, pop_size, i0, max_steps, dt=1.0,
                 hosp_rate=0.10, hosp_fixed_delay=0, hosp_avg_exdelay=0, hosp_max_delay=30):
        # SIR model parameters
        self.beta = beta
        self.mu = mu
        self.pop_size = pop_size
        self.i0 = i0
        self.max_steps = max_steps
        self.dt = dt

        # Postprocessing parameters
        self.hosp_rate = hosp_rate
        self.hosp_fixed_delay = hosp_fixed_delay
        self.hosp_avg_exdelay = hosp_avg_exdelay
        self.hosp_max_delay = hosp_max_delay


class SIRdata:
    """Includes dynamic variables and result containers for the SIR model"""

    def __init__(self, max_steps):

        # --- Dynamic variables
        # Current compartment counts
        self.num_s = 0
        self.num_i = 0
        self.num_r = 0

        # Changes in each compartment
        self.new_i = 0
        self.new_r = 0

        # Next step compartment counts
        self.num_s_next = 0
        self.num_i_next = 0
        self.num_r_next = 0

        # Misc
        self.i_t = 0  # Index of time step
        self.infec_probab = 0.

        # --- Output/result containers
        self.i_tseries = np.empty(max_steps + 1, int)  # a[i_t] | Number of infectious at the TIME STAMP i_t
        self.r_tseries = np.empty(max_steps + 1, int)
        self.incid_tseries = np.empty(max_steps + 1, int)  # a[i_t] | Number of infections that OCCURRED during [i_t, i_t+1]

    def reset_data(self, pop_size, i0):
        self.num_s = self.num_s_next = pop_size - i0
        self.num_i = self.num_i_next = i0
        self.num_r = self.num_r_next = 0

        self.i_t = 0

    def update_next_counters(self):
        """Add and subtract values from the num_x_next counters, based on new_x variables."""
        self.num_s_next -= self.new_i
        self.num_i_next += self.new_i - self.new_r
        self.num_r_next += self.new_r

    def consolidate_counts(self):
        """Transfer values from next step counters into current step counters."""
        self.num_s = self.num_s_next
        self.num_i = self.num_i_next
        self.num_r = self.num_r_next

    def register_data(self):
        """Saves current state into current position of the time series vectors."""
        self.i_tseries[self.i_t] = self.num_i
        self.r_tseries[self.i_t] = self.num_r

        self.incid_tseries[self.i_t] = self.new_i

    def register_fill_steps(self, num_steps):
        """
        Registers the current state into a number of afterward days (including the current).
        Useful to fill values after the outbreak is over.
        """
        self.i_tseries[self.i_t:self.i_t + num_steps] = self.num_i
        self.r_tseries[self.i_t:self.i_t + num_steps] = self.num_r

        self.incid_tseries[self.i_t:self.i_t + num_steps] = self.new_i


class PostData:
    """Postprocessing of an SIR model, and comparisson with "golden" data."""
    def __init__(self, num_days: int, max_delay: int):
        """
        num_days: int
            Total number of days (or time stamps) present in the time series.
            Assumed to be the length of the daily time series.
        max_delay: int
            Maximum delay between infection and hospitalization report.

        """

        self.num_days = num_days
        self.num_weeks = (num_days - 1) // 7 + 1
        self.max_delay = max_delay

        self.weekly_incid = np.empty(self.num_weeks, dtype=int)
        self.hosp_tseries_nodelay = np.empty(self.num_days, dtype=int)  # a[i_day] | Nr. of hospit. infected at i_day
        self.hosp_tseries = np.zeros(self.num_days + self.max_delay, dtype=int)  # Nr. of people hospit. at i_day

        # self.


# ----------------------------------------------------------------------------------------------------------------------
# --- GENERAL METHODS
# ----------------------------------------------------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    # --- Argument list
    parser.add_argument("output_dir")
    parser.add_argument("-p", "--prefix", default="sir",
                        help="Prefix for the output files.")
    parser.add_argument("-s", "--seed", default=DEFAULT_RND_SEED, type=int)

    return parser.parse_args()


def export_header_file(fname, param: Params):

    str = write_config_string(param.__dict__)
    with open(fname, "w") as fp:
        fp.write(str)


def export_tseries_file(fname, exd: SIRdata, psd: PostData, param: Params):

    num_steps = exd.i_t + param.hosp_max_delay

    out_df = pd.DataFrame(OrderedDict(
        num_i=exd.i_tseries[:num_steps],
        num_r=exd.r_tseries[:num_steps],
        new_cases=exd.incid_tseries[:num_steps],
        hosp=psd.hosp_tseries))

    return out_df.to_csv(fname)


# ----------------------------------------------------------------------------------------------------------------------
# --- SIR METHODS
# ----------------------------------------------------------------------------------------------------------------------

def aux_calc_limited_poisson(lamb, max_val, max_attempts=20):
    """Attempts to draw a Poisson random number that cannot exceed a certain value."""
    for i_trial in range(max_attempts):
        result = RNG.poisson(lamb)
        if result <= max_val:  # Success condition: value is acceptable
            break
    else:  # Acceptable value of new_i was never obtained!!
        raise RuntimeError(f"Hey, max attempts ({max_attempts}) for a physical Poisson value was reached.")

    return result


def calc_infec_probab(exd: SIRdata, param: Params):
    exd.infec_probab = 1. - (1. - param.beta / param.pop_size) ** exd.num_i
    return exd.infec_probab


def calc_new_infections(exd: SIRdata, param: Params):

    expec_ni = calc_infec_probab(exd, param) * exd.num_s

    # -()- ALWAYS POISSON
    exd.new_i = aux_calc_limited_poisson(expec_ni, exd.num_s)

    # -()- BINOMIAL/POISSON DECISION (not implemented)
    # if exd.num_s > 20 and exd.infec_probab < 0.01:  # Poisson condition
    #     pass
    #
    # else:  # Binomial condition
    #     pass

    return exd.new_i


def calc_new_removals(exd: SIRdata, param: Params):
    expec_nr = param.mu * exd.num_i

    # -()- BINOMIAL/POISSON DECISION
    if exd.num_i > 20 and param.mu < 0.01:  # Poisson condition
        exd.new_r = aux_calc_limited_poisson(expec_nr, exd.num_i)

    else:  # Binomial condition
        exd.new_r = RNG.binomial(exd.num_i, param.mu)

    return exd.new_r


# ----------------------------------------------------------------------------------------------------------------------
# --- POSTPROCESSING METHODS
# ----------------------------------------------------------------------------------------------------------------------

def calc_hosp_number(incid_tseries, p):
    """
    incid_tseries = time series of new infections (daily, per step).
    p = probability of hospitalization for each individual.
    """
    # -()- Simple binomial (no Poisson approximation).
    return RNG.binomial(incid_tseries, p)


def calc_hosp_delayed_series(psd: PostData, param: Params):
    """Calculates and applies the delays in hospitalization report for each individual."""
    inp = psd.hosp_tseries_nodelay  # Input data: number of people infected at day i_t that was hospitalized later.
    out = psd.hosp_tseries  # Output: number of people hospitalized in i_t
    max_hosp = inp.max()  # Maximum number of hospitalized individuals in the same day.
    aux_delays = np.empty(max_hosp, dtype=int)  # Stores the delays of all individuals in each day.

    out[:] = 0  # Sets all output values to zero.

    # MAIN LOOP
    for i_t, n in enumerate(inp):
        # n = number of infected people infected at time i_t and hospitalized at any time.

        # --------- Calculation of delays
        # -()- (Constant + Poisson) delay
        aux_delays[:n] = param.hosp_fixed_delay + RNG.poisson(param.hosp_avg_exdelay, size=n)

        # --------- Accounting of them into the output
        aux_delays[:n] = np.minimum(param.hosp_max_delay - 1, aux_delays[:n])  # Clamp output values to maximum.
        bc = np.bincount(aux_delays[:n], minlength=param.hosp_max_delay)   # Makes histogram of each delay count
        out[i_t: i_t + param.hosp_max_delay] += bc


def calc_hosp_tseries(exd: SIRdata, psd: PostData, param: Params):

    # For each new infection, draw wether it will be hospitalized. Use binomial
    psd.hosp_tseries_nodelay = calc_hosp_number(exd.incid_tseries[:exd.i_t], param.hosp_rate)
    calc_hosp_delayed_series(psd, param)


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
