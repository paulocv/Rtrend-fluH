"""Define instances of the ForecastOperator class to use in the 2023-2024
flusight season.
"""

import gc

import pandas as pd
import warnings

from rtrend_forecast import ForecastOperator
from rtrend_forecast.errchecking import McmcError
from rtrend_forecast.structs import RtData
from utils.parsel_utils import make_mcmc_prefixes


WEEKLEN = 7  # USE THIS LENGTH OF THE WEEK for parsel project.


class FluSight2023ForecastOperator(ForecastOperator):
    """
    This is the forecast operator for the 2023/2024 season of the CDC
    FluSight forecast.

    AS OF NOW... it only derivates from the Training operator, but a
    separate version shall be made at some point.
    """

    """A forecast operator designed for training the parameter
    selection machine.
    It starts from already preprocessed data (included MCMC R(t)
    estimation), and can be run through the parsel/synthesis,
    reconstruction and postprocessing multiple times.
    """
    is_aggr = False  # Refers to the INPUT DATA: it's granular
    gran_dt = pd.Timedelta("1d")  # Granular period
    aggr_nperiods = WEEKLEN  # Duration of a week

    # Declared members
    fore_quantiles: pd.DataFrame

    #  ^ ^ Redundant to self.inc.aggr_quantiles or self.inc.gran_quantiles

    def __init__(
            self,
            # main_roi_start: pd.Timestamp, main_roi_end: pd.Timestamp,
            day_pres: pd.Timestamp,  # "Present" day. One after end of the main ROI.
            nperiods_main_roi: int,  # Size of the main region-of-interest in aggr periods.
            population: int,
            # count_rates: pd.Series,
            *args,
            state_name="#nostate",
            **kwargs):
        super().__init__(*args, **kwargs)

        # Calculate and set the main ROI
        self.main_roi_len = nperiods_main_roi * self.aggr_dt
        self.main_roi_end = day_pres - 1 * self.gran_dt
        self.main_roi_start = self.main_roi_end - self.main_roi_len

        self.set_main_roi(self.main_roi_start, self.main_roi_end)

        # Remove the reconstruction phase: merged with synthesis
        self.remove_stage("inc_reconstruction")

        # Misc
        self.population = population  # Effective population size.
        self.state_name = state_name

    def callback_preprocessing(self):
        """Either runs the preprocessing defined in the
        PreprocessOperator or loads buffered data to achieve the
        same internal state.
        """

        # EXCEPTIONS (PREPROCESSING TIME)

        # 2024-03-13 Puerto, Alaska
        if (
                self.state_name in [
            "Puerto Rico",
            "Hawaii",
            "Alaska",
            "Montana",
            "Virginia",
        ]):
            self.sp["synth_method"] = "rnd_normal"
            self.sp["sigma"] = 0.006
            # Reduce sigma for Alaska
            if self.state_name == "Alaska":
                self.sp["sigma"] = 0.001
            self.logger.warning(f"Exception applied")

        # REGULAR RANDOM NORMAL
        if self.state_name in [
            "Alabama",
            "Idaho",
            "West Virginia",
            "North Carolina",
        ]:
            self.sp["synth_method"] = "rnd_normal"

        # Arizona is predicting huge increase and sharp turn
        # 2024-04-03 – Not too bad now
        if (
                "Arizona" in self.name
        ):
            # self.ep["scale_ref_inc"] = 50
            self.ep["scale_ref_inc"] = 180

        # Connecticut: weekly oscillating pattern. Strengthen the filter.
        if self.state_name in [
            "Connecticut",
            # "Alaska"
        ]:
            self.pp["denoise_cutoff"] = 0.07

        # 2024-02-24 reduce trend for WV
        # 2024-03-13 District of Columbia
        if self.state_name in [
            # "West Virginia",
            "District of Columbia",
        ]:
            self.sp["initial_bias"] = -0.15

        # 2024-03-13 New Mexico
        # 2024-03-20 Change to initial bias
        if self.state_name in [
            # "West Virginia",
            "New Mexico",
        ]:
            # self.sp["bias"] = -0.001
            self.sp["initial_bias"] = -0.01

        # # 2024-03-20 - States with some overconfidence-overestimating
        # if self.state_name in [
        #     # "West Virginia",
        #     "Michigan"
        #     "Ohio",
        #     "Virginia",
        #     "West Virginia",
        #     # "Wisconsin",
        # ]:
        #     self.sp["initial_bias"] = -0.2
        #     self.ep["scale_ref_inc"] *= 0.1

        # Too confident 2024-04-24
        # - Georgia, Colorado, Michigan, Mississippi, New York
        # 2024-04-17 Too narrow
        if self.state_name in [
            "Colorado",
            "Georgia",
            "Michigan",
            "Mississippi",
            "New York",
            # "North Carolina",
        ]:
            self.ep["scale_ref_inc"] *= 0.20

        # 2024-04-17 Too wide
        if self.state_name in [
            "Vermont",
        ]:
            self.ep["scale_ref_inc"] *= 100  # ?!?!?!?

        # 2024-03-20 – Utah - Reduce Uncertainty
        if self.state_name in [
            "Utah",
        ]:
            self.ep["scale_ref_inc"] = 5000

        # 2024-03-13 DC - Reduce uncertainty
        if self.state_name in [
            "District of Columbia",
        ]:
            self.ep["scale_ref_inc"] = 1000
            # self.sp["q_low"] = 0.40
            # self.sp["q_hig"] = 0.60

        # 2024-04-03 – Avoid late season rebound
        if self.state_name in ["Louisiana"]:
            self.ep["scale_ref_inc"] = 800

        if "New-Hampshire" in self.name:
            self.sp["drift_pop_coef"] *= 1.30
            self.logger.warning(f"Exception applied: too little drift")

        # 2024-03-27 – Oregon has too much uncertainty
        if self.state_name in ["Oregon"]:
            self.pp["denoise_cutoff"] = 0.07

        # ==============================================================

        # Store raw series
        self.extra["past_raw_sr"] = self.inc.past_gran_sr.copy()

        # Denoise
        self.denoise_and_fit(which="gran")
        # self.extra["past_float_sr"] = self.inc.past_gran_sr.copy()

        # Fix negative artifacts
        self.remove_negative_incidence(
            warn=True, method=self.pp["negative_method"])
        self.extra["past_denoised_sr"] = self.inc.past_gran_sr#.copy()

        # --- Debriefing
        self.set_stage_next()


        # if True:  # Placeholder: for now, does not load from file
        #     self.callback_preprocessing()

        # elif [load from files]:  #  # Placeholder for file loading
        #     function_that_leads_to_the_same_state()
        #     self.set_stage_next()

    def callback_rt_estimation(self):

        if True:  # TODO DECIDE LATER - Run, load or reuse-loaded
            # Briefing
            # --------
            # MCMC preprocessing
            scaled = self.get_scaled_past_incidence("gran")
            # self.extra["past_scaled_sr"] = scaled

            rounded = scaled.round()
            self.extra["past_scaled_int_sr"] = rounded

            # Execution
            # ---------
            # Modify prefixes to contain the fop name
            self.ep["in_prefix"] = f"mcmc/inputs/{self.name}"
            self.ep["out_prefix"] = f"mcmc/outputs/{self.name}"
            self.ep["out_prefix"] = f"mcmc/logs/{self.name}"

            try:
                self.estimate_rt_mcmc(
                    ct_array=rounded.values, sim_name="",
                )

            # Debriefing
            # ----------
            except McmcError as e:
                self.logger.error(f"An McmcError occurred: {e}")
                self.set_stage_error()
                return

            # --- OVERLY high reproduction number estimate:
            #     => use gaussian synth
            rt_roi = self.rt_past.array[:, -self.sp["nperiods_past"]:]
            if rt_roi.max() > self.sp["r_max"]:
                self.sp["synth_method"] = "rnd_normal"

            # Save the past estimation stats
            self.extra["rt_past_median"] = self.rt_past.get_median_pd()
            self.extra["rt_past_low_q"], self.extra["rt_past_high_q"] = (
                self.rt_past.get_quantiles_pd()
            )

        self.set_stage_next()

    def callback_rt_synthesis(self):

        # EXCEPTIONS
        # --------------------------------------------
        # ==============================================
        # --- TMP FIX: NOT ENOUGH CASES TO RECONSTRUCT
        n_past = 8  # How many days in the past to check
        c_min = 1.0  # Minimum number of cases expected
        c_add = 0.9  # How much to add
        avg = self.inc.past_gran_sr.iloc[-n_past:].mean()
        if avg < c_min:
            # --- Add cases
            add = c_add  # c_min - avg  # c_add
            self.inc.past_gran_sr.iloc[-n_past:] += add
            self.logger.info(
                f"Added {add} to the daily incidence so to have cases"
                f" to feed the renewal equation.")

            # --- Use rnd normal synth
            synth_method = self.sp["synth_method"] = "rnd_normal"
            self.logger.info(
                f"Changed method to `rnd_normal`.")

        # =====================================
        # Set population size as a parameter, for compatibility.
        self.sp["population"] = self.population

        # Call the combined synthesis/reconstruction method.
        self.synthesize_tg()
        self.synthesize_and_reconstruct()

        # --- Save the future R(t) estimation stats
        self.extra["rt_fore_median"] = self.rt_fore.get_median_pd()
        self.extra["rt_fore_low_q"], self.extra["rt_fore_high_q"] = (
            self.rt_fore.get_quantiles_pd()
        )

        self.set_stage_next()

    def callback_postprocessing(self):

        self.aggregate_past_incidence()
        self.aggregate_fore_incidence()

        # --- Make quantile for required aggregation levels
        # If both, `aggr` is set as the main `fore_quantiles` level.
        if self.op["use_aggr_level"] in ["gran", "both"]:
            self.fore_quantiles = self.make_quantiles_for("gran")
        if self.op["use_aggr_level"] in ["aggr", "both"]:
            self.fore_quantiles = self.make_quantiles_for("aggr")

        # OBS: Categorical rate change is calculated in FluSight script

        self.set_stage_next()

    def dump_heavy_stuff(self):
        """Manually discard structures that have high memory usage,
        but are possibly not useful for the next stages.

        OVERRIDES the base function by keeping weekly forecast
        trajectories.
        """
        self.rt_past = None
        self.rt_fore = None
        self.inc.fore_gran_df = None
        # self.inc.fore_agg_df = None

        gc.collect()
