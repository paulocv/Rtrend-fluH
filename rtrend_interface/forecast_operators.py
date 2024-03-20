"""Define instances of the ForecastOperator class to use within
specific forecast contexts.

Examples:
    * Daily data resolution, weekly forecasts
    * Automatic parameter selection
"""
import gc

import pandas as pd
import warnings

from rtrend_forecast import ForecastOperator
from rtrend_forecast.errchecking import McmcError
from rtrend_forecast.structs import RtData
from rtrend_interface.parsel_utils import make_mcmc_prefixes


WEEKLEN = 7  # USE THIS LENGTH OF THE WEEK for parsel project.


class ParSelPreprocessOperator(ForecastOperator):
    """Forecast operator that proceeds only to the MCMC R(t)
    estimation.
    """

    is_aggr = False  # Refers to the INPUT DATA: it's granular
    gran_dt = pd.Timedelta("1d")  # Granular period
    aggr_nperiods = WEEKLEN  # Duration of a week

    def __init__(
            self,
            main_roi_start: pd.Timestamp, main_roi_end: pd.Timestamp,
            *args,
            _sn: str = "", day_pres_str: str = "",  # State-date strings
            **kwargs):
        """
        Parameters
        ----------
        main_roi_start : pd.Timestamp
            Starting point of the main region-of-interest (ROI). The
            entire forecast pipeline will only be aware of this region
            of the data.
        main_roi_end : pd.Timestamp
            Last day of the main ROI. One day before the "present day".
        _sn : str (optional)
            File-friendyl name of the location or state
            Must not contain whitespaces.
        day_pres_str : str (optional)
            File-friendly string representation of the present day.
            Must not contain whitespaces.

        Other arguments are inherited from the base class
        `ForecastOperator`.

        """
        super(ParSelPreprocessOperator, self).__init__(*args, **kwargs)

        self.main_roi_start = main_roi_start
        self.main_roi_end = main_roi_end

        # Date and state info (Optional).
        self._sn = _sn
        self.day_pres_str = day_pres_str

        # --- ROI
        self.set_main_roi(self.main_roi_start, self.main_roi_end,
                          sort_data=True)

    def callback_preprocessing(self):

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

    def callback_rt_estimation(self):
        """"""
        # Briefing
        # --------
        # MCMC preprocessing
        scaled = self.get_scaled_past_incidence("gran")
        # self.extra["past_scaled_sr"] = scaled

        rounded = scaled.round()
        self.extra["past_scaled_int_sr"] = rounded

        # --- Create the paths to MCMC files (in, out, log)
        self.rt_estimation_params.update(
            make_mcmc_prefixes(
                self.day_pres_str,
                self._sn,
                self.general_params["preproc_data_dir"],
            )
        )

        # Execution
        # ---------

        try:
            self.estimate_rt_mcmc(
                ct_array=rounded, sim_name="",
            )

        # Debriefing
        # ----------
        except McmcError as e:
            self.logger.error(f"An McmcError occurred: {e}")
            self.set_stage_error()
            return

        self.set_stage_done()


#

#

#

#


class ParSelTrainOperator(ForecastOperator):
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
        super(ParSelTrainOperator, self).__init__(*args, **kwargs)

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

        # Iowa may have recent reporting issues - 2023-12-13
        # 2024-03-13 Puerto, Alaska
        if (
            self.state_name in [
                "Puerto Rico",
                "Hawaii",
                "Alaska",
        ]):
            self.sp["synth_method"] = "rnd_normal"
            self.sp["sigma"] = 0.006
            self.logger.warning(f"Exception applied")

        # Arizona is predicting huge increase and sharp turn
        if (
            "Arizona" in self.name
        ):
            # self.sp["drift_pop_coef"] *= 0.80
            self.ep["scale_ref_inc"] = 50
            self.logger.warning(f"Exception applied: too much drift")

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

        # 2024-03-20 - States with some overconfidence-overestimating
        if self.state_name in [
            # "West Virginia",
            "Michigan"
            "Ohio",
            "Virginia",
            "West Virginia",
            "Wisconsin",
        ]:
            self.sp["initial_bias"] = -0.2
            self.ep["scale_ref_inc"] *= 0.1

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

        # States with too little drift
        if ("Montana" in self.name
            or "New-York" in self.name
            # or "New-Jersey" in self.name
            or "North-Carolina" in self.name
            or "Delaware" in self.name
            or "Idaho" in self.name
            or "Arizona" in self.name
            or "Utah" in self.name
        ):
            self.sp["drift_pop_coef"] *= 1.50
            self.logger.warning(f"Exception applied: too little drift")

        if "New-Hampshire" in self.name:
            self.sp["drift_pop_coef"] *= 1.30
            self.logger.warning(f"Exception applied: too little drift")

        # # California forecast too confident and has no drift 2023-11-25
        # if "California" in self.name:
        #     self.sp["bias"] = 0.0
        #     self.sp["drift_pop_coef"] *= 1.5
        #     self.ep["scale_ref_inc"] /= 3.5
        #     self.logger.warning(f"Exception applied")

        if True:  # Placeholder: for now, does not load from file
            ParSelPreprocessOperator.callback_preprocessing(self)

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
        n_past = 8   # How many days in the past to check
        c_min = 1.0  # Minimum number of cases expected
        c_add = 0.9  # How much to add
        avg = self.inc.past_gran_sr.iloc[-n_past:].mean()
        if avg < c_min:
            # --- Add cases
            add = c_add  #  c_min - avg  # c_add
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


class FluSightForecastOperator(ParSelTrainOperator):
    """
    This is the forecast operator for the 2023/2024 season of the CDC
    FluSight forecast.

    AS OF NOW... it only derivates from the Training operator, but a
    separate version shall be made at some point.
    """

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


# --------------------------------------------------------------------
# ---------------------- LEGACY CLASSES --------------------------------
# --------------------------------------------------------------------

#
# class ParSelTrainOperatorLegacy(ForecastOperator):
#     """A forecast operator designed for training the parameter
#     selection machine.
#     It starts from already preprocessed data (included MCMC R(t)
#     estimation), and can be run through the parsel/synthesis,
#     reconstruction and postprocessing multiple times.
#
#     THIS SNAPSHOT WAS USED TO DEVELOP NEW FEATURES, now in `parsel_playground.py`.
#     """
#     is_aggr = False  # Refers to the INPUT DATA: it's granular
#     gran_dt = pd.Timedelta("1d")  # Granular period
#     aggr_nperiods = WEEKLEN  # Duration of a week
#
#     def __init__(
#             self,
#             main_roi_start: pd.Timestamp, main_roi_end: pd.Timestamp,
#             *args, **kwargs):
#         super(ParSelTrainOperatorLegacy, self).__init__(*args, **kwargs)
#
#         # Calculate and set the main ROI
#         self.main_roi_start = main_roi_start
#         self.main_roi_end = main_roi_end
#
#     def callback_preprocessing(self):
#         """Performs only basic preparation to the forecast.
#         The actual data preprocessing (such as filtering) is assumed
#         to have been done previously.
#         """
#
#         # --- ROI
#         self.set_main_roi(self.main_roi_start, self.main_roi_end,
#                           sort_data=True)
#
#         self.set_stage_done()
#         # self.set_stage_next()
#
#
# class FluParamSelectOperator(ForecastOperator):
#     """ForecastOperator for use during the parameter selection and
#     optimization of the Flu forecast using Rtrend.
#
#     Used in `eval_param_selection.py`
#
#     HUGE WARNING: This code was not finished before I changed the guideliens
#     for this project. This class is probably not in use.
#     """
#
#     is_aggr = False  # Refers to the INPUT DATA
#     gran_dt = pd.Timedelta("1d")
#     aggr_nperiods = 7
#
#     def __init__(
#             self,
#             main_roi_start: pd.Timestamp, main_roi_end: pd.Timestamp,
#             *args, **kwargs):
#         warnings.warn(
#             "This class is unfinished! Use `ParSel__` classes instead",
#             DeprecationWarning
#         )
#         super(FluParamSelectOperator, self).__init__(*args, **kwargs)
#         self.main_roi_start = main_roi_start
#         self.main_roi_end = main_roi_end
#         # Devnote: maybe the definition of the main_roi could move to the base class.
#
#     @FORE_XTT.track()
#     def callback_preprocessing(self):
#
#         # --- Briefing
#         pass
#
#         # --- ROI
#         self.set_main_roi(self.main_roi_start, self.main_roi_end,
#                           sort_data=True)
#
#         # --- Deal with missing data here
#         # NotImplemented
#
#         # Denoise
#         self.extra["past_raw_sr"] = self.inc.past_gran_sr
#         self.denoise_and_fit(which="gran")
#
#         #
#         # self.remove_negative_incidence(warn=True)
#         self.inc.past_gran_sr = self.inc.past_gran_sr.round()  # Devnote could be method
#
#         # --- Debriefing
#         self.set_stage_next()
#
#     @FORE_XTT.track()
#     def callback_rt_estimation(self):
#         self.estimate_rt_mcmc()
#         self.set_stage_next()
#
#     @FORE_XTT.track()
#     def callback_rt_synthesis(self):
#
#         # --- Briefing
#         # Simple criterion: use normal for low counts
#         if self.inc.past_gran_sr.sum() < 50:
#             self.sp["synth_method"] = "rnd_normal"
#
#         # --- Execution
#         self.logger.debug(
#             f"Rt synth with method = '{self.sp['synth_method']}'")
#         self.synthesize_tg()
#         self.synthesize_rt()
#
#         # --- Debriefing
#         # Simple dynamic method: if problem, rnd normal.
#         if self.rt_fore.df.shape[0] < 50:  # too little samples
#             self.sp["synth_method"] = "rnd_normal"
#             return  # Repeat step
#
#         self.set_stage_next()
#
#     @FORE_XTT.track()
#     def callback_inc_reconstruction(self):
#         self.reconstruct_incidence()
#         self.set_stage_next()
#
#     @FORE_XTT.track()
#     def callback_postprocessing(self):
#
#         # # self.inc.gran_quantiles = \
#         # self.aggregate_fore_incidence()
#
#         # # Incorporate noise into the aggregated series
#         # self.extra["fore_denoised_df"] = self.inc.fore_aggr_df
#         # self.incorporate_noise("aggr")
#
#         self.aggregate_fore_incidence()
#
#         # self.make_quantiles_for("gran")
#         self.make_quantiles_for("aggr")
#
#         # # T0D0  self.DUMP_ALL_THE_HEAVY_STUFF_OUT_OF_THE_MEMORY()!!
#
#         self.set_stage_next()
