#' Obtain influenza signal at daily or weekly scale
#'
#' @param as_of Date or string in format "YYYY-MM-DD" specifying the date which
#'   the data was available on or before. If `NULL`, the default returns the
#'   most recent available data.
#' @param locations optional list of FIPS or location abbreviations. Defaults to
#'   the US, the 50 states, DC, PR, and VI.
#' @param temporal_resolution "daily" or "weekly"
#' @param source either "covidcast" or "HealthData". HealthData only supports
#' `as_of` being `NULL` or the current date.
#' @param na.rm boolean indicating whether NA values should be dropped when
#'   aggregating state-level values and calculating weekly totals. Defaults to
#'   `FALSE`
#'
#' @return data frame of flu incidence with columns date, location,
#'   location_name, value

load_flu_hosp_data <- function(as_of = NULL,                               
                               locations = "*",
                               temporal_resolution = "daily",
                               source = "HealthData",
                               na.rm = FALSE) {
  
  # load location data
  location_data <- readr::read_csv(file = "population_data/locations.csv",
                                   show_col_types = FALSE) %>%
    dplyr::mutate(geo_value = tolower(abbreviation)) %>%
    dplyr::select(-c("population", "abbreviation"))
  
  # validate function arguments
  if (!(source %in% c("covidcast", "HealthData"))) {
    stop("`source` must be either covidcast or HealthData")
  } else if (source == "HealthData" && !is.null(as_of)) {
    if (as_of != Sys.Date()) {
      stop("`as_of` must be either `NULL` or the current date if source is HealthData")
    }
    
  }
  
  valid_locations <- unique(c(
    "*",
    location_data$geo_value,
    tolower(location_data$location)
  ))
  locations <-
    match.arg(tolower(locations), valid_locations, several.ok = TRUE)
  temporal_resolution <- match.arg(temporal_resolution,
                                   c("daily", "weekly"),
                                   several.ok = FALSE)
  if (!is.logical(na.rm)) {
    stop("`na.rm` must be a logical value")
  }
  
  # get geo_value based on fips if fips are provided
  if (any(grepl("\\d", locations))) {
    locations <-
      location_data$geo_value[location_data$location %in% locations]
  } else {
    locations <- tolower(locations)
  }
  # if US is included, fetch all states
  if ("us" %in% locations) {
    locations_to_fetch <- "*"
  } else {
    locations_to_fetch <- locations
  }
  
  # pull daily state data
  if (source == "covidcast") {
    state_dat <- covidcast::covidcast_signal(
      as_of = as_of,
      geo_values = locations_to_fetch,
      data_source = "hhs",
      signal = "confirmed_admissions_influenza_1d",
      geo_type = "state"
    ) %>%
      dplyr::mutate(
        epiyear = lubridate::epiyear(time_value),
        epiweek = lubridate::epiweek(time_value)
      ) %>%
      dplyr::select(geo_value, epiyear, epiweek, time_value, value) %>%
      dplyr::rename(date = time_value)
  } else {
    # temp <- httr::GET(
    #   "https://healthdata.gov/resource/qqte-vkut.json",
    #   config = httr::config(ssl_verifypeer = FALSE)
    # ) %>%
    #   as.character() %>%
    #   jsonlite::fromJSON() %>%
    #   dplyr::arrange(update_date)
    # csv_path <- tail(temp$archive_link$url, 1)
    temp <- RSocrata::read.socrata(url = "https://healthdata.gov/resource/qqte-vkut.json") %>% 
      dplyr::arrange(update_date)
    csv_path <- tail(temp$url, 1)
    data <- readr::read_csv(csv_path)
    state_dat <- data %>%
      dplyr::transmute(
        geo_value = tolower(state),
        date = date - 1,
        epiyear = lubridate::epiyear(date),
        epiweek = lubridate::epiweek(date),
        value = previous_day_admission_influenza_confirmed
      ) %>%
      dplyr::arrange(geo_value, date)
  }
  
  
  # creating US and bind to state-level data if US is specified or locations
  if (locations_to_fetch == "*") {
    us_dat <- state_dat %>%
      dplyr::group_by(epiyear, epiweek, date) %>%
      dplyr::summarize(value = sum(value, na.rm = na.rm), .groups = "drop") %>%
      dplyr::mutate(geo_value = "us") %>%
      dplyr::ungroup() %>%
      dplyr::select(geo_value, epiyear, epiweek, date, value)
    # bind to daily data
    if (locations != "*") {
      dat <- rbind(us_dat, state_dat) %>%
        dplyr::filter(geo_value %in% locations)
    } else {
      dat <- rbind(us_dat, state_dat)
    }
  } else {
    dat <- state_dat
  }
  
  # weekly aggregation
  if (temporal_resolution != "daily") {
    dat <- dat %>%
      dplyr::group_by(epiyear, epiweek, geo_value) %>%
      dplyr::summarize(
        date = max(date),
        num_days = n(),
        value = sum(value, na.rm = na.rm),
        .groups = "drop"
      ) %>%
      dplyr::filter(num_days == 7L) %>%
      dplyr::ungroup() %>%
      dplyr::select(-"num_days")
  }
  
  final_data <- dat %>%
    dplyr::left_join(location_data, by = "geo_value") %>%
    dplyr::select(date, location, location_name, value) %>%
    # drop data for locations retrieved from covidcast,
    # but not included in forecasting exercise -- mainly American Samoa
    dplyr::filter(!is.na(location))
  
  return(final_data)
}

# import libraries
library(dplyr)
library(readr)
library(covidcast)
library(lubridate)
library(httr)
library(jsonlite)
library(RSocrata)
library(optparse)

# --- Defines command line arguments
option_list = list(
  make_option(c("-t", "--truth-file"), type="character", 
              default="./hosp_data/truth_daily_latest.csv",
              help="Daily truth file path.", metavar="character"),
  make_option(c("--truth-file-weekly"), type="character", 
              default="./hosp_data/truth_weekly_latest.csv",
              help="Weekly truth file path.", metavar="character")
);

# --- Parsing and unpacking command-line arguments
arg_parser = OptionParser(option_list=option_list)
clargs = parse_args(arg_parser)

truth_file = clargs[[1]]
truth_file_weekly = clargs[[2]]


# 
# -()- DAILY: generate the latest weekly truth data file
print("--- --- Fecthing daily data")
truth <- load_flu_hosp_data(temporal_resolution = 'daily', na.rm = TRUE)
readr::write_csv(truth, file = truth_file)

# -()- WEEKLY: generate the latest weekly truth data file
print("--- --- Fecthing weekly data")
truth <- load_flu_hosp_data(temporal_resolution = 'weekly', na.rm = TRUE)
readr::write_csv(truth, file = truth_file_weekly)


# # ----
# 
# setwd("/Users/pventura/NonCloudStorage/Flu_forecast/programs/")
# 
# # -()- LOOP DAILY: loop over past
# as_of <- "2022-10-25"
# out_fname <- sprintf("./hosp_data/season_2022-2023/daily_inchosp_%s.csv", as_of)
# print(out_fname)
# 
# truth <- load_flu_hosp_data(as_of = as_of, temporal_resolution = 'daily', na.rm = TRUE, source="covidcast")
# readr::write_csv(truth, file = out_fname)

