"""
This module holds variables from the CDC requirements for the calculation and formatting
of the data.
"""
import numpy as np

CDC_QUANTILES_SEQ = np.round(np.append(np.append([0.01, 0.025], np.arange(0.05, 0.95 + 0.05, 0.050)), [0.975, 0.99]), 6)
NUM_QUANTILES = CDC_QUANTILES_SEQ.shape[0]
NUM_STATES = 53  # Number of locations: states and jurisdictions, excluding the "US" entry.
NUM_WEEKS_FORE = 4
WEEKDAY_TGT = 5  # 5 = Saturday
NUM_OUTP_LINES = NUM_WEEKS_FORE * (NUM_STATES + 1) * NUM_QUANTILES  # Expected output file number of lines
NUM_OUTP_LINES_WPOINTS = NUM_WEEKS_FORE * (NUM_STATES + 1) * (NUM_QUANTILES + 1)
