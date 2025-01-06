"""Utility methods for writing numba code.
Overcomes some limitations of numba.
"""
import math

import numba as nb


@nb.njit
def float_to_str_exp(value, decimals):
    """
    Formats a float value into a string in exponential notation.

    Parameters
    ----------
    value : float
        The float value to be formatted.
    decimals : int
        The number of significant digits to print.

    Returns
    -------
    str
        The formatted string representation of the float in exponential notation.
    """
    if value == 0.0:
        return "0.0E+00"

    sign = "-" if value < 0 else ""
    value = abs(value)

    exponent = int(math.floor(math.log10(value)))
    mantissa = value / 10.0**exponent

    # Adjust mantissa and exponent for desired precision
    while mantissa >= 10.0:
        mantissa /= 10.0
        exponent += 1
    while mantissa < 1.0 and exponent > 0:
        mantissa *= 10.0
        exponent -= 1

    # Format mantissa with desired precision
    # mantissa_str = str(round(mantissa, precision - 1))
    mantissa_str = float_to_str(mantissa, decimals - 1)

    # Handle cases where precision exceeds mantissa digits
    if len(mantissa_str) < decimals:
        mantissa_str += "0" * (decimals - len(mantissa_str))

    # return f"{sign}{mantissa_str}e{exponent:+}"
    return sign + mantissa_str + "E" + str(exponent)


@nb.njit
def float_to_str(value, decimals):
    """
    Formats a float value into a string with the specified number of decimal places.

    Parameters
    ----------
    value : float
        The float value to be formatted.
    decimals : int
        The number of decimal places to print.

    Returns
    -------
    str
        The formatted string representation of the float.
    """
    multiplier = 10 ** decimals
    integer_part = int(value)
    fractional_part = int((value - integer_part) * multiplier)

    # Handle negative values
    sign = "-" if value < 0 else ""

    # Convert integer parts to strings
    integer_str = str(abs(integer_part))
    fractional_str = str(fractional_part).zfill(decimals)

    # Construct the final string
    return sign + integer_str + "." + fractional_str


def wrap_numba_error(function):
    """
    Decorator to wrap a numba function with error handling. The last output of the wrapped
    function should be a string with an error message, if any. An empty
    string indicates that no error occurred.

    If an error message is present, an Exception is raised with the message.

    Otherwise, the function returns the outputs of the wrapped function,
    except the last one, which is the error message.

    Usage
    -----
    @wrap_numba_error
    @nb.njit
    def fun(*args, **kwargs):
        ...
        if problems:
            return ..., "Error message"
        else:
            return outputs, ""

    """
    """"""
    def res_fun(*args, **kwargs):
        outputs = function(*args, **kwargs)
        msg = outputs[-1]

        if msg:
            msg = f"Error on numba function {function.__name__}: {msg}"
            raise Exception(msg)

        results = outputs[:-1] if len(outputs) > 2 else outputs[0]
        return results

    return res_fun


import math
from numba import jit


@nb.njit
def log_negative_binomial_pmf(k, n, p):
    """
    Calculate the logarithm of the negative binomial PMF.

    Parameters:
        k (int): Number of failures until the experiment is stopped.
        n (int): Number of successes.
        p (float): Probability of success.

    Returns:
        float: Logarithm of the negative binomial PMF.
    """
    if k < 0 or n <= 0 or not (0 < p < 1):
        raise ValueError("Invalid parameters: k >= 0, n > 0, and 0 < p < 1 are required.")

    log_binom_coeff = (
            math.lgamma(k + n) - math.lgamma(k + 1) - math.lgamma(n)
    )
    log_pmf = log_binom_coeff + n * math.log(p) + k * math.log(1 - p)
    return log_pmf

