import pandas as pd
import numpy as np
import re

from collections import Counter

#########################
# helper functions
#########################


def clean_up(programs, swapper):
    """
    Cleans up program by converting all programs to its abbreviations
    """

    # Loop over keys
    for key in swapper.keys():

        r = re.compile(".*" + str(key) + ".*", re.IGNORECASE)
        newlist = list(filter(r.match, programs))
        programs = np.asarray([swapper[key] if x in newlist else x for x in programs])

    return programs


def delete_rest(programs, swapper):
    """
    Replaces all other data with NA
    """

    # Swaps everything that you don't want to keep with NA
    values = list(swapper.values())
    prog = np.asarray(['NA' if x not in values else x for x in programs])
    return prog

def calc_mean(series, min_val=None, max_val=None):
    """
    """
    series = pd.Series(series)
    if min_val and max_val:
        x_mean = np.asarray(pd.to_numeric(series, errors='coerce').dropna())
        x_mean = np.where(x_mean < min_val, min_val, x_mean)
        x_mean = np.where(x_mean > max_val, max_val, x_mean)
    else:
        x_mean = np.asarray(pd.to_numeric(series, errors='coerce').dropna()).astype(int)
    return np.mean(x_mean)


def get_count(to_plot):
    """
    Plots series
    """
    counted = Counter(to_plot)
    counted.pop("NA")

    keys, values = list(counted.keys()), list(counted.values())
    x = list(sorted(zip(keys, values)))
    keys, values = zip(*x)

    return keys, values


def impute_mean(series, mean, min_val=None, max_val=None):
    """
    """

    x = np.asarray(pd.to_numeric(series, errors='coerce')).astype(int)
    x = np.where(x < -100, mean, x).astype(int)

    if min_val and max_val:
        x = np.where(x < min_val, min_val, x)
        x = np.where(x > max_val, max_val, x)
    return x

def compare_series(string, compare, count=False):
    """
    Compares stress levels in series
    :return:
    """

    # Extract categorical variable
    series = getattr(df, string)
    programs = set(np.asarray(series))

    # Initialize dictionaries
    stress_means = dict()
    counters = dict()

    # Loop over series
    for program in programs:

        # Extract relevant subjects
        x = df.loc[df[string] == program]

        # Extract continuous variable and delete na's
        to_compare = getattr(x, compare)
        x_stress = np.asarray(to_compare)
        x_stress = x_stress[x_stress != 'NA'].astype(int)

        # Only add if more than 2 values, otherwise it's too crowded
        if len(x_stress) > 2:
            if count:
                counter = Counter(x_stress)
                counters[program] = counter
            else:
                stress_means[program] = np.mean(x_stress)

    if count:
        return counters
    return stress_means