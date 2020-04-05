import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from collections import Counter


def clean_up(programs, swapper):
    """
    Cleans up program by converting all programs to its abbreviations
    """
    for key in swapper.keys():
        r = re.compile(".*" + str(key) + ".*", re.IGNORECASE)
        newlist = list(filter(r.match, programs))
        programs = np.asarray([swapper[key] if x in newlist else x for x in programs])

    return programs


def delete_rest(programs, swapper):
    """
    Replaces all other data with NA
    """

    values = list(swapper.values())
    prog = np.asarray(['NA' if x not in values else x for x in programs])
    return prog

def program(prog):
    #### Clean up Program
    swapper = {"Artificial Intelligence": "AI",
               "AI": "AI",
               "Business Analytics": "BA",
               "Computer science": "CS",
               "QRM": "QRM",
               "Quantitative risk management": "QRM",
               "Econometrics": "Econometrics",
               "Business Administration": "Business administration",
               "Bioinformatics": "Bioinformatics",
               "Computational Science": "Computational science"}

    # swap all studies to one standard notation
    prog = clean_up(prog, swapper)

    return prog

def birthday(dates):
    #### Clean up Birthday

    # create dictionary
    birth_swap = dict()
    for x in range(1992, 1999):
        birth_swap[x] = str(2020 - x - 1) + '/' + str(2020 - x)

    # clean up array
    birth = clean_up(dates, birth_swap)
    # swap all other data with NA
    birth = delete_rest(np.asarray(birth), birth_swap)
    return birth