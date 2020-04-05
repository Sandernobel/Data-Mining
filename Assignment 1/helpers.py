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
               "CS": "CS",
               "QRM": "QRM",
               "Quantitative risk management": "QRM",
               "Econometrics": "Econometrics",
               "Business Administration": "Business administration",
               "Bioinformatics": "Bioinformatics",
               "Computational Science": "Computational science",
               "Human language technologies": "Human language technologies",
               "Digital business and innovation": "Digital business and innovation",
               "Information Sciences": "Information sciences",
               "Exchange": "Exchange"}

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

def stress(levels, cat):
    """
    Cleans up stress series
    :param levels:
    :return:
    """
    swap = {',': -50,
            '8-100': -50,
            '-': -50}

    stress = clean_up(np.asarray(levels), swap).astype(int)
    stress = np.asarray(['NA' if x == -50 else x for x in stress])

    if cat:
        breaks = [0, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
        values = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
        stress_swap = dict()

        for x in range(len(breaks) - 1):
            for y in range(breaks[x], breaks[x + 1]):
                stress_swap[str(y)] = values[x]

        stress = np.asarray([stress_swap[z] if z != 'NA' else z for z in stress])
    return stress
