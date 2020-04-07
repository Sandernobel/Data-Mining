import pandas as pd
import numpy as np

from helpers import *

##############################
# functions to clean up dataset
##############################

def program(prog):
    """
    Cleans up program
    :param prog:
    :return:
    """
    swapper = {"Artificial Intelligence": "AI",
               "AI": "AI",
               "Business Analytics": "BA",
               "Computer science": "CS",
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
    prog = np.asarray(['other' if x not in swapper.values() else x for x in prog])

    return prog


def birthday(dates):
    """
    Cleans up birthday
    :param dates:
    :return:
    """
    # create dictionary
    birth_swap = dict()
    for x in range(89, 99):
        y = '19' + str(x)
        birth_swap[x] = str(2020 - int(y) - 1) + '/' + str(2020 - int(y))
    birth_swap[99] = str(2020 - 1999 - 1) + '/' + str(2020 - 1999)

    birth = clean_up(np.asarray(dates), birth_swap)

    return birth


def continuous(series, min_val=None, max_val=None):
    """
    Cleans up continuous data
    :param neighbors:
    :return:
    """

    series_mean = calc_mean(series, min_val, max_val)
    new_series = impute_mean(series, series_mean, min_val, max_val)
    return new_series


def bedtime(times):
    """
    Cleans up bedtime series
    :param bedtime:
    :return:
    """

    # Create swapper
    bed_swap = {str(x) + ':': str(x % 12) + '-' + str(x % 12 + 1) for x in range(10, 24)}
    for x in range(7):
        bed_swap[str(x) + ':'] = str(x) + '-' + str(x + 1)
        bed_swap[str(x) + 'pm'] = str(x) + '-' + str(x + 1)
        bed_swap[str(x) + 'am'] = str(x) + '-' + str(x + 1)
        bed_swap[str(x) + ' pm'] = str(x) + '-' + str(x + 1)
        bed_swap[str(x) + ' am'] = str(x) + '-' + str(x + 1)
        bed_swap[str(x) + 'h'] = str(x) + '-' + str(x + 1)
    bed_swap['10'] = '10-11'
    bed_swap['01'] = '1-2'
    bed_swap['23'] = '11-12'
    bed_swap['24'] = '12-00'

    bedtimes = clean_up(np.asarray(times).astype(str), bed_swap)
    bedtimes = delete_rest(bedtimes, bed_swap)
    return bedtimes

def good_day_1(series):
    """
    Cleans up first good day series
    :param series:
    :return:
    """

    words = {'weather': ['weather', 'sun'],
             'sport': ['sport', 'workout', 'gym', 'exercise', 'work-out', 'football', 'match', 'running', 'golf',
                       'boxing'],
             'productivity': ['productive', 'productivity', 'progress', 'work', 'busy'],
             'friends': ['friends', 'company', 'people'],
             'food': ['food', 'breakfast', 'grapefruits', 'cake', 'meals', 'chocolate'],
             'fun': ['fun', 'laugh', 'lauging'],
             'drinks': ['alcohol', 'beer', 'drinks', 'coffee'],
             'sleep': ['sleep', 'slept', 'rest']}

    swapper = dict()
    for x in list(words.items()):
        for y in x[1]:
            swapper[y] = x[0]

    good2 = clean_up(series, swapper)
    good2 = np.asarray(["other" if x not in words.keys() else x for x in good2])
    return good2

def good_day_2(series):
    """
    Clean second good day series
    :param series:
    :return:
    """
    words = {
        'sport': ['sport', 'jogging', 'exercise', 'football', 'basketball', 'gym', 'running', 'dancing', 'skating'],
        'productivity': ['study', 'productive', 'work', 'done', 'achiev', 'full', 'accomplish', 'progress', 'useful',
                         'schedule'],
        'socialize': ['friends', 'family', 'people', 'social', 'company'],
        'weather': ['sun', 'rain', 'warm', 'temperature', 'sky', 'weather', 'temp', 'wheather'],
        'food': ['food', 'dinner', 'sugar', 'breakfast', 'pie', 'cooking', 'eat', 'chocolate', 'ice', 'snack'],
        'music': ['song', 'music', 'drumming'],
        'drinks': ['coffee', 'latte', 'beer'],
        'playing': ['chess', 'games', 'playing'],
        'sleep': ['sleep', 'wake'],
        'fun': ['fun', 'laugh', 'party'],
        'reading': ['read'],
        'outside': ['beach', 'outside']}

    swapper = dict()
    for x in list(words.items()):
        for y in x[1]:
            swapper[y] = x[0]

    good_3 = clean_up(series, swapper)
    good_3 = np.asarray(["other" if x not in words.keys() else x for x in good_3])
    return good_3