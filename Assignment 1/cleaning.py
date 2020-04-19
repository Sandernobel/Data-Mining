import pandas as pd
import numpy as np
import re

##############################
# functions to clean up dataset
##############################


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


def program(prog):
    """
    Cleans up program
    :param prog:
    :return:
    """
    swapper = {"Artificial Intelligence": "Beta",
               "AI": "Beta",
               "Business Analytics": "Business",
               "Computer science": "Beta",
               "QRM": "Business",
               "Quantitative risk management": "Business",
               "Econometrics": "Business",
               "Business Administration": "Business",
               "Bioinformatics": "Beta",
               "Computational Science": "Beta",
               "Digital business": "Beta",
               "Information Science": "Beta",
               "Information studies": "Beta",
               "Human language": "Other",
               "Human movement": "Other",
               "Datascience": "Beta",
               "Health sciences": "Other",
               "Fintech": "Business",
               "Erasmus": "Other",
               "Computing system": "Beta",
               "Finance": "Business",
               "Scientific computing": "Beta",
               "Big data engineering": "Beta",
               "EOR": "Business",
               "Exchange": "Other",
               "Physics": "Beta",
               "CPS": "Other",
               "CLS": "Other",
               "Computer systems": "Beta",
               "Forensic science": "Other",
               "MSc": "Other",
               "BA": "Business",
               "CS": "Beta"}

    # swap all studies to one standard notation
    prog = clean_up(prog, swapper)

    return prog


def birthday(dates):
    """
    Cleans up birthday
    :param dates:
    :return:
    """
    # create dictionary
    birth_swap = {str(x) : 2020-x for x in range(1970,2001)}
    birth_swap.update({'-'+str(x) : 120-x for x in range(70, 99)})
    birth_swap.update({'/' + str(x) : 120-x for x in range(70, 99)})

    birth = clean_up(np.asarray(dates), birth_swap)
    birth_pd = np.asarray(pd.to_numeric(pd.Series(birth), errors='coerce')).astype(int)
    birth_pd = pd.Series([np.NaN if (x < 0 or x > 100) else x for x in birth_pd])
    birth_pd.fillna(birth_pd.mean(), inplace=True)
    birth_pd = np.asarray(birth_pd).astype(int)

    return birth_pd


def continuous(series):
    """
    Cleans up continuous data
    :param neighbors:
    :return:
    """

    serie = pd.to_numeric(series, errors='coerce')
    serie.fillna(serie.mean(), inplace=True)
    serie = np.where(serie < 0, 0, serie)
    serie = np.where(serie > 100, 100, serie)
    return serie


def bedtime(times):
    """
    Cleans up bedtime series
    :param bedtime:
    :return:
    """

    # Create swapper
    bed_swap = {str(x) + ':': x%12 for x in range(10, 24)}
    for x in range(9):
        bed_swap[str(x) + ':'] = x%12
        bed_swap[str(x) + 'pm'] = x%12
        bed_swap[str(x) + 'am'] = x%12
        bed_swap[str(x) + ' pm'] = x%12
        bed_swap[str(x) + ' am'] = x%12
        bed_swap[str(x) + 'h'] = x%12
    bed_swap['10'] = 10
    bed_swap['01'] = 1
    bed_swap['22'] = 10
    bed_swap['23'] = 11
    bed_swap['24'] = 0
    bed_swap['Twelve'] = 0
    bed_swap['00'] = 0
    bed_swap['17'] = 5
    bed_swap['0,2'] = 0
    bed_swap['16'] = 4
    bed_swap['1,3'] = 1
    bed_swap['0,3'] = 0
    bed_swap['13'] = 1
    bed_swap['12'] = 0
    bed_swap['yes'] = "NA"
    bed_swap['30'] = "NA"
    bed_swap['dont know'] = "NA"
    bed_swap['nan'] = "NA"
    bed_swap['que'] = "NA"
    bed_swap['the beatles'] = "NA"
    bed_swap['troubles'] = "NA"
    bed_swap['didn'] = "NA"
    bed_swap['home'] = "NA"

    bedtimes = list(clean_up(np.asarray(times).astype(str), bed_swap))
    bedtimes[bedtimes.index('?')] = "NA"
    bedtimes = np.asarray(pd.to_numeric(pd.Series(bedtimes), errors='coerce')).astype(int)
    bedtimes = pd.Series(np.where(bedtimes < 0, np.NaN, bedtimes))
    bedtimes.fillna(method='ffill', inplace=True)
    bedtimes = np.asarray(pd.to_numeric(pd.Series(bedtimes), errors='coerce')).astype(int)

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

def clean(df):
    """

    :param df:
    :return:
    """
    # Change columns
    columns = ['Program', 'ML_course', 'IR_course', 'Stats_course', 'Databases_course', "Gender", "Chocolate", "Age",
               "Neighbors", "Stand", "Stress", "Euros", "Random_number", "Bedtime", "Good_day_1", "Good_day_2"]
    df.columns = columns

    # Clean up programs
    df.Program = program(np.asarray(df.Program))                            # categorical
    df.Age = birthday(df.Age)                                               # continuous
    df.Neighbors = continuous(df.Neighbors)                                 # continuous
    # Stand                                                                 # categorical
    df.Stress = continuous(df.Stress)                                       # continuous
    df.Euros = continuous(df.Euros)                                         # continuous
    df.Random_number = continuous(df.Random_number)                         # continuous
    df.Bedtime = bedtime(np.asarray(df.Bedtime))                            # continuous
    df.Good_day_1 = good_day_1(np.asarray(df.Good_day_1))                   # categorical
    df.Good_day_2 = good_day_2(np.asarray(df.Good_day_2))                   # categorical

    return df