import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from varname import varname

from helpers import *

def cat_vs_con(df, cat, con):
    """
    Plot categorical vs continuous series
    :param cat: categorical series (string)
    :param con: continuous series (string)
    :param mean: whether you want to plot frequency (false) or mean (true)
    :return:
    """

    # Get actual attributes
    cat_array = np.asarray(getattr(df, cat))
    con_array = np.asarray(getattr(df, con))
    cat_values = sorted(set(cat_array))

    counter = dict()

    # Loop over different categorical variables
    for value in cat_values:

        # Select relevant instances and get mean of continuous variable
        bool_mask = np.where(cat_array == value, 1, 0)
        con_values = con_array[bool_mask == 1]
        counter[value] = np.mean(con_values)

    # Plot it
    plt.bar(counter.keys(), counter.values())
    plt.xlabel(cat)
    plt.ylabel(con)
    plt.show()


def con_vs_con(df, con1, con2):
    """
    Gives a scatterplot of two continuous variables
    :param con1: first continuous series
    :param con2: second continuous series
    :return:
    """

    # Get attributes
    first_array = np.asarray(getattr(df, con1))
    second_array = np.asarray(getattr(df, con2))

    # Plot them against each other
    plt.scatter(first_array, second_array)
    plt.xlabel(con1)
    plt.ylabel(con2)
    plt.show()




def cat_vs_cat(df, cat1, cat2):
    """
    :param df: dataframe
    :param cat1: first categorical variable
    :param cat2: second categorical variable
    """

    # Get attributes
    first_array = np.asarray(getattr(df, cat1))
    second_array = np.asarray(getattr(df, cat2))

    first_set = sorted(set(first_array))

    counter = dict()

    # Loop over variables
    for var in range(len(first_set)):
        current_var = first_set[var]

        # Extract relevant subjects and get count of every second categorical variable
        bool_mask = np.where(first_array == current_var, 1, 0)
        second_values = second_array[bool_mask == 1]
        counter[current_var] = Counter(second_values)

        # Sort values
        keys, values = list(counter[current_var].keys()), list(counter[current_var].values())
        x = list(sorted(zip(keys, values)))
        keys, values = zip(*x)

        # Plot them (many plots after each other, one for each categorical level)
        plt.bar(keys, values)
        plt.title(current_var)
        plt.xlabel(cat2)
        plt.ylabel("Frequency")
        plt.show()