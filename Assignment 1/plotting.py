import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from varname import varname

def plot_mean(df, xlabel, ylabel):
    """
    Plot categorical vs continuous series
    :param xlabel: categorical series (string)
    :param ylabel: continuous series (string)
    :param mean: whether you want to plot frequency (false) or mean (true)
    :return:
    """

    # Get actual attributes
    cat_array = np.asarray(getattr(df, xlabel))
    con_array = np.asarray(getattr(df, ylabel))

    cat_array = np.asarray([cat_array[x] for x in range(len(con_array)) if con_array[x] > 0])
    con_array = np.asarray([con_array[x] for x in range(len(con_array)) if con_array[x] > 0]).astype(int)
    cat_values = sorted(set(cat_array))

    counter = dict()
    # Loop over different categorical variables
    for value in cat_values:

        # Select relevant instances and get mean of continuous variable
        bool_mask = np.where(cat_array == value, 1, 0)
        con_values = con_array[bool_mask == 1]
        print(con_values)
        counter[value] = np.mean(con_values)

    # Plot it
    plt.bar(counter.keys(), counter.values())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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




def plot_frequency(df, xlabel, ylabel, integer=False):
    """
    :param df: dataframe
    :param xlabel: first categorical variable
    :param ylabel: second variable
    """

    # Get attributes
    first_array = np.asarray(getattr(df, xlabel))
    second_array = np.asarray(getattr(df, ylabel))

    first_set = sorted(set(first_array))
    first_array = np.asarray([first_array[x] for x in range(len(second_array)) if second_array[x] > 0])

    if integer:
        second_array = np.asarray([second_array[x] for x in range(len(second_array)) if second_array[x] > 0]).astype(int)
    else:
        second_array = np.asarray([second_array[x] for x in range(len(second_array)) if second_array[x] > 0])
    counter = dict()

    # Loop over variables
    for var in range(len(first_set)):
        plt.subplot(1,len(first_set),var+1)
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
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
    plt.show()