import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from helpers import *

def plot_comparison(comparison):
    """
    """

    # Compare different categorical levels
    for x in range(len(list(comparison.keys()))):
        plt.subplot(1,len(list(comparison.keys())),x+1)
        plt.xlabel(list(comparison.keys())[x])
        plot_series(comparison[list(comparison.keys())[x]])

    plt.show()



def plot_series(to_plot):
    """
    Plots distribution of ages
    :param ages:
    :return:
    """
    # count vales and remove NA
    keys, values = get_count(to_plot)

    # plot
    plt.bar(keys, values)