import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def plot_series(to_plot):
    """
    Plots distribution of ages
    :param ages:
    :return:
    """
    # count vales and remove NA
    counted = Counter(to_plot)
    if 'NA' in counted.keys():
        counted.pop("NA")

    # sort keys
    keys, values = list(counted.keys()), list(counted.values())
    x = list(sorted(zip(keys, values)))
    keys, values = zip(*x)

    # plot
    plt.bar(keys, values)