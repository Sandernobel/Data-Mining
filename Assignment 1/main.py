import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from helpers import *
from plotting import *

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



    
if __name__ == "__main__":
    # Read in data
    file = "ODI-2020.csv"
    df = pd.read_csv(file, delimiter=';')

    # Change columns
    columns = ['Program', 'ML_course', 'IR_course', 'Stats_course', 'Databases_course', "Gender", "Chocolate", "Age",
               "Neighbors", "Stand", "Stress", "Euros", "Random_number", "Bedtime", "Good_day_1", "Good_day_2"]
    df.columns = columns

    # extract males and females
    male = df.loc[df['Gender'] == 'male']
    female = df.loc[df['Gender'] == 'female']

    # Clean up programs
    df.Program = program(np.asarray(df.Program))
    df.Age = birthday(np.asarray(df.Age))
    df.Stress = stress(np.asarray(df.Stress), cat=False)
    df.Bedtime = bedtime(np.asarray(df.Bedtime))

    gender_bedtimes = compare_series('Gender', 'Bedtime', count=True)
    plot_comparison(gender_bedtimes)

    program_bedtimes = compare_series('Program', "Bedtime", count=True)
    program_bedtimes = {key: value for (key, value) in program_bedtimes.items() if len(list(program_bedtimes[key].values())) > 3}
    plot_comparison(program_bedtimes)