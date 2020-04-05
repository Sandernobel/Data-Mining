import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from plotting import *

def compare_stress(series, string):
    """
    Compares stress levels among programs
    :return:
    """
    programs = set(np.asarray(series))

    stress_means = dict()
    for program in programs:
        x = df.loc[df[string] == program]
        x_stress = np.asarray(x.Stress)
        x_stress = x_stress[x_stress != 'NA'].astype(int)
        if len(x_stress) > 0:
            stress_means[program] = np.mean(x_stress)

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

    df.Program = program(np.asarray(df.Program))
    df.Age = birthday(np.asarray(df.Age))
    df.Stress = stress(np.asarray(df.Stress), cat=False)

    program_means = compare_stress(df.Program, 'Program')
    age_means = compare_stress(df.Age, 'Age')

    plt.subplot(1,2,1)
    plot_series(program_means)
    plt.subplot(1,2,2)
    plot_series(age_means)
    plt.show()