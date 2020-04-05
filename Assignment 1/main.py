import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import *
from plotting import *


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
    plot_age(df.Age)