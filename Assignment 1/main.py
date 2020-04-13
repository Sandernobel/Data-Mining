import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from plotting import *
from cleaning import *
from decision_tree import DecisionTree
from naive_bayes import *

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',16)

    
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
    df.Program = program(np.asarray(df.Program))                            # categorical
    # ML Course                                                             # categorical
    # IR Course                                                             # categorical
    # Stats Course                                                          # categorical
    # Databases Course                                                      # categorical
    # Gender                                                                # categorical
    # Chocolate                                                             # categorical
    df.Age = birthday(df.Age)                                               # continuous
    df.Neighbors = continuous(df.Neighbors)                                 # continuous
    # Stand                                                                 # categorical
    df.Stress = continuous(df.Stress)                                       # continuous
    df.Euros = continuous(df.Euros)                                         # continuous
    df.Random_number = continuous(df.Random_number)                         # continuous
    df.Bedtime = bedtime(np.asarray(df.Bedtime))                            # continuous
    df.Good_day_1 = good_day_1(np.asarray(df.Good_day_1))                   # categorical
    df.Good_day_2 = good_day_2(np.asarray(df.Good_day_2))                   # categorical

    df.dropna(inplace=True)
    print(df.describe(include='all'))

    plot_frequency(df, "Program", "Age")

    # label = "Program"
    # features = list(df.columns)
    # features.remove(label)
    # cls = DecisionTree(df, features, label)
    # cls.bottom_up()

    # separated = separate_by_class(df, "Age")
    # print(calc_prob(26, df.Age))