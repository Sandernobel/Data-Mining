import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from helpers import *
from plotting import *
from cleaning import *
from decision_tree import DecisionTree

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
    df.Age = birthday(np.asarray(df.Age))                                   # grouped
    df.Neighbors = continuous(np.asarray(df.Neighbors))                     # continuous
    # Stand                                                                 # categorical
    df.Stress = continuous(np.asarray(df.Stress), 0.001, 100)               # continuous
    df.Euros = continuous(np.asarray(df.Euros), 0.001, 100)                 # continuous
    df.Random_number = continuous(np.asarray(df.Random_number), 0.001, 100) # continuous
    df.Bedtime = bedtime(np.asarray(df.Bedtime))                            # grouped
    df.Good_day_1 = good_day_1(np.asarray(df.Good_day_1))                   # categorical
    df.Good_day_2 = good_day_2(np.asarray(df.Good_day_2))                   # categorical

    # gender_bedtimes = compare_series('Gender', 'Bedtime', count=True)
    # plot_comparison(gender_bedtimes)
    #
    # program_bedtimes = compare_series('Program', "Bedtime", count=True)
    # program_bedtimes = {key: value for (key, value) in program_bedtimes.items() if len(list(program_bedtimes[key].values())) > 3}
    # plot_comparison(program_bedtimes)

    # cat_vs_cat(df, "Good_day_1", "Bedtime")
    # cat_vs_cat(df, "Program", "ML_course")
    label = "Program"
    columns.remove(label)
    clsf = DecisionTree(df, columns, label)
    clsf.run_bottom_up()