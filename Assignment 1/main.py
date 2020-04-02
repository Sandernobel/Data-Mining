import pandas as pd
import numpy as np

if __name__ == "__main__":

    odi = pd.read_csv("ODI-2020.csv", sep=';')
    odi.columns = ['Program', 'ML_exp', 'IR_exp', 'Statistics_exp', 'Databases_exp', "Gender", "Chocolate", "Birthday", "Neighbors",
                   "Stand", "Stress", "100_euros", "Random_number", "Bedtime", "Good_day_1", "Good_day_2"]

    print(max(odi.Stress))
    print(odi.describe())
    print(odi.columns)
    print(odi.shape)