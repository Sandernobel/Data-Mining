import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from cleaning import *

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',100000)

def plot(response, cat_columns, num_columns):
    """

    :return:
    """
    for i in cat_columns:
        x, y = response, i
        (df
         .groupby(x)[y]
         .value_counts(normalize=True)
         .mul(100)
         .rename('percent')
         .reset_index()
         .pipe((sns.catplot, 'data'), x=x, y='percent', hue=y, kind='bar'))
        plt.show()

    for i in num_columns:
        sns.barplot(df_copy[response], df_copy[i])
        plt.show()

if __name__ == "__main__":

    file = "ODI-2020.csv"
    df = pd.read_csv(file, sep=';')

    df_copy = clean(df)

    response = 'Program'

    independent_vars = df_copy.drop(response, axis=1)
    num_columns = list(independent_vars.select_dtypes(include=['float64']).columns)
    cat_columns = list(independent_vars.select_dtypes(include=['object']).columns)

    # plot(response, cat_columns, num_columns)

    # Encode all categorical data to numerical data
    dummy_data = pd.get_dummies(data=independent_vars, columns=cat_columns)

    # First try with all data
    X_train, X_test, y_train, y_test = train_test_split(dummy_data, df_copy[response], test_size=0.3, random_state=42)

    # Make one decision tree with splits based on entropy(information gain) and one based on gini
    clf = DecisionTreeClassifier(criterion='entropy')
    clf2 = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    clf2.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    y_2_pred = clf2.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_2_pred)}")

    # Check which feature is most important
    X_new = SelectKBest(chi2, k=1).fit_transform(dummy_data, df_copy[response])

    # Test again with only that feature
    X_train, X_test, y_train, y_test = train_test_split(X_new, df_copy[response], test_size=0.3, random_state=15)

    clf = DecisionTreeClassifier(criterion='entropy')
    clf2 = DecisionTreeClassifier()

    clf.fit(X_train, y_train)
    clf2.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    y_2_pred = clf2.predict(X_test)

    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_2_pred)}")
    # Accuracy is better with only 1 feature (~60%)