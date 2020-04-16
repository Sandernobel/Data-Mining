import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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

    # Data inlezen
    file = "ODI-2020.csv"
    df = pd.read_csv(file, sep=';')

    # Data opschonen
    df_copy = clean(df)

    # Target variabele kiezen
    response = 'Program'

    # Features kiezen en onderscheiden in categorisch/continue
    independent_vars = df_copy.drop(response, axis=1)
    num_columns = list(independent_vars.select_dtypes(include=['float64']).columns)
    cat_columns = list(independent_vars.select_dtypes(include=['object']).columns)

    ## Eventueel plotten tegenover alle andere variabelen
    # plot(response, cat_columns, num_columns)

    # Categorische data continue maken
    dummy_data = pd.get_dummies(data=independent_vars, columns=cat_columns)

    # Volledige data opsplitten in train en test
    X_train, X_test, y_train, y_test = train_test_split(dummy_data, df_copy[response], test_size=0.3, random_state=15)

    # Kiezen welke modellen je wil testen met welke hyperparameters
    model_names = ["DecisionTree_entropy", "DecisionTree_gini", "RandomForest_gini", "RandomForest_entropy", "MLP", "KNN", "RandomForest_tuned"]
    models = [DecisionTreeClassifier(criterion='entropy'), DecisionTreeClassifier(), RandomForestClassifier(),
              RandomForestClassifier(criterion='entropy'), MLPClassifier(max_iter=1000), KNeighborsClassifier(n_neighbors=8),
              RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=2, min_samples_leaf=1)]

    # 5-fold cross validation, beste model kiezen
    cross_val_scores = {}
    for x in range(len(model_names)):
        cross_val_scores[model_names[x]] = cross_val_score(models[x], X_train, y_train, cv=10).mean()
    print(cross_val_scores)
    best_model = models[model_names.index(max(cross_val_scores, key=lambda x: cross_val_scores[x]))]

    # Beste model toepassen op volledige dataset
    best_model.fit(X_train, y_train)
    print(f"Ultimate accuracy for {max(cross_val_scores, key=lambda x: cross_val_scores[x])}: {best_model.score(X_test, y_test)}")

    ## Itereren over aantal features om te kijken welke meeste accuracy heeft
    features = {}
    feature_scores = []
    columns = []

    for x in range(1,len(list(dummy_data.columns))):
        kbest = SelectKBest(k=x)
        fit = kbest.fit(dummy_data, df_copy[response])
        X_new = kbest.fit_transform(dummy_data, df_copy[response])
        X_train, X_test, y_train, y_test = train_test_split(X_new, df_copy[response], test_size=0.3, random_state=15)
        best_model.fit(X_train, y_train)
        print(f"Final accuracy for {max(cross_val_scores, key=lambda y: cross_val_scores[y])} with {x} feature(s): {best_model.score(X_test, y_test)}")
        features[x] = best_model.score(X_test, y_test)
        feature_scores.append(features[x])
        columns.append(kbest.get_support(indices=True))

    plt.plot(range(1,len(list(dummy_data.columns))), feature_scores)
    plt.title(f"Accuracy op aantal features met {response} als target variabele")
    plt.xlabel("Aantal features")
    plt.ylabel("Accuracy")
    plt.show()

    print(features)
    print(max(features, key= lambda x: features[x]))
    important_columns = dummy_data.iloc[:,columns[max(features, key=lambda x: features[x])]]
    print(f"Most important features: {important_columns.columns}")