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
from sklearn.tree import plot_tree

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
        # x, y = response, i
        # (df
        #  .groupby(x)[y]
        #  .value_counts(normalize=True)
        #  .mul(100)
        #  .rename('percent')
        #  .reset_index()
        #  .pipe((sns.catplot, 'data'), x=x, y='percent', hue=y, kind='bar'))
        sns.barplot(df_copy[i], df_copy[response])
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
    sns.barplot(df_copy['Age'], df_copy['Stress'])
    plt.title('Mean stress per age')
    plt.show()
    # plot(response, cat_columns, num_columns)

    # Categorische data continue maken
    dummy_data = pd.get_dummies(data=independent_vars, columns=cat_columns)

    # Volledige data opsplitten in train en test
    X_train, X_test, y_train, y_test = train_test_split(dummy_data, df_copy[response], test_size=0.3, random_state=15)

    # Kiezen welke modellen je wil testen met welke hyperparameters
    model_names = ["RandomForest_entropy", "RandomForest_gini"]
    models = [DecisionTreeClassifier(criterion='entropy'), DecisionTreeClassifier(criterion='gini')]

    cross_val_scores = {}
    for x in range(len(model_names)):
        cross_val_scores[model_names[x]] = cross_val_score(models[x], X_train, y_train, cv=5).mean()
    print(cross_val_scores)
    best_model = models[model_names.index(max(cross_val_scores, key=lambda x: cross_val_scores[x]))]

    # 5-fold cross validation, beste model kiezen

    # Beste model toepassen op volledige dataset
    best_model.fit(X_train, y_train)
    print(f"Ultimate accuracy for {max(cross_val_scores, key=lambda x: cross_val_scores[x])}: {best_model.score(X_test, y_test)}")

    ## Itereren over aantal features om te kijken welke meeste accuracy heeft
    features = {}
    feature_scores = []
    columns = []
    trees = []

    for x in range(1,len(list(dummy_data.columns))):
        kbest = SelectKBest(k=x)
        fit = kbest.fit(dummy_data, df_copy[response])
        X_new = kbest.fit_transform(dummy_data, df_copy[response])
        X_train, X_test, y_train, y_test = train_test_split(X_new, df_copy[response], test_size=0.3, random_state=15)
        best_model = models[model_names.index(max(cross_val_scores, key=lambda y: cross_val_scores[y]))]
        best_model.fit(X_train, y_train)
        trees.append(best_model)
        print(f"Final accuracy for {max(cross_val_scores, key=lambda y: cross_val_scores[y])} with {x} feature(s): {best_model.score(X_test, y_test)}")
        features[x] = best_model.score(X_test, y_test)
        feature_scores.append(features[x])
        columns.append(kbest.get_support(indices=True))

    plt.plot(range(1,len(list(dummy_data.columns))), feature_scores)
    plt.title(f"Accuracy for amount of features with {response} as target variable")
    plt.xlabel("No of features")
    plt.ylabel("Accuracy")
    plt.show()

    print(features)
    print(max(features, key= lambda x: features[x]))
    conf_interval = [max(feature_scores) - 1.96 * np.sqrt((1/y_test.size * max(feature_scores)) * (1 - max(feature_scores))),
                     max(feature_scores) + 1.96 * np.sqrt((1/y_test.size * max(feature_scores)) * (1 - max(feature_scores)))]
    important_columns = dummy_data.iloc[:,columns[max(features, key=lambda x: features[x]) - 1]]
    print(f"Most important features: {list(important_columns.columns)}")
    print(f"Confidence interval: {conf_interval}")
    plot_tree(trees[max(features, key= lambda x: features[x]) - 1], filled=True)
    plt.show()