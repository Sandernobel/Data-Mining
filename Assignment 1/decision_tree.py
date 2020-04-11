import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

class DecisionTree(object):

    def __init__(self, df, columns, label):
        """
        init with dataframe, features and label
        :param df:
        :param features:
        :param label:
        """
        self.df = df
        self.columns = columns
        self.label = label
        self.X
        self.y = df[label]

        # Initialize metrics
        self.accuracys = dict()
        self.selected_features = set()
        self.scores = []

    def run_bottom_up(self):
        """

        :return:
        """

        # Loop over features
        for column in self.columns:
            # Decision tree function doesn't accept categorical variables, except when using this function...
            self.X = pd.get_dummies(self.df[column], drop_first=True)

            # Split in training and test
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)

            clf = DecisionTreeClassifier()
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            print(f"Accuracy with feature {column}:", metrics.accuracy_score(y_test, y_pred))
            self.accuracys[column] = metrics.accuracy_score(y_test, y_pred)

        # See which feature scored best
        best_feature = max(self.accuracys, key=self.accuracys.get)
        print(f"Max accuracy: {best_feature}")
        self.selected_features.add(best_feature)
        self.scores.append(max(self.accuracys.values()))


