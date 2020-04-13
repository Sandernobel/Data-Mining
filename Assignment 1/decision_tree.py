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
        self.y = df[label]

        # Initialize metrics
        self.accuracys = dict()
        self.selected_features = []
        self.scores = [0]


    def run(self, feature):
        """

        :param feature:
        :return:
        """

        self.X = pd.get_dummies(self.df[feature], drop_first=True)

        # Split in training and test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)

        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_test, y_pred

    def bottom_up(self):
        """

        :return:
        """

        # Loop over features
        for column in self.columns:
            y_test, y_pred = self.run(column)

            print(f"Accuracy with feature {column}:", metrics.accuracy_score(y_test, y_pred))
            self.accuracys[column] = metrics.accuracy_score(y_test, y_pred)

        # See which feature scored best
        best_feature = max(self.accuracys, key=self.accuracys.get)
        print(f"Max accuracy: {best_feature}\n")
        self.selected_features.append(best_feature)
        self.scores.append(max(self.accuracys.values()))
        self.columns.remove(self.selected_features[len(self.selected_features) - 1])
        self.accuracys.clear()

        while self.columns:
            print(f"Round {1+len(self.selected_features)}")
            print(f"Current features in model: {self.selected_features}")



            for column in self.columns:
                columns = self.selected_features + [column]
                y_test, y_pred = self.run(columns)
                print(f"Accuracy with feature {column}:", metrics.accuracy_score(y_test, y_pred))
                self.accuracys[column] = metrics.accuracy_score(y_test, y_pred)

            best_feature = max(self.accuracys, key=self.accuracys.get)
            print(f"Max accuracy: {best_feature}\n")
            self.selected_features.append(best_feature)
            self.scores.append(max(self.accuracys.values()))
            self.columns.remove(self.selected_features[len(self.selected_features) - 1])
            self.accuracys.clear()
        print(self.selected_features)
        print(f"\nMaximum accuracy: {max(self.scores)}")
        print(f"Features for max accuracy: {self.selected_features[:self.scores.index(max(self.scores))]}")


    def top_down(self):
        """

        :return:
        """

        self.columns = self.df.columns[1:]
        y_test, y_pred = self.run(self.columns)
        print(f"Accuracy with feature {self.columns}:", metrics.accuracy_score(y_test, y_pred))