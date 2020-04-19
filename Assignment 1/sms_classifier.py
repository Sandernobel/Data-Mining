import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def preprocess():
    """
    Reads in and preprocesses data
    :return:
    """
    fw = open("SmsClean.csv", 'w+')
    with open("SmsCollection.csv", 'r') as fr:
        for line in fr.readlines():
            line = line.replace(';', '\t', 1)
            fw.write(line)

    sms = pd.read_csv("SmsClean.csv", sep='\t')

    ham_texts = sms.loc[sms['label'] == 'ham', 'text']
    spam_texts = sms.loc[sms['label'] == 'spam', 'text']

    ham_lengths = np.asarray([len(x) for x in ham_texts])
    spam_lengths = np.asarray([len(x) for x in spam_texts])

    ham_cap = np.asarray([sum(map(str.isupper, string)) for string in ham_texts]) / ham_lengths
    spam_cap = np.asarray([sum(map(str.isupper, string)) for string in spam_texts]) / spam_lengths

    ham_num = np.asarray([sum(c.isdigit() for c in s) for s in ham_texts]) / ham_lengths
    spam_num = np.asarray([sum(c.isdigit() for c in s) for s in spam_texts]) / spam_lengths

    print(np.mean(ham_cap), np.mean(spam_cap))
    print(np.mean(ham_num), np.mean(spam_num))

    sms['length']  = pd.Series(dtype=int)
    sms['capital'] = pd.Series(dtype=int)
    sms['numbers'] = pd.Series(dtype=int)

    sms.loc[sms['label'] == 'ham', 'length'] = ham_lengths
    sms.loc[sms['label'] == 'spam', 'length'] = spam_lengths

    sms.loc[sms['label'] == 'ham', 'capital'] = ham_cap
    sms.loc[sms['label'] == 'spam', 'capital'] = spam_cap

    sms.loc[sms['label'] == 'ham', 'numbers'] = ham_num
    sms.loc[sms['label'] == 'spam', 'numbers'] = spam_num

    sms.loc[sms['label'] == 'ham', 'label'] = 0
    sms.loc[sms['label'] == 'spam', 'label'] = 1
    sms['label'] = np.asarray(sms['label']).astype(int)

    return sms


def cross_validate(sms):
    """
    cross-validation
    :param sms:
    :return:
    """

    features = sms[['length', 'capital', 'numbers']]
    label = sms['label']

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)

    clf = DecisionTreeClassifier()
    clf2 = DecisionTreeClassifier(criterion='entropy')
    rf = RandomForestClassifier()
    rf2 = RandomForestClassifier(criterion='entropy')

    dt_cvs = cross_val_score(clf, X_train, y_train, cv=10)
    clf2_cvs = cross_val_score(clf2, X_train, y_train, cv=10)
    rf_cvs = cross_val_score(rf, X_train, y_train, cv=10)
    rf2_cvs = cross_val_score(rf2, X_train, y_train, cv=10)

    print(np.mean(dt_cvs), np.mean(clf2_cvs), np.mean(rf_cvs), np.mean(rf2_cvs))

    clf = RandomForestClassifier(criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {clf.score(X_test, y_test)}")
    print(f"Cohen's kappa: {metrics.cohen_kappa_score(y_pred, y_test)}")
    metrics.plot_confusion_matrix(clf, features, label)
    plt.show()

def main():
    """

    :return:
    """
    sms = preprocess()
    cross_validate(sms)


if __name__ == "__main__":
    main()