import lightgbm as lgbm
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import KFold


def split_data(df: pd.DataFrame):
    """
    Splits data based on search id's
    :return:
    """

    kf = KFold(n_splits=5) # 5 splits ipv 10 want het duurt al best lang om te trainen en het zijn grote groepen
    unique_srch_ids = df['srch_id'].unique()

    train_ids, val_ids = [], []
    for train_index, val_index in kf.split(unique_srch_ids):
        train_ids.append(unique_srch_ids[train_index])
        val_ids.append(unique_srch_ids[val_index])
    return np.asarray(train_ids), np.asarray(val_ids)


def get_predictions(df: pd.DataFrame, model: lgbm.LGBMRanker):
    """

    :param df:
    :return:
    """

    print(f'\tPredicting relevance')
    test_pred = model.predict(df)
    df['relevance'] = test_pred
    df.sort_values(by=['srch_id', 'relevance'], ascending=[True, False], inplace=True)
    kaggle_answer = pd.DataFrame({'srch_id': df['srch_id'],
                                  'prop_id': df['prop_id']})
    print(f'\t Writing answers to csv')
    kaggle_answer.to_csv('expedia_answer.csv', index=False)

def save_model(model: lgbm.LGBMRanker, cv_scores: list):
    """
    Saves model
    :param model:
    :return:
    """

    out_file = open("model_"+str(np.mean(cv_scores)), 'wb+')
    pickle.dump(model, out_file)
    out_file.close()

def train_lgbm(df: pd.DataFrame):
    """

    :param df:
    :param seed: random seed
    :return:
    """

    print("\tSplitting data")
    df.drop(['click_bool', 'booking_bool', 'position'], axis=1, inplace=True)
    train_ids, val_ids = split_data(df)
    cv_scores = []

    for i, train_id in enumerate(train_ids):

        train_data = df.loc[df['srch_id'].isin(train_id)]
        val_data = df.loc[df['srch_id'].isin(val_ids[i])]

        y_train, y_val = train_data['relevance'], val_data['relevance']
        X_train, X_val = train_data.drop('relevance', axis=1), val_data.drop('relevance', axis=1)

        train_queries = list(Counter(np.asarray(X_train['srch_id'])).values())
        val_queries = list(Counter(np.asarray(X_val['srch_id'])).values())

        gbm = lgbm.LGBMRanker()
        print(f"\tTraining LGBM Ranker, fold {i+1}")
        gbm.fit(X_train, y_train, group=train_queries,
            eval_set=[(X_val, y_val)], eval_group = [val_queries],
            eval_at=[5], early_stopping_rounds=50)
        print('\n')
        cv_scores.append(gbm.best_score_['valid_0']['ndcg@5'])

    print(cv_scores)
    save_model(gbm, cv_scores)
    return gbm