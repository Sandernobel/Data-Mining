import lightgbm as lgbm
import numpy as np
import pandas as pd
from collections import Counter


def split_data(df: pd.DataFrame):
    """
    Splits data based on search id's
    :return:
    """
    unique_srch_ids = df['srch_id'].unique()
    train_ids = np.random.choice(unique_srch_ids, size=round(0.8*len(unique_srch_ids)), replace=False)
    val_ids = np.setdiff1d(unique_srch_ids, train_ids)

    train_data = df.loc[df['srch_id'].isin(train_ids)]
    val_data = df.loc[df['srch_id'].isin(val_ids)]

    return train_data, val_data


def test_lgbm(df: pd.DataFrame, model: lgbm.LGBMRanker):
    """

    :param df:
    :return:
    """

    test_pred = model.predict(df.drop('date_time', axis=1))
    df['relevance'] = test_pred

    return df


def train_lgbm(df: pd.DataFrame, seed: int = 15):
    """

    :param df:
    :param seed: random seed
    :return:
    """
    np.random.seed(seed)

    print("\tSplitting data")
    train_data, val_data = split_data(df.drop(['date_time', 'click_bool', 'booking_bool', 'position'], axis=1))
    y_train, y_val = train_data['relevance'], val_data['relevance']
    X_train, X_val = train_data.drop('relevance', axis=1), val_data.drop('relevance', axis=1)

    train_queries = list(Counter(np.asarray(X_train['srch_id'])).values())
    val_queries = list(Counter(np.asarray(X_val['srch_id'])).values())

    gbm = lgbm.LGBMRanker()
    print(f"\tTraining LGBM Ranker")
    print(gbm.fit(X_train, y_train, group=train_queries,
        eval_set=[(X_val, y_val)], eval_group = [val_queries],
        eval_at=[5], early_stopping_rounds=50))
    return gbm