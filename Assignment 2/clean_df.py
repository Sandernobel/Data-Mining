import numpy as np
import pandas as pd


def impute_with_dist(feature: pd.Series):
    """
    Imputes na's of column according to distribution
    :param df:
    :param feature:
    :return:
    """

    probs = feature.value_counts(normalize=True)
    isnull = feature.isnull()
    feature[isnull] = np.random.choice(probs.index, size=len(feature[isnull]), p=probs.values)
    return feature

def aggregate_competitors(df: pd.DataFrame, max_vals: int):
    """
    Aggregates all competitor columns
    :param df:
    :return:
    """
    for rate_inv in ['rate', 'inv']:

        comp = pd.Series(np.zeros(max_vals, ))

        for x in range(1,9):
            if 'comp_' + str(x) + rate_inv in df.columns:
                comp += np.where(df['comp'+str(x)+rate_inv].isnull(), 0, df['comp'+str(x)+rate_inv])

        comp = np.where(comp < -1, -1, comp)
        comp = pd.Series(np.where(comp > 1, 1, comp))
        df.loc[:, 'comp_' + rate_inv] = comp

    df = df.drop(['comp2_rate', 'comp3_rate', 'comp5_rate', 'comp8_rate',
                  'comp2_inv', 'comp3_inv', 'comp5_inv', 'comp8_inv',
                  'comp5_rate_percent_diff', 'comp8_rate_percent_diff', 'comp2_rate_percent_diff',
                  'orig_destination_distance'], axis=1)
    return df


def clean_up_df(df: pd.DataFrame, file: str, percent=0.90):
    """
    Cleans dataframe and deletes/imputes missing values
    :param df:
    :return:
    """
    max_vals = max(df.count().values)
    null_values = df.isnull().sum()

    print(f"\tDeleting every column with over {percent*100}% missing values")
    to_keep = null_values.index[np.asarray(null_values / max_vals < percent)]
    df_clean = df[to_keep]
    print(f"\t\tDeleting completed\n\tAggregating all competitor columns")
    df_clean = aggregate_competitors(df_clean, max_vals)
    print(f"\t\tAggregating completed\n\tImputing rest of na's")
    for feature in ['prop_review_score', 'prop_location_score2']:
        df_clean[feature] = impute_with_dist(df_clean[feature])
    print(f"\t\tImputing completed")
    df_clean['date_time'] = pd.to_datetime(df_clean['date_time'], infer_datetime_format=True)
    print(f'\t Writing df to csv file')
    df_clean.to_csv(f'clean_{file}_set.csv')
    return df_clean