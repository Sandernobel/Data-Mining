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

def delete_outliers(df: pd.DataFrame, column: str):
    """

    :param df:
    :param column:
    :return:
    """
    std = df[column].std()
    mean = df[column].mean()
    cut_off = std * 3
    lower, upper = mean - cut_off, mean + cut_off
    trimmed_df = df[(df[column] < upper) & (df[column] > lower)]
    return trimmed_df

def clean_up_df(df: pd.DataFrame, percent=0.90):
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
    print(f"\tTrimming outliers")
    for column in ['srch_length_of_stay', 'prop_location_score2', 'srch_booking_window']:
        df_clean = delete_outliers(df_clean, column)
        print(f"\t\tTrimming {column} completed")
    print(f'\tTrimming all outliers completed')
    df_clean = df_clean.drop('date_time', axis=1)

    return df_clean