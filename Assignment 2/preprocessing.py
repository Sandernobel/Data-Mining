import pandas as pd
import numpy as np


def aggregate(df: pd.DataFrame, aggregate_over: str, to_aggregate: list, methods=('mean', 'std', 'median')):
    """
    Adds aggregated columns
    :param df:
    :param aggregate_over:
    :param to_aggregate:
    :param methods:
    :return:
    """

    new_cols = dict()

    for column in to_aggregate:
        for method in methods:
            new_cols[column+'_'+method] = df.groupby(aggregate_over).agg({column: method})[column]
        print(f'\t\tAggregating {column} complete')
    agg_df = pd.DataFrame.from_dict(new_cols)

    return df.merge(agg_df, on=aggregate_over, suffixes=(False, False)).sort_values('srch_id')


def rank(df: pd.DataFrame, to_rank: list, rank_over: str = 'srch_id'):
    """
    Ranks all to_rank items over rank_over feature
    :param df:
    :param to_rank:
    :param rank_over:
    :return:
    """

    for col in to_rank:
        print(f"\t\tRanking {col}")
        df[to_rank + '_ranked'] = df.groupby(rank_over)[to_rank].rank()
        print(f"\t\t\tRanking {col} done")
    return df

def preprocess(df: pd.DataFrame):
    """
    Preprocessing dataframe
    :param df:
    :param file:
    :return:
    """

    print(f"\tAggregating columns")
    df = aggregate(df, 'prop_id', ['prop_location_score2', 'prop_log_historical_price', 'price_usd'])
    print('\tAggregating completed')

    # Ranking ipv normalizen want dat presteert beter
    print(f"\tRanking columns")
    df = rank(df, ['price_usd', 'prop_starrating', 'prop_review_score', 'prop_location_score1','prop_location_score2'])
    print(f"\tRanking done")

    return df