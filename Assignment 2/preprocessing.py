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
    agg_df = pd.DataFrame.from_dict(new_cols)

    return df.merge(agg_df, on=aggregate_over, suffixes=(False, False)).sort_values('srch_id')

def preprocess(df: pd.DataFrame, file: str):
    """
    Preprocessing dataframe
    :param df:
    :param file:
    :return:
    """

    pd.set_option("display.max_columns", None)

    if file == 'training':
        df['relevance'] = np.asarray(4*df['booking_bool'] + df['click_bool'])
    print(f"\tAggregating columns")

    to_aggregate = ['prop_location_score2', 'prop_log_historical_price', 'price_usd']
    df = aggregate(df, 'prop_id', to_aggregate)

    print(f"\tWriting to csv")
    df.to_csv(f'clean_{file}_set.csv')
    print(f"\t\tDone writing")

    return df