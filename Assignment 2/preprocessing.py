import pandas as pd
import numpy as np


def preprocess(df: pd.Series, file: str):
    """
    Preprocessing dataframe
    :param df:
    :return:
    """

    if file == 'train':
        df['relevance'] = np.asarray(4*df['booking_bool'] + df['click_bool'])