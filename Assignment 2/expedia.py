from os import path
from clean_df import *
from preprocessing import *
from train_model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd



train_test = ['training', 'test']

for file in train_test:
    print(f"Reading {file} datafile...")

    if path.exists(f'clean_{file}_set.csv'):
        print(f"\tClean {file} file exists")
        df = pd.read_csv(f'clean_{file}_set.csv', index_col=0)
        print("\tReading complete!")
    else:
        df = pd.read_csv(f'{file}_set_vu_DM.csv')
        print("\tReading complete!\nCleaning datafile...")
        df = clean_up_df(df, file)

    if 'prop_location_score2_mean' in df.columns:
        print(f"Preprocessing {file} file already done.")
    else:
        print(f"Starting preprocessing {file} file...")
        df = preprocess(df, file)
        print(f"\tDone preprocessing {file} file\n")

    if file == 'training':
        print(f"Preparing data for LGBM ranker")
        model = train_lgbm(df)
    else:
        print(f"Predicting relevance")
        df = test_lgbm(df, model)
        df.sort_values(by=['srch_id', 'relevance'], ascending=[True, False], inplace=True)
        kaggle_answer = pd.DataFrame({'srch_id': df['srch_id'],
                                      'prop_id': df['prop_id']})
        kaggle_answer.to_csv('expedia_answer.csv', index=False)