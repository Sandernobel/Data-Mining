from os import path
from clean_df import *
from preprocessing import *

train_test = ['training', 'test']

for file in train_test:
    print("Reading datafile...")
    if path.exists(f'clean_{file}_set.csv'):
        df = pd.read_csv(f'clean_{file}_set.csv')
        print("\tReading complete!")
    else:
        df = pd.read_csv(f'{file}_set_vu_DM.csv')
        print("\tReading complete!\nCleaning datafile...")
        df = clean_up_df(df, file)

    print(f"Starting preprocessing...")
    preprocessed_df = preprocess(df, file)