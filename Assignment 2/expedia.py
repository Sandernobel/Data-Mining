from os import path
from clean_df import *
from preprocessing import *
from train_model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


if not path.exists('train.csv') or not path.exists('test.csv'):
    print(f'Reading train file')
    train_file = pd.read_csv('training_set_VU_DM.csv')
    train_file['train'] = 'train'
    print(f'\tComplete\nReading test file')
    test_file = pd.read_csv('test_set_VU_DM.csv')
    test_file['train'] = 'test'
    print(f'\tComplete\nAppending')
    whole_file = test_file.append(train_file, sort=False)
    print('\tAppending done')

    if 'date_time' in whole_file.columns:
        print('\nCleaning up datafile')
        whole_file = clean_up_df(whole_file)
        print('\tFile cleaned.')

    print('\nStarting preprocessing file')
    whole_file = preprocess(whole_file)

    print('\tPreprocessing done\nSplitting back into train and test set')
    train_file = whole_file.loc[whole_file['train'] == 'train'].drop('train', axis=1)
    train_file['relevance'] = 4*train_file['booking_bool'] + train_file['click_bool']
    test_file = whole_file.loc[whole_file['train'] == 'test'].drop(['booking_bool', 'click_bool', 'position', 'train'], axis=1)
    del whole_file
    print('\tSplitting done')

    print(f'Writing cleaned training file to csv')
    out_file = open('train.csv', 'wb+')
    pickle.dump(train_file, out_file)
    out_file.close()
    print(f'Writing test file...')
    out_file = open('test.csv', 'wb+')
    pickle.dump(test_file, out_file)
    out_file.close()
    print('\tDone')
else:
    print(f'Reading clean train file')
    infile = open('train.csv', 'rb')
    train_file = pickle.load(infile)
    infile.close()
    print(f'\tComplete\nReading test file')
    infile = open('test.csv', 'rb')
    test_file = pickle.load(infile)
    infile.close()
    print(f'\tComplete')

print('\nPreparing data for ranker')
model = train_lgbm(train_file)
save_model(model)
get_predictions(test_file, model)