import argparse
import os
import numpy as np
import pandas as pd
import pickle
import time

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

import utils

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

    df = pd.read_csv(args.train_csv)
    df_y = df.target
    df_X = df.drop('target', axis=1)
    is_big = df_X.memory_usage().sum() > BIG_DATASET_SIZE

    print('Dataset read, shape {}'.format(df_X.shape))

    # drop constant features
    constant_columns = [
        col_name
        for col_name in df_X.columns
        if df_X[col_name].nunique() == 1
        ]
    df_X.drop(constant_columns, axis=1, inplace=True)

    # dict with data necessary to make predictions
    model_config = {}
    model_config['constant_columns'] = constant_columns
    model_config['categorical_values'] = {}
    model_config['is_big'] = is_big

    if is_big:
        # missing values
        if any(df_X.isnull()):
            model_config['missing'] = True
            df_X.fillna(-1, inplace=True)

        new_feature_count = min(df_X.shape[1],
                                int(df_X.shape[1] / (df_X.memory_usage().sum() / BIG_DATASET_SIZE)))
        # take only high correlated features
        correlations = np.abs([
            np.corrcoef(df_y, df_X[col_name])[0, 1]
            for col_name in df_X.columns if col_name.startswith('number')
            ])
        new_columns = df_X.columns[np.argsort(correlations)[-new_feature_count:]]
        df_X = df_X[new_columns]

    else:

        print("df_X shape before prefix: " + str(df_X.shape))
        # rename c_, d_, r_
        df_X = utils.add_prefix_to_colnames(df_X, ONEHOT_MAX_UNIQUE_VALUES)
        # missing values
        print("df_X shape before na replacement: " + str(df_X.shape))
        df_X = utils.replace_na_and_create_na_feature(df_X)

        # features from datetime
        print("df_X shape before adding datetime features: " + str(df_X.shape))
        df_X = utils.transform_datetime_features(df_X)



        # categorical encoding
        print("df_X shape before onehot: " + str(df_X.shape))
        model_config['categorical_to_onehot'], df_X = utils.onehot_encoding_train(df_X, ONEHOT_MAX_UNIQUE_VALUES)

        # selecting dummies using Random Forest
        print("df_X shape before selecting dummies: " + str(df_X.shape))
        model_config['important_dummies'], df_X = utils.select_important_dummies(df_X, df_y, args.mode, importance=0.05, n_estimators=10)
        print("df_X shape agter selecting dummies: " + str(df_X.columns))
        # real
        # transform df with numeric and dummy features by adding new features: x^2...x^k, log(x), 1/x, x1/x2, x1*x2.
        # Hyperparameters. degree: int (max degree of polynoms included)
        # num_mult: True for all multiplications, False for multiplications with dummies only
        print("df_X shape before real: " + str(df_X.shape))
        df_X = utils.numeric_feature_extraction(df_X, degree=4, num_mult=True)



    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('r_') or col_name.startswith('d_')
        ]
    df_X = df_X[used_columns].values
    model_config['used_columns'] = used_columns

    # scaling
    print("df_X shape before scaling: "+ str(df_X.shape))
    scaler = StandardScaler(copy=False)
    df_X = scaler.fit_transform(df_X)
    model_config['scaler'] = scaler

    # fitting
    model_config['mode'] = args.mode
    if args.mode == 'regression':
        model = Ridge()
    else:
        model = LogisticRegression()

    model.fit(df_X, df_y)
    model_config['model'] = model

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))
