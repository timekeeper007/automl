import argparse
import os
import pandas as pd
import pickle
import time
import numpy as np

import utils

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    ONEHOT_MAX_UNIQUE_VALUES = model_config['ONEHOT_MAX_UNIQUE_VALUES']
    # read dataset
    df = pd.read_csv(args.test_csv)
    print('Dataset read, shape {}'.format(df.shape))

    # drop train constant values
    df.drop(model_config['constant_columns'], axis=1, inplace=True)
    # rename c_, d_, r_
    df = utils.add_prefix_to_colnames(df, ONEHOT_MAX_UNIQUE_VALUES)
    # missing values
    _, df = utils.replace_na_and_create_na_feature(df, model_config['na_features'])

    if not model_config['is_big']:
        # features from datetime
        df = utils.transform_datetime_features(df)

        # categorical onehot encoding
        df = utils.onehot_encoding_test(df, model_config['categorical_to_onehot'])

        # real number feature extraction
        df = utils.numeric_feature_extraction(df, degree=4, num_mult=True)


    # filter columns
    used_columns = model_config['used_columns']

    # scale
    X_scaled = model_config['scaler'].transform(df[used_columns])

    model = model_config['model']
    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(X_scaled)
    elif model_config['mode'] == 'classification':
        df['prediction'] = model.predict_proba(X_scaled)[:, 1]

    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
