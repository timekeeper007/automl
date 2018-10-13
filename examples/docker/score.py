import argparse
import pandas as pd
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error, roc_auc_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-target-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    y_test = pd.read_csv(args.test_target_csv)
    y_pred = pd.read_csv(args.prediction_csv)
    df = y_test.merge(y_pred, how='inner')

    if args.mode == 'regression':
        print("Test_score: ", mean_squared_error(df['target'], df['prediction']))
    else:
        print("Test_score: ", roc_auc_score(df['target'], df['prediction']))
