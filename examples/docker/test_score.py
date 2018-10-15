import os
import pandas as pd

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    test_data_folders = [
        os.path.abspath(os.path.join('test_data', folder))
        for folder in os.listdir(os.path.join(current_path, 'test_data'))
    ]
    for test_data_path in test_data_folders:
        print(test_data_path)
        df = pd.read_csv(os.path.join(test_data_path, 'test-target.csv'))
        mode = 'classification' if df['target'].unique().shape[0] == 2 else 'regression'
        print(mode)
        os.system(
            "python train.py --mode {mode} --train-csv {train_csv} --model-dir {model_dir}".format(
                train_csv=os.path.join(test_data_path, 'train.csv'),
                model_dir=os.path.join(test_data_path),
                mode=mode)
        )
        os.system(
            "python predict.py --test-csv {test_csv} --prediction-csv {prediction_csv} --model-dir {model_dir}".format(
                test_csv=os.path.join(test_data_path, 'test.csv'),
                model_dir=os.path.join(test_data_path),
                prediction_csv=os.path.join(test_data_path, 'prediction.csv'))
        )
        os.system(
            "python score.py --test-target-csv {test_target_csv} --prediction-csv {prediction_csv} --mode {mode}".format(
                test_target_csv=os.path.join(test_data_path, 'test-target.csv'),
                prediction_csv=os.path.join(test_data_path, 'prediction.csv'),
                mode=mode)
        )
