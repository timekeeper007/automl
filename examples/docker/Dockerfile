FROM sberbank/python

COPY . .

EXPOSE 5000

CMD python ./train.py --train-csv test_data/check_1_r/train.csv --model-dir . --mode regression
CMD python predict.py --test-csv test_data/check_1_r/test.csv --prediction-csv prediction/prediction_1.csv --model-dir .
CMD python score.py --prediction-csv prediction/prediction_1.csv --test-target-csv test-target.csv --model-dir .
