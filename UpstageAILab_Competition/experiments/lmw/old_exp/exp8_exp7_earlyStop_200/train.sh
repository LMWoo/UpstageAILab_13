python train.py \
    --train_data_path=../../data/train_subway.csv \
    --test_data_path=../../data/test_subway.csv \
    --is_subway=True \
    --is_feature_reduction=True \
    --is_lightGBM=True \
    --model_name=exp8_exp7_earlyStop_200 \
    --early_stopping_rounds=200 \
    --n_estimator=300 