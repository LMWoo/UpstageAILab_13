python train.py \
    --train_data_path=../../data/train_subway.csv \
    --test_data_path=../../data/test_subway.csv \
    --is_subway=True \
    --is_feature_reduction=True \
    --is_lightGBM=True \
    --model_name=exp7_exp6_subway \
    --n_estimator=300 