import pandas as pd
import warnings;warnings.filterwarnings('ignore')

import argparse
import os

from sklearn.model_selection import train_test_split
from utils.util import dynamic_preprocessing_import

parser = argparse.ArgumentParser(description="Train parser")

# parser.add_argument("--is_feature_reduction", type=bool, default=False, help='50 features -> top 8 features')
# parser.add_argument("--is_feature_engineering", type=bool, default=False, help='gu -> High, Mid, Low')
# parser.add_argument("--is_logScale", type=bool, default=False, help='target -> logScale')
# parser.add_argument("--is_lightGBM", type=bool, default=False, help='model train LightGBM')
# parser.add_argument("--is_subway", type=bool, default=False, help='Add grade based on distance to subway station')
# parser.add_argument("--early_stopping_rounds", type=int, default=50, help='Only Using lightGBM')
# parser.add_argument("--n_estimator", type=int, default=100, help="RandomForest estimator num")
# parser.add_argument("--train_data_path", type=str, default="../../data/train.csv", help="train data path") 
# parser.add_argument("--test_data_path", type=str, default="../../data/test.csv", help="test data path") 

parser.add_argument("--data", type=str, default='BaselinePreprocess', help='Data Preprocessing ClassName')
parser.add_argument("--data_root_path", type=str, default='../data', help='data root path')
parser.add_argument("--model", type=str, default='BaselineModel', help='Select Train Model')
parser.add_argument("--model_name", type=str, default="save_model")
parser.add_argument("--mode", choices=['train', 'test'], required=True, help="Choose whether to run training or testing")
parser.add_argument("--test_result_file", type=str, default="output")

if __name__ == "__main__":

    args = parser.parse_args()    

    try:
        data_preprocessor_cls = dynamic_preprocessing_import("data", args.data)
    except:
        raise ImportError(f"Can not find {args.data} module")

    try:
        model_cls = dynamic_preprocessing_import("models", args.model)
    except:
        raise ImportError(f"Can not find {args.model} module")
    
    data_preprocessor = data_preprocessor_cls()
    data_preprocessor.preprocess_data()
    preprocessed_data = data_preprocessor.get_preprocessed_data()
    
    Y_train = preprocessed_data['X_train']['target']
    X_train = preprocessed_data['X_train'].drop(['target'], axis=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=2023)
    X_test = preprocessed_data['X_test']

    model = model_cls(X_train, X_val, Y_train, Y_val, X_test)

    if args.mode == 'train':
        model.train()

        print('finish train model')
        model.validation()


        weight_dir = "./weights"
        analysis_dir = "./plots"
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        else:
            pass
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        else:
            pass

        model.analysis_validation(os.path.join(analysis_dir, args.model_name + '.png'))
        model.save_model(os.path.join(weight_dir, args.model_name + '.pkl'))

        print('saved train model')
    elif args.mode == 'test':
        weight_dir = "./weights"
        result_dir = "./results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        else:
            pass

        model.load_model(os.path.join(weight_dir, args.model_name + '.pkl'))
        real_test_pred = model.test()

        # 앞서 예측한 예측값들을 저장합니다.
        preds_df = pd.DataFrame(real_test_pred.astype(int), columns=["target"])
        preds_df.to_csv(os.path.join(result_dir, args.test_result_file + '.csv'), index=False)

        print('finish test model')