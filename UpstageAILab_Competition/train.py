import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import warnings;warnings.filterwarnings('ignore')

# Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import eli5
from eli5.sklearn import PermutationImportance

from data.load import load_data

import argparse

parser = argparse.ArgumentParser(description="Train parser")

parser.add_argument("--feature_reduction", type=bool, default=False)

if __name__ == "__main__":

    args = parser.parse_args()

    X_train, y_train, X_val, y_val, categorical_columns_v2, label_encoders, dt_test = load_data(args.feature_reduction)

    # RandomForestRegressor를 이용해 회귀 모델을 적합시키겠습니다.
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    print(f'RMSE test: {np.sqrt(metrics.mean_squared_error(y_val, pred))}')

    # # 위 feature importance를 시각화해봅니다.
    # importances = pd.Series(model.feature_importances_, index=list(X_train.columns))
    # importances = importances.sort_values(ascending=False)

    # plt.figure(figsize=(10,8))
    # plt.title("Feature Importances")
    # sns.barplot(x=importances, y=importances.index)
    # plt.show()
    # 학습된 모델을 저장합니다. Pickle 라이브러리를 이용하겠습니다.
    with open('./weights/saved_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # # Permutation importance 방법을 변수 선택에 이용해보겠습니다.
    # perm = PermutationImportance(model,        # 위에서 학습된 모델을 이용하겠습니다.
    #                             scoring = "neg_mean_squared_error",        # 평가 지표로는 회귀문제이기에 negative rmse를 사용합니다. (neg_mean_squared_error : 음의 평균 제곱 오차)
    #                             random_state = 42,
    #                             n_iter=3).fit(X_val, y_val)
    # eli5.show_weights(perm, feature_names = X_val.columns.tolist())    # valid data에 대해 적합시킵니다.

    # Validation dataset에 target과 pred 값을 채워주도록 하겠습니다.
    X_val['target'] = y_val
    X_val['pred'] = pred

    # Squared_error를 계산하는 함수를 정의하겠습니다.
    def calculate_se(target, pred):
        squared_errors = (target - pred) ** 2
        return squared_errors

    # RMSE 계산
    squared_errors = calculate_se(X_val['target'], X_val['pred'])
    X_val['error'] = squared_errors

    # Error가 큰 순서대로 sorting 해 보겠습니다.
    X_val_sort = X_val.sort_values(by='error', ascending=False)       # 내림차순 sorting

    print(X_val_sort.head())

    X_val_sort_top100 = X_val.sort_values(by='error', ascending=False).head(100)        # 예측을 잘 하지못한 top 100개의 data
    X_val_sort_tail100 = X_val.sort_values(by='error', ascending=False).tail(100)       # 예측을 잘한 top 100개의 data

    # 해석을 위해 레이블인코딩 된 변수를 복원해줍니다.
    error_top100 = X_val_sort_top100.copy()
    for column in categorical_columns_v2 :     # 앞서 레이블 인코딩에서 정의했던 categorical_columns_v2 범주형 변수 리스트를 사용합니다.
        error_top100[column] = label_encoders[column].inverse_transform(X_val_sort_top100[column])

    best_top100 = X_val_sort_tail100.copy()
    for column in categorical_columns_v2 :     # 앞서 레이블 인코딩에서 정의했던 categorical_columns_v2 범주형 변수 리스트를 사용합니다.
        best_top100[column] = label_encoders[column].inverse_transform(X_val_sort_tail100[column])

    # display(error_top100.head(1))
    # display(best_top100.head(1))

    # sns.boxplot(data = error_top100, x='target')
    # plt.title('The worst top100 prediction의 target 분포')
    # plt.show()

    # sns.boxplot(data = best_top100, x='target', color='orange')
    # plt.title('The best top100 prediction의 target 분포')
    # plt.show()

    # sns.histplot(data = error_top100, x='전용면적', alpha=0.5)
    # sns.histplot(data = best_top100, x='전용면적', color='orange', alpha=0.5)
    # plt.title('전용면적 분포 비교')
    # plt.show()

    print('finish train')