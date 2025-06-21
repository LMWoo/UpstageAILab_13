import os
import pandas as pd
from typing import final
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class BaseModel:
    def __init__(self, X_train, X_val, Y_train, Y_val, X_test):
        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.X_test = X_test
        pass

    # 모댈 학습 구현 함수
    @abstractmethod
    def train(self):
        pass

    # 모델 평가 구현 함수
    @abstractmethod
    def validation(self):
        pass

    # 모델 테스트 구현 함수
    @abstractmethod
    def test(self):
        pass

    # 모델 평가 분석 함수
    #  - Feature Importance, Permutation Importance
    #  - top, worst sample comparision
    @abstractmethod
    def analysis_validation(self, save_root_path, data_preprocessor):
        pass

    # 모델 저장 함수
    @abstractmethod
    def save_model(self, save_path):
        pass

    # 모델 로딩 함수
    @abstractmethod
    def load_model(self, load_path):
        pass