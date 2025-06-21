import os
import pandas as pd
from typing import final
from abc import ABC, abstractmethod

class BasePreprocess:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.concat = None

        self.basic_columns = ['target', 'is_test'] 
        self.label_encoders = {}

        # None 필수로 채우기, main에서 X_train, X_test를 모델에 넣음
        self.preprocessed_data = {'X_train': None, 'X_test': None}
      
    @final
    def preprocess_data(self):
        self.feature_selection()
        self.feature_cleaning()
        self.feature_engineering()
        self.feature_encoding()
        
    # EDA 이후 원본 데이터에서 사용할 변수 고르는 함수
    @abstractmethod
    def feature_selection(self):
        pass

    # 결측치, 이상치 처리 함수
    @abstractmethod
    def feature_cleaning(self):
        pass

    # 파생 변수 구현 함수
    @abstractmethod
    def feature_engineering(self):
        pass
    
    # 범주형 변수 인코딩 함수
    @abstractmethod
    def feature_encoding(self):
        pass
    
    @abstractmethod
    def get_preprocessed_data(self):
        return self.preprocessed_data