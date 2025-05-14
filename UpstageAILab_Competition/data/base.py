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
        self.preprocessed_data = {'X_train': None, 'X_test': None}

    @final
    def preprocess_data(self):
        self.feature_selection()
        self.feature_cleaning()
        self.feature_engineering()
        self.feature_encoding()
    
    @abstractmethod
    def feature_selection(self):
        pass

    @abstractmethod
    def feature_cleaning(self):
        pass

    @abstractmethod
    def feature_engineering(self):
        pass

    @abstractmethod
    def feature_encoding(self):
        pass

    @abstractmethod
    def get_preprocessed_data(self):
        return self.preprocessed_data