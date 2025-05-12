import os
import pandas as pd
from typing import final
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class BaseModel:
    def __init__(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor
        pass

    @abstractmethod
    def encoding(self):
        pass

    @abstractmethod
    def splitdata(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validation(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def analysis_validation(self, save_path):
        pass

    @abstractmethod
    def save_model(self, save_path):
        pass

    @abstractmethod
    def load_model(self, load_path):
        pass