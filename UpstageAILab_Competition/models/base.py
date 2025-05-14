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