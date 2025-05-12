import os
import pandas as pd
from typing import final
from abc import ABC, abstractmethod

class BasePreprocess:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.concat = None

    @final
    def preprocess_data(self):
        self.feature_selection()
        self.feature_engineering()
    
    @abstractmethod
    def feature_selection(self):
        pass

    @abstractmethod
    def feature_engineering(self):
        pass