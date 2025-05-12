import pandas as pd
import pickle
from data.base import BasePreprocess
from data.lmw.baseline_dataprocess import BaselinePreprocess
from tqdm import tqdm
import numpy as np
import os

class AddressToXYPreprocess(BasePreprocess):
    def __init__(self):
        super().__init__()

        self.baselinePreprocess = BaselinePreprocess()

        with open("../data/adres_to_geo.pickle", "rb") as f:
            adres_to_geo = pickle.load(f)

        def address_to_xy(dt):
            for i in tqdm(range(dt.shape[0])):
                
                try:
                    add = dt['시군구'].iloc[i] + ' ' + dt['번지'].iloc[i]
                    dt.loc[i, '좌표X'] = adres_to_geo[add][1]
                    dt.loc[i, '좌표Y'] = adres_to_geo[add][0]
                except:
                    pass
            return dt

        if os.path.exists('../data/train_xy.csv') and os.path.exists('../data/test_xy.csv') :
            self.train_data = pd.read_csv("../data/train_xy.csv")
            self.test_data = pd.read_csv("../data/test_xy.csv")
        else:
            self.train_data = pd.read_csv("../data/train.csv")
            self.test_data = pd.read_csv("../data/test.csv")
            self.train_data = address_to_xy(self.train_data)
            self.test_data = address_to_xy(self.test_data)
            self.train_data.to_csv('../data/train_xy.csv')
            self.test_data.to_csv('../data/test_xy.csv')
        
        self.baselinePreprocess.train_data = self.train_data
        self.baselinePreprocess.test_data = self.test_data

    def feature_selection(self):
        self.baselinePreprocess.feature_selection()

    def feature_engineering(self):
        self.baselinePreprocess.feature_engineering()

        self.train_data = self.baselinePreprocess.train_data
        self.test_data = self.baselinePreprocess.test_data
        self.concat = self.baselinePreprocess.concat