import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def targetEncoding(df, feature_name):
    encoding_map = df[df['is_test'] == 0].groupby(feature_name)['target'].mean() # Target 데이터만 사용

    df.loc[df['is_test'] == 0, feature_name] = df.loc[df['is_test'] == 0, feature_name].map(encoding_map) # Train 데이터 target encoding
    df.loc[df['is_test'] == 1, feature_name] = df.loc[df['is_test'] == 1, feature_name].map(encoding_map) # Test 데이터 target encoding
    df.loc[df['is_test'] == 1, feature_name] = df.loc[df['is_test'] == 1, feature_name].fillna(df[df['is_test'] == 0]['target'].mean()) # Unseen 데이터 처리

