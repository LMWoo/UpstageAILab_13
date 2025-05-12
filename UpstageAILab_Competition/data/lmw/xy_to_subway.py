import pandas as pd
from scipy.spatial import distance_matrix

if __name__ == "__main__":
    train_df = pd.read_csv("../../../data/train_xy.csv")
    subway_df = pd.read_csv("../../../data/subway_feature.csv")

    subway_df["좌표Y"] = subway_df["위도"]
    subway_df["좌표X"] = subway_df["경도"]

    train_coords = train_df[["좌표X", "좌표Y"]]
    subway_coords = subway_df[["좌표X", "좌표Y"]]

    distances = distance_matrix(train_coords, subway_coords)

    min_distances = distances.min(axis=1)

    train_df["is_subway_near"] = (min_distances <= 150).astype(int)

    train_df.to_csv('../../../data/train_subway.csv')