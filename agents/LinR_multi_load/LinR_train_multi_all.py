import lightgbm as lgb
import os
import numpy as np
import time

from LinR_feature_engineering import *
from sklearn.multioutput import MultiOutputRegressor
import pickle as pk
from sklearn.linear_model import Ridge

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    path_to_bu1 = "data/citylearn_challenge_2022_phase_1/Building_1.csv"
    path_to_bu2 = "data/citylearn_challenge_2022_phase_1/Building_2.csv"
    path_to_bu3 = "data/citylearn_challenge_2022_phase_1/Building_3.csv"
    path_to_bu4 = "data/citylearn_challenge_2022_phase_1/Building_4.csv"
    path_to_bu5 = "data/citylearn_challenge_2022_phase_1/Building_5.csv"

    path_to_bu = [path_to_bu1, path_to_bu2, path_to_bu3, path_to_bu4, path_to_bu5]
    path_to_weather = "data/citylearn_challenge_2022_phase_1/weather.csv"

    x_train, y_train = pd.DataFrame(), pd.DataFrame()
    for i in range(5):
        x, y = get_train_data(path_to_bu[i], path_to_weather)
        x_train = x_train.append(x)
        y_train = y_train.append(y)
    targets = [item for item in y_train.columns if 'Equipment_Electric_Power_' in item]

    start_time = time.time()
    print("------------------train---------------------------")
    linR = Ridge(alpha=1.0, normalize=False, solver='svd')
    linR.fit(x_train, y_train)
    pkl_filename = "agents/checkpoints/LinR_Load.pkl"
    with open(pkl_filename, 'wb') as file:
        pk.dump(linR, file)

    train_score = linR.score(x_train, y_train)
    print('train_score', train_score)