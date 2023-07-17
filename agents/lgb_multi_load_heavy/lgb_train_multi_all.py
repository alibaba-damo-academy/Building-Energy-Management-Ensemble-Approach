import lightgbm as lgb
import os
import numpy as np
import time

from lgb_feature_engineering import *
from sklearn.multioutput import MultiOutputRegressor
import pickle as pk

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
    gbm = MultiOutputRegressor(
        lgb.LGBMRegressor(objective='regression', n_estimators=750, learning_rate=0.01, num_leaves=16, random_state=2022,
                          verbose=-1, n_jobs=1), n_jobs=24)
    gbm.fit(x_train, y_train)
    print('Training time: {}'.format(time.time() - start_time))

    # Save to file in the current working directory
    pkl_filename = "agents/checkpoints/lgb_load_multi_speedup_new_heavy.pkl"
    with open(pkl_filename, 'wb') as file:
        pk.dump(gbm, file)

    pred_train = pd.DataFrame(gbm.predict(x_train), columns=targets)
    pred_train.index = y_train.index
    print('Building 1-5 WMAPE:')
    print((y_train - pred_train).abs().mean() / y_train.abs().mean())