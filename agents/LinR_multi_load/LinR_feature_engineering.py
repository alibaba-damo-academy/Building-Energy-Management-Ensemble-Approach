import pandas as pd
import numpy as np
import os
import warnings
import copy

warnings.simplefilter('ignore')

def get_train_data(path_to_bu, path_to_weather):
    print("Loading train data")

    df_bu, df_we = pd.read_csv(path_to_bu), pd.read_csv(path_to_weather)
    df = pd.merge(df_bu, df_we, left_index=True, right_index=True)

    cols = ['Month', 'Hour', 'Day_Type', 'Daylight_Savings_Status', 'Indoor_Temperature',
            'Average_Unmet_Cooling_Setpoint_Difference',
            'Indoor_Relative_Humidity',
            'Equipment_Electric_Power',
            'DHW_Heating',
            'Cooling_Load',
            'Heating_Load',
            'Solar_Generation',
            'Outdoor_Drybulb_Temperature',
            'Relative_Humidity',
            'Diffuse_Solar_Radiation',
            'Direct_Solar_Radiation',
            '6h_Prediction_Outdoor_Drybulb_Temperature',
            '12h_Prediction_Outdoor_Drybulb_Temperature',
            '24h_Prediction_Outdoor_Drybulb_Temperature',
            '6h_Prediction_Relative_Humidity',
            '12h_Prediction_Relative_Humidity',
            '24h_Prediction_Relative_Humidity',
            '6h_Prediction_Diffuse_Solar_Radiation',
            '12h_Prediction_Diffuse_Solar_Radiation',
            '24h_Prediction_Diffuse_Solar_Radiation',
            '6h_Prediction_Direct_Solar_Radiation',
            '12h_Prediction_Direct_Solar_Radiation',
            '24h_Prediction_Direct_Solar_Radiation',
            ]
    df.columns = cols

    selected_cols = ['Month', 'Hour', 'Day_Type',
            'Equipment_Electric_Power',
            'Solar_Generation',
            'Outdoor_Drybulb_Temperature',
            'Relative_Humidity',
            'Diffuse_Solar_Radiation',
            'Direct_Solar_Radiation',
            ]
    df = df[selected_cols]

    df['Solar_Generation'] = df['Solar_Generation'] / 1000.  # * pv_cap / 1000.0
    df['Day'] = df.index // 24
    df['Hour'] = df.Hour % 24
    N = 24
    for i in range(N):
        df['Outdoor_Drybulb_Temperature_{}'.format(i)] = df['Outdoor_Drybulb_Temperature'].shift(-i - 1)
        df['Relative_Humidity_{}'.format(i)] = df['Relative_Humidity'].shift(-i - 1)
        df['Diffuse_Solar_Radiation_{}'.format(i)] = df['Diffuse_Solar_Radiation'].shift(-i - 1)
        df['Direct_Solar_Radiation_{}'.format(i)] = df['Direct_Solar_Radiation'].shift(-i - 1)
    for i in range(N * 14):
        df['Load_Past_{}'.format(i)] = df['Equipment_Electric_Power'].shift(i + 1)
    for i in range(N):
        df['Equipment_Electric_Power_{}'.format(i)] = df['Equipment_Electric_Power'].shift(-i - 1)
    df_drop = df.dropna(inplace=False)

    targets = [item for item in df_drop.columns if 'Equipment_Electric_Power_' in item]
    x_train = df_drop.drop(targets, axis=1)
    y_train = df_drop[targets]

    print("Loading train data finish")
    return x_train, y_train

