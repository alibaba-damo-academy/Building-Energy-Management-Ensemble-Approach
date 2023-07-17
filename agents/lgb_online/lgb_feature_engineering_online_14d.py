import pandas as pd
import numpy as np
import os
import warnings
import copy

warnings.simplefilter('ignore')

def init_test_data():
    print("Init test data")
    path_to_bu1 = "agents/data_saved/citylearn_challenge_2022_phase_1/Building_1.csv"
    path_to_bu = path_to_bu1
    path_to_weather = "agents/data_saved/citylearn_challenge_2022_phase_1/weather.csv"
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

    df['Solar_Generation'] = 0.
    df['Equipment_Electric_Power'] = 0.
    df['Day'] = df.index // 24
    df['Hour'] = df.Hour % 24

    N = 24
    for i in range(N):
        df['Outdoor_Drybulb_Temperature_{}'.format(i)] = df['Outdoor_Drybulb_Temperature'].shift(-i - 1)
        df['Relative_Humidity_{}'.format(i)] = df['Relative_Humidity'].shift(-i - 1)
        df['Diffuse_Solar_Radiation_{}'.format(i)] = df['Diffuse_Solar_Radiation'].shift(-i - 1)
        df['Direct_Solar_Radiation_{}'.format(i)] = df['Direct_Solar_Radiation'].shift(-i - 1)
    for i in range(N * 14):
        df['Solar_Past_{}'.format(i)] = 0.
        df['Load_Past_{}'.format(i)] = 0.
    print('init df shape:', df.shape)
    return df
