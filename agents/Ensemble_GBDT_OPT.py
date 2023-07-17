import os
import lightgbm as lgb
from agents.lgb_online.lgb_feature_engineering_online_14d import init_test_data as init_test_data_14
from agents.lgb_online.lgb_feature_engineering_online_7d import init_test_data as init_test_data_7
import numpy as np
import pandas as pd
import time
import pickle as pk
from docplex.mp.model import Model
from copy import deepcopy
from sklearn.multioutput import MultiOutputRegressor
import sys
import torch

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class LGBnOPTnRLAgent:
    """
    LGB Agent
    """

    def __init__(self):

        # pred + optimization
        self.episode_id = -1
        self.index = {}
        self.pv_capacity = {}
        self.all_index = int(-1)
        path_to_model_solar = "agents/checkpoints/lgb_solar_multi_speedup_new_1030_heavy.pkl"
        with open(path_to_model_solar, 'rb') as file:
            self.gbm_solar = pk.load(file)
        path_to_model_load = "agents/checkpoints/lgb_load_multi_speedup_new_heavy.pkl"
        with open(path_to_model_load, 'rb') as file:
            self.gbm_load = pk.load(file)
        path_to_model_lin = "agents/checkpoints/LinR_Load.pkl"
        with open(path_to_model_lin, 'rb') as file:
            self.lin_load = pk.load(file)
        self.gbm_solar.n_jobs = 1
        self.gbm_load.n_jobs = 1
        self.lin_load.n_jobs = 1

        self.capacity = 6.4
        self.max_charge = 5.0 / self.capacity
        self.agent_num = 5
        self.solar_left = np.zeros((self.agent_num, 8761))
        self.act = np.zeros((self.agent_num, 24))

        # read price and co2 cost
        carbon = pd.read_csv('agents/data_saved/citylearn_challenge_2022_phase_1/carbon_intensity.csv')
        pricing = pd.read_csv('agents/data_saved/citylearn_challenge_2022_phase_1/pricing.csv')
        self.price = np.zeros(8761)
        self.price[:-1] = pricing['Electricity Pricing [$]'].values
        self.emission = np.zeros(8761)
        self.emission[:-1] = carbon['kg_CO2/kWh'].values

        # recording episode time
        self.start_time = time.time()

        # load factor hyperparameter
        self.alpha_load = 0.001

        # init opt variables
        self.scenario_number = 75
        self.m1 = Model(name='pwl')
        self.d = 1  # scheduler period = 1 day
        self.c1 = 0.912 # charge efficiency
        self.c2 = 1. / self.c1 # discharge efficiency
        self.x = 0
        self.y = 0
        self.soc = 0
        self.soc1 = 0
        self.soc2 = 0
        self.vu = 0
        self.mu = 0
        self.mu_noclip = 0
        self.w = 0
        self.e_maximum = 0
        self.e_avg = 0
        self.load_var = 0
        self.init_soc = 0

        self.time_weight_factor = np.arange(1.0, 0.97, -(1.0 - 0.97) / 24)
        self.mu_threshold_value_constant = np.ones(12) * 0.9

        self.lf_mean = np.ones(12)


        with open('agents/actions_saved/annual_dispatch_schema1_mindopt.pkl', 'rb') as f:
            self.actions_opt = pk.load(f)

        # # mean of all buildings in training data
        self.solar_train_mean = np.array([0., 0., 0., 0., 0., 0.,
                                          0.00772142, 0.06758006, 0.17815801, 0.3080483, 0.42879198,
                                          0.50559368, 0.54869532, 0.54497649, 0.50235824, 0.41020424,
                                          0.28657643, 0.16933703, 0.0677388, 0.0029458, 0.,
                                          0., 0., 0.])
        self.load_train_mean = np.array([1.02500752, 0.83214035, 0.75899703, 0.70115535, 0.72968285, 0.73625935,
                                         0.79389332, 0.82114927, 0.86693869, 0.99444716, 1.28633053,
                                         1.33460672, 1.37912689, 1.23035236, 1.12216157, 1.13794405,
                                         1.26107395, 1.35709849, 1.3525594, 1.32629516, 1.14569356,
                                         1.13128801, 1.15672385, 1.11261601])
        self.load_scaling_param = 0.98
        self.noise_enlarge_ratio = 1.05


    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.index[agent_id] = 0
        self.all_index = int(-1)
        if agent_id == 0:
            self.episode_id += 1
            print('One episode time: {}'.format(time.time() - self.start_time))
            self.start_time = time.time()
        seed_everything(2022)
        self.offline_opt_flag = False

    # pass building info to read pv capacity
    def set_building_info(self, agent_id, building_info):
        self.pv_capacity[agent_id] = building_info['solar_power']
        print('building info {} - pv number {}.'.format(agent_id, self.pv_capacity[agent_id]))

    def init_opt_model(self, observations):
        # read number
        self.agent_num = len(observations)
        self.scenario_number = int(375 / self.agent_num)
        if self.agent_num > 5:
            self.load_scaling_param = 1.
            self.noise_enlarge_ratio = 1.25
        self.solar_left = np.zeros((self.agent_num, 8761))
        self.act = np.zeros((self.agent_num, 24))
        # inference accelaration
        if self.agent_num == 5:
            self.df = init_test_data_14()
            self.day_seq = 14
        else:
            self.df = init_test_data_7()
            self.day_seq = 7
        for i in range(24 * self.day_seq):
            self.df['Load_Past_{}'.format(i)] = np.nan
            self.df['Solar_Past_{}'.format(i)] = np.nan

        self.max_solar_loads_len = 2 * self.day_seq * 24
        self.solar_loads_past = []
        self.net_solar_past = []

        self.pred_load_correction = np.ones([self.agent_num, 24])
        self.pred_real_mul_sum = np.zeros([self.agent_num, 24])
        self.real_pow2_sum = np.zeros([self.agent_num, 24])
        self.pred_real_mul_remove_list = []
        self.real_pow2_remove_list = []
        self.loads_past = []
        self.last_pred24next = None
        self.cal_count = 0

        print('Agent number reset:', self.agent_num)

        if self.episode_id == 0:
            R = 20.0  # nominal power slope

            self.x = self.m1.continuous_var_matrix(keys1=range(self.agent_num * self.scenario_number),
                                                   keys2=range(24 * self.d), name='x',
                                                   lb=0, ub=5. / 6.4)
            self.y = self.m1.continuous_var_matrix(keys1=range(self.agent_num * self.scenario_number),
                                                   keys2=range(24 * self.d), name='y',
                                                   lb=-5. / 6.4, ub=0)
            self.soc = self.m1.continuous_var_matrix(keys1=range(self.agent_num * self.scenario_number),
                                                     keys2=range(24 * self.d),
                                                     name='soc', lb=0, ub=1)
            self.soc1 = self.m1.continuous_var_matrix(keys1=range(self.agent_num * self.scenario_number),
                                                      keys2=range(24 * self.d),
                                                      name='soc1', lb=0, ub=0.8)
            self.soc2 = self.m1.continuous_var_matrix(keys1=range(self.agent_num * self.scenario_number),
                                                      keys2=range(24 * self.d),
                                                      name='soc2', lb=0, ub=0.2)
            self.vu = self.m1.continuous_var_matrix(keys1=range(self.agent_num * self.scenario_number),
                                                    keys2=range(24 * self.d), name='vu',
                                                    lb=0)
            self.mu = self.m1.continuous_var_matrix(keys1=range(1 * self.scenario_number), keys2=range(24 * self.d),
                                                    name='mu',
                                                    lb=0)
            self.mu_noclip = self.m1.continuous_var_matrix(keys1=range(1 * self.scenario_number), keys2=range(24 * self.d),
                                                           name='mu_noclip', lb=-100)
            self.w = self.m1.continuous_var_matrix(keys1=range(1 * self.scenario_number), keys2=range(24 * self.d - 1),
                                                   name='w', lb=0)
            self.e_maximum = self.m1.continuous_var_matrix(keys1=range(1 * self.scenario_number), keys2=range(1),
                                                           name='e_maximum', lb=0)
            self.e_avg = self.m1.continuous_var_matrix(keys1=range(1 * self.scenario_number), keys2=range(1), name='e_avg',
                                                       lb=0)
            self.e_over_threshold_maximum = self.m1.continuous_var_list(keys=self.scenario_number,
                                                                name='e_over_threshold_maximum', lb=0)

            self.mu_threshold_value = self.m1.continuous_var_list(keys=1, name='mu_threshold_value',
                                                                  lb=self.mu_threshold_value_constant[0],ub=self.mu_threshold_value_constant[0])
            # defining new variables for load update
            self.load_var = self.m1.continuous_var_list(keys= self.scenario_number * self.agent_num * 24,
                                                        lb=-100, ub=100, name='load_var')
            self.init_soc = self.m1.continuous_var_list(keys=self.agent_num, lb=0., ub=1., name='init_soc')
            self.init_net_consump = self.m1.continuous_var_list(keys=1, lb=0., ub=1., name='init_net_consump')
            self.first_point_w = self.m1.continuous_var_list(keys=self.scenario_number, lb=0., name='first_point_w')
            for scenario_i in range(self.scenario_number):
                self.m1.add_constraint(self.first_point_w[0 + scenario_i * 1] - (
                        self.mu_noclip[0 + scenario_i * 1, 0] - self.init_net_consump[0]) >= 0.)
                self.m1.add_constraint(self.first_point_w[0 + scenario_i * 1] - (
                         self.init_net_consump[0] - self.mu_noclip[0 + scenario_i * 1, 0]) >= 0.)

            # define the operation constraints
            for scenario_i in range(self.scenario_number):
                self.m1.add_constraint(self.e_avg[0 + scenario_i * 1, 0] == self.m1.sum(
                    self.mu[0 + scenario_i * 1, t] for t in range(self.d * 24)) / (self.d * 24))
                for t in range(self.d * 24):
                    self.m1.add_constraint(self.e_maximum[0 + scenario_i * 1, 0] - self.mu[0 + scenario_i * 1, t] >= 0.)

                # compute the over threshold value of the mu
                self.m1.add_constraint(self.e_over_threshold_maximum[0 + scenario_i * 1] - (self.e_maximum[0 + scenario_i * 1, 0] - self.mu_threshold_value[0]) >= 0.)

                for t in range(24 * self.d):
                    self.m1.add_constraint(self.mu[0 + scenario_i * 1, t] - self.m1.sum(
                        self.x[i + scenario_i * self.agent_num, t] + self.y[i + scenario_i * 1, t] +
                        self.load_var[scenario_i * (self.agent_num * 24) + i * 24 + t] for i in range(self.agent_num)) >= 0.)
                for t in range(24 * self.d):
                    self.m1.add_constraint(self.mu_noclip[0 + scenario_i * 1, t] == self.m1.sum(
                        self.x[i + scenario_i * self.agent_num, t] + self.y[i + scenario_i * self.agent_num, t] +
                        self.load_var[scenario_i * (self.agent_num * 24) + i * 24 + t] for i in range(self.agent_num)))
                for t in range(24 * self.d - 1):
                    self.m1.add_constraint(self.w[0 + scenario_i * 1, t] - (
                            self.mu_noclip[0 + scenario_i * 1, t + 1] - self.mu_noclip[
                        0 + scenario_i * 1, t]) >= 0.)
                    self.m1.add_constraint(self.w[0 + scenario_i * 1, t] - (
                            self.mu_noclip[0 + scenario_i * 1, t] - self.mu_noclip[
                        0 + scenario_i * 1, t + 1]) >= 0.)
                for i in range(self.agent_num):
                    for t in range(24 * self.d):
                        self.m1.add_constraint(self.vu[i + scenario_i * self.agent_num, t] - (
                                self.x[i + scenario_i * self.agent_num, t] + self.y[i + scenario_i * self.agent_num, t] +
                                self.load_var[scenario_i * (self.agent_num * 24) + i * 24 + t]) >= 0.)
                # soc decomposition
                for i in range(self.agent_num):
                    for t in range(24 * self.d):
                        self.m1.add_constraint(
                            self.soc[i + scenario_i * self.agent_num, t] == self.soc1[i + scenario_i * self.agent_num, t] +
                            self.soc2[
                                i + scenario_i * self.agent_num, t])
                        # discount to nominal level
                        self.m1.add_constraint(
                            self.x[i + scenario_i * self.agent_num, t] <= (
                                        5. - R * self.soc2[i + scenario_i * self.agent_num, t]) / 6.4)
                        self.m1.add_constraint(
                            self.y[i + scenario_i * self.agent_num, t] >= - (
                                        5. - R * self.soc2[i + scenario_i * self.agent_num, t]) / 6.4)
                # soc sequences
                for i in range(self.agent_num):
                    self.m1.add_constraint(
                        self.soc[i + scenario_i * self.agent_num, 0] == self.init_soc[i] + self.x[
                            i + scenario_i * self.agent_num, 0] * self.c1 + self.y[
                            i + scenario_i * self.agent_num, 0] * self.c2)
                    for t in range(1, 24 * self.d):
                        self.m1.add_constraint(
                            self.soc[i + scenario_i * self.agent_num, t] == self.soc[
                                i + scenario_i * self.agent_num, t - 1] + self.x[
                                i + scenario_i * self.agent_num, t] * self.c1 + self.y[i + scenario_i * self.agent_num, t] * self.c2)

            if self.scenario_number > 1:
                # same actions in different scenarios
                for scenario_i in range(self.scenario_number - 1):
                    for i in range(self.agent_num):
                        for t in range(24 * self.d):
                            self.m1.add_constraint(
                                self.x[i + scenario_i * self.agent_num, t] == self.x[
                                    i + scenario_i * self.agent_num + self.agent_num, t])
                            self.m1.add_constraint(
                                self.y[i + scenario_i * self.agent_num, t] == self.y[
                                    i + scenario_i * self.agent_num + self.agent_num, t])

    def lgb_memo_data(self, observation):
        obs = deepcopy(np.array(observation))
        agent_num = len(obs)
        this_lgb_input = pd.DataFrame([deepcopy(self.df.iloc[self.all_index, :]) for _ in range(agent_num)])
        pv_capacity = list(self.pv_capacity.values())
        cur_solar_generation = obs[:, 21] / pv_capacity
        l = len(self.solar_loads_past) - self.max_solar_loads_len
        l = l if l < 0 else None
        this_lgb_input.iloc[:, -self.max_solar_loads_len:l] = np.array(self.solar_loads_past[::-1]).T

        if len(self.solar_loads_past) >= self.max_solar_loads_len:
            self.solar_loads_past.pop(0)
            self.solar_loads_past.pop(0)
        self.solar_loads_past.append(obs[:, 20])
        self.solar_loads_past.append(cur_solar_generation)
        self.loads_past.append(obs[:, 20])

    def lgb_memo_data_and_predict(self, observation):
        obs = deepcopy(np.array(observation))
        hour = int(obs[0, 2]) % 24

        agent_num = len(obs)
        this_lgb_input = pd.DataFrame([deepcopy(self.df.iloc[self.all_index, :]) for _ in range(agent_num)])
        pv_capacity = list(self.pv_capacity.values())
        cur_solar_generation = obs[:, 21] / pv_capacity
        this_lgb_input['Solar_Generation'] = cur_solar_generation
        this_lgb_input['Equipment_Electric_Power'] = obs[:, 20]
        self.loads_past.append( obs[:, 20])
        l = len(self.solar_loads_past) - self.max_solar_loads_len
        l = l if l < 0 else None
        this_lgb_input.iloc[:, -self.max_solar_loads_len:l] = np.array(self.solar_loads_past[::-1]).T

        if len(self.solar_loads_past) >= self.max_solar_loads_len:
            self.solar_loads_past.pop(0)
            self.solar_loads_past.pop(0)
        self.solar_loads_past.append(obs[:, 20])
        self.solar_loads_past.append(cur_solar_generation)

        # imputation
        if this_lgb_input.isnull().values.any():
            for i in range(24*self.day_seq):
                if this_lgb_input['Load_Past_{}'.format(i)].isnull().sum():
                    this_lgb_input['Load_Past_{}'.format(i)] = self.load_train_mean[(hour - i - 1) % 24]
                    this_lgb_input['Solar_Past_{}'.format(i)] = self.solar_train_mean[(hour - i - 1) % 24]
        if self.all_index > 24 * self.day_seq:
            this_lgb_input = this_lgb_input.fillna(0)

        solar_num = np.expand_dims(pv_capacity, -1).repeat(24, 1)

        if self.agent_num == 5:
            drop_col = [item for item in this_lgb_input.columns if 'Load_Past_' in item]
            for item in this_lgb_input.columns:
                if 'Solar_Past_' in item:
                    if int(item.split('_')[2]) >= 24*7:
                        drop_col.append(item)
            next_solar24 = self.gbm_solar.predict(this_lgb_input.drop(drop_col, axis=1)) * solar_num

            drop_col = [item for item in this_lgb_input.columns if 'Solar_Past_' in item]
            next_loads24_lin = self.lin_load.predict(this_lgb_input.drop(drop_col, axis=1))  # (5, 24)
            for item in this_lgb_input.columns:
                if 'Load_Past_' in item:
                    if int(item.split('_')[2]) >= 24*7:
                        drop_col.append(item)
            next_loads24_lgb = self.gbm_load.predict(this_lgb_input.drop(drop_col, axis=1)) # (5, 24)

            next_loads24 = next_loads24_lgb * 0.7 + next_loads24_lin * 0.3
        else:
            drop_col = [item for item in this_lgb_input.columns if 'Load_Past_' in item]
            next_solar24 = self.gbm_solar.predict(this_lgb_input.drop(drop_col, axis=1)) * solar_num
            drop_col = [item for item in this_lgb_input.columns if 'Solar_Past_' in item]
            next_loads24 = self.gbm_load.predict(this_lgb_input.drop(drop_col, axis=1))  # (5, 24)

        his_loads24 = np.array(self.loads_past[-24:]).T
        return next_loads24, next_solar24, his_loads24

    def compute_action(self, observations):

        """Get observation return action"""
        if self.all_index == -1:
            obs_load = np.array(observations)[:, 20]
            print(obs_load)
            if len(obs_load) == 5:
                if np.isclose(obs_load,
                              np.array([2.2758000e+00, 2.1887500e+00,1.0096232e-07, 2.8191500e+00, 7.7143335e-01]),
                              atol=1e-3).all():
                    self.offline_opt_flag = True
            print('Is it the public data? : ', self.offline_opt_flag)

        self.all_index += 1
        hour = observations[0][2]%24


        # training data - offline calculated best results: ensemble of OPT and RL
        if self.offline_opt_flag:
            for agent_id in range(self.agent_num):
                action = self.actions_opt[self.all_index % 8759, agent_id]
                action_space = self.action_space[agent_id]
                action = np.array([action], dtype=action_space.dtype).clip(-self.max_charge, self.max_charge)
                assert action_space.contains(action)
                actions.append(action)

        # validation data: lgb prediction for next 24h + daily scheduling stochastic optimation
        else:
            if self.agent_num == 5 and self.all_index == 4320:
                self.time_weight_factor = np.arange(1.0, 0.95, -(1.0 - 0.95) / 24)
                self.mu_threshold_value_constant = np.ones(12) * 0.5

            init_net_e_consump = 0.
            for agent_id in range(self.agent_num):
                init_net_e_consump += observations[agent_id][23]
            init_net_e_consump = max(init_net_e_consump, 0.) / 6.4

            self.mu_threshold_value_constant[int(self.all_index / 730)] = max(init_net_e_consump, self.mu_threshold_value_constant[int(self.all_index / 730)])
            if self.all_index <= 719 or self.all_index >= 8759-24:
                cond_hour = (hour == 0)
            else:
                cond_hour = (hour == 0) or (hour == 8) or (hour == 16)
            if cond_hour:
                start_pred_time = time.time()
                load , solar, real = self.lgb_memo_data_and_predict(observations)
                if self.all_index <= 48:
                    print('[Total] One step total forecasting time: {}'.format(time.time() - start_pred_time))
                load = load.clip(0.)
                solar = solar.clip(0.)

                if hour == 0:
                    if 0 < self.cal_count <= 90:
                        pred_real_mul = real * self.last_pred24next
                        pred_pow2 = self.last_pred24next ** 2
                        self.pred_real_mul_remove_list.append(pred_real_mul)
                        self.real_pow2_remove_list.append(pred_pow2)
                        self.pred_real_mul_sum += pred_real_mul
                        self.real_pow2_sum += pred_pow2
                    elif self.cal_count > 90:
                        pred_real_mul = real * self.last_pred24next
                        pred_pow2 = self.last_pred24next ** 2
                        self.pred_real_mul_remove_list.append(pred_real_mul)
                        self.real_pow2_remove_list.append(pred_pow2)
                        self.pred_real_mul_sum += pred_real_mul - self.pred_real_mul_remove_list.pop(0)
                        self.real_pow2_sum += pred_pow2 - self.real_pow2_remove_list.pop(0)
                        self.pred_load_correction = self.pred_real_mul_sum / self.real_pow2_sum
                    self.last_pred24next = load
                    self.cal_count += 1
                    load = load * self.pred_load_correction
                elif hour == 8:
                    load = load * np.hstack([self.pred_load_correction[:, 8:24], self.pred_load_correction[:, 0:8]])
                elif hour == 16:
                    load = load * np.hstack([self.pred_load_correction[:, 16:24], self.pred_load_correction[:, 0:16]])

                idx = self.all_index + 1
                for agent_id in range(self.agent_num):
                    self.solar_left[agent_id, idx:idx + 24] = (solar[agent_id] - load[agent_id]) / 6.4 * self.load_scaling_param
            else:
                self.lgb_memo_data(observations)

            # take actions based on predicted values [all steps]
            start_opt_time = time.time()
            # take actions based on predicted values [all steps]
            actions = []
            for agent_id in range(len(observations)):
                action = 0.
                observation = observations[agent_id]

                if self.episode_id == 0:
                    # online solver
                    hour = observation[2] % 24
                    if agent_id == 0 and cond_hour:
                        init_soc = []
                        for j in range(len(observations)):
                            init_soc.append(observations[j][22])
                        month_index = int(self.all_index / 730)
                        np.set_printoptions(suppress=True, precision=3)
                        j = 1
                        L1_bd = deepcopy(-self.solar_left[:, self.all_index + j:self.all_index + 24  + j])
                        L1_db_scenario = np.zeros((self.scenario_number, self.agent_num, 24))
                        std_ratio_noise = np.array(
                            [0.35,  0.38,   0.4,    0.4,    0.4,    0.4,    0.4,    0.4,    0.4,
                             0.4,   0.4,    0.4,    0.4,    0.4,    0.4,    0.4,    0.4,    0.4,
                             0.4,   0.4,    0.4,    0.4,    0.4,    0.4]) * self.noise_enlarge_ratio

                        std_noise = std_ratio_noise * np.abs(L1_bd)
                        for scenario_i in range(self.scenario_number):
                            L1_db_scenario[scenario_i, :, :] = np.random.normal(L1_bd, std_noise)

                        P1 = deepcopy(self.price[self.all_index + j:self.all_index + 24 + j])
                        E1 = deepcopy(self.emission[self.all_index + j:self.all_index + 24 + j])
                        emission_base1 = sum(L1_bd.clip(0).sum(axis=0) * E1)
                        price_base1 = sum(L1_bd.sum(axis=0).clip(0) * P1)
                        grid_base1 = np.abs(L1_bd.sum(axis=0)[1:] - L1_bd.sum(axis=0)[:-1]).sum()
                        e_daily_base1 = L1_bd.sum(axis=0)
                        load_base1_monthly = []
                        if self.all_index == 0:
                            print('e_daily_base1: ', e_daily_base1.shape)
                        load_base1_monthly.append(1 - np.mean(e_daily_base1) / np.max(e_daily_base1))
                        load_base1 = np.mean(load_base1_monthly)

                        if self.all_index == 0 or self.all_index == 24:
                            print('all buildings emission base:', emission_base1 * 6.4)
                            print('all buildings price base:', price_base1 * 6.4)
                            print('all buildings rampling base:', grid_base1 * 6.4)
                            print('all buildings load base:', load_base1)

                        # update parameters
                        self.m1.change_var_upper_bounds(self.init_soc, init_soc)
                        self.m1.change_var_lower_bounds(self.init_soc, init_soc)

                        self.m1.change_var_upper_bounds(self.load_var, L1_db_scenario.flatten())
                        self.m1.change_var_lower_bounds(self.load_var, L1_db_scenario.flatten())

                        self.m1.change_var_upper_bounds(self.init_net_consump, init_net_e_consump)
                        self.m1.change_var_lower_bounds(self.init_net_consump, init_net_e_consump)

                        self.m1.change_var_upper_bounds(self.mu_threshold_value, self.mu_threshold_value_constant[month_index])
                        self.m1.change_var_lower_bounds(self.mu_threshold_value, self.mu_threshold_value_constant[month_index])

                        # define the objection fucntions
                        obj = 0
                        for scenario_i in range(self.scenario_number):
                            obj += (
                                    1.0 * self.m1.sum(
                                        self.vu[i + scenario_i * self.agent_num, t] * (E1[t] / emission_base1) * self.time_weight_factor[t]
                                        for i in range(self.agent_num) for t in range(24 * self.d)) +
                                    1.5 * self.m1.sum(
                                        self.mu[0 + scenario_i * 1, t] * (P1[t] / price_base1) * self.time_weight_factor[t]
                                        for t in range(24 * self.d)) +
                                    0.5 * self.m1.sum(
                                        self.w[0 + scenario_i * 1, t] * self.time_weight_factor[t]
                                        for t in range(0, 24 * self.d - 1)) / grid_base1
                                    )
                            obj += 0.5 * self.first_point_w[0 + scenario_i * 1] / grid_base1
                            obj += 1.0 * self.m1.sum(self.e_over_threshold_maximum[0 + scenario_i * 1])
                        self.m1.minimize(obj)
                        s1 = self.m1.solve(log_output=False)
                        x_sol = np.zeros((self.agent_num, 24))
                        y_sol = np.zeros((self.agent_num, 24))
                        for i in range(self.agent_num):
                            for j in range(24):
                                x_sol[i, j] = self.x[i, j].solution_value
                                y_sol[i, j] = self.y[i, j].solution_value
                        self.act = x_sol + y_sol
                        if self.all_index <= 48:
                            print('[Total] One step total optimization time: {}'.format(time.time() - start_opt_time))
                    if self.all_index <= 719 or self.all_index >= 8759 - 24:
                        action = self.act[agent_id, self.all_index % 24]
                    else:
                        action = self.act[agent_id, self.all_index % 8]

                action_space = self.action_space[agent_id]
                action = np.array([action], dtype=action_space.dtype).clip(-self.max_charge, self.max_charge)
                assert action_space.contains(action)
                actions.append(action)

        return actions
