import numpy as np
import pandas as pd
import pickle as pk
import copy
import time
from mindoptpy import *


class PiecewiseLPAgent:
    """
    Memorized DP Agent
    """

    def __init__(self):
        self.episode_id = -1
        self.agent_num = 5
        self.action_space = {}
        self.index = {}
        self.capacity = 6.4
        self.max_charge = 1.0
        self.solar_left = np.zeros((self.agent_num, 8761))
        self.all_index = int(-1)
        path_to_bu1 = "data/citylearn_challenge_2022_phase_1/Building_1.csv"
        path_to_bu2 = "data/citylearn_challenge_2022_phase_1/Building_2.csv"
        path_to_bu3 = "data/citylearn_challenge_2022_phase_1/Building_3.csv"
        path_to_bu4 = "data/citylearn_challenge_2022_phase_1/Building_4.csv"
        path_to_bu5 = "data/citylearn_challenge_2022_phase_1/Building_5.csv"
        path_to_bu = [path_to_bu1, path_to_bu2, path_to_bu3, path_to_bu4, path_to_bu5]
        for i in range(5):
            df = pd.read_csv(path_to_bu[i])
            self.solar_left[i,:-1] = np.array(df['Solar Generation [W/kW]'] * (5.0 if i == 3 else 4.0) / 1000.0
                                          - df['Equipment Electric Power [kWh]']) / self.capacity

        # read price and co2 cost
        carbon = pd.read_csv('data/citylearn_challenge_2022_phase_1/carbon_intensity.csv')
        pricing = pd.read_csv('data/citylearn_challenge_2022_phase_1/pricing.csv')
        self.price = np.zeros(8761)
        self.price[:-1] = pricing['Electricity Pricing [$]'].values
        self.emission = np.zeros(8761)
        self.emission[:-1] = carbon['kg_CO2/kWh'].values

        self.act = np.zeros(5)

        # recording episode time
        self.start_time = time.time()

        # load factor hyperparameter
        self.alpha_load = 0.40

        self.pv_capacity = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.index[agent_id] = 0
        self.all_index = int(-1)
        if agent_id == 0:
            self.episode_id += 1
            print('One episode time: {}'.format(time.time() - self.start_time))
            self.start_time = time.time()

    # pass building info to read pv capacity
    def set_building_info(self, agent_id, building_info):
        self.pv_capacity[agent_id] = building_info['solar_power']
        print('building info {} - pv number {}.'.format(agent_id, self.pv_capacity[agent_id]))

    def init_opt_model(self, observations):
         print('schema 1 opt')


    def compute_action(self, observations):

        actions = []
        for agent_id in range(len(observations)):
            action = 0.
            observation = observations[agent_id]
            if agent_id == 0:
                self.all_index += 1
            pt = 730
            m_load = 12 # montly calculation unit for load factor
            d_pt = pt * m_load
            if self.episode_id == 0:
                # online solver
                if agent_id == 0 and self.all_index % d_pt == 0:
                    init_soc = []
                    for j in range(len(observations)):
                        init_soc.append(observations[j][22])
                    np.set_printoptions(suppress=True,precision=3)
                    j = 1
                    L1_bd = copy.deepcopy(-self.solar_left[:, self.all_index +j : self.all_index + j + d_pt])
                    P1 = copy.deepcopy(self.price[self.all_index +j :self.all_index + j + d_pt])
                    E1 = copy.deepcopy(self.emission[self.all_index + j:self.all_index + j + d_pt])
                    base_rate = 1.
                    emission_base1 = sum((-self.solar_left).clip(0).sum(axis=0) * self.emission) / base_rate
                    price_base1 = sum((-self.solar_left).sum(axis=0).clip(0) * self.price) / base_rate
                    grid_base1 = np.abs((-self.solar_left).sum(axis=0)[1:]- (-self.solar_left).sum(axis=0)[:-1]).sum() / base_rate
                    load_base1_monthly = []
                    for m in range(12):
                        e_daily_base1 = (-self.solar_left)[:,m*pt:(m+1)*pt].sum(axis=0)
                        load_base1_monthly.append(1 - np.mean(e_daily_base1) / np.max(e_daily_base1))
                    load_base1 = np.mean(load_base1_monthly) / base_rate
                    if self.all_index==0:
                        print('5 buildings emission base:', emission_base1 * 6.4 * base_rate)
                        print('5 buildings price base:', price_base1 * 6.4 * base_rate)
                        print('5 buildings rampling base:', grid_base1 * 6.4 * base_rate)
                        print('5 buildings load base:', load_base1 * base_rate)
                    c1 = 0.912
                    c2 = 1. / c1
                    R = 20.0
                    buildmodel_start_time = time.time()
                    m1 = MdoModel()
                    INF = MdoModel.get_infinity()
                    x = m1.add_vars((5, d_pt), name='x', lb=0, ub=5 / 6.4)
                    y = m1.add_vars((5, d_pt), name='y', lb=-5 / 6.4, ub=0)
                    soc = m1.add_vars((5, d_pt), name='soc', lb=0, ub=1)
                    soc1 = m1.add_vars((5, d_pt), name='soc1', lb=0, ub=0.8)
                    soc2 = m1.add_vars((5, d_pt), name='soc2', lb=0, ub=0.2)
                    vu = m1.add_vars((5, d_pt), name='vu', lb=0, ub=INF)
                    mu = m1.add_vars((1, d_pt), name='mu', lb=0, ub=INF)
                    mu_noclip = m1.add_vars((1, d_pt), name='mu_noclip', lb=-100, ub=INF)
                    w = m1.add_vars((1, d_pt - 1), name='w', lb=0, ub=INF)
                    e_maximum = m1.add_vars((1, m_load), name='e_maximum', lb=0, ub=INF)
                    e_avg = m1.add_vars((1, m_load), name='e_avg', lb=0, ub=INF)
                    for m in range(m_load):
                        m1.add_cons(e_avg[0, m] - quicksum(mu[0, t] for t in range(pt*m, pt*(m+1))) / pt == 0.)
                        for t in range(pt*m, pt*(m+1)):
                            m1.add_cons(e_maximum[0, m] - mu[0, t] >= 0.)

                    for t in range(d_pt):
                        m1.add_cons(mu[0, t] - quicksum(x[i, t] + y[i, t] for i in range(5)) >= L1_bd[:, t].sum())
                    for t in range(d_pt):
                        m1.add_cons(mu_noclip[0, t] - quicksum(x[i, t] + y[i, t] for i in range(5)) == L1_bd[:, t].sum())
                    for t in range(d_pt - 1):
                        m1.add_cons(w[0, t] - (mu_noclip[0, t + 1] - mu_noclip[0, t]) >= 0.)
                        m1.add_cons(w[0, t] - (mu_noclip[0, t] - mu_noclip[0, t + 1]) >= 0.)
                    for i in range(5):
                        for t in range(d_pt):
                            m1.add_cons(vu[i, t] - (x[i, t] + y[i, t]) >= L1_bd[i, t])
                    # soc decomposition
                    for i in range(5):
                        for t in range(d_pt):
                            m1.add_cons(soc[i, t] - (soc1[i, t] + soc2[i, t]) == 0.)
                            # discount to nominal level
                            m1.add_cons(x[i, t] + R * soc2[i, t] / 6.4 <= 5. / 6.4)
                            m1.add_cons(y[i, t] - R * soc2[i, t] / 6.4 >= - 5. / 6.4)
                    # soc sequences
                    for i in range(5):
                        m1.add_cons(soc[i, 0] - (x[i, 0] * c1 + y[i, 0] * c2) == init_soc[i])
                        for t in range(1, d_pt):
                            m1.add_cons(soc[i, t] - (soc[i, t - 1] + x[i, t] * c1 + y[i, t] * c2) == 0.)
                    print('building variable and cosntraint time: {}'.format(time.time() - buildmodel_start_time))
                    setobj_start_time = time.time()
                    m1_obj_load = np.zeros(289103)
                    m1_obj_load[219000:219000 + 8760 * 5] = np.repeat(E1[:, np.newaxis], 5,
                                                                      1).T.flatten() / emission_base1  # vu
                    m1_obj_load[262800:262800 + 8760] = P1 / price_base1  # mu
                    m1_obj_load[280320:280320 + 8759] = 0.5 / grid_base1  # w
                    m1_obj_load[289079:289079 + 12] = 0.5 * self.alpha_load / m_load / load_base1  # e_maximum
                    m1_obj_load[289079 + 12:289079 + 24] = - 0.5 * self.alpha_load / m_load / load_base1  # e_maximum
                    m1.set_objs(list(m1_obj_load))
                    m1.set_min_obj_sense()
                    print('set objective time: {}'.format(time.time() - setobj_start_time))
                    solve_start_time = time.time()
                    m1.set_int_param("Method", -1) #-1=parallel opt; 0=simplex; 1=dual simplex; 2=interior point
                    m1.solve_prob()
                    print('solve pb time: {}'.format(time.time() - solve_start_time))
                    m1.display_results()

                    self.act = np.zeros((5, d_pt))
                    retrieve_start_time = time.time()
                    for i in range(5):
                        for j in range(d_pt):
                            self.act[i, j] = x[i, j].get_real_attr(MDO_REAL_ATTR.PRIMAL_SOLN) + y[i, j].get_real_attr(
                                MDO_REAL_ATTR.PRIMAL_SOLN)
                    print('retrieve solution time: {}'.format(time.time() - retrieve_start_time))
                    action = self.act[agent_id, self.index[agent_id] % d_pt]
                else:
                    action = self.act[agent_id, self.index[agent_id] % d_pt]
                self.index[agent_id] += 1

            action_space = self.action_space[agent_id]
            action = np.array([action], dtype=action_space.dtype).clip(-self.max_charge, self.max_charge)
            assert action_space.contains(action)
            actions.append(action)

        return actions