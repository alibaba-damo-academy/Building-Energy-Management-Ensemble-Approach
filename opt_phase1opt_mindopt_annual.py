import numpy as np
import time
import pickle as pk

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper_phase1opt_mindopt_annual import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv


class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations}
    return obs_dict


def evaluate(schema_path):
    env = CityLearnEnv(schema=schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    saved_actions = []
    try:
        while True:
            observations, _, done, _ = env.step(actions)
            saved_actions.append(actions)
            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"score": np.average(metrics_t),
                           "price_cost": metrics_t[0],
                           "emmision_cost": metrics_t[1],
                           "grid_cost": metrics_t[2]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                agent_time_elapsed += time.perf_counter() - step_start

            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")
                metrics_t = env.evaluate()
                metrics = {"score": np.average(metrics_t),
                           "price_cost": metrics_t[0],
                           "emmision_cost": metrics_t[1],
                           "grid_cost": metrics_t[2]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

            if episodes_completed >= Constants.episodes:
                break
    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True

    if not interrupted:
        print("=========================Completed=========================")
    print(f"Total time taken by agent: {agent_time_elapsed}s")

    with open('agents/actions_saved/annual_dispatch_schema1_mindopt.pkl', 'wb') as f:
        pk.dump(np.array(saved_actions).squeeze(), f)

    return metrics["score"]



if __name__ == '__main__':
    print("======================== Evaluation Start ========================= ")
    score1 = evaluate(Constants.schema_path)
    print("======================== Evaluation End ========================= ")