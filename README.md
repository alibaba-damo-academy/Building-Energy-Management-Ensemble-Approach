
### How to reproduce the results

Solution for NeurIPS Citylearn Challenge 2022: https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge

Firstly, please install the necessary packages. For example, you can create an environment from the provided .yml file, then activate the environment.
```
conda env create -f environment.yml
conda activate citylearn_together
```
**Hardware specification:** We run all the experiments on a machine equipped with a CPU: Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz, and a GPU: Nvidia Tesla v100 GPU with 16G memory.

The following steps should be executed at the main directory (citylearn-2022-starter-kit) to reproduce the results.
#### Reproduce Phase 1 results

To solve the LP problem, we tested two solvers: cplex and mindopt. Their performance are comparable (difference less than 0.0005).

Here we provide the example using mindopt:
```
python opt_phase1opt_mindopt_annual.py
python local_evaluation.py
```

Public users can apply the licenses to run the solvers for large-scale problem. These standard solvers are free to apply for academic purpose.

In this competition, we ensemble the actions obtained from optimization and RL to achieve the best performance for schema1 (0.6456); due to the issue of data desensitization within our organization, we provide the solution of optimization here (0.6593).




#### Reproduce Phase 2 & 3 results
1. Forecasting: Training GBDT and Ridge regression
```
python agents/lgb_multi_load_heavy/lgb_train_multi_all.py
python agents/lgb_multi_solar_1pv_new_heavy_1030/lgb_train_multi_all.py
python agents/LinR_multi_load/LinR_train_multi_all.py 
```

2. Stochastic optimization
Please put the data and schema.json files into `./data/citylearn_challenge_2022_phase_1`, then execute:
```
python local_evaluation.py
```
