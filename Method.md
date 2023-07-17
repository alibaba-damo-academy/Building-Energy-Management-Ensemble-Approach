## Buildings' Energy Management through an Ensemble Approach of Forecasting, Optimization and Reinforcement Learning

### Phase 1: Optimization for Full-year Dispatch with Complete Information
#### Optimization Objective and Constraint
We aim to optimze the average of price cost, emission cost and grid cost of a microgrid community of a full year. Assume the total number of timesteps is $`T`$ and the number of building is $`I`$, then we denote that, given the difference between building's load and solar generation $`L = (l_{it})  \in {\mathbb{R}}^{I \times T}`$, the price cost unit $`P = (p_t)  \in {\mathbb{R}^+}^{T}`$ and emission cost unit $`E = (e_t)  \in {\mathbb{R}^+}^{T}`$ .

The optimal actions $`X = (x_{it})  \in [-5/6.4,\, 5/6.4]^{I\times T}`$ satisfy:
```math
\min_{X} \quad \frac{\sum_{t=1}^T \max\left(\sum_{i=1}^I\left(l_{it}+x_{it}\right), 0\right)\cdot p_t}{C_{\rm price-no\,bat}} + \frac{\sum_{t=1}^T\left(\sum_{i=1}^I \max \left(l_{it}+x_{it}, 0\right)\right) \cdot e_t}{C_{\rm emission-no\,bat}} + 
 \frac{1}{2} \cdot\frac{\sum_{t=2}^T{|\sum_{i=1}^I(l_{it} + x_{it}) -\sum_{i=1}^I(l_{i,t-1} + x_{i,t-1})|}}{C_{\rm rampling-no\,bat}} + 
 \frac{1}{2} \cdot\frac{1- \sum_{m=1}^{12}\frac{\text{avg}\left(\sum_{i=1}^I(l_{it} + x_{it})\right)}{\max \left (\sum_{i=1}^I(l_{it} + x_{it})\right)}}{C_{\rm load\,factor-no\,bat}}  \\
\text{s.t.:} \quad 0 \leqslant \sum_{t=0}^{T_0} x_{it} \cdot \left(c \cdot \mathbb{I}({x\ge 0)} + \frac{1}{c}\cdot\mathbb{I}(x<0)\right) \leqslant 1 \quad \text{for all }i \text{ and } 0 \leqslant T_0<T.
```



#### Model Linearization 
To facilitate the optimization solving process, we linearzie the model by adding and replacing variables and constraint, as follows:
1. We define separately the charge actions $`X^{+} = (x^+_{it})  \in [0,1]^{I\times T}`$ and the discharge actions $`X^- = (x^-_{it})  \in [-1,0]^{I\times T}`$
2. We add the soc variables: $`S = (s_{it})  \in [0,1]^{I\times T}`$ and given the initial soc $`s_{i0}`$.
3. We linearize the term of load factor as: $`\alpha \cdot(\max - \text{avg})`$, with the hyperparameter $`\alpha`$ to tune.
4. We linearize the objective function by adding:
(1) Building net consumption $`\Nu= (\nu_{it}) \in [0,+\infty]^{I\times T}`$ and district net consumption $`\Mu = (\mu_{t})\in [0,+\infty]^{T}`$.
(2) Difference between consumpitions of each two adjacent timesteps $`\Omega = (\omega_{t})\in [0,+\infty]^{T}`$
(3) Mensual average district consumption $`e_{\rm avg, m}`$ and mensual maximum district consumption $`e_{\rm max, m}`$, for $`m=1,2, \cdots, 12`$.
 
After that, we fully linearize the objective and constraint, as follows:
```math
\min \quad\frac{\sum_{t=1}^{T}\mu_t\cdot p_t}{C_{\rm price-no\,bat}} +
\frac{\sum_{t=1}^{T} \sum_{i=1}^I \nu_{it}\cdot e_t}{C_{\rm emission-no\,bat}} + \frac{1}{2} \cdot\frac{\sum_{t=2}^{T}\omega_t}{C_{\rm rampling-no\,bat}} +
 \frac{\alpha}{2} \cdot \frac{\sum_{m=1}^{12}(e_{\text{max},m} - e_{\text{avg}, m})}{C_{\rm load\,factor-no\,bat}} \\
\text{s.t.:} \quad \mu_t \ge 0\,, \mu_t \ge \sum_{i}(l_{it} + x^{+}_{it}+x^{-}_{it}) \\
\nu_{it} \ge l_{it} +x^+_{it}+x^-_{it} \\
s_{it} = s_{i,t-1} + c \cdot x^+_{it} + \frac{1}{c} \cdot x^-_{it} \\
-\omega_t \le \sum_{i}(l_{it} +x^+_{it}+x^-_{it}) - \sum_{i}(l_{i,t-1} +x^+_{i,t-1}+x^-_{i,t-1}) \le \omega_t \\
e_{\rm avg, m} = \frac{1}{730} \sum_{t=(m-1)\cdot730}^{m\cdot730} \sum_{i}(l_{it} +x^+_{it}+ x^-_{it}) \\
e_{\rm max, m} \ge\sum_{i}(l_{it} + x^+_{it}+ x^-_{it})
```
Optimal charge or discharge actions can be obtained by solving the LP problem above.

#### Ensemble of optimization and reinforcement learning
We have also conducted reinforcement learning, in parallel of optimization. We observe that the two methods exhibit their own advantages in different period throughout a year. Therefore, an ensemble of these two methods according to timesteps has been applied. For more details of reinforcement learning, please refer to the document of detailed implementation.

### Phase 2&3: Stochastic Optimization for Rolling-Horizon Dispatch with Solar Generation and Load Forecasting
For the validation and test set, since the future information is not observed in current step, we propose a two-stage method: a day-ahead forecasting task, followed by stochastic optimization.

#### Self-Adaptive Forecasting
We aim to predict the solar generation and load of the future 24 hours. Indepedent GBDT model has been setup for each timescale. Essential features include timestamps, next-day weather information, historical value of solar generation or load. In addition, due to the data drift issue, we propose a self-adaptive prediction adjustment, by comparing the historical real value $`L = (l_{it})  \in {\mathbb{R}}^{I \times T}`$ and predicted value $`\hat{L} = \left(\hat{l}_{it}\right)  \in {\mathbb{R}}^{I \times T}`$. Then self-adaptive prediction adjustment parameter is defined as:
```math
\hat{\beta} = \underset{\beta}{\operatorname{argmin}} \quad \sum_{t=1}^T \left(\beta \cdot \hat{l}_{it} - l_{it}\right)^2 \quad \Rightarrow \quad
\hat{\beta} = \frac{\sum_{t=1}^Tl_{it}\cdot\hat{l}_{it}}{\sum_{t=1}^T {\hat{l}_{it}}^2}\,.
```

#### Rolling-Horizon Optimization
We aim to apply model predictive control in a framework of rolling-horizon optimzation:

1. At timestamp $`t`$, based on the predicted solar generation and load of next 24 hours, we perform a daily dispatch. Optimization formula of LP problem is the same as in Phase1, but we replace the annual dispatch with a daily basis, where the true values of solar generation and load are replaced by the predicted values.

2. At timestamp $`t+T`$, we update the predicted values, and repeat solving the optimization problem for the next 24 hours.

#### Stochastic Optimization
Regarding the uncertainty of forecasting, we perform a stochasitic optimization to improve the generalization ability. Gaussian noise is added on the predicted value, to generate a multi-scenes optimization problem which we solve coordinately.

To simplify the notations, we denote the orginal deterministic optimization problem as $`\min_{X} F(X, Y; \hat{L})`$, where $`X`$ represents the objective charge/discharge varibles, $`Y`$ for other auxillary variables and $`L`$ the predicted value for (load - solar generation). Then, the stochastic optimization with $`N`$ scenes can be formulated as:
```math
\min_{X_1 = X_2 = \cdots = X_N} \quad \sum_{n=1}^{N} F(X_n, Y_n; \hat{L}_n) \,,
```
where $`\hat{L}_n`$ represents the predicted value with gaussian noise.

In this way, the rolling-horizon optimal actions can be obtained by solving the stochastic optimization with forecasting.



