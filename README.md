# Decision-Aware MBRL Papers
A curated list of decision-aware MBRL papers in survey (as of September 2023): [A Unified View of Solving Objective Mismatch in Model-Based Reinforcement Learning](https://arxiv.org/abs/2310.06253). We identify 4 categories of approaches to address the [Objective Mismatch](https://arxiv.org/abs/2002.04523) problem in MBRL:

1. [Distribution Correction](#distribution-correction) (policy and model shifts): Shifting the learning of model or policy to state/actions that matter!
2. [Control-As-Inference](#control-as-inference): Unifying model and policy learning by maximizing the likelihood of optimal imagined trajectories!
3. [Value-Equivalence](#value-equivalence) (value-prediction and robust control): Predict value rather than features and rurther learn robust dynamics with signed predictions! 
4. [Differentiable Planning](#differentiable-planning): Solving all control components with gradients directly!

![alt text](/img/mbrl_tree.png)
![alt text](/img/tb1.png)
![alt text](/img/tb2.png)

```
@article{wei2023unified,
    title={A Unified View on Solving Objective Mismatch in Model-Based Reinforcement Learning},
    author={Wei, Ran and Lambert, Nathan and McDonald, Anthony and Garcia, Alfredo and Calandra, Roberto},
    year={2023},
    eprint={2310.06253},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Distribution Correction
### Model-Shift
* Discriminator Augmented Model-Based Reinforcement Learning, Haghgoo et al, 2021, [paper](https://arxiv.org/abs/2103.12999)

### Policy-Shift
* A Unified Framework for Alternating Offline Model Training and Policy Learning, Yang et al, 2022, [paper](https://arxiv.org/abs/2210.05922)
* Gradient-aware model-based policy search, D'Oro et al, 2020, [paper](https://arxiv.org/abs/1909.04115)
* Live in the Moment: Learning Dynamics Model Adapted to Evolving Policy, Wang et al, 2020, [paper](https://arxiv.org/abs/2207.12141)
* Policy-Aware Model Learning via Transition Occupancy Matching, Ma et al, 2023, [paper](https://arxiv.org/abs/2305.12663)
* Weighted model estimation for offline model-based reinforcement learning, Hishinuma et al, 2021, [paper](https://proceedings.neurips.cc/paper/2021/hash/949694a5059302e7283073b502f094d7-Abstract.html)

## Control-As-Inference
* Mismatched No More:
Joint Model-Policy Optimization for Model-Based RL, Eysenbach et al, 2021, [paper](https://arxiv.org/abs/2110.02758)
* Simplifying Model-based RL: Learning Representations, Latent-space Models, and Policies with One Objective, Ghugare et al, 2023, [paper](https://arxiv.org/abs/2209.08466)
* Variational model-based policy optimization, Chow et al, 2020, [paper](https://arxiv.org/abs/2006.05443)

## Value-Equivalence
### Value Prediction
* Approximate Value Equivalence, Grimm et al, 2022, [paper](https://openreview.net/forum?id=S2Awu3Zn04v)
* Bridging Worlds in Reinforcement Learning with Model-Advantage, Modhe et al, 2020, [paper](https://openreview.net/forum?id=XBRYX4c_xFQ)
* Equivalence between wasserstein and value-aware loss for model-based reinforcement learning, Asadi et al, 2018, [paper](https://arxiv.org/abs/1806.01265)
* Iterative value-aware model learning, Farahmand et al, 2018, [paper](https://papers.nips.cc/paper_files/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html)
* Lambda-AC: Learning latent decision-aware models for reinforcement learning in continuous state-spaces, Voelcker et al, 2023, [paper](https://arxiv.org/abs/2306.17366)
* Mastering atari, go, chess and shogi by planning with a learned model, Schrittwieser et al, 2020, [paper](https://arxiv.org/abs/1911.08265)
* Minimax model learning, Voloshin et al, 2021, [paper](https://arxiv.org/abs/2103.02084)
* Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice, Modhe et al, 2021, [paper](https://arxiv.org/abs/2106.14080)
* Model-based reinforcement learning with value-targeted regression, Ayoub et al, 2020, [paper](https://arxiv.org/abs/2006.01107)
* Online and offline reinforcement learning by planning with a learned model, Schrittwieser et al, 2021, [paper](https://arxiv.org/abs/2104.06294)
* Policy-aware model learning for policy gradient methods, Abachi et al, 2020, [paper](https://arxiv.org/abs/2003.00030)
* Proper value equivalence, Grimm et al, 2021, [paper](https://arxiv.org/abs/2106.10316)
* The predictron: End-to-end learning and planning, Silver et al, 2017, [paper](https://arxiv.org/abs/1612.08810)
* The value equivalence principle for model-based reinforcement learning, Grimm et al, 2020, [paper](https://arxiv.org/abs/2011.03506)
* Value gradient weighted model-based reinforcement learning, Voelcker et al, 2022, [paper](https://arxiv.org/abs/2204.01464)
* Value prediction network, Oh et al, 2017, [paper](https://arxiv.org/abs/1707.03497)
* Value-aware loss function for model-based reinforcement learning, Farahmand et al, 2017, [paper](https://proceedings.mlr.press/v54/farahmand17a.html)

### Robust Control
* RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning, Rigter et al, 2022, [paper](https://arxiv.org/abs/2204.12581)
* The Virtues of Laziness in Model-based RL: A Unified Objective and Algorithms, Vemula et al, 2023, [paper](https://arxiv.org/abs/2303.00694)

## Differentiable Planning
### RL
* Control-oriented model-based reinforcement learning with implicit differentiation, Nikishin et al, 2022, [paper](https://arxiv.org/abs/2106.03273)
* Goal-driven dynamics learning via Bayesian optimization, Bansal et al, 2017, [paper](https://arxiv.org/abs/1703.09260)
* Learning mdps from features: Predict-then-optimize for sequential decision making by reinforcement learning, Wang et al, 2021, [paper](https://arxiv.org/abs/2106.03279)
* Reinforcement learning with misspecified model classes, Joseph et al, 2013, [paper](https://ieeexplore.ieee.org/document/6630686)
* Robust Decision-Focused Learning for Reward Transfer, Sharma et al, 2023, [paper](https://arxiv.org/abs/2304.03365)
* TaskMet: Task-Driven Metric Learning for Model Learning, Bansal et al, 2023, [paper](https://openreview.net/forum?id=Yr7XOXCPQR&referrer=%5Bthe%20profile%20of%20Ricky%20T.%20Q.%20Chen%5D(%2Fprofile%3Fid%3D~Ricky_T._Q._Chen1))
* The differentiable cross-entropy method, Amos et al, 2020, [paper](https://arxiv.org/abs/1909.12830)
* Treeqn and atreec: Differentiable tree-structured models for deep reinforcement learning, Farquhar et al, 2018, [paper](https://arxiv.org/abs/1710.11417)
* Understanding End-to-End Model-Based Reinforcement Learning Methods as Implicit Parameterization, Gehring et al, 2021, [paper](https://openreview.net/forum?id=xj2sE--Q90e)

### IRL & IL
* A Bayesian Approach to Robust Inverse Reinforcement Learning, Wei et al, 2023, [paper](https://arxiv.org/abs/2309.08571)
* Differentiable mpc for end-to-end planning and control, Amos et al, 2018, [paper](https://arxiv.org/abs/1810.13400)
* Path integral networks: End-to-end differentiable optimal control, Okada et al, 2017, [paper](https://arxiv.org/abs/1706.09597)
* Qmdp-net: Deep learning for planning under partial observability, Karkus et al, 2017, [paper](https://arxiv.org/abs/1703.06692)
* Universal planning networks: Learning generalizable representations for visuomotor control, 2018, Srinivas et al, [paper](https://arxiv.org/abs/1804.00645)
* Value iteration networks, Tamar et al, 2016, [paper](https://arxiv.org/abs/1602.02867)