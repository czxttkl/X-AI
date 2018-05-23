A Collectible Card Game (CCG) Deck Recommender just for fun. The project is expected to complete in May, 2018.

The paper is available here: Q-DeckRec: a Fast Deck Recommendation System for Collectible Card Games

All deck recommendation codes are available under `combinatorial_search` directory.

<br>

### Usage

We mainly have two models: "rl_prtr" refers to Q-DeckRec, and "ga" refers to Genetic Algorithm.

#### Synthetic Neural Network Problem

The synthetic neural network problem assumes the win rate function f() is a neural network. It also assumes x_o is fixed. Therefore, Q-DeckRec will learn a search policy to find the best x_p against the fixed x_o, while x_p is initialized randomly. This problem is the best starting point to experiment Q-DeckRec and other algorithms before moving to real deck recommendation problems.

Generate problems
```
# generate a problem where there are 20 total cards and deck size is 6
# the problem will be serialized in test_probs folder
python3.6 problem_generator.py --k=20 --d=6 --env=env_nn --pv=0 --env_seed=303
```

Test different methods
```
# final results will be stored in test_result.csv in the problem folder
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_k20_d6_t5000/optimizer_model_fixedxoTrue/qlearning"
python3.6 experimenter.py --method="ga" --wall_time_limit=5 --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303"
```

Before test `rl_prtr`, we need to generate a pre-training RL model
```
# the model will be saved in prtr_models/
python3.6 Q_comb_search.py --env_name="env_nn" --k=20 --d=6 --test_period=100 --load=0 --fixed_xo=1 --env_dir="test_probs/prob_env_nn_pv0_envseed303" --learn_wall_time_limit=5000 --root_dir="prtr_models"
# use tensorboard to check progress
tensorboard --logdir=prtr_models/rl_prtr_env_nn_k20_d6_t5000/
```
What you will see in tensorboard is:

![Image of Tensorflow](combinatorial_search/resource/tf_res.png)

x axis represents learning episode and y axis represents the win rate of x_p^d against x_o. The learned search policy will be more and more stable to find the best x_p.

#### Deck Recommendation Using MetaStone + GreedyMove AI
Generate problems
`combinatorial_search/commands/env_greedymove/prob_generate.bash`

Test different methods: see different directories in `combinatorial_search/commands/env_greedymove/`

<br>

### File Structure in `combinatorial_search` directory

`commands`
slurm commands to run experiments

`environment`
Combinatorial Optimization environment set up. `shadow.jar` is the simulator for simulating Hearthstone matches controlled by greedy-based AI.
See https://github.com/czxttkl/metastone for how `shadow.jar` is generated.

`GA`
Contains an example implemented by [DEAP library](https://github.com/DEAP/deap)

`prioritized_exp`
A folder which contains all needed functions for prioritized experience replay (together with `prioritized_memory.py`)

`prtr_models`
Not checked in the repository. But this directory stores all trained models.

`resource`
Record what cards included in the simulator

`slurm`
Slurm outputs files for debugging

`test_probs`
Serialized test problems and test results

`experimenter.py`
A helper to evaluate different algorithms on test problem instances.

`experimenter_cpu_time.py`
A helper to evaluate CPU time usage of `experimenter.py`

`genetic_algorithm.py`
Helper code to implement genetic algorithm's mutate and crossover operators

`logger.py`
A logger helper to log test statistics on files.

`Q_comb_search.py`
A helper to kick off QLearning training. It allows the tuning of various hyperparamters.

`QLearning.py`
Implement Q-Learning with MLP-based function approximator. It has two main functions: `collect_samples` keeps trying different x_o vs. x_p and store experiences (s, a, r, s', a') into a prioritized experience replay; `learn` keeps using the stored experiences to update MLP paramters.

`random_search.py`
Randomly search for decks (each random deck needs a win-rate evaluation, which is costly)

`report.py`
Report tool. For example, you can use `python3.6 report.py --env=env_greedymove` to check the current results for the experiment in our paper.

`supervise_learning.py`
Learn a win-rate predictor and then use Monte Carlo simulation to sample random decks and pick the one with the highest predicted win rate

`supervise_learning_cpu_time.py`
Only for measuring CPU time of `supervise_learning.py`

`tfboard.py`
Tensorboard helper



<br>

### Requirement

Python 3.6. Please also see `requirements.txt`


Credits to https://github.com/jleclanche/fireplace
