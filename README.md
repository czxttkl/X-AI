A Collectible Card Game (CCG) Deck Recommender just for fun. The project is expected to complete in May, 2018.

The paper is available here:

All deck recommendation codes are available under `combinatorial_search` directory.

<br>

### Usage

#### Synthetic Neural Network Problem
Generate problems
```
python3.6 problem_generator.py --k=20 --d=6 --env=env_nn_noisy --pv=0 --env_seed=303
```
Test different methods
```
python3.6 experimenter.py --method="rl" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303"
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k20_d6_t500/optimizer_model_fixedxoFalse/qlearning"
python3.6 experimenter.py --method="random" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303"
python3.6 experimenter.py --method="ga" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303"
```
Generate pre-training RL model
```
python3.6 Q_comb_search.py --env_name="env_nn_noisy" --k=20 --d=6 --test_period=100 --load=0 --env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --learn_wall_time_limit=500 --root_dir="prtr_models"
```

#### Deck Recommendation Using MetaStone + GreedyMove AI
Generate problems
`combinatorial_search/commands/env_greedymove/prob_generate.bash`

Test different methods
See different directories in `combinatorial_search/commands/env_greedymove/`

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

Python 3.6


Credits to https://github.com/jleclanche/fireplace
