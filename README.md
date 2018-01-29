

A Collectible Card Game (CCG) Deck Recommender just for fun. The project is expected to complete in May, 2018.



### Usage

#### Synthetic Neural Network Problem
Generate problems
```
python3.6 problem_generator.py --k=20 --d=6 --env=env_nn --pv=0 --env_seed=303
```
Test different methods
```
python3.6 experimenter.py --method="rl" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303"
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_k20_d6_t500/optimizer_model_fixedxoFalse/qlearning"
python3.6 experimenter.py --method="random" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303"
python3.6 experimenter.py --method="rbf" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303"
python3.6 experimenter.py --method="ga" --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303"
```
Generate pre-training RL model
```
python3.6 Q_comb_search.py --env_name="env_nn" --test_period=100 --load=0 --env_dir="test_probs/prob_env_nn_pv0_envseed303" --learn_wall_time_limit=500 --root_dir="prtr_models"
```

### Requirement

Python 3.6


Credits to https://github.com/jleclanche/fireplace
