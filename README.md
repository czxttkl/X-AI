

A Collectible Card Game (CCG) Deck Recommender just for fun. The project is expected to complete in May, 2018.



### Usage

#### Synthetic Neural Network Problem
```
python3.6 problem_generator.py --k=20 --d=6 --env=env_nn --pv=0 --env_seed=303
python3.6 experimenter.py --wall_time_limit=500 --prob_env_dir="test_probs/prob_env_nn_pv0_envseed303" --method="rl"
```



### Requirement

Python 3.6


Credits to https://github.com/jleclanche/fireplace
