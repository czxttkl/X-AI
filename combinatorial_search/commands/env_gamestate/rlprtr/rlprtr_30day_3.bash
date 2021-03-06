#!/bin/bash

#SBATCH --job-name=rlprtr_30day3
#SBATCH --error=slurm/rlprtr_30day3_err
#SBATCH --out=slurm/rlprtr_30day3_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u Q_comb_search.py --env_name="env_gamestate" --k=312 --d=15 --test_period=99999999 --load=1 --env_dir="test_probs/prob_env_gamestate_pv0_envseed303" --learn_wall_time_limit=2591000 --root_dir="prtr_models" > slurm/rlprtr_30day3_out

# replace rlprtr_x to rlprtr_y
# change wall time limit

# check test period
# check env_dir
# check env_name
# check load