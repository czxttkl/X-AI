#!/bin/bash

#SBATCH --job-name=rlprtr_1hr
#SBATCH --error=slurm/rlprtr_1hr_err
#SBATCH --out=slurm/rlprtr_1hr_out
#SBATCH --exclusive
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u Q_comb_search.py --env_name="env_nn_noisy" --k=312 --d=15 --test_period=100 --load=0 --env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --learn_wall_time_limit=3600 --root_dir="prtr_models" > slurm/rlprtr_1hr_out

# replace rlprtr_x to rlprtr_y
# change wall time limit