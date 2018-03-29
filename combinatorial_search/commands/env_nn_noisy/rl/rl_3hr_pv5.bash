#!/bin/bash

#SBATCH --job-name=rl_3hr_pv5
#SBATCH --error=slurm/rl_3hr_pv5_err
#SBATCH --out=slurm/rl_3hr_pv5_out
#SBATCH --exclusive
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="rl" --wall_time_limit=10800 --prob_env_dir="test_probs/prob_env_nn_noisy_pv5_envseed303" > slurm/rl_3hr_pv5_out

# replace pvx to pvy
# replace rl_xx to rl_yy
# check wall_time_limit
