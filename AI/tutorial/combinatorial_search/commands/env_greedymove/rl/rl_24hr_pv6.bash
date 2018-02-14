#!/bin/bash

#SBATCH --job-name=rl_24hr_pv6
#SBATCH --error=slurm/rl_24hr_pv6_err
#SBATCH --out=slurm/rl_24hr_pv6_out
#SBATCH --exclusive
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="rl" --wall_time_limit=85400 --rl_load=0 --prob_env_dir="test_probs/prob_env_greedymove_pv6_envseed303" > slurm/rl_24hr_pv6_out

# replace pvx to pvy
# replace rl_xx to rl_yy
# check wall_time_limit
# check rl_load