#!/bin/bash

#SBATCH --job-name=rl_48hr1_pv3
#SBATCH --error=slurm/rl_48hr1_pv3_err
#SBATCH --out=slurm/rl_48hr1_pv3_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="rl" --wall_time_limit=171800 --rl_load=0 --prob_env_dir="test_probs/prob_env_greedymove_pv3_envseed303" > slurm/rl_48hr1_pv3_out

# replace pvx to pvy
# replace rl_xx to rl_yy
# check wall_time_limit
# check rl_load