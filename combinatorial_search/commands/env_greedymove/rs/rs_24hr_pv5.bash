#!/bin/bash

#SBATCH --job-name=rs_24hr_pv5
#SBATCH --error=slurm/rs_24hr_pv5_err
#SBATCH --out=slurm/rs_24hr_pv5_out
#SBATCH --exclusive
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="random" --wall_time_limit=85400 --prob_env_dir="test_probs/prob_env_greedymove_pv5_envseed303" > slurm/rs_24hr_pv5_out

# replace pvx to pvy
# replace rs_xx to rs_yy
# check wall_time_limit