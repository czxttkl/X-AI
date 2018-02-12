#!/bin/bash

#SBATCH --job-name=rl_12hr_pv0
#SBATCH --error=slurm/rl_12hr_pv0_err
#SBATCH --out=slurm/rl_12hr_pv0_out
#SBATCH --exclusive
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="rl" --wall_time_limit=43200 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" > slurm/rl_12hr_pv0_out

# replace pvx to pvy
# replace rl_xx to rl_yy
# check wall_time_limit