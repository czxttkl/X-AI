#!/bin/bash

#SBATCH --job-name=ml_30day1
#SBATCH --error=slurm/ml_30day1_err
#SBATCH --out=slurm/ml_30day1_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=fullnode
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u multilabel_learning.py --env_name="env_greedymove" --k=312 --d=15 --load=0 --wall_time_limit=2591000 > slurm/ml_30day1_out

# replace ml_x to ml_y
# change copy directory

# check env_name
# check load