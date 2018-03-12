#!/bin/bash

#SBATCH --job-name=sl_30day3
#SBATCH --error=slurm/sl_30day3_err
#SBATCH --out=slurm/sl_30day3_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

cp -R prtr_models/sl_env_greedymove_k312_d15_t2591000 prtr_models/sl_env_greedymove_k312_d15_t2591000_2
python -u supervise_learning.py --env_name="env_greedymove" --k=312 --d=15 --load=1 --wall_time_limit=2591000 > slurm/sl_30day3_out

# replace sl_x to sl_y
# change copy directory

# check env_name
# check load