#!/bin/bash

#SBATCH --job-name=ml_test_pv0
#SBATCH --error=slurm/ml_test_pv0_err
#SBATCH --out=slurm/ml_test_pv0_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=fullnode
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="ml" --prob_env_dir="test_probs/prob_env_greedymove_pv0_envseed303" --prtr_model_dir="prtr_models/ml_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 > slurm/ml_test_pv0_out


# replace pvx to pvy
