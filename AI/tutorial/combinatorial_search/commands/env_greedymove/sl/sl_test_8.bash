#!/bin/bash

#SBATCH --job-name=sl_test_pv8
#SBATCH --error=slurm/sl_test_pv8_err
#SBATCH --out=slurm/sl_test_pv8_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv8_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=67 > slurm/sl_test_pv8_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv8_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=670 > slurm/sl_test_pv8_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv8_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=6700 > slurm/sl_test_pv8_out


# replace sl_x to sl_y
# change copy directory

# check env_name
# check load