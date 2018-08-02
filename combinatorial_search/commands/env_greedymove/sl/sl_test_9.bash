#!/bin/bash

#SBATCH --job-name=sl_test_pv9
#SBATCH --error=slurm/sl_test_pv9_err
#SBATCH --out=slurm/sl_test_pv9_out
#SBATCH --exclusive
#SBATCH --time 24:00:00
#SBATCH --partition=fullnode
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv9_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=67 > slurm/sl_test_pv9_67_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv9_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=670 > slurm/sl_test_pv9_670_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv9_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=6700 > slurm/sl_test_pv9_6700_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv9_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=67000 > slurm/sl_test_pv9_67000_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv9_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=670000 > slurm/sl_test_pv9_670000_out && \
python -u experimenter.py --method="sl" --prob_env_dir="test_probs/prob_env_greedymove_pv9_envseed303" --prtr_model_dir="prtr_models/sl_env_greedymove_k312_d15_t2591000_3" --wall_time_limit=2591000 --sl_num_trial=6700000 > slurm/sl_test_pv9_6700000_out


# replace pvx to pvy
