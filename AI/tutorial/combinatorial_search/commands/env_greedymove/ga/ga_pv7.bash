#!/bin/bash

#SBATCH --job-name=ga_pv7
#SBATCH --error=slurm/ga_pv7_err
#SBATCH --out=slurm/ga_pv7_out
#SBATCH --exclusive
#SBATCH --partition=ser-par-10g-5
#SBATCH -N 1
#SBATCH -D /home/chen.zhe/combinatorial_search

source activate myenv
work=/home/chen.zhe/combinatorial_search
cd $work

python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1380 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303" && \
python -u experimenter.py --method="ga" --wall_time_limit=1500 --prob_env_dir="test_probs/prob_env_greedymove_pv7_envseed303"

# replace pvx to pvy
# check wall_time_limit