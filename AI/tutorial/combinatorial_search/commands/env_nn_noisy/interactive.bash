# 1 hr prtr
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t3600/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t3600/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t3600/optimizer_model_fixedxoFalse/qlearning"

# 3 hr prtr
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t10800/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t10800/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t10800/optimizer_model_fixedxoFalse/qlearning"

# 6 hr prtr
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t21600/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t21600/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t21600/optimizer_model_fixedxoFalse/qlearning"

# 12 hr prtr
python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t43200/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t43200/optimizer_model_fixedxoFalse/qlearning" \
&& python3.6 experimenter.py --method="rl_prtr" --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303" --prtr_model_dir="prtr_models/rl_prtr_env_nn_noisy_k200_d30_t43200/optimizer_model_fixedxoFalse/qlearning"

python3.6 experimenter.py --method="ga" --wall_time_limit=50 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" \
&& python3.6 experimenter.py --method="ga" --wall_time_limit=50 --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" \
&& python3.6 experimenter.py --method="ga" --wall_time_limit=50 --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303"

python3.6 experimenter.py --method="ga" --wall_time_limit=25 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" \
&& python3.6 experimenter.py --method="ga" --wall_time_limit=25 --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" \
&& python3.6 experimenter.py --method="ga" --wall_time_limit=25 --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303"

python3.6 experimenter.py --method="random" --wall_time_limit=50 --prob_env_dir="test_probs/prob_env_nn_noisy_pv0_envseed303" \
&& python3.6 experimenter.py --method="random" --wall_time_limit=50 --prob_env_dir="test_probs/prob_env_nn_noisy_pv1_envseed303" \
&& python3.6 experimenter.py --method="random" --wall_time_limit=50 --prob_env_dir="test_probs/prob_env_nn_noisy_pv2_envseed303"
