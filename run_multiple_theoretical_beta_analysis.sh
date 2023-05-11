#!/bin/bash

nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 0 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 1 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 2 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 3 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 4 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 5 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 6 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 7 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 8 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 500 --theo_data 1 --slot_beta 9 &