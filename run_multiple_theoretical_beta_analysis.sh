#!/bin/bash

nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 0 --overhead 15 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 1 --overhead 15 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 2 --overhead 15 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 3 --overhead 15 &
nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 4 --overhead 15 &
#nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 5 --overhead 15 &
#nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 6 --overhead 15 &
#nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 7 --overhead 15 &
#nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 8 --overhead 15 &
#nohup python3 theoretical_beta_analysis.py --model_name mobilenet --n_branches 3 --model_id 1 --max_iter 20 --theo_data 1 --slot_beta 9 --overhead 15 &