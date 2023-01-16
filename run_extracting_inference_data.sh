#!/bin/bash

python3 extracting_inference_data.py --n_branches 1 --model_name mobilenet --model_id 1
python3 extracting_inference_data.py --n_branches 1 --model_name mobilenet --model_id 2
python3 extracting_inference_data.py --n_branches 1 --model_name mobilenet --model_id 3
python3 extracting_inference_data.py --n_branches 3 --model_name mobilenet --model_id 1
python3 extracting_inference_data.py --n_branches 3 --model_name mobilenet --model_id 3
python3 extracting_inference_data.py --n_branches 5 --model_name mobilenet --model_id 1
python3 extracting_inference_data.py --n_branches 5 --model_name mobilenet --model_id 3
