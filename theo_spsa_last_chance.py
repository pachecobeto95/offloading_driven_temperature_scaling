import os, time, sys, json, os, argparse, torch
import config, utils, temperature_scaling, ee_nn
import numpy as np
import pandas as pd
import spsa2 as spsa
import torch.nn as nn

def save_beta_results(savePath, beta_theta, beta_acc, beta_inf_time, ee_prob, threshold, n_branches_edge, max_branches, beta, overhead, calib_mode, mode):
	result = {"beta_acc": beta_acc, "beta_inf_time": beta_inf_time, "ee_prob": ee_prob, "threshold": threshold, "n_branches_edge": n_branches_edge, "beta": beta, 
	"calib_mode": calib_mode, "overhead": overhead, "mode": mode}

	for i in range(max_branches):

		temp_branch = beta_theta[i] if (i < max_branches) else np.nan

		result["temp_branch_%s"%(i+1)] = temp_branch


	df = pd.DataFrame([result])
	df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))


def run_theoretical_beta_analysis(args, df_inf_data, df_inf_data_device, threshold, n_branches_edge, beta_list, savePath, overhead, calib_mode, mode="theo"):
	"""
	This function runs our proposed theoretical model for different beta
	
	In:
	args: contains the arguments inserted via command line
	df_inf_data: dataframe containing inference data with inference measured in the cloud server.
	df_inf_data_device: dataframe containing inference data with inference measured in the edge device.	
	threshold: the confidence threshold used in early-exit DNN model
	n_branches_edge: number of side branches implemented at the edge device
	beta_list: used to iterate and use a beta value for compute the optmization function
	savePath: path to save the results
	overhead: cost to offload to the cloud
	calib_mode: this is the calibration mode used to identify the line in the savePath

	Out:
	None
	"""

	max_exits = args.n_branches + 1

	for beta in beta_list:
		print("Beta: %s"%(beta))

		#This function runs our theoretical model
		#This function returns the parameters chosen for our model 
		beta_theta_theo, beta_opt_loss_theo = spsa.run_theoretical_beta_opt(df_inf_data, 
			df_inf_data_device, beta, threshold, args.max_patience, n_branches_edge, args.n_branches, 
			args.a0, args.c, args.alpha, args.gamma, overhead, mode)

		#This function computes the on-device accuracy using the parameters found previously
		beta_acc_theo, beta_ee_prob_theo = spsa.accuracy_edge(beta_theta_theo, n_branches_edge, 
			threshold, df_inf_data)


		#This function computes the inference time using the parameters found previously
		if(n_branches_edge == 1):
			beta_inf_time, _ = spsa.compute_inference_time(beta_theta_exp, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)
		else:
			beta_inf_time_theo, a = spsa.compute_inference_time_multi_branches(beta_theta_theo, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)

		print("Acc: %s, Inf Time: %s, Exit Prob: %s"%(beta_acc_theo, beta_inf_time_theo, beta_ee_prob_theo))

		#Saves the results
		save_beta_results(savePath, beta_theta_theo, beta_acc_theo, beta_inf_time_theo, beta_ee_prob_theo, threshold, n_branches_edge, args.n_branches, beta, overhead, calib_mode, mode="theo")


def main(args):

	n_classes = 257

	input_dim, dim = 330, 300

	#mode = "theo" if(args.theo_data) else "exp"

	#The next row specifies the file name that contains the inference data measured in cloud server. 
	inf_data_cloud_path = os.path.join(config.DIR_NAME, "last_chance_inf_data_%s_%s_branches.csv"%(args.model_name, args.n_branches))

	#The next row specifies the file name that contains the inference data measured in edge device. 	
	inf_data_device_path = os.path.join(config.DIR_NAME, "new_inference_data", "inference_data_%s_%s_branches_%s_jetson_nano.csv"%(args.model_name, args.n_branches, args.model_id))


	#The next row specifies the file name that saves the results. 	
	resultsPath = os.path.join(config.DIR_NAME, "theo_beta_analysis_%s_%s_branches_overhead_%s_rodrigo_version_test_2.csv"%(args.model_name, args.n_branches, args.overhead))

	threshold_list = [0.8]

	#Defines a list of beta to evaluate the optimization problem. 
	beta_list = np.arange(0, 202, 2)
	beta_list = [100]	
	
	#The next rows reads the inference data 
	df_inf_data_cloud = pd.read_csv(inf_data_cloud_path)
	df_inf_data_device = pd.read_csv(inf_data_device_path)

	for threshold in threshold_list:
		print("Overhead: %s, Threshold: %s"%(args.overhead, threshold))

		#This row runs our proposed theoretical model for a given threhsold and a beta list
		run_theoretical_beta_analysis(args, df_inf_data_cloud, df_inf_data_device, 
			threshold, args.n_branches, beta_list, resultsPath, args.overhead, 
			calib_mode="beta_calib")


if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Learning the Temperature driven for offloading.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, choices=["mobilenet", "resnet18", "resnet152", "vgg16"], 
		help='DNN model name (default: mobilenet)')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))
	parser.add_argument('--batch_size_test', type=int, default=config.batch_size_test, 
		help='Test Batch Size. Default: %s'%(config.batch_size_test))


	# This argument defines the seed for random operations.
	parser.add_argument('--seed', type=int, default=config.seed, 
		help='Seed. Default: %s'%(config.seed))

	# This argument defines the backbone DNN is pretrained.
	parser.add_argument('--pretrained', type=bool, default=config.pretrained, 
		help='Is backbone DNN pretrained? Default: %s'%(config.pretrained))

	# This argument defines Offloading-drive TS uses GPU board.
	parser.add_argument('--cuda', type=bool, default=config.cuda, 
		help='Cuda? Default: %s'%(config.cuda))

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution. Default: %s'%(config.distribution))

	parser.add_argument('--n_branches', type=int, default=config.n_branches, 
		help='Number of side branches. Default: %s'%(config.n_branches))

	parser.add_argument('--max_patience', type=int, default=config.max_patience, 
		help='Number of epochs. Default: %s'%(config.max_patience))

	parser.add_argument('--read_inf_data', type=bool, default=config.read_inf_data, 
		help='Do you read inference data. Default: %s'%(config.read_inf_data))

	parser.add_argument('--a0', type=int, default=config.a0, 
		help='a0. Default: %s'%(config.a0))

	parser.add_argument('--c', type=int, default=config.c, 
		help='c. Default: %s'%(config.c))

	parser.add_argument('--alpha', type=int, default=config.alpha, 
		help='alpha. Default: %s'%(config.alpha))

	parser.add_argument('--gamma', type=int, default=config.gamma, 
		help='gamma. Default: %s'%(config.gamma))

	parser.add_argument('--threshold', type=float, default=config.threshold, 
		help="Threshold that decides if the prediction if confidence enough. Default: %s"%(config.threshold))	

	parser.add_argument('--step', type=float, default=config.step, 
		help="Step of beta. Default: %s"%(config.step))	

	parser.add_argument('--model_id', type=int, default=1)	

	parser.add_argument('--input_dim', type=int, default=330)

	parser.add_argument('--dim', type=int, default=300, help='Dim. Default: %s')
	#parser.add_argument('--theo_data', type=int, default=1, help='Default: True')
	#parser.add_argument('--slot_beta', type=int)
	parser.add_argument('--overhead', type=int)


	args = parser.parse_args()

	main(args)
