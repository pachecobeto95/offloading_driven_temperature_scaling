import os, time, sys, json, os, argparse, torch
import config, utils, spsa, temperature_scaling, ee_nn
import numpy as np
import pandas as pd

def extractGlobalTSTemperature(args, temp_data_path, threshold, n_branches_edge):

	df_temp = pd.read_csv(temp_data_path)
	df_temp = df_temp[(df_temp.threshold==threshold) & (df_temp.n_branches == n_branches_edge)]

	temp_list = ["temp_branch_%s"%(i) for i in range(1, args.n_branches+1)]

	#return [df_temp["temperature"].values[0]]*n_branches_edge
	#return df_temp[temp_list].values[0]
	return df_temp[temp_list].values


def save_beta_results(savePath, beta_theta, beta_acc, beta_inf_time, ee_prob, threshold, n_branches_edge, max_branches, beta, overhead, calib_mode):
	result = {"beta_acc": beta_acc, "beta_inf_time": beta_inf_time, "ee_prob": ee_prob, "threshold": threshold, "n_branches_edge": n_branches_edge, "beta": beta, 
	"calib_mode": calib_mode, "overhead": overhead}

	for i in range(max_branches):

		temp_branch = beta_theta[i] if (i < max_branches) else np.nan

		result["temp_branch_%s"%(i+1)] = temp_branch


	df = pd.DataFrame([result])
	df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))


def run_overall_acc_theoretical_beta_analysis(args, df_inf_data, df_val_inf_data, df_inf_data_device, threshold, n_branches_edge, beta_list, savePath, overhead, mode, calib_mode):

	max_exits = args.n_branches + 1

	for beta in beta_list:
		print("Beta: %s"%(beta))

		beta_theta, beta_opt_loss = spsa.run_overall_acc_theoretical_beta_opt(df_val_inf_data, df_inf_data_device, beta, threshold, args.max_iter, n_branches_edge, args.n_branches, 
			args.a0, args.c, args.alpha, args.gamma, overhead, mode)

		beta_acc, beta_ee_prob = spsa.overall_accuracy(beta_theta, n_branches_edge, threshold, df_inf_data)
		#beta_inf_time, _ = spsa.compute_inference_time(beta_theta, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)

		if(n_branches_edge == 1):
			beta_inf_time, _ = spsa.compute_inference_time(beta_theta, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)
		else:
			beta_inf_time, _ = spsa.compute_inference_time_multi_branches(beta_theta, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)

		print("Test")
		print("Acc: %s, Inf Time: %s, Exit Prob: %s"%(beta_acc, beta_inf_time, beta_ee_prob))

		save_beta_results(savePath, beta_theta, beta_acc, beta_inf_time, beta_ee_prob, threshold, n_branches_edge, args.n_branches, beta, overhead, calib_mode)


def runNoCalibInference(args, df_inf_data, df_val_inf_data, df_inf_data_device, threshold, n_branches_edge, savePath, overhead, calib_mode):

	temp_list = np.ones(n_branches_edge)

	max_exits = args.n_branches + 1

	beta = 0

	no_calib_acc, no_calib_ee_prob = spsa.overall_accuracy(temp_list, n_branches_edge, threshold, df_inf_data)

	if(n_branches_edge == 1):
		no_calib_inf_time, _ = spsa.compute_inference_time(temp_list, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)
	else:
		no_calib_inf_time, _ = spsa.compute_inference_time_multi_branches(temp_list, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)

	save_beta_results(savePath, temp_list, no_calib_acc, no_calib_inf_time, no_calib_ee_prob, threshold, n_branches_edge, args.n_branches, beta, overhead, calib_mode)

def runGlobalTemperatureScalingInference(args, df_inf_data, df_val_inf_data, df_inf_data_device, threshold, n_branches_edge, savePath, global_ts_path, overhead, calib_mode):

	max_exits = args.n_branches + 1

	beta = 0

	temperature_overall_list = extractGlobalTSTemperature(args, global_ts_path, threshold, n_branches_edge)			


	for temperature_overall in temperature_overall_list:

		global_ts_acc, global_ts_ee_prob = spsa.overall_accuracy(temperature_overall, n_branches_edge, threshold, df_inf_data)

		if(n_branches_edge == 1):
			global_ts_inf_time, ee_prob = spsa.compute_inference_time(temperature_overall, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)
		else:
			global_ts_inf_time, ee_prob = spsa.compute_inference_time_multi_branches(temperature_overall, n_branches_edge, max_exits, threshold, df_inf_data, df_inf_data_device, overhead)

		print(global_ts_acc, global_ts_ee_prob, global_ts_inf_time, ee_prob)

		save_beta_results(savePath, temperature_overall, global_ts_acc, global_ts_inf_time, global_ts_ee_prob, threshold, n_branches_edge, args.n_branches, beta, overhead, calib_mode)


def main(args):

	n_classes = 257

	input_dim, dim = 330, 300

	mode = "theo" if(args.theo_data) else "exp"

	inf_data_cloud_path = os.path.join(config.DIR_NAME, "new_inference_data", "inference_data_%s_%s_branches_%s_local_server.csv"%(args.model_name, args.n_branches, args.model_id))
	inf_data_device_path = os.path.join(config.DIR_NAME, "new_inference_data", "inference_data_%s_%s_branches_%s_jetson_nano.csv"%(args.model_name, args.n_branches, args.model_id))

	#resultsPath = os.path.join(config.DIR_NAME, "overall_acc_adaTS_%s_%s_branches_%s_with_overhead_with_nano_pos_3_review_%s_overhead_%s.csv"%(args.model_name, args.n_branches, args.model_id, mode, args.overhead))
	resultsPath = os.path.join(config.DIR_NAME, "overall_acc_adaTS_mobilenet_1_branches_1_with_overhead_with_nano_pos_6_review_theo.csv")
	global_ts_path = os.path.join(config.DIR_NAME, "alternative_temperature_%s_%s_branches_id_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	#threshold_list = [0.7, 0.8, 0.9]
	threshold = 0.8
	#beta_list = np.arange(0, config.max_beta+config.step_beta, 0.1)
	#beta_list = [np.arange(10*i, 10*(i+1), 0.1) for i in range(10)]
	#beta_list = [np.arange(0, 51, 1), np.arange(51, 101, 1)]	
	#beta_list = beta_list[args.slot_beta]
	beta_list = [[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], [10000, 60, 65, 70, 75, 85, 95, 100]]
	#beta_list = [np.arange(0, 51, 1), np.arange(51, 101, 1)]
	beta_list = beta_list[args.slot_beta]


	df_inf_data_cloud = pd.read_csv(inf_data_cloud_path)
	df_inf_data_device = pd.read_csv(inf_data_device_path)

	#overhead_list = np.arange(0, config.max_overhead+config.step_overhead, config.step_overhead)
	overhead_list = [args.overhead]

	for overhead in overhead_list:

		#for n_branches_edge in reversed(range(1, args.n_branches+1)):
		
		print("Overhead: %s, Nr Branches: %s, Threshold: %s"%(overhead, args.n_branches, threshold))

		run_overall_acc_theoretical_beta_analysis(args, df_inf_data_cloud, df_inf_data_cloud, df_inf_data_device, threshold, args.n_branches, 
			beta_list, resultsPath, overhead, mode, calib_mode="overall_acc_beta_calib")


		#runNoCalibInference(args, df_inf_data_cloud, df_inf_data_cloud, df_inf_data_device, threshold, args.n_branches, resultsPath, overhead, calib_mode="no_calib")

		runGlobalTemperatureScalingInference(args, df_inf_data_cloud, df_inf_data_cloud, df_inf_data_device, threshold, args.n_branches, resultsPath, global_ts_path, 
			overhead, calib_mode="global_TS")


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

	parser.add_argument('--max_iter', type=int, default=config.max_iter, 
		help='Number of epochs. Default: %s'%(config.max_iter))

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
	parser.add_argument('--theo_data', type=int, help='Default: True')
	parser.add_argument('--slot_beta', type=int)
	parser.add_argument('--overhead', type=int)


	args = parser.parse_args()

	main(args)
