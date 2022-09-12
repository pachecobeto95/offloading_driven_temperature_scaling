import os, time, sys, json, os, argparse, torch
import config, utils, spsa
from early_exit_dnn import Early_Exit_DNN
import numpy as np
import pandas as pd



def main(args):


	model_id = config.models_id_dict[args.model_name]

	n_classes = config.nr_class_dict[args.dataset_name]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	model_path = os.path.join(config.DIR_NAME, "models", args.model_name, "models", 
		"ee_mobilenet_branches_%s_id_%s.pth"%(args.n_branches, model_id))

	inference_data_path = os.path.join(config.DIR_NAME, "models", args.model_name, "results", 
		"no_calib_exp_data_%s.csv"%(model_id))

	# Instantiate LoadDataset class
	dataset = utils.LoadDataset(args, model_id)

	dataset_path = config.dataset_path_dict[args.dataset_name]
	idx_path = config.idx_path_dict[args.dataset_name]

	_, _, test_loader = dataset.getDataset(dataset_path, args.dataset_name, idx_path)

	#Instantiate the Early-exit DNN model.
	ee_model = Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.input_dim, 
		args.exit_type, device, args.distribution)
	#Load the trained early-exit DNN model.
	ee_model = ee_model.to(device)
	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

	# Obtain the confidences and predictions running an early-exit DNN inference. It returns as a Dataframe
	df_preds = utils.eval_ee_dnn_inference(test_loader, ee_model, args.n_branches, device, inference_data_path, args.read_inf_data)

	# Obtain the average inference time to process up to each side branch.
	inference_time_branch = utils.collect_avg_inference_time_branch(ee_model, test_loader, args.n_branches, args.threshold, device)

	#theta_opt_acc, loss_opt_acc = utils.read_temp(config.filePath_acc)
	#theta_opt_inf_time, loss_opt_inf_time = utils.read_temp(config.filePath_inf_time)

	beta_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

	for beta in beta_list:
		print("Start Joint Optimization")
		theta_opt_joint, loss_opt_joint = spsa.run_multi_obj_analysis(df_preds, inference_time_branch, args.threshold, args.max_iter, args.n_branches, args.a0, args.c, args.alpha, args.gamma, beta)


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

	# This argument defines the input dimension.
	parser.add_argument('--input_dim', type=int, default=config.input_dim, 
		help='Input Dimension. Default: %s'%(config.input_dim))

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

	args = parser.parse_args()

	main(args)
