import os, time, sys, json, os, argparse, torch
import config, utils, spsa, ee_nn, temperature_scaling
from early_exit_dnn import Early_Exit_DNN
import numpy as np
import pandas as pd

def calibrating_early_exit_dnns(model, data_loader, threshold, max_iter, n_branches, device):

	global_ts_model = temperature_scaling.GlobalTemperatureScaling(model, device, 1.5, max_iter, n_branches, 0.8)
	global_ts_model.run(data_loader)


	#per_branch_ts_model = temperature_scaling.PerBranchTemperatureScaling(model, device, 1.5, max_iter, n_branches, 0.8)
	#per_branch_ts_model.run(data_loader)

	
	return global_ts_model, 0



def extracting_global_ts_ee_inference_data(data_loader, calib_model, n_branches, device):

	temperature_overall = calib_model.temperature_overall

	temperature_overall_list = [temperature_overall for i in range(n_branches)]

	df = utils.extracting_ee_inference_data(data_loader, calib_model, temperature_overall_list, args.n_branches, device, mode="global_ts")

	return df	


def extracting_per_branch_ts_ee_inference_data(data_loader, calib_model, n_branches, device):

	temperature_branches = calib_model.temperature_branches

	df = utils.extracting_ee_inference_data(data_loader, calib_model, temperature_branches, args.n_branches, device, mode="per_branch_ts")

	return df

def extracting_ee_inference_data(data_loader, model, n_branches, device):

	no_calib_temperature = np.ones(n_branches)

	df = utils.extracting_ee_inference_data(data_loader, model, no_calib_temperature, args.n_branches, device, mode="no_calib")

	return df


def main(args):

	n_classes = 257

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	model_path = os.path.join(config.DIR_NAME, "new_models", "models", "ee_%s_%s_branches_id_%s.pth"%(args.model_name, args.n_branches, args.model_id) )	

	multi_branch_model_path = os.path.join(config.DIR_NAME, "new_models", "models", "ee_%s_%s_branches_id_%s.pth"%(args.model_name, args.n_branches, args.model_id) )	

	dataset_path = config.dataset_path_dict[args.dataset_name]

	#inf_data_path = os.path.join(config.DIR_NAME, "new_inference_data", "inference_data_%s_%s_branches_%s_local_server.csv"%(args.model_name, args.n_branches, args.model_id))

	indices_path = os.path.join(config.DIR_NAME, "indices", "caltech256", "validation_idx_caltech256_id_1.npy")

	model_dict = torch.load(model_path, map_location=device)
	multi_model_dict = torch.load(multi_branch_model_path, map_location=device)

	val_idx, test_idx = model_dict["val"], model_dict["test"]

	val_idx = np.load(indices_path)

	#Load Early-exit DNN model.	
	ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.dim, device, args.exit_type, args.distribution)
	ee_model.load_state_dict(multi_model_dict["model_state_dict"])
	ee_model = 	ee_model.to(device)
	ee_model.eval()

	#Load Dataset 
	val_loader = utils.load_caltech256_test_inference(args, dataset_path, val_idx)

	global_ts_model, _ = calibrating_early_exit_dnns(ee_model, val_loader, 0.8, args.max_iter, args.n_branches, device)
	print(global_ts_model.temperature_overall)

	#df_global_calib = extracting_global_ts_ee_inference_data(val_loader, global_ts_model, n_branches, device)

	#df_per_branch_ts_inference_data = extracting_per_branch_ts_ee_inference_data(test_loader, per_branch_ts_model, args.n_branches, device)


	#df_no_calib = extracting_ee_inference_data(val_loader, ee_model, args.n_branches, device)

	# Obtain the confidences and predictions running an early-exit DNN inference. It returns as a Dataframe
	#df_inference_data = utils.extracting_ee_inference_data(test_loader, ee_model, args.n_branches, device)

	#df_inference_data.to_csv(inf_data_path)

	#df_global_calib.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))
	#df_per_branch_ts_inference_data.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))
	#df_no_calib.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))



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

	parser.add_argument('--step', type=float, default=config.step, 
		help="Step of beta. Default: %s"%(config.step))	

	parser.add_argument('--model_id', type=int, default=1)	

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim. Default: %s'%config.input_dim)

	parser.add_argument('--dim', type=int, default=300, help='Dim. Default: %s')


	args = parser.parse_args()

	main(args)






