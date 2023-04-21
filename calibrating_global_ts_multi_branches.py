import os, time, sys, json, os, argparse, torch, config, utils
import numpy as np
import pandas as pd
import temperature_scaling, ee_nn


def main(args):

	n_classes, input_dim, dim = 257, 330, 300

	temp_save_path = os.path.join(config.DIR_NAME, "alternative_temperature_%s_%s_branches_id_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	model_path = os.path.join(config.DIR_NAME, "new_models", "models", "ee_%s_%s_branches_id_%s.pth"%(config.model_name, args.n_branches, args.model_id))

	dataset_path = config.dataset_path_dict[args.dataset_name]
	
	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	model_dict = torch.load(model_path, map_location=device)

	val_idx, test_idx = model_dict["val"], model_dict["test"]

	ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.dim, device, args.exit_type, args.distribution)
	ee_model.load_state_dict(model_dict["model_state_dict"])
	ee_model = ee_model.to(device)
	ee_model.eval()

	#Load Dataset 
	val_loader = utils.load_caltech256_test_inference(args, dataset_path, val_idx)
	test_loader = utils.load_caltech256_test_inference(args, dataset_path, test_idx)

	data_loader = val_loader if (args.valid_indices) else test_loader

	threshold_list = [0.7, 0.8, 0.9]

	theta_initial = 1.5

	for threshold in threshold_list:

		# Instantiate SPSA class to initializes the parameters
		global_ts = temperature_scaling.GlobalTemperatureScaling(ee_model, device, theta_initial, args.max_iter, args.n_branches, threshold)

		global_ts.run(data_loader)

		temperature_overall = [global_ts.temperature_overall.item()]*args.n_branches

		result = {"threshold": threshold, "n_branches": args.n_branches, "calib_mode": "global_TS"}

		for i in range(args.n_branches):
			result["temp_branch_%s"%(i+1)] = temperature_overall[i]


		df = pd.DataFrame([result])
		df.to_csv(temp_save_path, mode='a', header=not os.path.exists(temp_save_path))




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

	parser.add_argument('--model_id', type=int, default=1)	

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim. Default: %s'%config.input_dim)

	parser.add_argument('--dim', type=int, default=300, help='Dim. Default: %s')

	parser.add_argument('--valid_indices', type=bool, default=True, help='Use Validation indices. default: True')


	args = parser.parse_args()

	main(args)

