import os, time, sys, json, os, argparse, torch, config, utils
from early_exit_dnn import Early_Exit_DNN
import numpy as np
import pandas as pd
import temperature_scaling


def save_calibration_results(results, calib_results_path):
	df = pd.DataFrame([results])
	df.to_csv(saveTempPath, mode='a', header=not os.path.exists(calib_results_path))


def early_exit_inference(model, data_loader, p_tar, n_exits, device, id_run):
	df_result = pd.DataFrame()

	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list = [], []

	model.model.eval()

	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)

			conf_branches, infered_class_branches = model.forwardGlobalCalibration(data)      
			conf_branches_list.append(conf_branches), infered_class_branches_list.append(infered_class_branches)     
			target_list.append(target.item())      
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)
	id_run_list = correct_list.shape[0]*[id_run]

	results = {"target": target_list, "id": id_run_list}
	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results




def main(args):
	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	model_path = os.path.join(config.DIR_NAME, "models", args.model_name, "models", 
		"ee_mobilenet_branches_%s_id_%s.pth"%(args.n_branches, model_id))

	saveTempPath = os.path.join(resultsPath, "global_temperature_scaling.csv")
	calib_results_path = os.path.join(resultsPath, "global_inference_data.csv")

	# Instantiate LoadDataset class
	dataset = utils.LoadDataset(args, model_id)

	dataset_path = config.dataset_path_dict[args.dataset_name]
	idx_path = config.idx_path_dict[args.dataset_name]

	_, _, test_loader = dataset.getDataset(dataset_path, args.dataset_name, idx_path)

	#Instantiate the Early-exit DNN model.
	ee_model = Early_Exit_DNN(args.model_name, n_classes, args.pretrained, config.n_branches, config.input_dim, 
		config.exit_type, device, config.distribution)
	#Load the trained early-exit DNN model.
	ee_model = ee_model.to(device)
	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

	p_tar_list = [0.7, 0.75, 0.8, 0.85, 0.9]

	for p_tar in p_tar_list:
		global_ts = temperature_scaling.GlobalTemperatureScaling(ee_model, device, config.n_exits, saveTempPath, config.temp_init)
		global_ts.set_temperature(test_loader, p_tar)
		calibration_results = early_exit_inference(global_ts, test_loader, p_tar, config.n_exits, device, args.id)
		print("Success")
		sys.exit()	
		save_calibration_results(calibration_results, calib_results_path)





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

	parser.add_argument('--cuda', type=bool, default=config.cuda, 
		help='Cuda? Default: %s'%(config.cuda))

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--id', type=int)

	args = parser.parse_args()

	main(args)

