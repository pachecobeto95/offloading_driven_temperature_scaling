import os, config, time, requests, sys, json, logging, torch, utils
import numpy as np
from PIL import Image
import pandas as pd
import argparse
#from utils import LoadDataset
from requests.exceptions import HTTPError, ConnectTimeout
from glob import glob
#from load_dataset import load_test_caltech_256
from torchvision.utils import save_image

def load_dataset(args, dataset_path, savePath_idx):
	return load_test_caltech_256(config.input_dim, dataset_path, args.split_ratio, savePath_idx, config.model_id_dict[args.model_name])




def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	p_tar_list = [0.8, 0.81, 0.82, 0.83, 0.84, 0.85]
	dataset_path = config.dataset_path_dict["caltech256"]

	logPath = os.path.join(config.DIR_NAME, "logTest_%s.log"%(args.model_name))

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	
	#This line defines the number of side branches processed at the edge
	nr_branch_edge = np.arange(3, config.n_branches+1)

	logging.debug("Sending Configuration")

	utils.sendModelConf(config.urlConfModelEdge, config.n_branches, args.dataset_name, args.model_name, args.location)
	#utils.sendModelConf(config.urlConfModelCloud, config.n_branches, args.dataset_name, args.model_name, args.location)
	
	logging.debug("Finish Configuration")

	#test_loader = load_dataset(args, dataset_path, save_indices_path)
	#inferenceTimeExperiment(test_loader, p_tar_list, nr_branch_edge, logPath)






if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance in terms of missed deadline probability.")

	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default=config.model_name, 
		choices=["mobilenet", "resnet18", "resnet152", "vgg16"], help='DNN model name (default: MobileNet)')

	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	parser.add_argument('--location', type=str, choices=["ohio", "sp"], help='Location of Cloud Server')

	args = parser.parse_args()


	main(args)
