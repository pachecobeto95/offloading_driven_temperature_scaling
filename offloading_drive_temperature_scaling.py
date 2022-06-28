import os, time, sys, json, os, argparse, torch
import config, utils
from early_exit_dnn import Early_Exit_DNN


def main(args):

	model_id = config.models_id_dict[args.model_name]

	n_classes = config.nr_class_dict[args.dataset_name]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	# Instantiate LoadDataset class
	dataset = utils.LoadDataset(args, model_id)

	dataset_path = config.dataset_path_dict[args.dataset_name]
	idx_path = config.idx_path_dict[args.dataset_name]

	_, _, test_loader = dataset.getDataset(dataset_path, args.dataset_name, idx_path)

	#Instantiate the Early-exit DNN model.
	ee_model = Early_Exit_DNN(args.model_name, n_classes, args.pretrained, 5, args.input_dim, 
		args.exit_type, device, args.distribution)

	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):

			data, target = data.to(device), target.to(device)
			ee_model.measuring_inference_time_block_wise(data)

	#Load the train early-exit DNN model.
	print("Success")


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


	args = parser.parse_args()

	main(args)






