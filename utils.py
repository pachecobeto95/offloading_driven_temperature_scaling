from torchvision import datasets, transforms
import torch, os, sys
import numpy as np
import config
import pandas as pd

class LoadDataset():
	def __init__(self, args, model_id):

		#Th following lines sets the initial and global hyperparameters.
		self.input_dim = args.input_dim
		self.batch_size_train = args.batch_size_train
		self.batch_size_test = args.batch_size_test
		self.model_id = model_id
		self.split_ratio = args.split_ratio
		self.seed = args.seed

		# Note that we apply data augmentation in the training dataset.
		self.transformations_train = transforms.Compose([
			transforms.Resize(self.input_dim),
			transforms.RandomChoice([transforms.ColorJitter(brightness=config.brightness)]),
			transforms.RandomHorizontalFlip(p = config.h_flip_prob),
			transforms.RandomRotation(config.rotation_angle),
			transforms.ToTensor(), 
			transforms.Normalize(mean = config.mean, std = config.std),
			])

		# Note that we do not apply data augmentation in the test dataset.
		self.transformations_test = transforms.Compose([
			transforms.Resize(self.input_dim), 
			transforms.ToTensor(), 
			transforms.Normalize(mean = config.mean, std = config.std),
			])


	def caltech256(self, dataset_path, idx_path):

		# This method loads the Caltech-256 dataset.

		torch.manual_seed(self.seed)
		np.random.seed(seed=self.seed)

		# This block receives the dataset path and applies the transformation data. 
		train_set = datasets.ImageFolder(dataset_path, transform=self.transformations_train)

		val_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)
		test_set = datasets.ImageFolder(dataset_path, transform=self.transformations_test)

		train_idx_path = os.path.join(idx_path, "training_idx_caltech256_id_%s.npy"%(self.model_id))
		val_idx_path = os.path.join(idx_path, "validation_idx_caltech256_id_%s.npy"%(self.model_id))
		test_idx_path = os.path.join(idx_path, "test_idx_caltech256_id_%s.npy"%(self.model_id))


		if( os.path.exists(train_idx_path) ):
			#Load the indices to always use the same indices for training, validating and testing.
			train_idx = np.load(train_idx_path)
			val_idx = np.load(val_idx_path)
			test_idx = np.load(test_idx_path, allow_pickle=True)
			test_idx = np.array(list(test_idx.tolist()))

		else:
			# This line get the indices of the samples which belong to the training dataset and test dataset. 
			train_idx, val_idx, test_idx = self.get_indices(train_set, self.split_ratio)

			#Save the training, validation and testing indices.
			np.save(train_idx_path, train_idx)
			np.save(val_idx_path, val_idx)
			np.save(test_idx_path, test_idx)


		train_data = torch.utils.data.Subset(train_set, indices=train_idx)
		val_data = torch.utils.data.Subset(val_set, indices=val_idx)
		test_data = torch.utils.data.Subset(test_set, indices=test_idx)

		train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_data, batch_size=1)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

		return train_loader, val_loader, val_loader



	def getDataset(self, dataset_path, dataset_name, idx_path):
		# This method loads the image dataset provided in the argument dataset_name.

		def func_not_found():
			print("No dataset %s is found."%(dataset_name))

		#The following line loads the required image dataset.
		func_name = getattr(self, dataset_name, func_not_found)
		train_loader, val_loader, test_loader = func_name(dataset_path, idx_path)
		return train_loader, val_loader, test_loader

def eval_ee_dnn_inference(test_loader, model, n_branches, device, data_path, read_inf_data):

	# Checks if data_path exists
	if (os.path.exists(data_path) and read_inf_data):
		#If so, read the confidences and predictions in the file given by the data_path.
		predictions = get_confs_predictions(data_path, n_branches)

	else:
		#Otherwise, we run an early-exit dnn to gather confidences and predictions.
		predictions = run_ee_dnn_inference(test_loader, model, n_branches, device)

	return predictions

def get_confs_predictions(data_path, n_branches):
	
	# Reads .csv file and returns a DataFrame. 
	df = pd.read_csv(data_path)

	# Create a list containing the columns of the dataframe. 
	conf_columns_list = ["conf_branch_%s"%(i) for i in range(1, n_branches+1+1)]
	correct_columns_list = ["correct_branch_%s"%(i)  for i in range(1, n_branches+1+1)]
	
	data_column_list = conf_columns_list + correct_columns_list
	
	# Extract the required column
	df_data = df[data_column_list]

	# Returns confidences and predictions into a DataFrame.
	return df_data

def run_ee_dnn_inference(test_loader, model, n_branches, device):
	"""
	This function gathers the processing time to run up to each block layer.
	Then, this function repeats this procedure for other inputs on test sets.
	Finally, we compute the average processing time.

	Inputs:
	test_loader -> contains the DataLoader of the test set.
	model -> early-exit DNN model.
	device -> device CPU or GPU that will run the EE-DNN model.

	Outputs:
	avg_inference_time_dict -> dictionary that contains the average inference time computed previously
	"""

	n_exits = n_branches + 1
	conf_list, correct_list = [], []
	conf_columns_list = ["conf_branch_%s"%(i) for i in range(1, n_exits+1)]
	correct_columns_list = ["correct_branch_%s"%(i)  for i in range(1, n_branches+1+1)]
	
	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):
			print(i)
			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			confs, predictions = model.evaluating_prediction(data)
			
			correct_list.append([predictions[i].eq(target.view_as(predictions[i])).sum().item() for i in range(n_exits)])

			conf_list.append(confs)


	conf_list, correct_list = np.array(conf_list).T, np.array(correct_list).T

	conf_dict = dict(zip(conf_columns_list, conf_list))
	prediction_dict = dict(zip(correct_columns_list, prediction_list))

	df_confs, df_corrects = pd.DataFrame(conf_dict), pd.DataFrame(prediction_dict)

	df_data = pd.concat([a, b], axis=1)

	# Returns confidences and predictions into a DataFrame.
	return df_data

def collect_avg_inference_time_branch(model, test_loader, n_branches, threshold, device):

	n_exits = n_branches + 1

	inf_time_list = []
	
	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):
			print(i)

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			inf_time = model.run_measuring_inference_time_branch(data)
			inf_time_list.append(inf_time)

	# The next line computes the average inference time
	avg_inf_time = np.mean(inf_time_list, axis=0)

	# Returns the average inference time
	return avg_inf_time


"""
def evaluating_early_exit_dnn_inference(test_loader, model, n_branches, device):
	
	This function gathers the processing time to run up to each block layer.
	Then, this function repeats this procedure for other inputs on test sets.
	Finally, we compute the average processing time.

	Inputs:
	test_loader -> contains the DataLoader of the test set.
	model -> early-exit DNN model.
	device -> device CPU or GPU that will run the EE-DNN model.

	Outputs:
	avg_inference_time_dict -> dictionary that contains the average inference time computed previously
	
	inf_time_list, conf_list, prediction_list = [], [], []
	exit_list = ["exit_%s"%(i) for i in range(1, n_branches+1)]

	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			inference_time_dict = model.measuring_inference_time_block_wise(data)

			# The next line gathers the dictionary of the inference time for running the current input data.
			confs, predictions = model.evaluating_prediction(data)

			inf_time_list.append(list(inference_time_dict.values()))
			conf_list, prediction_list = np.array(conf_list).T, np.array(prediction_list).T
			break

	inf_time_list = np.array(inf_time_list)
	conf_list, prediction_list = np.array(conf_list).T, np.array(prediction_list).T

	# Compute the average of the inference time of each block layer.
	avg_inference_time = np.mean(inf_time_list, axis=0)


	block_layer_list = list(inference_time_dict.keys())

	avg_inference_time_dict = dict(zip(block_layer_list, avg_inference_time))

	conf_dict = dict(zip(exit_list, conf_list))
	prediction_dict = dict(zip(exit_list, prediction_list))

	return avg_inference_time_dict, conf_dict, prediction_dict	
"""	