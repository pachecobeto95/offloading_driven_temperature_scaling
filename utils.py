from torchvision import datasets, transforms
import torch, os, sys, requests
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

		print(self.input_dim)

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

		return train_loader, val_loader, test_loader



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
	correct_columns_list = ["correct_branch_%s"%(i)  for i in range(1, n_exits+1)]
	result_dict = {}
	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):
			#print(i)
			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			confs, predictions = model.evaluating_prediction(data)

			correct_list.append([predictions[i].eq(target.view_as(predictions[i])).sum().item() for i in range(n_exits)])

			conf_list.append(confs)


	conf_list, correct_list = np.array(conf_list), np.array(correct_list)

	#print([sum( correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)])
	print(correct_list[:, -1])

	for i in range(n_exits):
		result_dict["conf_branch_%s"%(i+1)] = conf_list[:, i]
		result_dict["correct_branch_%s"%(i+1)] = correct_list[:, i]


	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df

def collect_avg_inference_time_branch(model, test_loader, n_branches, threshold, device):

	n_exits = n_branches + 1

	inf_time_list, cumulative_inf_time_list = [], []

	avg_inf_time_dict, avg_cumulative_inf_time_dict = {}, {}
	
	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			inf_time, cumulative_inf_time = model.run_measuring_inference_time_branch(data)
			inf_time_list.append(inf_time), cumulative_inf_time_list.append(cumulative_inf_time)

	# The next line computes the average inference time
	avg_inf_time = np.mean(inf_time_list, axis=0)
	avg_cumulative_inf_time = np.mean(cumulative_inf_time_list, axis=0)

	#avg_inf_time, avg_cumulative_inf_time = np.array(avg_inf_time).T, np.array(avg_cumulative_inf_time).T

	
	for i in range(n_exits):
		avg_inf_time_dict["inf_time_branch_%s"%(i+1)] = [avg_inf_time[i]]
		avg_cumulative_inf_time_dict["cumulative_inf_time_branch_%s"%(i+1)] = [avg_cumulative_inf_time[i]]		

	df_inf_time, df_cumulative_inf_time = pd.DataFrame(avg_inf_time_dict), pd.DataFrame(avg_cumulative_inf_time_dict)

	df_inf_time_branches = pd.concat([df_inf_time, df_cumulative_inf_time], axis=1)

	# Returns confidences and predictions into a DataFrame.
	return df_inf_time_branches


def sendData(url, data):
	try:
		r = requests.post(url, json=data, timeout=config.timeout)
		r.raise_for_status()
	except requests.HTTPError as http_err:
		logging.warning("HTTPError")
		pass
	except requests.Timeout:
		logging.warning("Timeout")
		pass
	#except ConnectTimeout as timeout_err:
	#	print("Timeout error: ", timeout_err)



def sendImage(img_path, idx, url, target, p_tar, nr_branch_edge):

	data_dict = {"p_tar": p_tar, "nr_branch": int(nr_branch_edge), "target": target.item(), "id": idx}

	files = [
	('img', (img_path, open(img_path, 'rb'), 'application/octet')),
	('data', ('data', json.dumps(data_dict), 'application/json')),]

	try:
		r = requests.post(url, files=files, timeout=config.timeout)
		r.raise_for_status()
	
	except HTTPError as http_err:
		logging.warning("HTTPError")
		pass

	except ConnectTimeout as timeout_err:
		logging.warning("Timeout")
		pass


def sendModelConf(url, n_branches, dataset_name, model_name, location):
		
	#The next row mounts a dictionary to configure the model's parameters. 
	data_dict = {"n_branches": n_branches, "model_name": model_name, "location": location}
	sendData(url, data_dict)


def read_temp(filepath):

	df = pd.read_csv(filepath)
	temp_list = np.array([df["temp_branch_%s"%(i+1)][0] for i in range(config.n_exits)])
	loss = df["loss"][0]
	return temp_list, loss
