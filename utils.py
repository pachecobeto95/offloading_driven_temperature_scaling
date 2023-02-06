from torchvision import datasets, transforms
import torch, os, sys, requests, early_exit_dnn, b_mobilenet, ee_nn
import numpy as np
import config
import pandas as pd

def get_indices(dataset, split_ratio):
	
	nr_samples = len(dataset)
	indices = list(range(nr_samples))
	np.random.shuffle(indices)

	train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))


	train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]

	np.random.shuffle(train_val_idx)

	train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

	train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

	return train_val_idx, val_idx, test_idx


#def load_caltech256(args, dataset_path, save_indices_path, input_dim, dim):
def load_caltech256(args, dataset_path):

	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)

	transformations_train = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.RandomHorizontalFlip(p=0.25),
		transforms.RandomRotation(25),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)


	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	#train_idx_path = os.path.join(save_indices_path, "training_idx_caltech256_id_%s.npy"%(args.model_id))
	#val_idx_path = os.path.join(save_indices_path, "validation_idx_caltech256_id_%s.npy"%(args.model_id))
	#test_idx_path = os.path.join(save_indices_path, "test_idx_caltech256_id_%s.npy"%(args.model_id))
	#test_idx_path = os.path.join(save_indices_path, "test_idx_caltech256.npy")

	#train_idx = np.load(train_idx_path)
	#val_idx = np.load(val_idx_path)
	#test_idx = np.load(test_idx_path, allow_pickle=True)
	#test_idx = np.array(list(test_idx.tolist()))

	train_idx, val_idx, test_idx = get_indices(train_set, args.split_ratio)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

	return train_loader, val_loader, test_loader, {"train": train_idx, "val": val_idx, "test": test_idx}



#def load_caltech256(args, dataset_path, save_indices_path, input_dim, dim):
def load_caltech256_test_inference(args, dataset_path, test_idx):

	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)

	transformations_train = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.RandomHorizontalFlip(p=0.25),
		transforms.RandomRotation(25),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

	return test_loader



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


def extracting_ee_inference_data(test_loader, model, temp_list, n_branches, device, mode):
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
	conf_list, correct_list, inference_time_list = [], [], []
	result_dict = {}

	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# Obtain confs and predictions for each side branch.
			confs, predictions, inf_time_branches = model.forwardCalibration(data, temp_list)

			correct_list.append([predictions[i].eq(target.view_as(predictions[i])).sum().item() for i in range(n_exits)])

			conf_list.append(confs), inference_time_list.append(inf_time_branches)

	conf_list, correct_list, inference_time_list = np.array(conf_list), np.array(correct_list), np.array(inference_time_list)

	accuracy_branches = [sum( correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)]

	print("Accuracy: %s"%(accuracy_branches))

	#result_dict = {"calib_mode": mode*}

	for i in range(n_exits):
		result_dict["conf_branch_%s"%(i+1)] = conf_list[:, i]
		result_dict["correct_branch_%s"%(i+1)] = correct_list[:, i]
		result_dict["inferente_time_branch_%s"%(i+1)] = inference_time_list[:, i]


	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df


def extracting_ee_inference_time(model, test_loader, n_branches, threshold_list, device):

	df_inf_time = pd.DataFrame()

	for threshold in threshold_list:
		df_inf_time_branches = collect_avg_inference_time_branch(model, test_loader, n_branches, threshold, device)
		
		df_inf_time = pd.concat([df_inf_time, df_inf_time_branches], ignore_index=True)

	return df_inf_time



def collect_avg_inference_time_branch(model, test_loader, n_branches, threshold, device):

	n_exits = n_branches + 1

	inf_time_list = []
	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			starter.record()

			# The next line gathers the conf, prediction, early classification for running the current input data.
			_, _, _, _ = model.forwardInference(data, threshold)

			ender.record()
			torch.cuda.synchronize()
			curr_time = starter.elapsed_time(ender)

			inf_time_list.append(curr_time)

	result_dict = {"threshold": [threshold]*len(inf_time_list), "inference_time": inf_time_list}

	return pd.DataFrame(result_dict)


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


def load_ee_model(args, model_path, n_classes, dim, device):


	if (args.n_branches == 1):

		ee_model =  ee_nn.Early_Exit_DNN(args.model_name, 257, args.pretrained, args.n_branches, 300, device, "bnpool", "predefined")

	elif(args.n_branches == 3):
		ee_model = b_mobilenet.B_MobileNet(n_classes, args.pretrained, args.n_branches, dim, args.exit_type, device)

	elif(args.n_branches == 5):
		ee_model = early_exit_dnn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, dim, args.exit_type, device, 
			args.distribution)

	else:
		raise Exception("The number of early-exit branches is not available yet.")


	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

	ee_model = ee_model.to(device)
	ee_model.eval()

	return ee_model
