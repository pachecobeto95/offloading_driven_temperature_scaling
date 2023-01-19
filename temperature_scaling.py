import torchvision, torch
import os, sys, time, math, os
from torch.utils.data import random_split
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import product
import pandas as pd
from torch import Tensor
import functools
import torch.nn.functional as F
from tqdm import tqdm
#torch.multiprocessing.set_sharing_strategy('file_system')

class ECE(nn.Module):
	"""This method computes ECE metric to measure model's miscalibration"""

	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(ECE, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

	def forward(self, logits, labels):
		softmaxes = F.softmax(logits, dim=1)
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)

		ece = torch.zeros(1, device=logits.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if (prop_in_bin.item() > 0):
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
				#print(bin_lower, bin_upper, ece)
		return ece


class GlobalTemperatureScaling(nn.Module):
	"""This class implements Global Temperature Scaling for early-exit DNNs"""

	def __init__(self, model, device, temp_init, max_iter, n_branches_edge, threshold, lr=0.001):
		super(GlobalTemperatureScaling, self).__init__()
    
		self.model = model #the model to be calibrated
		self.device = device
		self.temperature_overall = nn.Parameter((temp_init*torch.ones(1)).to(self.device)) #initial temperature to be optimized
		self.lr = lr #learning rate
		self.max_iter = 50 #maximum iteration to the optimization method
		self.n_branches_edge = n_branches_edge
		self.threshold = threshold


	def temperature_scale(self, logits):
		temperature = self.temperature_overall.unsqueeze(1).expand(logits.size(0), logits.size(1))
		return logits / temperature


	def forwardCalibrationInference(self, x, threshold):
		return self.model.forwardGlobalCalibrationInference(x, threshold, self.temperature_overall)


	def forwardGlobalTS(self, x):
		return self.model.forwardGlobalCalibration(x, self.temperature_overall)


	def save_temperature(self, p_tar, before_temp_nll, after_temp_nll, before_temp_ece, after_temp_ece):
		# This function probably should live outside of this class, but whatever
		# This method sves the learned temperature parameters.


		result = {"p_tar": round(p_tar, 2), "before_nll": before_temp_nll, "after_nll": after_temp_nll, "before_ece": before_temp_ece, 
		"after_ece": after_temp_ece, "temperature": self.temperature_overall.item()}

		df = pd.DataFrame([result])
		df.to_csv(self.saveTempPath, mode='a', header=not os.path.exists(self.saveTempPath))

	def run(self, valid_loader):
		"""
		Tune the tempearature of the model (using the validation set).
		We're going to set it to optimize NLL.
		valid_loader (DataLoader): validation set loader
		p_tar: confidence threshold to decide wheter an input should be classified earlier or not.
		"""
        
		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece_criterion = ECE().to(self.device)

		# First: collect all the logits and labels for the validation set
		logits_list, labels_list = [], []

		self.model.eval()
		with torch.no_grad():
			#Run inference over samples from validation dataset
			for data, label in tqdm(valid_loader):
				data, label = data.to(self.device), label.to(self.device)  
				#Check the next row to confirm 
				logits, confs, _, _ = self.model.forwardInference(data, self.threshold)

				logits_list.append(logits), labels_list.append(label)

		logits = torch.cat(logits_list).to(self.device)
		labels = torch.cat(labels_list).to(self.device)

		# Calculate NLL and ECE before temperature scaling
		before_temperature_nll = nll_criterion(logits, labels).item()
		#before_temperature_ece = ece_criterion(logits, labels).item()
		#print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

		# Next: optimize the temperature w.r.t. NLL
		optimizer = optim.LBFGS([self.temperature_overall], lr=self.lr, max_iter=self.max_iter)

		def eval():
			optimizer.zero_grad()
			loss = nll_criterion(self.temperature_scale(logits), labels)
			loss.backward()
			return loss
		optimizer.step(eval)

		# Calculate NLL and ECE after temperature scaling
		after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
		#after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
		#print('Optimal temperature: %.3f' % self.temperature_overall.item())
		#print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

		#self.save_temperature(p_tar, before_temperature_nll, after_temperature_nll, before_temperature_ece, after_temperature_ece)
		
		#return self.temperature_overall
		return self


class PerBranchTemperatureScaling(nn.Module):

	def __init__(self, model, device, n_branches, temp_init, threshold, lr=0.01, max_iter=1000):
		super(PerBranchTemperatureScaling, self).__init__()

		self.model = model
		self.device = device
		self.n_exits = n_branches + 1
		self.temperature_branches = [nn.Parameter((temp_init*torch.ones(1)).to(self.device)) for i in range(self.n_exits)]
		self.lr = lr
		self.max_iter = max_iter
		self.temp_init = temp_init
		self.threshold = threshold

	def forwardCalibrationInference(self, x, threshold):
		return self.model.forwardPerBranchCalibrationInference(x, threshold, self.temperature_branches)


	def forwardBranchesCalibration(self, x):
		return self.model.forwardBranchesCalibration(x, self.temperature_branches)

	def temperature_scale_branches(self, logits):
		temperature = self.temperature_branch.unsqueeze(1).expand(logits.size(0), logits.size(1))
		return logits / temperature

	def save_temperature(self, p_tar, before_temp_nll_list, before_ece_list, after_temp_nll_list, after_ece_list):

		result = {"p_tar": p_tar}

		for i in range(self.n_exits):
			result.update({"before_nll_branch_%s"%(i+1): before_temp_nll_list[i], 
				"before_ece_branch_%s"%(i+1): before_ece_list[i],
				"after_nll_branch_%s"%(i+1): after_temp_nll_list[i],
				"after_ece_branch_%s"%(i+1): after_ece_list[i],
				"temperature_branch_%s"%(i+1): self.temperature_branches[i]})

		df = pd.DataFrame([result])
		df.to_csv(self.saveTempPath, mode='a', header=not os.path.exists(self.saveTempPath))

	def run(self, valid_loader):

		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece = ECE()

		logits_list = [[] for i in range(self.n_exits)]
		labels_list = [[] for i in range(self.n_exits)]
		#idx_sample_exit_list = [[] for i in range(self.n_exits)]
		before_temp_nll_list, after_temp_nll_list = [], []
		before_ece_list, after_ece_list = [], []

		self.model.eval()
		with torch.no_grad():
			for (data, target) in tqdm(valid_loader):
				data, target = data.to(self.device), target.to(self.device)
				logits, _, inf_class, exit_branch = self.model.forwardInference(data, self.threshold)

				logits_list[exit_branch].append(logits), labels_list[exit_branch].append(target)

		for i in range(self.n_exits):
			print("Exit: %s"%(i+1))

			if (len(logits_list[i]) == 0):
				before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
				before_ece_list.append(None), after_ece_list.append(None)
				continue

			self.temperature_branch = nn.Parameter((self.temp_init*torch.ones(1)*self.temp_init).to(self.device))
			optimizer = optim.LBFGS([self.temperature_branch], lr=self.lr, max_iter=self.max_iter)

			logit_branch = torch.cat(logits_list[i]).to(self.device)
			label_branch = torch.cat(labels_list[i]).to(self.device)

			before_temp_nll = nll_criterion(logit_branch, label_branch).item()
			before_temp_nll_list.append(before_temp_nll)

			before_ece = ece(logit_branch, label_branch).item()
			before_ece_list.append(before_ece)

			def eval():
				optimizer.zero_grad()
				loss = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch)
				loss.backward()
				return loss

			optimizer.step(eval)

			after_temp_nll = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch).item()
			after_temp_nll_list.append(after_temp_nll)

			after_ece = ece(self.temperature_scale_branches(logit_branch), label_branch).item()
			after_ece_list.append(after_ece)

			self.temperature_branches[i] = self.temperature_branch

			#print("Branch: %s, Before NLL: %s, After NLL: %s"%(i+1, before_temperature_nll, after_temperature_nll))
			#print("Branch: %s, Before ECE: %s, After ECE: %s"%(i+1, before_ece, after_ece))
			#print("Temp Branch %s: %s"%(i+1, self.temperature_branch.item()))

		self.temperature_branches = [temp_branch.item() for temp_branch in self.temperature_branches]


		# This saves the parameter to save the temperature parameter for each side branch
		#self.save_temperature(p_tar, before_temp_nll_list, before_ece_list, after_temp_nll_list, after_ece_list)
		return self


def run_global_TS_opt(model, valid_loader, threshold, max_iter, n_branches_edge, n_branches, device):


	#theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)
	theta_initial = 2.0

	# Instantiate SPSA class to initializes the parameters
	ts = GlobalTemperatureScaling(model, device, theta_initial, max_iter, n_branches_edge, threshold)

	ts.run(valid_loader)

	return ts.temperature_overall, ts


def run_per_branch_TS_opt(model, valid_loader, threshold, max_iter, n_branches_edge, n_branches, device):


	#theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)
	theta_initial = 2.0

	# Instantiate SPSA class to initializes the parameters
	ts = PerBranchTemperatureScaling(model, device, n_branches_edge, theta_initial, threshold)

	ts.run(valid_loader)

	return ts.temperature_branches, ts




def run_early_exit_inference(calib_model, valid_loader, ts_theta, n_branches_edge, threshold, device):

	conf_list, correct_list, is_early_exit_list, inference_time_list = [], [], [], []

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	calib_model.model.eval()

	with torch.no_grad():
		for data, target in valid_loader:

			data, target = data.to(device), target.to(device)

			starter.record()

			#MODIFICAR LINHA A SEGUIR. PENSAR SE CONF É CONF OU CONF.ITEM()
			_, conf, infered_class, is_early_exit = calib_model.forwardCalibrationInference(data, threshold)

			ender.record()
			torch.cuda.synchronize()
			inference_time = starter.elapsed_time(ender)

			conf_list.append(conf), correct_list.append(infered_class.eq(target.view_as(infered_class)).sum().item())
			is_early_exit_list.append(is_early_exit), inference_time_list.append(inference_time)

			del data, target
			torch.cuda.empty_cache()

	accuracy = float(sum(correct_list))/len(correct_list)
	avg_inference_time = float(sum(inference_time_list))/len(inference_time_list)
	ee_prob = float(sum(is_early_exit_list))/len(is_early_exit_list)

	return accuracy, avg_inference_time, ee_prob