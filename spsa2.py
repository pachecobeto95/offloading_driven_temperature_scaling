from itertools import tee, islice
import random, sys, torch, os, logging, copy
import numpy as np
import config, math
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.special import softmax

class Bernoulli(object):
	'''
	Bernoulli Perturbation distributions.
	'''
	# This class generates a bernoulli vector

	def __init__(self, dim, r=1):
		#dim - provides the dimension of the bernoulli vector
		#r - he values thar the bernoulli vector may assume. 

		self.dim = dim
		self.r = r

	def __call__(self):
		# When this method is called, it returns the Bernoulli vector that works as delta vector to estimate
		# the gradient.
		return np.array([random.choice((-self.r, self.r)) for _ in range(self.dim)])



class SPSA (object):

	def __init__(self, function, theta_initial, nr_iter, n_branches, a0, c, alpha, gamma,  min_bounds, args=(), 
		function_tol=None, param_tol=None, ens_size=5, seed=42):

		""" Simultaneous Perturbation Stochastic Approximation. (SPSA)"""

		# function - a objective function of theta that returns a scalar
		# a0 - a parameter to create ak
		# A - other parameter to create ak
		# c - a parameter to create ck
		# delta - a function of no parameters which creates the delta vector
		# min_bounds -  A vector with minimum bounds for parameters theta
		# ens_size - Number of computations to approximate the gradient
		# function_tol - this parameter specifies a threshold the function can shifts after theta modifications.
		# param_tol - this parameter specifies a threshold the parameters can shifts after an update.
		# epsilon - threshold to detect convergence

		self.function = function
		self.theta_initial = theta_initial
		self.nr_iter = nr_iter
		self.n_branches = n_branches
		self.a0 = a0
		self.alpha = alpha
		self.gamma = gamma
		self.c = 0.1 # a small number
		self.min_bounds = min_bounds
		self.args = args
		self.ens_size = ens_size
		self.function_tol = 0.1
		self.param_tol = 0.01

		# Defining the seed to have same results
		np.random.seed(seed)


	def init_hyperparameters(self):

		# A is <= 10% of the number of iterations
		A = self.nr_iter*0.1

		# order of magnitude of first gradients
		#magnitude_g0 = np.abs(self.grad(self.function, self.theta_initial, self.c).mean())
		
		grad, _, _ = self.estimate_grad(self.theta_initial, self.c)
		magnitude_g0 = np.abs(grad.mean())

		# the number 2 in the front is an estimative of
		# the initial changes of the parameters,
		# different changes might need other choices
		a =  0.1*((A+1)**self.alpha)/magnitude_g0

		return a, A, self.c

	def compute_loss(self, theta):
		return self.function(theta, self.n_branches, *(self.args) )

	def estimate_grad(self, theta, ck):

		grad_hat = 0.0

		y_list, theta_list = [], []

		for i in range(self.ens_size):

			# bernoulli-like distribution
			deltak = np.random.choice([-1, 1], size=self.n_branches)
			ck_deltak = ck * deltak

			#Stochastic perturbantions
			theta_plus = theta + ck_deltak
			theta_minus = theta - ck_deltak

			theta_minus = np.maximum(theta_minus, self.min_bounds)

			y_plus, _ = self.compute_loss(theta_plus) 
			y_minus, _ = self.compute_loss(theta_minus)

			theta_list.append(theta_plus), theta_list.append(theta_minus)
			y_list.append(y_plus), y_list.append(y_minus) 

			delta_y_pred = y_plus - y_minus

			grad_hat += (delta_y_pred)/(2*ck_deltak)

		avg_grad_hat = grad_hat/float(self.ens_size)

		idx_min = np.argmin(y_list)
		theta, y = theta_list[idx_min], y_list[idx_min]

		return avg_grad_hat, theta, y

	def compute_ak(self, a, A, k):
		#return 0.1*a/((k+A)**(self.alpha))
		return a/((k+A)**(self.alpha))

	def compute_ck(self, c, k):
		#return 0.1*c/(k**(self.gamma))
		return c/(k**(self.gamma))


	def check_function_tolerance(self, theta, old_theta, k):
	
		j_old, _ = self.compute_loss(old_theta)
		j_new, _ = self.compute_loss(theta)

		j_delta = np.abs(j_new - j_old)

		return False if(j_delta > self.function_tol) else True


	def check_theta_tolerance(self, theta, old_theta, k):

		delta_theta = np.abs (theta - old_theta)

		return False if((np.any(delta_theta) > self.param_tol).any() ) else True



	def check_violation_step(self, theta, old_theta, k):

		is_function_step_ok, is_theta_step_ok = True, True

		if (self.function_tol is not None):
			is_function_step_ok = self.check_function_tolerance(theta, old_theta, k)


		if (self.param_tol is not None):
			is_theta_step_ok = self.check_theta_tolerance(theta, old_theta, k)


		#print(is_function_step_ok, is_theta_step_ok)
		if(is_function_step_ok and is_theta_step_ok):
			return theta, k + 1
		else:
			return old_theta, k

	def min(self):

		theta = copy.copy(self.theta_initial)
		best_theta = copy.copy(theta)

		a, A, c = self.init_hyperparameters()

		k = 1
		max_patience = 50
		best_loss, best_ee_prob = self.compute_loss(theta)
		patience = 0
		#while (k <= self.nr_iter):
		while (patience < max_patience):
			print(patience)

			old_theta = copy.copy(theta)

			#Computes the parameters for each iteration
			ak = self.compute_ak(a, A, k)
			ck = self.compute_ck(c, k)

			#Estimate Gradient
			grad_hat, theta_t, y_t = self.estimate_grad(theta, ck)

			# update parameters
			theta -= ak*grad_hat			

			#theta, k = self.check_violation_step(theta, old_theta, k)	
			theta = np.maximum(theta, self.min_bounds)

			y_k, ee_prob = self.compute_loss(theta)

			y_alt_list, theta_alt_list = [y_t, y_k], [theta_t, theta]

			idx_k = np.argmin(y_alt_list)
			y_k, theta = y_alt_list[idx_k], theta_alt_list[idx_k]

			if (y_k < best_loss):
				best_loss = y_k
				best_theta = copy.copy(theta)
				best_ee_prob = ee_prob
				patience = 0
			else:
				patience += 1

			k += 1
			#print("Iter: %s, Parameter: %s, Function: %s, EE Prob: %s"%(k, best_theta, best_loss, best_ee_prob))
		#sys.exit()
		return best_theta, best_loss 


def theoretical_beta_function(temp_list, n_branches, max_exits, threshold, df, df_device, beta, overhead, mode):

	if(mode == "exp"):
		acc_current, ee_prob = accuracy_edge(temp_list, n_branches, threshold, df)
	else:
		acc_current, ee_prob = theoretical_accuracy_edge(temp_list, n_branches, threshold, df)

	if(n_branches == 1):
		inf_time_current, _ = compute_inference_time(temp_list, n_branches, max_exits, threshold, df, df_device, overhead)
	else:
		inf_time_current, _ = compute_inference_time_multi_branches(temp_list, n_branches, max_exits, threshold, df, df_device, overhead)

	f = inf_time_current - beta*acc_current

	print(accuracy_edge(temp_list, n_branches, threshold, df), acc_current, temp_list)
	sys.exit()

	return f, ee_prob

def theoretical_accuracy_edge(temp_list, n_branches, threshold, df):
	num = 0
	acc_edge, early_classification_prob = accuracy_edge(temp_list, n_branches, threshold, df)

	numexits, correct_list = np.zeros(n_branches), np.zeros(n_branches)
	acc_device = np.zeros(n_branches)
	theo_acc_device = np.zeros(n_branches)
	n_samples = len(df)

	remaining_data = df

	for i in range(n_branches):
		logit_branch = getLogitBranches(remaining_data, i)
		conf_branch, _ = get_confidences(logit_branch, i, temp_list)

		early_exit_samples = conf_branch >= threshold
		df_branch = remaining_data[early_exit_samples]
		
		numexits[i] = df_branch["conf_branch_%s"%(i+1)].count()
		correct_list[i] = df_branch["correct_branch_%s"%(i+1)].sum()

		acc_device[i] = correct_list[i]/numexits[i]
		theo_acc_device[i] = estimate_expectation(df_branch, i, threshold, temp_list) 
		
		print(acc_device[i], theo_acc_device[i])

		remaining_data = remaining_data[~early_exit_samples]

	prob = numexits/sum(numexits)
	acc_dev = sum(acc_device*prob)

	print(acc_dev, acc_edge)
	sys.exit()
		
	return acc_edge, early_classification_prob


def estimate_expectation(df_branch, idx_branch, threshold, temp_list, n_bins=100):
	bin_boundaries = np.linspace(threshold, 1, n_bins)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	#acc_list, prop_in_bin_list = [], []
	acc_list = []
	
	logit_branch = getLogitBranches(df_branch, idx_branch)
	conf_branch, _ = get_confidences(logit_branch, idx_branch, temp_list)
	conf_branch_pdf = conf_branch[:, np.newaxis]

	pdf, _ = np.histogram(conf_branch, bins=1000, density=True)

	#model = KernelDensity(kernel='gaussian', bandwidth=0.1)
	#model.fit(conf_branch_pdf)
	#log_dens = model.score_samples(conf_branch_pdf)

	#pdf_kde = np.exp(log_dens)

	print(pdf)
	#sys.exit()

	correct = df_branch["correct_branch_%s"%(idx_branch+1)].values

	for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
		in_bin = np.where((conf_branch > bin_lower) & (conf_branch <= bin_upper), True, False)
		prop_in_bin = np.mean(in_bin)
		confs_in_bin, correct_in_bin = conf_branch[in_bin], correct[in_bin] 
		avg_confs_in_bin = np.mean(confs_in_bin) if (len(confs_in_bin)>0) else 0
		avg_acc_in_bin = np.mean(correct_in_bin) if (len(correct_in_bin)>0) else 0
		acc_list.append(avg_confs_in_bin)#, prop_in_bin_list.append(prop_in_bin)
		#print(avg_confs_in_bin, prop_in_bin)
	
	#print(sum(np.array(avg_confs_in_bin)*np.array(prop_in_bin_list)))
	print(acc_list)
	sys.exit()


def compute_prob_success_branch(temp_list, idx_branch, threshold, df, n_bins=100):
	d_confs = np.linspace(threshold, 1.0, n_bins)

	pdf_values = compute_pdf_values(temp_list, idx_branch, threshold, df)
	expectations, pdf_values = compute_expectation(temp_list, idx_branch, threshold, df, pdf_values)

	product = expectations*pdf_values
	result = np.sum([(d_confs[i+1] - d_confs[i])*product[i] for i in range(len(product) - 1) ])
	return result



def accuracy_edge(temp_list, n_branches, threshold, df):
	
	numexits, correct_list = np.zeros(n_branches), np.zeros(n_branches)
	n_samples = len(df)

	remaining_data = df

	for i in range(n_branches):
		current_n_samples = len(remaining_data)

		confs = remaining_data["conf_branch_%s"%(i+1)]
		calib_confs = confs/temp_list[i]
		early_exit_samples = calib_confs >= threshold

		numexits[i] = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		correct_list[i] = remaining_data[early_exit_samples]["correct_branch_%s"%(i+1)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	acc_edge = sum(correct_list)/sum(numexits) if(sum(numexits) > 0) else 0
	early_classification_prob = sum(numexits)/n_samples

	return acc_edge, early_classification_prob


def compute_inference_time_multi_branches(temp_list, n_branches, max_exits, threshold, df, df_device, overhead):
	
	avg_inference_time = 0
	n_samples = len(df)
	n_exits_device_list = []
	n_remaining_samples = n_samples
	remaining_data = df

	for i in range(n_branches):

		logit_branch = getLogitBranches(remaining_data, i)

		conf_list, infered_class_list = get_confidences(logit_branch, i, temp_list)

		early_exit_samples = conf_list >= threshold
		
		n_exit_branch = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		n_exits_device_list.append(n_exit_branch)

		inf_time_branch_device = df_device["inferente_time_branch_%s"%(i+1)].mean()

		avg_inference_time += n_exit_branch*inf_time_branch_device

		n_remaining_samples -= n_exit_branch
		inf_time_previous_branch = inf_time_branch_device

		remaining_data = remaining_data[~early_exit_samples]


	inf_time_branch_cloud = df["inferente_time_branch_%s"%(n_branches+1)].mean()-df["inferente_time_branch_%s"%(n_branches)].mean()

	avg_inference_time += n_remaining_samples*(df_device["inferente_time_branch_%s"%(n_branches)].mean()+overhead+inf_time_branch_cloud)

	avg_inference_time = avg_inference_time/float(n_samples)
	early_classification_prob = sum(n_exits_device_list)/float(n_samples)

	return avg_inference_time, early_classification_prob

def getLogitBranches(df, idx_branch):
	n_classes = 257
	logit_data = np.zeros((len(df), n_classes))

	for j in range(n_classes):
		logit_data[:, j] = df["logit_branch_%s_class_%s"%(idx_branch+1, j+1)].values
	return logit_data

def getLogitPreviousBranches(df, idx_branch):
	n_classes = 257
	logit_data = np.zeros((len(df), n_classes))

	for j in range(n_classes):
		logit_data[:, j] = df["logit_branch_%s_class_%s"%(idx_branch, j+1)].values
	return logit_data


def get_confidences(logit_branch, idx_branch, temp_list):
	n_rows, n_classes = logit_branch.shape
	softmax = nn.Softmax(dim=1)
	conf_list, infered_class_list = [], []

	for n_row in range(n_rows):
		calib_logit_branch = logit_branch[n_row, :]/temp_list[idx_branch]

		tensor_logit_branch = torch.from_numpy(calib_logit_branch)
		tensor_logit_branch = torch.reshape(tensor_logit_branch, (1, n_classes))
		
		softmax_data = softmax(tensor_logit_branch)
		conf, infered_class = torch.max(softmax_data, 1)
		conf_list.append(conf.item()), infered_class_list.append(infered_class.item())

	return np.array(conf_list), np.array(infered_class_list)

def get_previous_confidences(logit_branch, idx_branch, temp_list):
	n_rows, n_classes = logit_branch.shape
	softmax = nn.Softmax(dim=1)
	conf_list, infered_class_list = [], []

	for n_row in range(n_rows):
		calib_logit_branch = logit_branch[n_row, :]/temp_list[idx_branch-1]

		tensor_logit_branch = torch.from_numpy(calib_logit_branch)
		tensor_logit_branch = torch.reshape(tensor_logit_branch, (1, n_classes))
		
		softmax_data = softmax(tensor_logit_branch)
		conf, infered_class = torch.max(softmax_data, 1)
		conf_list.append(conf.item()), infered_class_list.append(infered_class.item())

	return np.array(conf_list), np.array(infered_class_list)


def run_theoretical_beta_opt(df_inf_data, df_inf_data_device, beta, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, 
	gamma, overhead, mode, epsilon=0.00001):

	max_exits = max_branches + 1

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)+epsilon

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(theoretical_beta_function, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, 
		args=(max_exits, threshold, df_inf_data, df_inf_data_device, beta, overhead, mode))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	return theta_opt, loss_opt