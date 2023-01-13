from itertools import tee, islice
import random, sys, torch, os, logging
import numpy as np
import config
import pandas as pd

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
		self.nr_iter = 100000
		self.n_branches = n_branches
		self.a0 = a0
		self.alpha = alpha
		self.gamma = gamma
		self.c = 1e-2 # a small number
		self.min_bounds = min_bounds
		self.args = args
		self.ens_size = ens_size
		self.function_tol = function_tol
		self.param_tol = param_tol

		# Defining the seed to have same results
		np.random.seed(seed)


	def init_hyperparameters(self):

		# A is <= 10% of the number of iterations
		A = self.nr_iter*0.1

		# order of magnitude of first gradients
		#magnitude_g0 = np.abs(self.grad(self.function, self.theta_initial, self.c).mean())
		magnitude_g0 = np.abs(self.estimate_grad(self.theta_initial, self.c))

		# the number 2 in the front is an estimative of
		# the initial changes of the parameters,
		# different changes might need other choices
		a = 2*((A+1)**self.alpha)/magnitude_g0

		return a, A, c

	def compute_loss(self, theta):
		return self.function(theta, self.n_branches, *(self.args) )

	def estimate_grad(self, theta, ck):

		grad_hat = 0.0

		for i in range(self.ens_size):

			# bernoulli-like distribution
			deltak = np.random.choice([-1, 1], size=self.n_branches+1) # TROCAR ATÉ O FINAL DO DIA.

			ck_deltak = ck * deltak

			#Stochastic perturbantions
			theta_plus = theta + ck_deltak
			theta_minus = theta - ck_deltak

			delta_y_pred = self.compute_loss(theta_plus) - self.compute_loss(theta_minus)

			grad_hat += (delta_y_pred)/(2*ck_deltak)

		avg_grad_hat = grad_hat/float(self.ens_size)

		return avg_grad_hat

	def compute_ak(self, a, A, k):
		return a/((k+A)**(self.alpha))

	def compute_ck(self, c, k):
		return c/(k**(self.gamma))


	def check_function_tolerante(self, theta, old_theta, k):
	
		j_old = self.compute_loss(old_theta)
		j_new = self.compute_loss(theta)

		j_delta = np.abs( j_new - j_old)


		return False if(j_delta > self.function_tol) else True


	def check_theta_tolerante(self, theta, old_theta, k):

		delta_theta = np.abs (theta - old_theta)

		return False if(delta_theta > self.param_tol) else True



	def check_violation_step(self, theta, old_theta, k):

		is_function_step_ok, is_theta_step_ok = True, True

		if (self.function_tol is not None):
			is_function_step_ok = check_function_tolerante(theta, old_theta, k)


		if (self.param_tol is not None):
			is_theta_step_ok = check_theta_tolerante(theta, old_theta, k)


		if(is_function_step_ok and is_theta_step_ok):
			return theta, k + 1
		else:
			return old_theta, k

	def min(self):

		theta = self.theta_initial

		a, A, c = self.init_hyperparameters()

		k = 1
		#for k in range(1, self.nr_iter):
		while (k <= self.nr_iter):

			old_theta = theta

			#Computes the parameters for each iteration
			ak = self.compute_ak(a, A, k)
			ck = self.compute_ck(c, k)

			#Estimate Gradient
			grad_hat = self.estimate_grad(theta, ck)

			# update parameters
			theta -= ak*grad_hat

			#Avoid for constraint violation
			theta = max(theta, self.min_bounds)

			
			theta, k = self.check_violation_step(theta, old_theta, k)


		y_final = self.compute_loss(theta)

		print("Parameter: %s, Function: %s"%(theta, y_final))

		return theta, y_final 


def measure_inference_time(temp_list, n_branches, threshold, test_loader, model, device):

	n_exits = n_branches + 1

	inf_time_list = []
	
	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):
			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			inf_time = model.measuring_inference_time(data, temp_list, threshold)
			inf_time_list.append(inf_time)

	# The next line computes the average inference time
	avg_inf_time = np.mean(inf_time_list)

	# Returns the average inference time
	return avg_inf_time


def joint_function(temp_list, n_branches, threshold, df, inf_time_branch, loss_acc, loss_time):

	acc_current = accuracy_edge(temp_list, n_branches, threshold, df)
	inf_time_current = compute_avg_inference_time(temp_list, n_branches, threshold, df, inf_time_branch)

	f1 = (acc_current - loss_acc)/loss_acc
	f2 = (inf_time_current - loss_time)/loss_time	

	return f1+f2


def compute_avg_inference_time(temp_list, n_branches, threshold, df, inf_time_branch):

	avg_inference_time = 0
	total_samples = 0
	n_samples = len(df)
	remaining_data = df

	#print("Total Samples: %s"%(n_samples))

	# somatorio P[fl-1 < threshold, fl > threshold]* time_l

	for i in range(n_branches+1):

		current_n_samples = len(remaining_data)

		if (i == config.max_exits-1):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			confs = remaining_data["conf_branch_%s"%(i+1)]
			calib_confs = confs/temp_list[i]
			early_exit_samples = calib_confs >= threshold

		#print(temp_list[i])
		#print(sum(calib_confs >= threshold), current_n_samples)
		#numexits = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		numexits = float(sum(early_exit_samples))
		#numexits = float(remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count())
		total_samples += numexits
		#print("Calib Confs: %s, Number of Exits: %s, Total Samples: %s"%(calib_confs.mean(), numexits, total_samples))

		avg_inference_time += numexits*inf_time_branch[i]

		#prob = numexits/current_n_samples if(current_n_samples > 0) else 0

		#avg_inference_time +=  prob*inf_time_branch[i]
		#print("Branch: %s, Prob: %s, Inf_time: %s, S_prob: %s"%(i+1, prob, prob*inf_time_branch[i], s_prob))
		remaining_data = remaining_data[~early_exit_samples]

	#print("Total TIMe: %s"%(avg_inference_time))
	#avg_inference_time = avg_inference_time/float(n_samples)
	#print("Avg Time: %s"%(avg_inference_time))

	return avg_inference_time


def accuracy_edge(temp_list, n_branches, threshold, df):

	"""
	This function computes the accuracy on the edge

	Inputs:
		temp_list:  temperature vector
		df:         this DataFrame contains the confidences, predictions and a boolean that indicates if the predictions is correct or not.
		threshold:  this threshold that decides whether the prediction is confidence to classify earlier on the 
					side branches at the edge device.
		n_branches: number of side branches that is placed at the edge device.

	Outputs:
		acc_edge: is the accuracy obtained by the side branches at the edge device.
	"""
	
	numexits, correct_list = np.zeros(n_branches), np.zeros(n_branches)
	n_samples = len(df)

	remaining_data = df

	for i in range(n_branches+1):
		current_n_samples = len(remaining_data)

		#if (i == config.max_exits):
		#	early_exit_samples = np.ones(current_n_samples, dtype=bool)
		#else:
		confs = remaining_data["conf_branch_%s"%(i+1)]
		calib_confs = confs/temp_list[i]
		early_exit_samples = calib_confs >= threshold

		numexits[i] = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		correct_list[i] = remaining_data[early_exit_samples]["correct_branch_%s"%(i+1)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	acc_edge = sum(correct_list)/sum(numexits) if(sum(numexits) > 0) else 0
	print("Neg Accuracy: %s" %( - acc_edge))

	sys.exit()
	return - acc_edge


def run_spsa(function, max_iter, dim, min_bounds, max_bounds, a0, c, alpha, gamma):
	
	theta_initial = np.ones(dim)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(function, max_iter, dim, a0, c, alpha, gamma,  min_bounds, max_bounds)
	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses, n_iter = optim.min(theta_initial)

	return theta_opt, loss_opt


def run_SPSA_accuracy(df_inf_data, threshold, max_iter, n_branches, a0, c, alpha, gamma):

	n_exits = n_branches + 1
	theta_initial = np.ones(n_exits)

	min_bounds = np.zeros(n_exits)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(accuracy_edge, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, args=(threshold, df_inf_data))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses = optim.min()

	#optim.save_temperature(config.filePath_acc, loss_opt, theta_opt, n_exits)

	return theta_opt, loss_opt

def run_SPSA_inf_time_old_version(model, test_loader, threshold, max_iter, n_branches, a0, c, alpha, gamma, device): 

	theta_initial = np.ones(n_branches+1)
	min_bounds = np.zeros(n_branches+1)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(measure_inference_time, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, 
		args=(threshold, test_loader, model, device))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses = optim.min()

	#optim.save_temperature(config.filePath_inf_time, loss_opt, theta_opt, n_exits)

	return theta_opt, loss_opt


def run_SPSA_inf_time(df_preds, avg_inf_time, threshold, max_iter, n_branches, a0, c, alpha, gamma):

	n_exits = n_branches + 1
	theta_initial = np.ones(n_exits)
	min_bounds = np.zeros(n_exits)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(compute_avg_inference_time, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, 
		args=(threshold, df_preds, avg_inf_time))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses = optim.min()

	#optim.save_temperature(config.filePath_inf_time, loss_opt, theta_opt, n_exits)

	return theta_opt, loss_opt

def run_multi_obj(df_preds, avg_inf_time, loss_acc, loss_time, threshold, max_iter, n_branches, a0, c, alpha, gamma, beta):

	n_exits = n_branches + 1
	theta_initial = np.ones(n_exits)
	min_bounds = np.zeros(n_exits)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(joint_function, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, 
		args=(threshold, df_preds, avg_inf_time, loss_acc, loss_time, beta))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses = optim.min()

	optim.save_temperature(config.filePath_joint_opt, loss_opt, theta_opt, n_exits)

	return theta_opt, loss_opt

def run_multi_obj_analysis(df_preds, avg_inf_time, threshold, max_iter, n_branches, a0, c, alpha, gamma, beta):

	n_exits = n_branches + 1
	theta_initial = np.ones(n_exits)
	min_bounds = np.zeros(n_exits)

	logging.basicConfig(level=logging.DEBUG, filename=config.logFile, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(joint_function_analysis, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, 
		args=(threshold, df_preds, avg_inf_time, beta))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, f_acc, f_inf_time = optim.min()

	optim.save_temperature_analysis(config.filePath_joint_opt, theta_opt, f_acc, f_inf_time, n_exits, beta)

	return theta_opt, loss_opt

def joint_function_analysis(temp_list, n_branches, threshold, df, inf_time_branch, beta):

	acc_current = beta*accuracy_edge(temp_list, n_branches, threshold, df)
	inf_time_current = (1-beta)*compute_avg_inference_time(temp_list, n_branches, threshold, df, inf_time_branch)
	joint_f = acc_current + inf_time_current

	return joint_f, acc_current, inf_time_current


