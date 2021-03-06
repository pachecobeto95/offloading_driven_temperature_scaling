from itertools import tee, islice
import random, sys, torch
import numpy as np
import config

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

	def __init__(self, function, theta_initial, max_iter, dim, a0, c, alpha, gamma,  min_bounds, args=(), 
		max_patience=20, function_tol=None, param_tol=None, ens_size=2, epsilon=1e-4):

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
		#self.threshold = threshold
		self.max_iter = max_iter
		self.dim = dim
		self.a = a0
		self.A = max_iter/10
		self.alpha = alpha
		self.gamma = gamma
		self.c = c
		self.min_bounds = min_bounds
		self.args = args
		self.ens_size = ens_size
		self.function_tol = function_tol
		self.param_tol = param_tol
		self.epsilon = epsilon
		self.max_patience = max_patience


	def compute_ak(self, k):
		# This method returns the parameter ak for each iteration
		# k - is the current number of iterations
		return self.a/(k+1+self.A)**self.alpha

	def compute_ck(self, k):
		# This method computes the parameter ck for each iteration
		# k - is the current number of iterations
		return self.c/(k+1)**self.gamma

	def estimate_gradient(self, theta, ck, delta):

		# This method estimates the gradient based on two measuments of the objective function self.function.

		ghat = 0

		for i in range(1):
			
			# Compute the Delta vector
			delta_k = delta()

			#Stochastic perturbantions
			theta_plus = theta + ck * delta_k
			theta_minus = theta - ck * delta_k
			
			#This block forces the theta parameters after stochastic perturbations to be inside the limits 
			#defined by max_bounds and min_bounds.
			#theta_plus = np.minimum(theta_plus, self.max_bounds)
			theta_minus = np.maximum(theta_minus, self.min_bounds)

			y_plus = self.compute_loss(theta_plus)
			y_minus = self.compute_loss(theta_minus)

			ghat += (y_plus-y_minus)/(2*ck*delta_k)

		
		ghat = ghat/float(self.ens_size)

		return ghat

	def check_boundaries(self, theta):
		""" This method checks whether the current theta is inside the boundaries
			In this case, we check only the min_bounds"""

		theta = np.maximum(theta, self.min_bounds)
		return theta

	def adjusting_to_bounds(self, theta, ghat, ak):

		not_all_pass = True
		this_ak = ( theta*0 + 1 )*ak
		theta_new = theta
		while (not_all_pass):
			#out_of_bounds = np.where ( np.logical_or (theta_new - this_ak*ghat > self.max_bounds, 
			#	theta_new - this_ak*ghat < self.min_bounds ) )[0]
			out_of_bounds = np.where (theta_new - this_ak*ghat < self.min_bounds)[0]
			theta_new = theta - this_ak*ghat
			if len ( out_of_bounds ) == 0:
				theta = theta - this_ak*ghat
				not_all_pass = False
			else:
				this_ak[out_of_bounds] = this_ak[out_of_bounds]/2.

		return theta

	def compute_distance_theta(self, theta_old, theta):
		return np.linalg.norm(theta_old-theta)/np.linalg.norm(theta_old)
	
	
	def check_function_tolerance(self, loss_new, loss_old, theta, theta_saved):

		reject_iter = False

		if (self.function_tol is not None):

			if (np.abs(loss_new-loss_old) > self.function_tol):
				theta = theta_saved
				reject_iter = True

			else:
				loss_old = loss_new

		return theta, loss_old, reject_iter


	def check_param_tolerance(self, loss_new, loss_old, theta, theta_saved):
		"""
		This method rejects an iteration if a theta update results in a shifts too much the objective function.
		This procedure aims to decrease slowly to avoid deconvergence.

		Outputs:
			theta: this is the current parameter
			reject_iter: this output is a bool that denotes if we must reject the iteration
		"""

		reject_iter = False

		if (self.param_tol is not None):
			delta_theta = theta_dif = np.abs ( theta - theta_saved )

			if (not np.all ( delta_theta < self.param_tol )):
				theta = theta_saved
				reject_iter = True

		return theta, loss_old, reject_iter

	def compute_loss(self, theta):
		return self.function(theta, self.dim, *(self.args) )

	def min(self, report_interval=100):

		n_iter, patience = 0, 0
		losses = []
		theta = self.theta_initial
		delta = Bernoulli(dim=self.dim)

		loss_old = self.compute_loss(theta)

		# The optimisation runs until the solution has converged, or the maximum number of itertions has been reached.
		#Convergence means that the theta is not significantly changes until max_patience times in a row.

		while ((patience < self.max_patience) and (n_iter < self.max_iter)):

			# Store theta at the start of the interation. We update theta later.
			theta_saved = theta

			#Obtains the ak, ck for the current iteration
			ak = self.compute_ak(n_iter)
			ck = self.compute_ck(n_iter)

			# Get estimated gradient
			ghat = self.estimate_gradient(theta, ck, delta)

			theta = self.adjusting_to_bounds(theta, ghat, ak)

			# The new loss value evaluating the objective function.
			loss = self.compute_loss(theta)

			# Saves the loss in a list to create a loss history
			losses += [loss]

			# Function tolerance: 
			# You can ignore theta values that result in large shifts in the function value.
			# This procedure aims to decrease slowly to avoid deconvergence.
			theta, loss_old, reject_iter = self.check_function_tolerance(loss, loss_old, theta, theta_saved)

			# Parameter tolerance: 
			# You can ignore iteration if a theta update results in a shifts too much the objective function.
			# This procedure aims to decrease slowly to avoid deconvergence.			
			theta, loss_old, reject_iter = self.check_param_tolerance(loss, loss_old, theta, theta_saved)

			patience = patience + 1 if(self.compute_distance_theta(theta_saved, theta) < self.epsilon) else 0

			if (not reject_iter): 
				n_iter += 1

			# Be friendly to the user, tell him/her how it's going on...
			if(n_iter%report_interval == 0):
				print("Iter: %s, Loss: %s, Best Theta: %s."%(n_iter, loss, theta))

		print("Iter: %s, Loss: %s, Best Theta: %s."%(n_iter, loss, theta))

		return theta, loss, losses, n_iter

	def minimize(self, theta_0, report_interval=100):

		# Initializes counter n_iter
		n_iter = 0
		theta = theta_0
		losses = []

		delta = Bernoulli(dim=self.dim)
		
		for n_iter in range(self.max_iter):

			#Obtains the ak, ck for the current iteration
			ak = self.compute_ak(n_iter)
			ck = self.compute_ck(n_iter)

			# Get estimated gradient
			ghat = self.estimate_gradient(theta, ck, delta)
			
			# Adjust theta using SA
			theta = theta - ak*ghat
			theta = self.check_boundaries(theta)

			#Computes the new loss using the current theta parameter.
			loss = self.function(theta)

			#Saves the history of losses values in a list.
			losses += [loss]
			
			#Report to user the current loss. 
			if(n_iter%report_interval == 0):
				print("Iter: %s, Loss: %s, Best Theta: %s."%(n_iter, loss, theta))



		return theta, loss, losses


def objective_function(x):
	return x[0]**2 + x[1]**2



def measure_inference_time(temp_list, n_branches, threshold, test_loader, model, device):

	n_exits = n_branches + 1

	inf_time_list = []
	
	model.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(test_loader, 1):
			print(i)
			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# The next line gathers the dictionary of the inference time for running the current input data.
			inf_time = model.measuring_inference_time(data, temp_list, threshold)
			inf_time_list.append(inf_time)

	# The next line computes the average inference time
	avg_inf_time = np.mean(inf_time_list)

	# Returns the average inference time
	return avg_inf_time

def compute_avg_inference_time(temp_list, n_branches, threshold, df, inf_time_branch):

	avg_inference_time = 0
	n_samples = len(df)
	remaining_data = df

	print(n_branches, temp_list)

	# somatorio P[fl-1 < threshold, fl > threshold]* time_l

	for i in range(n_branches+1):
		current_n_samples = len(remaining_data)

		if (i == config.max_exits):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			confs = remaining_data["conf_branch_%s"%(i+1)]
			calib_confs = confs/temp_list[i]
			early_exit_samples = calib_confs >= threshold

		print(sum(calib_confs >= threshold), current_n_samples)
		#numexits = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		numexits = float(sum(calib_confs >= threshold))

		prob = numexits/current_n_samples
		print(prob)

		avg_inference_time +=  prob*inf_time_branch[i]

		remaining_data = remaining_data[~early_exit_samples]
	sys.exit()
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

	for i in range(n_branches):
		current_n_samples = len(remaining_data)

		if (i == config.max_exits):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			confs = remaining_data["conf_branch_%s"%(i+1)]
			calib_confs = confs/temp_list[i]
			early_exit_samples = calib_confs >= threshold

		numexits[i] = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		correct_list[i] = remaining_data[early_exit_samples]["correct_branch_%s"%(i+1)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	acc_edge = sum(correct_list)/sum(numexits)

	return - acc_edge


def run_spsa(function, max_iter, dim, min_bounds, max_bounds, a0, c, alpha, gamma):
	
	theta_initial = np.ones(dim)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(function, max_iter, dim, a0, c, alpha, gamma,  min_bounds, max_bounds)
	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses, n_iter = optim.min(theta_initial)

	return theta_opt, loss_opt

def run_SPSA_accuracy(model, df_preds, threshold, max_iter, n_branches, a0, c, alpha, gamma):

	theta_initial = np.ones(n_branches)

	min_bounds = np.zeros(n_branches)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(accuracy_edge, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, args=(threshold, df_preds))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses, n_iter = optim.min()

	return theta_opt, loss_opt

def run_SPSA_inf_time(model, test_loader, threshold, max_iter, n_branches, a0, c, alpha, gamma, device): 

	theta_initial = np.ones(n_branches)
	min_bounds = np.zeros(n_branches)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(measure_inference_time, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, 
		args=(threshold, test_loader, model, device))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses, n_iter = optim.min()

	return theta_opt, loss_opt


def run_SPSA_inf_time2(df_preds, avg_inf_time, threshold, max_iter, n_branches, a0, c, alpha, gamma):

	theta_initial = np.ones(n_branches)
	min_bounds = np.zeros(n_branches)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(compute_avg_inference_time, theta_initial, max_iter, n_branches, a0, c, alpha, gamma, min_bounds, 
		args=(threshold, df_preds, avg_inf_time))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses, n_iter = optim.min()

	return theta_opt, loss_opt



















