from itertools import tee, islice
import random, sys, torch, os, logging, copy
import numpy as np
import config
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt

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
		best_loss, best_ee_prob = self.compute_loss(theta)

		while (k <= self.nr_iter):

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

	#f = (1-beta)*inf_time_current - beta*acc_current
	f = inf_time_current - beta*acc_current

	return f, ee_prob


def run_theoretical_beta_opt(df_inf_data, df_inf_data_device, beta, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, 
	gamma, overhead, mode, epsilon=0.00001):

	max_exits = max_branches + 1

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)+epsilon

	print("oi")
	sys.exit()

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(theoretical_beta_function, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, 
		args=(max_exits, threshold, df_inf_data, df_inf_data_device, beta, overhead, mode))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	return theta_opt, loss_opt
