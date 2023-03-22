from itertools import tee, islice
import random, sys, torch, os, logging, copy
import numpy as np
import config
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, gaussian_kde

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
		a = 0.1*((A+1)**self.alpha)/magnitude_g0

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
		return a/((k+A)**(self.alpha))

	def compute_ck(self, c, k):
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
		
		return best_theta, best_loss 

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


def joint_function(temp_list, n_branches, max_exits, threshold, df, loss_acc, loss_time):

	acc_current, _ = accuracy_edge(temp_list, n_branches, threshold, df)
	inf_time_current, _ = compute_inference_time(temp_list, n_branches, max_exits, threshold, df)

	f1 = (acc_current - loss_acc)/loss_acc
	f2 = (inf_time_current - loss_time)/loss_time	

	return f1+f2, _


def theoretical_beta_function(temp_list, n_branches, max_exits, threshold, df, df_device, loss_acc, loss_time, beta, overhead):

	acc_current, ee_prob = theoretical_accuracy_edge(temp_list, n_branches, threshold, df)

	inf_time_current, _ = compute_inference_time(temp_list, n_branches, max_exits, threshold, df, df_device, overhead)

	f = beta*acc_current + (1-beta)*inf_time_current 

	return f, ee_prob


def beta_function(temp_list, n_branches, max_exits, threshold, df, df_device, loss_acc, loss_time, beta, overhead):

	acc_current, ee_prob = accuracy_edge(temp_list, n_branches, threshold, df)
	inf_time_current, _ = compute_inference_time(temp_list, n_branches, max_exits, threshold, df, df_device, overhead)

	f = beta*acc_current + (1-beta)*inf_time_current 

	return f, ee_prob


def compute_inference_time(temp_list, n_branches, max_exits, threshold, df, df_device, overhead):

	avg_inference_time = 0
	n_samples = len(df)
	#remaining_data = df

	# somatorio P[fl-1 < threshold, fl > threshold]* time_l
	numexits = np.zeros(n_branches)

	confs = df["conf_branch_1"]

	calib_confs = confs/temp_list[0]
	early_exit_samples = calib_confs >= threshold

	n_exit_edge = df[early_exit_samples]["conf_branch_1"].count()
	n_exit_cloud = n_samples - n_exit_edge
	
	inf_time_branch_device = df_device["inferente_time_branch_1"].mean()

	inf_time_branch_cloud = df["inferente_time_branch_2"].mean() - df["inferente_time_branch_1"].mean()

	avg_inference_time += n_samples*inf_time_branch_device

	avg_inference_time += n_exit_cloud*(overhead+inf_time_branch_cloud)

	avg_inference_time = avg_inference_time/float(n_samples)
	early_classification_prob = n_exit_edge/float(n_samples)

	return avg_inference_time, early_classification_prob


	#for i in range(n_branches):

	#	current_n_samples = len(remaining_data)

		#if (i == n_branches):
		#	early_exit_samples = np.ones(current_n_samples, dtype=bool)
		#else:
	#	confs = remaining_data["conf_branch_%s"%(i+1)]
	#	calib_confs = confs/temp_list[i]
	#	early_exit_samples = calib_confs >= threshold

	#	numexits[i] = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()

	#	inf_time_branch = df_device["inferente_time_branch_%s"%(i+1)].mean()

		#print(inf_time_branch)

	#	avg_inference_time += numexits[i]*inf_time_branch

	#	remaining_data = remaining_data[~early_exit_samples]


	#inf_time_backbone = df["inferente_time_branch_%s"%(max_exits)].mean()
	#print(inf_time_backbone)
	#avg_inference_time += len(remaining_data)*(inf_time_backbone+overhead)
	#sys.exit()

	#avg_inference_time = avg_inference_time/float(n_samples)

	#early_classification_prob = sum(numexits)/n_samples
	#print(avg_inference_time, early_classification_prob)

	#return avg_inference_time, early_classification_prob

def theoretical_accuracy_edge(temp_list, n_branches, threshold, df):

	n_samples = len(df)
	num = 0

	for i in range(n_branches):
		
		#prob_success = compute_prob_success_branch(temp_list, i, threshold, df)
		
		#if (isinstance(prob_success, str)):
		#	return "error", "error"
		#else:
		#	num += prob_success	

		num += compute_prob_success_branch(temp_list, i, threshold, df)
	
	den = compute_theoretical_edge_prob(temp_list, n_branches, threshold, df)

	acc = num/den if (den>0) else 0

	#print("Acc: %s"%(acc))

	return - acc, den

def compute_prob_success_branch(temp_list, idx_branch, threshold, df):

	n_samples = len(df)

	if(idx_branch == 0):
		confs = df["conf_branch_%s"%(idx_branch+1)].values

	else:
		confs = df[df["conf_branch_%s"%(idx_branch)]/temp_list[idx_branch-1] < threshold]["conf_branch_%s"%(idx_branch+1)].values
	
	temp_list[idx_branch] = temp_list[idx_branch]+0.0001 if (temp_list[idx_branch] == 0) else temp_list[idx_branch]
	data_conf = confs/temp_list[idx_branch] 
	#data_conf = np.float64(data_conf)
	#print(data_conf)
	#data_conf = data_conf[data_conf < 1E308]
	data_conf = data_conf[:, np.newaxis]
	#print(data_conf)

	conf_d = np.linspace(threshold, 1, 100)
	conf_col = conf_d[:, np.newaxis]

	model = KernelDensity(kernel='gaussian', bandwidth=0.1)
	model.fit(data_conf)
	log_dens = model.score_samples(conf_col)

	pdf_values = np.exp(log_dens)
	#print(pdf_values.shape)

	#kde = gaussian_kde(data_conf)

	#conf_d = np.linspace(threshold, 1, 100)

	#pdf_values = kde.evaluate(conf_d)

	#print(pdf_values.shape)

	#expected_correct, pdf_values = compute_P_l(df, pdf_values, conf_d, idx_branch, temp_list)
	expected_correct, pdf_values = compute_reliability_diagram(df, pdf_values, conf_d, idx_branch, temp_list)

	product = expected_correct*pdf_values

	#Integrate
	prob_success_branch = np.sum([(conf_d[i+1] - conf_d[i])*product[i] for i in range(len(product) - 1) ])

	return prob_success_branch


def compute_reliability_diagram(df, pdf, confs, idx_branch, temp_list, delta_step=0.01, n_bins=15):

	bin_boundaries = np.linspace(0, 1, n_bins)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	conf_list, acc_list = [], [] 
	
	correct = df["correct_branch_%s"%(idx_branch+1)].values
	confs = df["conf_branch_%s"%(idx_branch+1)].values

	bin_size = 1/n_bins
	positions = np.arange(0+bin_size/2, 1+bin_size/2, bin_size)

	for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
		in_bin = np.where((confs > bin_lower) & (confs <= bin_upper), True, False)
		prop_in_bin = np.mean(in_bin)
		confs_in_bin, correct_in_bin = confs[in_bin], correct[in_bin] 
		avg_confs_in_bin = sum(confs_in_bin)/len(confs_in_bin) if (len(confs_in_bin)>0) else 0
		avg_acc_in_bin = sum(correct_in_bin)/len(correct_in_bin) if (len(confs_in_bin)>0) else 0
		#avg_acc_in_bin += delta
		conf_list.append(avg_confs_in_bin), acc_list.append(avg_acc_in_bin)

	return np.array(acc_list), np.array(conf_list)



def compute_P_l(df, pdf, confs, idx_branch, temp_list, delta_step=0.01):

	expected_correct_list, pdf_list = [], []

	for i, conf in enumerate(confs):
		#data = df[(df["conf_branch_%s"%(idx_branch+1)]/temp_list[idx_branch]  > conf) & (df["conf_branch_%s"%(idx_branch+1)]/temp_list[idx_branch] < conf+delta_step)]
		data = df[(df["conf_branch_%s"%(idx_branch+1)] > conf) & (df["conf_branch_%s"%(idx_branch+1)] < conf+delta_step)]

		correct = data["correct_branch_%s"%(idx_branch+1)].sum()

		n_samples = len(data["correct_branch_%s"%(idx_branch+1)].values)

		expected_correct = correct/n_samples if (n_samples>0) else 0
		expected_correct = data["conf_branch_%s"%(idx_branch+1)].mean()

		if (expected_correct is not np.nan):
			expected_correct_list.append(expected_correct), pdf_list.append(pdf[i])

	return np.array(expected_correct_list), np.array(pdf_list)


def compute_theoretical_edge_prob(temp_list, n_branches, threshold, df):

	n_samples = len(df)

	confs = df["conf_branch_%s"%(n_branches)]
	calib_confs = confs/temp_list[n_branches-1]
	early_exit_samples = calib_confs >= threshold

	numexits = df[early_exit_samples]["conf_branch_%s"%(n_branches)].count()

	prob = numexits/n_samples

	return prob

def accuracy_edge(temp_list, n_branches, threshold, df):

	"""
	This function computes the accuracy on the edge
	return avg_inf_time

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

		confs = remaining_data["conf_branch_%s"%(i+1)]
		calib_confs = confs/temp_list[i]
		early_exit_samples = calib_confs >= threshold

		numexits[i] = remaining_data[early_exit_samples]["conf_branch_%s"%(i+1)].count()
		correct_list[i] = remaining_data[early_exit_samples]["correct_branch_%s"%(i+1)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	acc_edge = sum(correct_list)/sum(numexits) if(sum(numexits) > 0) else 0
	early_classification_prob = sum(numexits)/n_samples

	return - acc_edge, early_classification_prob


def run_spsa(function, max_iter, dim, min_bounds, max_bounds, a0, c, alpha, gamma):
	
	theta_initial = np.ones(dim)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(function, max_iter, dim, a0, c, alpha, gamma,  min_bounds, max_bounds)
	# Run SPSA to minimize the objective function
	theta_opt, loss_opt, losses, n_iter = optim.min(theta_initial)

	return theta_opt, loss_opt

def run_SPSA_accuracy(df_inf_data, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, gamma, savePath):

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(accuracy_edge, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, args=(threshold, df_inf_data))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	save_temperature(savePath, theta_opt, loss_opt, threshold, n_branches_edge, max_branches, metric="acc")

	return theta_opt, loss_opt




def run_SPSA_inf_time(df_inf_data, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, gamma, savePath):

	max_exits = max_branches + 1

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(compute_inference_time, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, 
		args=(max_exits, threshold, df_inf_data))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	#theta_opt = 1./np.array(theta_opt)

	save_temperature(savePath, theta_opt, loss_opt, threshold, n_branches_edge, max_branches, metric="inf_time")

	return theta_opt, loss_opt


def run_multi_obj(df_inf_data, loss_acc, loss_time, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, gamma, savePath):

	max_exits = max_branches + 1

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(joint_function, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, 
		args=(max_exits, threshold, df_inf_data, loss_acc, loss_time))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	save_temperature(savePath, theta_opt, loss_opt, threshold, n_branches_edge, max_branches, metric="joint")

	return theta_opt, loss_opt


def run_beta_opt(df_inf_data, df_inf_data_device, beta, opt_acc, opt_inf_time, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, gamma, overhead):

	max_exits = max_branches + 1

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(beta_function, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, 
		args=(max_exits, threshold, df_inf_data, df_inf_data_device, opt_acc, opt_inf_time, beta, overhead))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	return theta_opt, loss_opt

def run_theoretical_beta_opt(df_inf_data, df_inf_data_device, beta, opt_acc, opt_inf_time, threshold, max_iter, n_branches_edge, max_branches, a0, c, alpha, gamma, overhead):

	max_exits = max_branches + 1

	theta_initial, min_bounds = np.ones(n_branches_edge), np.zeros(n_branches_edge)

	# Instantiate SPSA class to initializes the parameters
	optim = SPSA(theoretical_beta_function, theta_initial, max_iter, n_branches_edge, a0, c, alpha, gamma, min_bounds, 
		args=(max_exits, threshold, df_inf_data, df_inf_data_device, opt_acc, opt_inf_time, beta, overhead))

	# Run SPSA to minimize the objective function
	theta_opt, loss_opt = optim.min()

	return theta_opt, loss_opt

def save_temperature(savePath, theta_opt, loss_opt, threshold, n_branches, max_branches, metric):

	result_temp_dict = {"threshold": threshold, "opt_loss": loss_opt, "n_branches": n_branches, "metric": metric, "max_branches": max_branches}

	for i in range(max_branches):

		temp_branch = theta_opt[i] if (i < max_branches) else np.nan

		result_temp_dict["temp_branch_%s"%(i+1)] = temp_branch

	df_temp = pd.DataFrame([result_temp_dict])

	df_temp.to_csv(savePath, mode='a', header=not os.path.exists(savePath))