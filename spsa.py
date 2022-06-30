import numpy as np
from itertools import islice, tee, count
import random, sys


identity = lambda x: x

def SPSA(y, t0, a, c, delta, constraint=identity):

	'''
	Creates an Simultaneous Perturbation Stochastic Approximation (SPSA) iterator.
	f_loss - a function of theta that evaluates the parameter theta and returns a scalar
	t0 - initial value of the parameter theta
	a - an iterable of a_k values. This is an hyperparameter of the optimization method.
	c - an iterable of c_k values. This is an hyperparameter of the optimization method.
	delta - a function of no parameters which creates the delta vector
	constraint - a function of theta that returns theta
	'''

	theta = t0

	for ak, ck in zip(a, c):

		# Estimate gradient
		gk = estimate_gk(y, theta, delta, ck)

		# Adjust theta using SA
		theta = [t - ak * gkk for t, gkk in zip(theta, gk)]

		# Constrain
		theta = constraint(theta)

	yield theta # This makes this function become an iterator

def estimate_gk(f_loss, theta, delta, ck):
	'''
	This function estimates gradient for SPSA method based on two evaluations on loss functions.
	f_loss - a function of theta that evaluates the parameter theta and returns a scalar
	theta - this input contains the current parameter theta, i.e., theta(t).
	delta - a function of no parameters which creates the delta vector 
	ck - current parameter c. This is an hyperparameter of the optimization method.
	'''

	# Generate Delta vector
	delta_k = delta()

	# Get the two perturbed values of theta. list comprehensions like this are quite nice
	ta = [t + ck * dk for t, dk in zip(theta, delta_k)]
	tb = [t - ck * dk for t, dk in zip(theta, delta_k)]

	# Calculate g_k(theta_k)
	ya, yb = f_loss(ta), f_loss(tb)
	gk = [(ya-yb) / (2*ck*dk) for dk in delta_k]

	return gk


def standard_ak(a, A, n, alpha):
	'''
	Create a generator for values of a_k in the standard form.
	'''
	# count() is an infinite iterator as 0, 1, 2, ... 
	return ( a / (k + 1 + A) ** alpha for k in range(n))

def standard_ck(c, n, gamma):
	'''Create a generator for values of c_k in the standard form.'''
	return ( c / (k + 1) ** gamma for k in range(n) )

class Bernoulli:
	'''
	Bernoulli Perturbation distributions.
	p is the dimension
	+/- r are the alternate values
	'''
	def __init__(self, r=1, p=2):
		self.p = p
		self.r = r

	def __call__(self):
		return [random.choice((-self.r, self.r)) for _ in range(self.p)]

class LossFunction:
	''' A base class for loss functions which defines y as L+epsilon '''
	def y(self, theta):
		return self.L(theta) + self.epsilon(theta)


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

class SkewedQuarticLoss(LossFunction):
	'''
	Skewed Quartic Loss function.
	Initialize with vector length p.
	Functions, L, y, and epsilon available
	'''
	def __init__(self, p, sigma):
		x = 1./p
		self.B = [[x if i >= j else 0 for i in range(p)] for j in range(p)]
		self.sigmasq = sigma ** 2

	def L(self, theta):
		bt = [np.dot(Br, theta) for Br in self.B]
		return np.dot(bt,bt) + sum((.1 * b**3 + .01 * b**4 for b in bt))

	def epsilon(self, theta):
		return random.gauss(0, self.sigmasq) # multiply by stdev

def run_spsa(n=1000, replications=40):
	p = 2
	loss = SkewedQuarticLoss(p, sigma=1)
	theta0 = [1 for _ in range(p)]
	c = standard_ck(c=1, n=n, gamma=.101)
	a = standard_ak(a=1, n=n, A=100, alpha=.602)
	delta = Bernoulli(p=p)
  
	# tee is a useful function to split an iterator into n independent runs of that iterator
	ac = zip(tee(a,n),tee(c,n))
  
	losses = []
  
	for i, (a, c) in enumerate(islice(ac, replications)):
		print(i)
		theta_iter = SPSA(a=a, c=c, y=loss.y, t0=theta0, delta=delta)
		#print(list(theta_iter))
		terminal_theta = list(theta_iter)[i] # Get 1000th theta
		print(terminal_theta)
		terminal_loss = loss.L(terminal_theta)
		losses += [terminal_loss]
	
	return losses # You can calculate means/variances from this data.


run_spsa()















