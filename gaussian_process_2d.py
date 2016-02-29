from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad,grad
from scipy.optimize import minimize
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from builtins import range
from autograd.scipy.misc import logsumexp
import itertools
import sys
from optparse import OptionParser
from autograd.util import quick_grad_check

def build_checker_dataset(n_data = 6, noise_std =0.1):
	rs = npr.RandomState(0)
	inputs = np.array([np.array([x,y]) for x in np.linspace(-1,1,n_data) for y in np.linspace(-1,1,n_data)])
	targets = np.sign([np.prod(input) for input in inputs]) + rs.randn(n_data**2)*noise_std
	return inputs, targets

def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1) - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2)) 


def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_kernel_params(params):
        mean        = params[0]
        noise_scale = np.exp(params[1]) + 0.001
        cov_params  = params[2:]

        return mean, cov_params, noise_scale

    def predict(params, x0, y0, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x0, xstar)
       	cov_y_y = cov_func(cov_params, x0, x0) + noise_scale * np.eye(len(y0))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y0 - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def predict_with_noise(params, x0, y0, xstar):
        pred_mean, pred_cov = predict(params, x0, y0, xstar)
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        return pred_mean, pred_cov + noise_scale*np.eye(len(xstar)) 

    def log_marginal_likelihood(params, x, y):
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        return mvn.logpdf(y, prior_mean, cov_y_y)   

    return num_cov_params + 2, predict, predict_with_noise, unpack_kernel_params, log_marginal_likelihood

if __name__ == '__main__':

	input_dimension = 2
	num_params, predict, predict_with_noise, unpack_kernel_params, log_marginal_likelihood = make_gp_funs(rbf_covariance,input_dimension+1)

	X, y = build_checker_dataset(n_data=16)
	rs = npr.RandomState(0)
	objective = lambda params: -log_marginal_likelihood(params, X, y)

	# Set up figure.
	fig = plt.figure(figsize=(12,8), facecolor='white')
	ax = fig.add_subplot(111, frameon=False)
	plt.show(block=False)

	def callback(params):

	    print("Log likelihood {}".format(-objective(params)))
	    plt.cla()
	    pred_mean, pred_cov = predict(params, X, y, plot_xs)
	    pred_mean = pred_mean.reshape(40,40)
	    ax.contourf(np.linspace(-1,1,40),np.linspace(-1,1,40), pred_mean)
	    ax.scatter(X[:,0],X[:,1],c=y)
	    ax.set_xticks([])
	    ax.set_yticks([])
	    plt.draw()
	    plt.pause(1.0/60.0)

    # Initialize covariance parameters
	plot_xs = np.array([np.array([a,b]) for a in np.linspace(-1,1,40) for b in np.linspace(-1,1,40)])
	init_params = 0.1 * rs.randn(num_params)
	print("Optimizing covariance parameters...")
	cov_params = minimize(value_and_grad(objective), init_params, jac=True,
	                      method='CG', callback=callback)
	plt.pause(10.0)

