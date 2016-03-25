from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad,grad
from scipy.optimize import minimize
import autograd.scipy.stats.multivariate_normal as mvn
from sklearn.cross_validation import train_test_split
from autograd.scipy.misc import logsumexp
from autograd.util import quick_grad_check
import time
import pandas as pd

from deep_gaussian_process_6 import rbf_covariance, pack_gp_params, pack_layer_params, pack_deep_params, build_deep_gp, initialize,build_step_function_dataset

def test_log_likelihood(all_params, X, y, n_samples):
    rs = npr.RandomState(0)
    samples = [sample_mean_cov_from_deep_gp(all_params, X, True, rs, FITC = True) for i in xrange(n_samples)]
    return logsumexp(np.array([mvn.logpdf(y,mean,var) for mean,var in samples]))

def test_squared_error(all_params, X, y, n_samples):
    rs = npr.RandomState(0)
    samples = np.array([sample_mean_cov_from_deep_gp(all_params, X, True, rs, FITC = True)[0] for i in xrange(n_samples)])
    return np.mean((y - np.mean(samples,axis = 0)) ** 2)

def callback(params):
    print("Log likelihood {}, MSE {}".format(-objective(params),squared_error(params,X,y,n_samples)))

def plot_single_gp(ax, params, layer, unit, plot_xs, n_samples_to_plot):
    ax.cla()
    rs = npr.RandomState(0)

    deep_map = create_deep_map(params)
    gp_details = deep_map[layer][unit]
    gp_params = pack_gp_params(gp_details)

    pred_mean, pred_cov = predict_layer_funcs[layer][unit](gp_params, plot_xs, with_noise = False, FITC = False)
    x0 = deep_map[layer][unit]['x0']
    y0 = deep_map[layer][unit]['y0']
    noise_scale = deep_map[layer][unit]['noise_scale']

    # Show samples from posterior.
    sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov*(random), size=n_samples_to_plot)
    ax.plot(plot_xs, sampled_funcs.T)
    ax.plot(plot_xs,pred_mean,'r--')
    ax.plot(x0, y0, 'ro')
    ax.set_xlim([-5,5])
    ax.set_xticks([])
    ax.set_yticks([])

def plot_deep_gp(ax, params, plot_xs, n_samples_to_plot):
    ax.cla()
    rs = npr.RandomState(0)
    
    sampled_means_and_covs = [sample_mean_cov_from_deep_gp(params, plot_xs, rs = rs, with_noise = False, FITC = False) for i in xrange(50)]
    sampled_means, sampled_covs = zip(*sampled_means_and_covs)
    avg_pred_mean = np.mean(sampled_means, axis = 0)
    avg_pred_cov = np.mean(sampled_covs, axis = 0)

    sampled_means_and_covs_2 = [sample_mean_cov_from_deep_gp(params, plot_xs, rs = rs, with_noise = False, FITC = False) for i in xrange(n_samples_to_plot)]
    sampled_funcs = np.array([rs.multivariate_normal(mean, cov*(random)) for mean,cov in sampled_means_and_covs_2])
    ax.plot(plot_xs,sampled_funcs.T)
    ax.plot(X, y, 'kx')
    ax.plot(plot_xs,avg_pred_mean,'r--')
    #ax.set_ylim([-1.5,1.5])
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_title("Full Deep GP, inputs to outputs")


if __name__ == '__main__':
    random = 1 
    
    n_samples = 10 
    n_samples_to_test = 100
    num_pseudo_params = 50 

    dimensions =[1,1,1]
    n_layers = len(dimensions)-1 

    npr.seed(0) #Randomness comes from KMeans
    rs = npr.RandomState(0)

    motor = np.genfromtxt('motor.csv', delimiter=',',skip_header = True)
    X = motor[:,1]
    X = (X - np.mean(X))/(np.std(X))
    X = X.reshape(len(X),1)

    y = motor[:,2]
    y = (y-np.mean(y))/(np.std(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

    total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, create_deep_map = \
            build_deep_gp(dimensions, rbf_covariance, num_pseudo_params, random)

    init_params = .1 * npr.randn(total_num_params)
    deep_map = create_deep_map(init_params)

    init_params = initialize(deep_map,X,num_pseudo_params)
    print("Optimizing covariance parameters...")
    objective = lambda params: -log_likelihood(params,X,y,n_samples)

    #print("Initial value",value_and_grad(objective)(init_params))
    #print("Quick grad check", quick_grad_check(objective,init_params))

    params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='BFGS', callback=callback,options={'maxiter':500})

    test_log_lik_prior = log_likelihood(params['x'],X_test,y_test,n_samples_to_test)
    test_log_lik = test_log_likelihood(params['x'],X_test,y_test,n_samples_to_test)
    test_error = squared_error(params['x'],X_test,y_test,n_samples_to_test)
    print("Test Log Likelihood {}, Test Log Likelihood with Prior {}, Test MSE {}".format(\
        test_log_lik,test_log_lik_prior,test_error))

    params = params['x']
    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))

    fig = plt.figure(figsize=(16,8), facecolor='white')
    ax_full = fig.add_subplot(311, frameon=False)
    ax_one = fig.add_subplot(312, frameon=False)
    ax_two = fig.add_subplot(313, frameon=False)

    n_samples_to_plot = 5

    plot_deep_gp(ax_full, params, plot_xs, n_samples_to_plot)
    plot_single_gp(ax_one,params,0,0,plot_xs,n_samples_to_plot)
    plot_single_gp(ax_two,params,1,0,plot_xs,n_samples_to_plot)
    ax_full.set_title('Full deep GP', fontsize = 20)
    ax_one.set_title("Input to Hiddens", fontsize = 20)
    ax_two.set_title("Hiddens to Outputs", fontsize = 20)

    plt.savefig('motorcycle.pdf', format='pdf', bbox_inches='tight',dpi=200)

    plt.pause(40.0)



