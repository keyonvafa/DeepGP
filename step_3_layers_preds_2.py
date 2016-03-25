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
    print("Num Layers {}, Log likelihood {}, MSE {}".format(n_layers,-objective(params),squared_error(params,X,y,n_samples)))


if __name__ == '__main__':
    random = 1 
    
    n_samples = 10 
    n_samples_to_test = 100
    num_pseudo_params = 10 

    dimensions =[1,1,1,1]
    n_data = 60

    npr.seed(1) #Randomness comes from KMeans
    rs = npr.RandomState(1)

    X, y = build_step_function_dataset(D=1, n_data=n_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
        
    n_layers = len(dimensions)-1 

    total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, create_deep_map = \
            build_deep_gp(dimensions, rbf_covariance, num_pseudo_params, random)

    init_params = .1 * rs.randn(total_num_params)
    deep_map = create_deep_map(init_params)

    init_params = initialize(deep_map,X,num_pseudo_params)
    print("Optimizing covariance parameters...")
    objective = lambda params: -log_likelihood(params,X,y,n_samples)

    params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='BFGS', callback=callback,options={'maxiter':1000})

    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False) 

    n_samples_to_plot = 2000
    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
    sampled_means_and_covs = [sample_mean_cov_from_deep_gp(params['x'],plot_xs,with_noise=False,FITC=False) for i in xrange(n_samples_to_plot)]
    sampled_funcs = np.array([rs.multivariate_normal(mean, cov + 1e-6*np.eye(len(cov))) for mean,cov in sampled_means_and_covs])

    for sampled_func in sampled_funcs:
        ax.plot(plot_xs,np.array(sampled_func),linewidth=.01,c='black')

    ax.plot(X,y,'rx')
    ax.set_ylim([-2,2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('3-Layer Deep GP', fontsize = 20)
    ax.set_xlabel(r'$x$',fontsize = 20)
    ax.set_ylabel(r'$f(x)$',fontsize = 20)
    plt.savefig('step_3_layers_preds.png', format='png', bbox_inches='tight',dpi=200)



