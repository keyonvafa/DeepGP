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
    print("Num Layers {}, Log likelihood {}, Squared Error {}".format(n_layers,-objective(params),squared_error(params,X,y,n_samples)))


if __name__ == '__main__':
    random = 1 
    
    n_samples = 10 
    n_samples_to_test = 100
    n_data = 10 #150 #80 Worked!
    num_pseudo_params = 10 

    dimension_set = [[1,1],[1,1,1]] # Architecture of the GP. Last layer should always be 1 

    npr.seed(0)
    rs = npr.RandomState(0)

    X, y = build_step_function_dataset(D=1, n_data=n_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for dimensions in dimension_set:
        n_layers = len(dimensions)-1 
        start_time = time.time()

        total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, create_deep_map = \
            build_deep_gp(dimensions, rbf_covariance, num_pseudo_params, random)

        npr.seed(0)
        init_params = .1 * npr.randn(total_num_params)
        deep_map = create_deep_map(init_params)

        init_params = initialize(deep_map,X,num_pseudo_params)
        print("Optimizing covariance parameters...")
        objective = lambda params: -log_likelihood(params,X,y,n_samples)

        params = minimize(value_and_grad(objective), init_params, jac=True,
                              method='BFGS', callback=callback,options={'maxiter':500})
        duration = time.time()-start_time
        print("Test Log Likelihood {}, Test Squared Error {}, Duration {} seconds".format(test_log_likelihood(params['x'],X_test,y_test,\
            n_samples_to_test),squared_error(params['x'],X_test,y_test,n_samples_to_test),duration))
        