from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad,grad
from scipy.optimize import minimize

from deep_gaussian_process_6 import rbf_covariance, pack_gp_params, pack_layer_params, pack_deep_params, build_deep_gp, initialize,build_step_function_dataset

if __name__ == '__main__':
    random = 1 
    n_samples = 10 
    n_samples_to_plot = 10 
    
    n_data = 20 
    input_dimension = 1
    num_pseudo_params = 10 
    X, y = build_step_function_dataset(D=input_dimension, n_data=n_data)

    dimension_set = [[1,1,1,1]] # Architecture of the GP. Last layer should always be 1 
    
    fig = plt.figure(figsize=(10,8), facecolor='white')
    ax_three_layer = fig.add_subplot(111, frameon=False)
    plt.show(block=False) 

    npr.seed(0)

    for dimensions in dimension_set:
        n_layers = len(dimensions)-1 

        total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, create_deep_map = \
            build_deep_gp(dimensions, rbf_covariance, num_pseudo_params, random)

        def callback(params):
            print("Log likelihood {}, Squared Error {}".format(-objective(params),squared_error(params,X,y,n_samples)))

        rs = npr.RandomState(0)
        init_params = .1 * rs.randn(total_num_params)
        deep_map = create_deep_map(init_params)
        init_params = initialize(deep_map, X, num_pseudo_params)

        print("Optimizing covariance parameters...")
        objective = lambda params: -log_likelihood(params,X,y,n_samples)

        params = minimize(value_and_grad(objective), init_params, jac=True,
                              method='BFGS', callback=callback,options={'maxiter':1000})
        
        n_samples_to_plot = 2000
        plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
        sampled_means_and_covs = [sample_mean_cov_from_deep_gp(params['x'],plot_xs,with_noise=False,FITC=False) for i in xrange(n_samples_to_plot)]
        sampled_funcs = np.array([rs.multivariate_normal(mean, cov*(random)) for mean,cov in sampled_means_and_covs])

        if dimensions == [1,1,1,1]:
            ax = ax_three_layer
            title = "3-Layer Deep GP"
        else:
            ax = ax_three_layer   
            title = "2-Layer Deep GP"
        for sampled_func in sampled_funcs:
            ax.plot(plot_xs,np.array(sampled_func),linewidth=.01,c='black')
        ax.plot(X,y,'rx')
        ax.set_ylim([-2,2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize = 20)
        ax.set_xlabel(r'$x$',fontsize = 20)
        ax.set_ylabel(r'$f(x)$',fontsize = 20)
    plt.savefig('step_3_layers_preds_2.png', format='png', bbox_inches='tight',dpi=200)

    plt.pause(80.0)
