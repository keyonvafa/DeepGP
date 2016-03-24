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
    n_samples_to_plot = 5 
    
    n_data = 20 
    input_dimension = 1
    num_pseudo_params = 10 
    X, y = build_step_function_dataset(D=input_dimension, n_data=n_data)

    initialization_set = [False, True] # Architecture of the GP. Last layer should always be 1 
    dimensions = [1,1,1]
    n_layers = len(dimensions)-1 

    total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, create_deep_map = \
            build_deep_gp(dimensions, rbf_covariance, num_pseudo_params, random)

    def plot_single_gp(ax, params, layer, unit, plot_xs):
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
        ax.plot(x0, y0, 'ro')
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_deep_gp(ax, params, plot_xs):
        ax.cla()
        rs = npr.RandomState(0)
        
        sampled_means_and_covs = [sample_mean_cov_from_deep_gp(params, plot_xs, rs = rs, with_noise = False, FITC = False) for i in xrange(n_samples_to_plot)]
        sampled_funcs = np.array([rs.multivariate_normal(mean, cov*(random)) for mean,cov in sampled_means_and_covs])
        ax.plot(plot_xs,sampled_funcs.T)
        ax.plot(X, y, 'kx')
        #ax.set_ylim([-1.5,1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title("Full Deep GP, inputs to outputs")

    def callback(params):
        print("Log likelihood {}, Squared Error {}".format(-objective(params),squared_error(params,X,y,n_samples)))
    
    rs = npr.RandomState(0)
    
    fig = plt.figure(figsize=(20,16), facecolor='white')
    ax_random_full = fig.add_subplot(321, frameon=False)
    ax_random_one = fig.add_subplot(323, frameon=False)
    ax_random_two = fig.add_subplot(325, frameon=False)
    ax_smart_full = fig.add_subplot(322, frameon=False)
    ax_smart_one = fig.add_subplot(324, frameon=False)
    ax_smart_two = fig.add_subplot(326, frameon=False)
    plt.show(block=False) 

    for initialization in initialization_set:

        init_params = .1 * rs.randn(total_num_params)
        deep_map = create_deep_map(init_params)
        if initialization:
            init_params = initialize(deep_map, X, num_pseudo_params)

        print("Optimizing covariance parameters...")
        objective = lambda params: -log_likelihood(params,X,y,n_samples)

        params = minimize(value_and_grad(objective), init_params, jac=True,
                              method='BFGS', callback=callback,options={'maxiter':1000})
        
        params = params['x']
        plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
        if initialization:
            ax_full = ax_smart_full
            ax_one = ax_smart_one
            ax_two = ax_smart_two
            title = "Two Layers, Smart Initialization"
        else:
            ax_full = ax_random_full
            ax_one = ax_random_one
            ax_two = ax_random_two  
            title = "Two Layers, Random Initialization"
        
        plot_deep_gp(ax_full, params, plot_xs)
        plot_single_gp(ax_one,params,0,0,plot_xs)
        plot_single_gp(ax_two,params,1,0,plot_xs)
        ax_full.set_title(title, fontsize = 20)
        ax_full.set_xlabel(r'$x$',fontsize = 20)
        ax_full.set_ylabel(r'$g(f(x))$',fontsize = 20)
        ax_one.set_title("Input to Hiddens", fontsize = 20)
        ax_one.set_xlabel(r'$x$',fontsize = 20)
        ax_one.set_ylabel(r'$f(x)$',fontsize = 20)
        ax_two.set_title("Hiddens to Outputs", fontsize = 20)
        ax_two.set_xlabel(r'$f(x)$',fontsize = 20)
        ax_two.set_ylabel(r'$g(f(x))$',fontsize = 20)
    
    plt.savefig('step_initialization.png', format='png', bbox_inches='tight',dpi=200)

    plt.pause(80.0)
