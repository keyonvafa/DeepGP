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

def build_step_function_dataset(D=1, n_data=40, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-2, 2, num=n_data)
    targets = np.sign(inputs) + rs.randn(n_data) * noise_std
    inputs  = inputs.reshape((len(inputs), D))
    return inputs, targets

def build_parabola(D=1, n_data=20, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-4, 4, num=n_data)
    targets = inputs ** 2 + rs.randn(n_data) * noise_std
    inputs  = inputs.reshape((len(inputs), D))
    return inputs, targets

def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
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

    return num_cov_params + 2, predict, predict_with_noise, unpack_kernel_params

def build_deep_gp(input_dimension, hidden_dimension, n_layers, covariance_function, num_pseudo_params, random): # make build deep gp and gp the same

    layer_details = [make_gp_funs(covariance_function, num_cov_params = input_dimension + 1) for x in xrange(n_layers)]
    num_params_each_layer, predict_layer_funcs, predict_funcs_with_noise, unpack_kernel_params = zip(*layer_details)
    unpack_kernel_params = unpack_kernel_params[0]

    # Psuedo params defined as X0, y0 (inducing points). We have 10 of these for each layer    
    total_num_params = sum(num_params_each_layer) + n_layers*2*num_pseudo_params

    def unpack_all_params(all_params):
        layer_params = np.array_split(all_params[:sum(num_params_each_layer)], n_layers)
        pseudo_params = all_params[sum(num_params_each_layer):]
        x0, y0 = np.array_split(pseudo_params, 2)
        x0 = np.array_split(x0, n_layers)
        y0 = np.array_split(y0, n_layers)
        return layer_params, x0, y0

    def pack_all_params(layer_params, x0, y0):
        all_params = np.ndarray.flatten(np.array(layer_params))
        all_params = np.concatenate([all_params, np.ndarray.flatten(np.array(x0))])
        all_params = np.concatenate([all_params, np.ndarray.flatten(np.array(y0))])
        return all_params

    def sample_from_mvn(mu, sigma): # make sure we return 2d, also make sure data is 2d
        rs = npr.RandomState(0)
        return np.dot(np.linalg.cholesky(sigma+1e-6*np.eye(len(sigma))*np.max(np.diag(sigma))),rs.randn(len(sigma)))+mu if random == 1 else mu

    def sample_mean_cov_from_deep_gp(all_params, X, with_noise = False):
        predict = predict_funcs_with_noise if with_noise else predict_layer_funcs
        X_star = X
        layer_params, x0, y0 = unpack_all_params(all_params)
        n_layers = len(x0)
        for layer in xrange(n_layers):
            layer_mean, layer_cov = predict[layer](layer_params[layer],np.atleast_2d(x0[layer]).T, y0[layer],X_star)
            X_star = np.atleast_2d(sample_from_mvn(layer_mean, layer_cov)).T
        return layer_mean,layer_cov

    def squared_error(all_params):
        n_samples = 10
        samples = np.array([sample_mean_cov_from_deep_gp(all_params, X, False)[0] for i in xrange(n_samples)])
        return np.mean((y - np.mean(samples,axis = 0)) ** 2)

    def evaluate_prior(all_params): # clean up code so we don't compute matrices twice
        layer_params, x0, y0 = unpack_all_params(all_params)
        log_prior = 0
        for layer in xrange(n_layers):
            #import pdb; pdb.set_trace()
            mean, cov_params, noise_scale = unpack_kernel_params(layer_params[layer])
            cov_y_y = covariance_function(cov_params, np.atleast_2d(x0[layer]).T, np.atleast_2d(x0[layer]).T) + noise_scale * np.eye(len(y0[layer]))
            log_prior += mvn.logpdf(y0[layer],np.ones(len(cov_y_y))*mean,cov_y_y+np.eye(len(cov_y_y))*1e-6*np.max(np.diag(cov_y_y)))
        return log_prior

    def log_likelihood(all_params):  # implement mini batches later?
        n_samples = 1
        layer_params, x0, y0 = unpack_all_params(all_params)
        samples = [sample_mean_cov_from_deep_gp(all_params, X, True) for i in xrange(n_samples)]
        return logsumexp(np.array([mvn.logpdf(y,mean,var+1e-6*np.eye(len(var))*np.max(np.diag(var))) for mean,var in samples])) - np.log(n_samples) \
            + evaluate_prior(all_params)

    return total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, unpack_all_params, predict_layer_funcs, squared_error, pack_all_params

if __name__ == '__main__':


    parser = OptionParser()
    parser.add_option("--n_layers",
                      dest="n_layers", default="3", type = "int",
                      help="Set the number of layers")
    parser.add_option("--random",
                      dest="random", default=0, type="int",
                      help="Set whether we are drawing random functions")
    parser.add_option("--smart_init",
                      dest="smart_init", default=1, type="int",
                      help="Set whether we are initializing intelligently")
    parser.add_option("--n_samples",
                      dest="n_samples", default=1, type="int",
                      help="Set number of samples")

    (options, args) = parser.parse_args()
    n_layers = options.n_layers
    random = options.random
    smart_init = options.smart_init
    n_samples = options.n_samples

    n_data = 20
    input_dimension = 1
    hidden_dimension = 1
    num_pseudo_params = 10
    X, y = build_step_function_dataset(D=input_dimension, n_data=20)
    #X, y = build_parabola(D = input_dimension, n_data = 20)

    total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, unpack_all_params, predict_layer_funcs, squared_error, pack_all_params = \
        build_deep_gp(input_dimension, hidden_dimension, n_layers, rbf_covariance, num_pseudo_params, random)

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax_first = fig.add_subplot(411, frameon=False)
    ax_end_to_end = fig.add_subplot(412, frameon=False)
    ax_x_to_h = fig.add_subplot(413, frameon=False)
    ax_h_to_y = fig.add_subplot(414, frameon=False)
    plt.show(block=False)


    def plot_full_gp(ax, params, plot_xs):
        ax.cla()
        rs = npr.RandomState(0)
        
        sampled_means_and_covs = [sample_mean_cov_from_deep_gp(params, plot_xs) for i in xrange(n_samples)]
        sampled_means, sampled_covs = zip(*sampled_means_and_covs)
        avg_pred_mean = np.mean(sampled_means, axis = 0)
        avg_pred_cov = np.mean(sampled_covs, axis = 0)
        marg_std = np.sqrt(np.diag(avg_pred_cov))
        if n_samples > 1:
            ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
            np.concatenate([avg_pred_mean - 1.96 * marg_std,
                           (avg_pred_mean + 1.96 * marg_std)[::-1]]),
                           alpha=.15, fc='Blue', ec='None')
            ax.plot(plot_xs, avg_pred_mean, 'b')

        sampled_funcs = np.array([rs.multivariate_normal(mean, cov*(random)) for mean,cov in sampled_means_and_covs])
        ax.plot(plot_xs,sampled_funcs.T)
        ax.plot(X, y, 'kx')
        ax.set_ylim([-1.5,1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Full GP, X to Y")


    def plot_gp(ax, X, y, pred_mean, pred_cov, plot_xs):
        ax.cla()
        marg_std = np.sqrt(np.diag(pred_cov))
        if n_samples > 1:
            ax.plot(plot_xs, pred_mean, 'b')
            ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
            np.concatenate([pred_mean - 1.96 * marg_std,
                           (pred_mean + 1.96 * marg_std)[::-1]]),
                           alpha=.15, fc='Blue', ec='None')

        # Show samples from posterior.
        rs = npr.RandomState(0)
        sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov*(random), size=n_samples)
        ax.plot(plot_xs, sampled_funcs.T)
        ax.plot(X, y, 'kx')
        #ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([])
        ax.set_yticks([])

    def callback(params):
        print("Log likelihood {}, Squared Error {}".format(-objective(params),squared_error(params)))
        layer_params, x0, y0 = unpack_all_params(params)

        # Show posterior marginals.
        plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
        plot_full_gp(ax_end_to_end, params, plot_xs)
        if n_layers == 1:
            ax_end_to_end.plot(x0[0],y0[0], 'ro')
        else:
            hidden_mean, hidden_cov = predict_layer_funcs[0](layer_params[0], np.atleast_2d(x0[0]).T, y0[0], plot_xs)
            plot_gp(ax_x_to_h, x0[0], y0[0], hidden_mean, hidden_cov, plot_xs)
            ax_x_to_h.set_title("X to hiddens, with inducing points")

            y_mean, y_cov = predict_layer_funcs[1](layer_params[1], np.atleast_2d(x0[1]).T, y0[1], plot_xs)
            plot_gp(ax_h_to_y, x0[1], y0[1], y_mean, y_cov, plot_xs)
            ax_h_to_y.set_title("hiddens to layer 2, with inducing points")

        plt.draw()
        plt.pause(1.0/60.0)

    def smart_initialize_params(init_params):
        layer_params, x0, y0 = unpack_all_params(init_params)
        # Initialize the first length scale parameter as the median distance between points
        pairs = itertools.combinations(X, 2)
        dists = np.array([np.linalg.norm(np.array([p1])- np.array([p2])) for p1,p2 in pairs])
        layer_params[0][2] = np.log(np.var(y))
        layer_params[0][3] = np.log(np.median(dists))

        # Initialize the pseudo inputs for the first layer by sampling from the data, the pseudo outputs equal to the inputs
        x0[0] = np.ndarray.flatten(np.array(X)[rs.choice(len(X), num_pseudo_params, replace=False),:])
        y0[0] = x0[0]
        
        # For every other layer, set the inducing outputs to the inducing inputs (which are sampled from N(0,.01)) and lengthscale large 
        for layer in xrange(1,n_layers):
            y0[layer] = x0[layer]
            layer_params[layer][3] = np.log(1)

        return pack_all_params(layer_params, x0, y0)

    # Initialize covariance parameters and hiddens.
    rs = npr.RandomState(1234)
    init_params = .1 * rs.randn(total_num_params) 

    print("Optimizing covariance parameters...")
    objective = lambda params: -log_likelihood(params)

    if smart_init == 1:
        init_params = smart_initialize_params(init_params)

    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
    plot_full_gp(ax_first, init_params, plot_xs)
    ax_first.set_title("Initial full predictions")
    print("Objective: ",objective(init_params))

    cov_params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='BFGS', callback=callback)

    plt.pause(10.0)