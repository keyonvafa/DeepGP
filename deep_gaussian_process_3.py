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

def build_step_function_dataset(D=1, n_data=40, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-2, 2, num=n_data)
    targets = np.sign(inputs) + rs.randn(n_data) * noise_std
    inputs  = inputs.reshape((len(inputs), D))
    return inputs, targets

def build_toy_dataset(D=1, n_data=20, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = (np.cos(inputs) + rs.randn(n_data) * noise_std) / 2.0
    inputs = (inputs - 4.0) / 2.0
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
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2)) # median across all data points of np.sum(diffs**2, axis=2) should be 1


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

    return num_cov_params + 2, predict, predict_with_noise

def build_deep_gp(input_dimension, hidden_dimension, n_layers, covariance_function):

    layer_details = [make_gp_funs(covariance_function, num_cov_params = input_dimension + 1) for x in xrange(n_layers)]
    num_params_each_layer, predict_layer_funcs, predict_funcs_with_noise = zip(*layer_details)

    # Psuedo params defined as X0, y0 (inducing points). We have 10 of these for each layer    
    num_pseudo_params = 5
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
        #print(np.ndarray.flatten(np.array(x0)).shape)
        #print(all_params.shape)
        #print(np.concatenate([np.ndarray.flatten(np.array(x0)),all_params]))
        all_params = np.concatenate([all_params, np.ndarray.flatten(np.array(x0))])
        all_params = np.concatenate([all_params, np.ndarray.flatten(np.array(y0))])
        return all_params


    def sample_from_mvn(mu, sigma): # make sure we return 2d, also make sure data is 2d
        rs = npr.RandomState(0)
        return np.dot(np.linalg.cholesky(sigma+1e-6*np.eye(len(sigma))),rs.randn(len(sigma)))+mu 

    def sample_from_deep_gp(all_params, X, with_noise = False):
        predict = predict_funcs_with_noise if with_noise else predict_layer_funcs
        X_star = X
        layer_params, x0, y0 = unpack_all_params(all_params)
        n_layers = len(x0)
        layer_mean, layer_cov = predict[0](layer_params[0],np.atleast_2d(x0[0]).T, y0[0],X_star)
        for layer in xrange(n_layers-1):
            layer_mean, layer_cov = predict[layer](layer_params[layer],np.atleast_2d(x0[layer]).T, y0[layer],X_star)
            hiddens = sample_from_mvn(layer_mean, layer_cov)
            X_star = np.atleast_2d(hiddens).T
        h_2_mean, h_2_cov = predict[n_layers-1](layer_params[n_layers-1],np.atleast_2d(x0[n_layers-1]).T,y0[n_layers-1],X_star)
        return h_2_mean,h_2_cov

    def squared_error(all_params):
        n_samples = 1
        samples = np.array([sample_from_deep_gp(all_params, X, False)[0] for i in xrange(n_samples)])
        return np.mean((y - np.mean(samples,axis = 0)) ** 2)

    def log_likelihood(all_params): 
        n_samples = 1
        layer_params, x0, y0 = unpack_all_params(all_params)
        #print("Layer 1 Noise Scale", np.exp(layer_params[0][1]) + 0.001)
        #print("Layer 2 Noise Scale", np.exp(layer_params[1][1]) + 0.001)
        #print("Layer 2 Mean: ", layer_params[1][0])
        # It looks like the layer 2 noise scale is going toward 0
        samples = [sample_from_deep_gp(all_params, X, True) for i in xrange(n_samples)]
        return logsumexp(np.array([mvn.logpdf(y,sample[0],sample[1]+1e-6*np.eye(len(sample[1]))) for sample in samples])) - np.log(n_samples) 

    return total_num_params, log_likelihood, sample_from_deep_gp, unpack_all_params, predict_layer_funcs, squared_error, pack_all_params

if __name__ == '__main__':

    n_data = 20
    input_dimension = 1
    hidden_dimension = 1
    n_layers = 3
    X, y = build_step_function_dataset(D=input_dimension, n_data=20)
    #X, y = build_parabola(D = input_dimension, n_data = 10)

    total_num_params, log_likelihood, sample_from_deep_gp, unpack_all_params, predict_layer_funcs, squared_error, pack_all_params = \
        build_deep_gp(input_dimension, hidden_dimension, n_layers, rbf_covariance)

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax_first = fig.add_subplot(411, frameon=False)
    ax_end_to_end = fig.add_subplot(412, frameon=False)
    ax_x_to_h = fig.add_subplot(413, frameon=False)
    ax_h_to_y = fig.add_subplot(414, frameon=False)
    plt.show(block=False)


    def plot_full_gp(ax, params, plot_xs):
        # How do I plot predictive means and error bars when I'm sampling each function from a different distribution?
        # Right now I'm averaging each sample's mean and covariance matrix
        
        ax.cla()
        rs = npr.RandomState(0)
        n_samples = 1
        
        sampled_means_and_covs = [sample_from_deep_gp(params, plot_xs) for i in xrange(n_samples)]
        sampled_means, sampled_covs = zip(*sampled_means_and_covs)
        avg_pred_mean = np.mean(sampled_means, axis = 0)
        avg_pred_cov = np.mean(sampled_covs, axis = 0)
        marg_std = np.sqrt(np.diag(avg_pred_cov))
        #ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
        #np.concatenate([avg_pred_mean - 1.96 * marg_std,
        #               (avg_pred_mean + 1.96 * marg_std)[::-1]]),
        #               alpha=.15, fc='Blue', ec='None')
        #ax.plot(plot_xs, avg_pred_mean, 'b')

        sampled_funcs = np.array([rs.multivariate_normal(sample[0], sample[1]) for sample in sampled_means_and_covs])
        ax.plot(plot_xs,sampled_funcs.T)
        ax.plot(X, y, 'kx')
        #ax.set_ylim([-1.5,1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Full GP, X to Y")


    def plot_gp(ax, X, y, pred_mean, pred_cov, plot_xs):
        ax.cla()
        marg_std = np.sqrt(np.diag(pred_cov))
        #ax.plot(plot_xs, pred_mean, 'b')
        #ax.fill(np.concatenate([plot_xs, plot_xs[::-1]]),
        #        np.concatenate([pred_mean - 1.96 * marg_std,
        #                       (pred_mean + 1.96 * marg_std)[::-1]]),
        #        alpha=.15, fc='Blue', ec='None')

        # Show samples from posterior.
        rs = npr.RandomState(0)
        sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov, size=1)
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
        #ax_end_to_end.plot(x0[0],y0[0], 'ro')

        hidden_mean, hidden_cov = predict_layer_funcs[0](layer_params[0], np.atleast_2d(x0[0]).T, y0[0], plot_xs)
        plot_gp(ax_x_to_h, x0[0], y0[0], hidden_mean, hidden_cov, plot_xs)
        ax_x_to_h.set_title("X to hiddens, with inducing points")

        y_mean, y_cov = predict_layer_funcs[1](layer_params[1], np.atleast_2d(x0[1]).T, y0[1], plot_xs)
        plot_gp(ax_h_to_y, x0[1], y0[1], y_mean, y_cov, plot_xs)
        ax_h_to_y.set_title("hiddens to y, with inducing points")

        plt.draw()
        plt.pause(1.0/60.0)

    # Initialize covariance parameters and hiddens.
    rs = npr.RandomState(1234)
    init_params = .1 * rs.randn(total_num_params) # change initialization (maybe initializing by sampling from our training data), length scale is median of Euclidean distances between data points
    # initialize pseudo x0 by subsampling, l by the median distance thing, outputs at the same value as the inputs for each layer

    print("Optimizing covariance parameters...")
    objective = lambda params: -log_likelihood(params)
    layer_params, x0, y0 = unpack_all_params(init_params)
    
    pairs = itertools.combinations(X, 2)
    dists = np.array([np.linalg.norm(np.array([p1])- np.array([p2])) for p1,p2 in pairs])
    layer_params[0][3] = np.log(np.median(dists))
    x0[0] = np.ndarray.flatten(np.array(X)[rs.choice(len(X), 5, replace=False),:])
    y0[0] = x0[0]
    
    y0[1] = x0[1]
    layer_params[1][3] = np.log(1)
    
    y0[2] = x0[2]
    layer_params[2][3] = np.log(1)
    init_params = pack_all_params(layer_params, x0, y0)

    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
    plot_full_gp(ax_first, init_params, plot_xs)
    ax_first.set_title("Initial full predictions")


    cov_params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='BFGS', callback=callback)

    plt.pause(10.0)