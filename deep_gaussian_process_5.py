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

def build_step_function_dataset(D=1, n_data=40, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.linspace(-2, 2, num=n_data)
    targets = np.sign(inputs) + rs.randn(n_data) * noise_std
    inputs  = inputs.reshape((len(inputs), D))
    return inputs, targets

def rbf_covariance(cov_params, x, xp):
    output_scale = np.exp(cov_params[0])
    lengthscales = np.exp(cov_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2)) 

def build_single_gp(cov_func, num_cov_params, num_pseudo_params, input_dimension):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_gp_params(params):
        mean          = params[0]
        noise_scale   = np.exp(params[1]) + 0.001
        cov_params    = params[2:2+num_cov_params]
        pseudo_params = params[2+num_cov_params:]
        x0, y0 = np.split(pseudo_params,[num_pseudo_params*input_dimension])
        x0     = x0.reshape((num_pseudo_params,input_dimension))
        return mean, cov_params, noise_scale, x0, y0

    def pack_gp_params(mean, cov_params, noise_scale, x0, y0):
        params = np.append(mean,noise_scale)
        params = np.concatenate([params,cov_params])
        params = np.concatenate([params,np.ndarray.flatten(np.array(x0))])
        params = np.concatenate([params,np.ndarray.flatten(np.array(y0))])
        return params#np.append(mean, noise_scale, cov_params, np.ndarray.flatten(np.array(x0)), np.ndarray.flatten(np.array(y0)))

    def predict(params, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale, x0, y0 = unpack_gp_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x0, xstar)
        cov_y_y = cov_func(cov_params, x0, x0) + noise_scale * np.eye(len(y0))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y0 - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov 

    def predict_with_noise(params, xstar):
        pred_mean, pred_cov = predict(params, xstar)
        mean, cov_params, noise_scale, x0, y0 = unpack_gp_params(params)
        return pred_mean, pred_cov + noise_scale*np.eye(len(xstar))

    num_gp_params = 2 + num_cov_params + num_pseudo_params*input_dimension + num_pseudo_params    

    return num_gp_params, predict, predict_with_noise, unpack_gp_params, pack_gp_params

def build_single_layer(input_dimension, output_dimension, num_pseudo_params, covariance_function, random):
    layer_details = [build_single_gp(covariance_function, input_dimension + 1, num_pseudo_params, input_dimension) for i in xrange(output_dimension)]
    num_params_each_output, predict_layer_funcs, predict_funcs_with_noise, unpack_gp_params_layer, pack_gp_params_layer = zip(*layer_details)
    total_params_layer = sum(num_params_each_output)

    def unpack_layer_params(params):
        gp_params = np.array_split(params, output_dimension) # assuming all parameters have equal dims, change to what we had below
        return gp_params

    def sample_from_mvn(mu, sigma):
        rs = npr.RandomState(0)
        return np.dot(np.linalg.cholesky(sigma+1e-6*np.eye(len(sigma))*np.max(np.diag(sigma))),rs.randn(len(sigma)))+mu if random == 1 else mu

    def sample_mean_cov_from_layer(layer_params, xstar, with_noise = False):
        predict = predict_funcs_with_noise if with_noise else predict_layer_funcs
        gp_params = unpack_layer_params(layer_params)
        samples = [predict[i](gp_params[i],xstar) for i in xrange(output_dimension)]
        return samples

    def sample_values_from_layer(layer_params, xstar, with_noise = False): # should return len(x*)
        samples = sample_mean_cov_from_layer(layer_params, xstar, with_noise)
        outputs = [sample_from_mvn(mean,cov) for mean,cov in samples]
        return np.array(outputs).T

    # Implement Prior regularization

    return total_params_layer, sample_mean_cov_from_layer, sample_values_from_layer, predict_layer_funcs, unpack_gp_params_layer, unpack_layer_params, pack_gp_params_layer

def build_deep_gp(dimensions, covariance_function, num_pseudo_params, random):

    deep_details = [build_single_layer(dimensions[i],dimensions[i+1],num_pseudo_params,covariance_function,random) for i in xrange(len(dimensions)-1)]
    num_params_each_layer, sample_mean_cov_funcs, sample_value_funcs, predict_layer_funcs, unpack_gp_params_all, unpack_layer_params, pack_gp_params_all = zip(*deep_details)
    total_params_gp = sum(num_params_each_layer)

    def unpack_all_params(all_params):
        all_layer_params = np.array_split(all_params,np.cumsum(num_params_each_layer))
        return all_layer_params

    def sample_mean_cov_from_deep_gp(all_params, xstar, with_noise = False):
        xtilde = xstar 
        all_layer_params = unpack_all_params(all_params)
        for layer in xrange(len(dimensions)-2):
            #layer_params = all_params[sum(num_params_each_layer[:layer]):sum(num_params_each_layer[:layer+1])]
            layer_params = all_layer_params[layer]
            xtilde = sample_value_funcs[layer](layer_params, xtilde, with_noise)
        final_layer = len(dimensions)-2
        final_layer_params = all_layer_params[final_layer]
        #final_layer_params = all_params[sum(num_params_each_layer[:final_layer]):sum(num_params_each_layer[:final_layer+1])]
        final_mean, final_cov = sample_mean_cov_funcs[final_layer](final_layer_params, xtilde, with_noise)[0] # index into 0 because final layer has one dimension
        return final_mean, final_cov  

    def log_likelihood(all_params):
        samples = [sample_mean_cov_from_deep_gp(all_params, X, True) for i in xrange(n_samples)]
        return logsumexp(np.array([mvn.logpdf(y,mean,var+1e-6*np.eye(len(var))*np.max(np.diag(var))) for mean,var in samples])) - np.log(n_samples)

    def squared_error(all_params):
        samples = np.array([sample_mean_cov_from_deep_gp(all_params, X, False)[0] for i in xrange(n_samples)])
        return np.mean((y - np.mean(samples,axis = 0)) ** 2)

    return total_params_gp, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, unpack_gp_params_all, unpack_layer_params, unpack_all_params,\
        pack_gp_params_all

if __name__ == '__main__':

    random = 0
    n_samples = 1
    dimensions = [1,2,2,1] # Architecture of the GP. Last layer should always be 1

    n_data = 20
    input_dimension = dimensions[0]
    n_layers = len(dimensions)-1
    num_pseudo_params = 10
    X, y = build_step_function_dataset(D=input_dimension, n_data=20)

    total_num_params, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, unpack_gp_params_all, unpack_layer_params, unpack_all_params, \
        pack_gp_params_all = build_deep_gp(dimensions, rbf_covariance, num_pseudo_params, random)

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax_first = fig.add_subplot(411, frameon=False)
    ax_end_to_end = fig.add_subplot(412, frameon=False)
    ax_x_to_h = fig.add_subplot(413, frameon=False)
    ax_h_to_y = fig.add_subplot(414, frameon=False)
    plt.show(block=False)


    def plot_deep_gp(ax, params, plot_xs):
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
        #ax.set_ylim([-1.5,1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Full Deep GP, inputs to outputs")

    def plot_single_gp(ax, x0, y0, pred_mean, pred_cov, plot_xs):
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
        ax.plot(x0, y0, 'ro')
        ax.set_xticks([])
        ax.set_yticks([])


    def callback(params):
        print("Log likelihood {}, Squared Error {}".format(-objective(params),squared_error(params)))
        
        # Show posterior marginals.
        plot_deep_gp(ax_end_to_end, params, plot_xs)
        if dimensions == [1,1]:
            ax_end_to_end.plot(params[4:14],params[14:24], 'ro')
        elif dimensions == [1,1,1]:
            hidden_mean, hidden_cov = predict_layer_funcs[0][0](params[0:24], plot_xs)
            plot_single_gp(ax_x_to_h, params[4:14], params[14:24], hidden_mean, hidden_cov, plot_xs)
            ax_x_to_h.set_title("Inputs to hiddens, inducing points in red")

            y_mean, y_cov = predict_layer_funcs[0][0](params[24:48], plot_xs)
            plot_single_gp(ax_h_to_y, params[28:38], params[38:48], y_mean, y_cov, plot_xs)
            ax_h_to_y.set_title("Hiddens to outputs, inducing points in red")
        plt.draw()
        plt.pause(1.0/60.0)

    rs = npr.RandomState(1234)
    init_params = .1 * rs.randn(total_num_params)

    # SMART INITIALIZATION
    # Admittedly, this is not a good way to do it
    # If you have any tips on how to make this better please let me know

    smart_params = np.array([])
    all_layer_params = unpack_all_params(init_params)
    for layer in xrange(n_layers):
        layer_params = all_layer_params[layer]
        layer_gp_params = unpack_layer_params[layer](layer_params)
        for dim in xrange(dimensions[layer+1]):
            gp_params = layer_gp_params[dim]
            mean, cov_params, noise_scale, x0, y0 = unpack_gp_params_all[layer][dim](gp_params)
            lengthscales = cov_params[1:]
            if layer == 0:
                pairs = itertools.combinations(X, 2)
                dists = np.array([np.abs(p1-p2) for p1,p2 in pairs])
                smart_lengthscales = np.array([np.log(np.median(dists[:,i])) for i in xrange(len(lengthscales))])
                smart_x0 = np.array(X)[rs.choice(len(X), num_pseudo_params, replace=False),:]
                smart_y0 = np.ndarray.flatten(smart_x0)
            else:
                smart_x0 = x0
                smart_y0 = np.ndarray.flatten(x0)
                smart_lengthscales = np.array([np.log(1) for i in xrange(len(lengthscales))])
            cov_params = np.append(cov_params[0],smart_lengthscales)
            params = pack_gp_params_all[layer][dim](mean, cov_params, noise_scale, smart_x0, smart_y0)
            smart_params = np.append(smart_params, params)

    init_params = smart_params



    print("Optimizing covariance parameters...")
    objective = lambda params: -log_likelihood(params)

    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
    plot_deep_gp(ax_first, init_params, plot_xs)
    ax_first.set_title("Initial full predictions")

    params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='BFGS', callback=callback)

    plt.pause(10.0)
