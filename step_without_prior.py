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
from sklearn.cluster import KMeans
from scipy.stats import norm

import cProfile
import re
import pstats

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

def initialize(deep_map, X,num_pseudo_params):
    smart_map = {}
    for layer,layer_map in deep_map.iteritems():
        smart_map[layer] = {}
        for unit,gp_map in layer_map.iteritems():
            smart_map[layer][unit] = {}
            cov_params = gp_map['cov_params']
            lengthscales = cov_params[1:]
            if layer == 0:
                pairs = itertools.combinations(X, 2)
                dists = np.array([np.abs(p1-p2) for p1,p2 in pairs])
                smart_lengthscales = np.array([np.log(np.median(dists[:,i])) for i in xrange(len(lengthscales))])
                kmeans = KMeans(n_clusters = num_pseudo_params, init = 'k-means++')
                fit = kmeans.fit(X)
                smart_x0 = fit.cluster_centers_
                #inds = npr.choice(len(X), num_pseudo_params, replace = False)
                #smart_x0 = np.array(X)[inds,:]
                smart_y0 = np.ndarray.flatten(smart_x0) 
                #smart_y0 = np.array(y)[inds]
                smart_noise_scale = np.log(np.var(smart_y0))
            else:
                smart_x0 = gp_map['x0']
                smart_y0 = np.ndarray.flatten(smart_x0[:,0])
                smart_lengthscales = np.array([np.log(1) for i in xrange(len(lengthscales))])
                smart_noise_scale = np.log(np.var(smart_y0))
            gp_map['cov_params'] = np.append(cov_params[0],smart_lengthscales)
            gp_map['x0'] = smart_x0
            gp_map['y0'] = smart_y0
            #gp_map['noise_scale'] = smart_noise_scale
            smart_map[layer][unit] = gp_map
    smart_params = pack_deep_params(smart_map)
    return smart_params

def rbf_covariance(cov_params, x, xp):
    output_scale = np.exp(cov_params[0])
    lengthscales = np.exp(cov_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2)) 

def pack_gp_params(gp_details):
    params = np.append(gp_details['mean'],gp_details['noise_scale'])
    params = np.concatenate([params,gp_details['cov_params']])
    params = np.concatenate([params,np.ndarray.flatten(np.array(gp_details['x0']))])
    params = np.concatenate([params,np.ndarray.flatten(np.array(gp_details['y0']))])
    return params

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

    def create_gp_map(params):
        mean, cov_params, noise_scale, x0, y0 = unpack_gp_params(params)
        gp_map = {'mean': mean, 'noise_scale': noise_scale, 'cov_params': cov_params, 'x0': x0, 'y0': y0}
        return gp_map

    def predict(params, xstar, with_noise = False, FITC = False):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale, x0, y0 = unpack_gp_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x0, xstar)
        cov_y_y = cov_func(cov_params, x0, x0) + noise_scale * np.eye(len(y0))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y0 - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        if FITC:
            pred_cov = np.diag(np.diag(pred_cov))
        if with_noise:
            pred_cov = pred_cov + noise_scale*np.eye(len(xstar))
        return pred_mean, pred_cov

    num_gp_params = 2 + num_cov_params + num_pseudo_params*input_dimension + num_pseudo_params    

    return num_gp_params, predict, create_gp_map

def pack_layer_params(layer_map):
    params = np.array([])
    for unit,gp_map in layer_map.iteritems():
        params = np.concatenate([params,pack_gp_params(gp_map)])
    return params

def build_single_layer(input_dimension, output_dimension, num_pseudo_params, covariance_function, random):
    layer_details = [build_single_gp(covariance_function, input_dimension + 1, num_pseudo_params, input_dimension) for i in xrange(output_dimension)]
    num_params_each_output, predict_layer_funcs, create_gp_map = zip(*layer_details)
    total_params_layer = sum(num_params_each_output)

    def unpack_layer_params(params):
        gp_params = np.array_split(params, np.cumsum(num_params_each_output))
        return gp_params

    def create_layer_map(params):
        gp_params = unpack_layer_params(params)
        layer_map = {}
        for unit in xrange(output_dimension):
            layer_map[unit] = create_gp_map[unit](gp_params[unit])
        return layer_map

    def sample_from_mvn(mu, sigma,rs = npr.RandomState(0)):
        return np.dot(np.linalg.cholesky(sigma+1e-6*np.eye(len(sigma))),rs.randn(len(sigma)))+mu if random == 1 else mu

    def sample_mean_cov_from_layer(layer_params, xstar, with_noise = True, FITC = False):
        gp_params = unpack_layer_params(layer_params)
        samples = [predict_layer_funcs[i](gp_params[i],xstar,with_noise, FITC) for i in xrange(output_dimension)]
        return samples

    def sample_values_from_layer(layer_params, xstar, with_noise = True, rs = npr.RandomState(0), FITC = False): 
        samples = sample_mean_cov_from_layer(layer_params, xstar, with_noise, FITC)
        outputs = [sample_from_mvn(mean,cov,rs) for mean,cov in samples]
        return np.array(outputs).T

    return total_params_layer, sample_mean_cov_from_layer, sample_values_from_layer, predict_layer_funcs, create_layer_map

def pack_deep_params(deep_map):
    params = np.array([])
    for layer,layer_map in deep_map.iteritems():
        params = np.concatenate([params,pack_layer_params(layer_map)])
    return params

def build_deep_gp(dimensions, covariance_function, num_pseudo_params, random):

    deep_details = [build_single_layer(dimensions[i],dimensions[i+1],num_pseudo_params,covariance_function,random) for i in xrange(len(dimensions)-1)]
    num_params_each_layer, sample_mean_cov_funcs, sample_value_funcs, predict_layer_funcs, create_layer_map = zip(*deep_details)
    total_params_gp = sum(num_params_each_layer)
    n_layers = len(dimensions)-1

    def unpack_all_params(all_params):
        all_layer_params = np.array_split(all_params,np.cumsum(num_params_each_layer))
        return all_layer_params

    def create_deep_map(all_params):
        all_layer_params = unpack_all_params(all_params)
        deep_map = {}
        for layer in xrange(n_layers):
            deep_map[layer] = create_layer_map[layer](all_layer_params[layer])
        return deep_map

    def sample_mean_cov_from_deep_gp(all_params, xstar, with_noise = True, rs=npr.RandomState(0), FITC = False):
        xtilde = xstar 
        all_layer_params = unpack_all_params(all_params)
        for layer in xrange(len(dimensions)-2):
            layer_params = all_layer_params[layer]
            xtilde = sample_value_funcs[layer](layer_params, xtilde, with_noise, rs, FITC)
        final_layer = len(dimensions)-2
        final_layer_params = all_layer_params[final_layer]
        final_mean, final_cov = sample_mean_cov_funcs[final_layer](final_layer_params, xtilde, with_noise, FITC)[0] # index into 0 because final layer has one unit
        return final_mean, final_cov  

    def evaluate_prior(all_params): # clean up code so we don't compute matrices twice
        all_layer_params = unpack_all_params(all_params)
        log_prior = 0
        deep_map = create_deep_map(all_params)
        for layer,layer_map in deep_map.iteritems():
            for unit,gp_map in layer_map.iteritems():
                cov_y_y = covariance_function(gp_map['cov_params'],gp_map['x0'],gp_map['x0']) + gp_map['noise_scale'] * np.eye(len(gp_map['y0']))
                log_prior += mvn.logpdf(gp_map['y0'],np.ones(len(cov_y_y))*gp_map['mean'],cov_y_y + np.diag(np.diag(cov_y_y))*0) # CHANGE
                ##log_prior += mvn.logpdf(gp_map['y0'],np.ones(len(cov_y_y))*gp_map['mean'],cov_y_y + np.eye(len(cov_y_y))*tuning_param)
                ###log_prior += mvn.logpdf(gp_map['y0'],np.ones(len(cov_y_y))*gp_map['mean'],np.diag(np.diag(cov_y_y))*10)
        return 0#log_prior

    def log_likelihood(all_params, X, y, n_samples):
        rs = npr.RandomState(0)
        samples = [sample_mean_cov_from_deep_gp(all_params, X, True, rs, FITC = True) for i in xrange(n_samples)]
        return logsumexp(np.array([mvn.logpdf(y,mean,var) for mean,var in samples])) - np.log(n_samples) \
            + evaluate_prior(all_params)
        #return logsumexp(np.array([mvn.logpdf(y,mean,np.diag(np.diag(var))) for mean,var in samples])) - np.log(n_samples) \
        #    + evaluate_prior(all_params)

    def squared_error(all_params, X, y, n_samples):
        rs = npr.RandomState(0)
        samples = np.array([sample_mean_cov_from_deep_gp(all_params, X, True, rs, FITC = True)[0] for i in xrange(n_samples)])
        return np.mean((y - np.mean(samples,axis = 0)) ** 2)

    return total_params_gp, log_likelihood, sample_mean_cov_from_deep_gp, predict_layer_funcs, squared_error, create_deep_map

if __name__ == '__main__':
    random = 1 
    n_samples = 10 
    n_samples_to_plot = 5 
    
    n_data = 20 
    input_dimension = 1
    num_pseudo_params = 10 
    X, y = build_step_function_dataset(D=input_dimension, n_data=n_data)

    dimensions = [1,1]
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
    npr.seed(0)
    
    fig = plt.figure(figsize=(20,8), facecolor='white')
    ax_large = fig.add_subplot(121, frameon=False)
    ax_small = fig.add_subplot(122, frameon=False)
    plt.show(block=False) 

    axes_set = [False, True] # Architecture of the GP. Last layer should always be 1 

    init_params = .1 * rs.randn(total_num_params)
    deep_map = create_deep_map(init_params)
    init_params = initialize(deep_map, X, num_pseudo_params)

    print("Optimizing covariance parameters...")
    objective = lambda params: -log_likelihood(params,X,y,n_samples)

    params = minimize(value_and_grad(objective), init_params, jac=True,
                          method='BFGS', callback=callback,options={'maxiter':200})
    
    params = params['x']
    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))

    deep_map = create_deep_map(params)
    for axes in axes_set:
        if axes:
            ax = ax_small
            title = "Close up"
        else:
            ax = ax_large
            title = "Far"
        plot_deep_gp(ax, params, plot_xs)
        ax.plot(np.ndarray.flatten(deep_map[0][0]['x0']),deep_map[0][0]['y0'], 'ro')

        if axes:
            ax.set_ylim([-1.5,1.5])
        else:
            ax.set_ylim([-21,5])
        #ax.set_title(title, fontsize = 20)
        ax.set_xlabel(r'$x$',fontsize = 20)
        ax.set_ylabel(r'$f(x)$',fontsize = 20)
    print(deep_map[0][0]['cov_params'])
    print(deep_map[0][0]['noise_scale'])
    
    plt.savefig('step_without_prior.png', format='png', bbox_inches='tight',dpi=200)

    plt.pause(80.0)