# -*- coding: utf-8 -*-
"""

Backend to implement a general Gaussian-Process based emulator with the underlying gpytorch kernels, and perform the relevant GP operations.
Allows for a possible PCA decomposition of the training set as well.

Created 2019

@author: Michael Walther (@Waelthus)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import functools as ft
import torch
import gpytorch
import numpy as np
import scipy.optimize as op
from tqdm import trange

def predict_weights_func(x, npc, gparr, output_cov=False):
    """
    Predicts the principal component weights from a gaussian process (Most input values are set when creating the emulator with create_emulator)

    input:
      x:
        new position in parameter space to be evaluated
      npc:
        number of pca components to be used for this evaluation
      gparr:
        list of gaussian processes to use
      ww:
        array of old weights for each PCA component and each grid point
    """
    new_weights = []
    new_var = []
    x = torch.tensor(x, dtype=torch.float32)
    for gp_model, gp_likelihood in gparr[:npc]:
      gp_model.eval()
      gp_likelihood.eval()
      with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = gp_likelihood(gp_model(x))
        new_weights.append(pred.mean.numpy())  
        if output_cov:
          new_var.append(pred.variance.numpy())     #could also do the full covariances when changing the model to a multioutput using pred.covariance_matrix

    return np.array(new_weights), np.array(new_var)


def emulator_func(x, mean, std, pca, predict_new_weights, npcmax, npc=None, output_cov=True):
    """
    This function computes the real function from the PCA component prediction made by the GP. (Most input values are set when creating the emulator with create_emulator)

    input:
        pars:
          parameters in the same units as when calling create_emulator
        npc:
          by default use the number of PCA components from creating the emulator, setting this allows getting an emulator with fewer components
        other keywords:
          are fixed at create_emulator and a partial func is returned
      output:
        recon:
          reconstructed statistics for this parameter combination
    """
    if npc is None:
        npc = npcmax
    em = 0
    # get the maximum number of PCA components possible and set npc to this if higher
    npc = min([npc, mean.size])
    # actually get the new PC-weights (and their covariance)
    new_weights, weights_var = predict_new_weights(x, npc)
    # multiply each pca vector with the respective standard deviation (because the PCA was generated for std normalized components)
    res_pca = pca[0:npc, :] * std[np.newaxis, :]
    # sum up weigts*rescaled pca vectors
    em = np.einsum('ij,ik->kj', res_pca[0:npc, :], new_weights)
    # get the covariance matrix by summing up PCA-vector^T * variance in weights * PCA-vector
    if output_cov:
        Cov_em = np.einsum('ij,ik,il->kjl', res_pca[0:npc, :], weights_var, res_pca[0:npc, :])

    # add the mean to the emulated sum of PCA components
    out = em + mean[np.newaxis, :]

    if out.shape[0] == 1:
        out = out[0, ...]
        if output_cov:
            Cov_em = Cov_em[0, ...]
    # return the emulated statistics and cov_matrix
    return out, (Cov_em if output_cov else None)


def PCA_analysis(stat_grid, npc=5):
    """
    This function converts some grid of training data (e.g. statistics) and returns the corresponding PC vectors, weights, as well as the mean and standard deviation of the grid

    input:
        stat_grid:
          grid of training data
      output:
        PC:
          Principle component Analysis eigenvectors (Principle components)
        ww:
          Principle component Analysis eigenvalues
        meangrid:
          The mean of the grid
        stdgrid:
          The standard deviation of the grid
    """
    meangrid = np.mean(stat_grid, axis=0)
    stdgrid = np.std(stat_grid, axis=0)
    # subtract the mean from the statistics and normalize by standard deviation
    normgrid = (stat_grid.T - meangrid[:, np.newaxis]) / stdgrid[:, np.newaxis]
    nbins = normgrid.shape[0]
    nm = normgrid.shape[1]
    npc = min(min(nm, nbins), npc)
    # svd and pca decomposition
    U, D, V = np.linalg.svd(normgrid, full_matrices=False)
    # combine U and D and rescale to get the principal component vectors
    PC = np.dot(U, np.diag(D))[:, :npc].T / np.sqrt(nm)
    # rescale V to get the principal component weights
    ww = V[:npc, :] * np.sqrt(nm)
    # Variance accounted for by each component
    variance = D**2 / np.sum(D**2)
    # Save to file
    np.savez('PCA_results.npz', PCA=PC, weights=ww,
             meangrid=meangrid, stdgrid=stdgrid, variance=variance)
    return PC, ww, meangrid, stdgrid

# Choose kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kerneltype='SE', smooth_lengths=1.0, sigma_l=None, sigma_0=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        if kerneltype == 'SE':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=smooth_lengths))
        elif kerneltype == 'M52':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, lengthscale=smooth_lengths))
        elif kerneltype == 'M32':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, lengthscale=smooth_lengths))
        
        if sigma_0 is not None:
            self.covar_module *= sigma_0 ** 2
        if sigma_l is not None:
            self.covar_module += sigma_l ** 2 * gpytorch.kernels.LinearKernel()
            
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def create_emulator(par_grid, stat_grid, smooth_lengths, noise=None, npc=5, optimize=True, output_cov=True,sigma_l=None,sigma_0=None, noPCA=False, kerneltype='SE',make_deriv_emulator=False):
    """
      generates the emulator given a grid of parameters and the statistics for it

      input:
        par_grid:
          grid of parameters

          par_grid.shape[0]: number of models in grid
          par_grid.shape[1]: number of parameters used for each point
        stat_grid:
          grid of statistics at the positions given by par_grid

          stat_grid.shape[0]: number of points in statistic to emulate
          stat_grid.shape[1]: number of parameters used for each point

        smooth_length:
          correlation length in each of the model parameters used as initial
          guess. This is refined later in this routine.

        noise:
          additional noise assumed in the statistics points (if None uses a default value of the george package)

        npc:
          number of principal components to use

        optimize:
          should an optimization be performed on the hyperpars (using downhill simplex algorithm to find maximum likelihood values)

        output_cov:
          should the covariance matrix of the results also be returned from the created emulator?

        sigma_l:
          Kernel length scale for a dot-product kernel

        sigma_0:
          Kernel length scale for a constant kernel

        noPCA:
          Disable the PCA compression

        kerneltype:
          Type of the GP kernel

        make_deriv_emulator:
          Compute additional derivatives of the emulated mean prediction w.r.t. the input parameters
    """
    ndim = par_grid.shape[1]
    nbins = stat_grid.shape[1]
    if not noPCA:
      # compute the PCA, mean and std of the statistics
      PC, ww, mean_stat, std_stat = PCA_analysis(stat_grid, npc)
    else:
      mean_stat = np.mean(stat_grid, axis=0)
      std_stat = np.std(stat_grid, axis=0)
      PC = np.diag(np.ones(nbins))
      ww = (stat_grid.T - mean_stat[:, np.newaxis]) / std_stat[:, np.newaxis]
    # get the maximum possible number of PCA components and dimensionality of the problem

    npcmax = len(PC)

    if noise is None:
        noise = 1e-5  # A small default noise value

    # Convert data to torch tensors
    train_x = torch.tensor(par_grid, dtype=torch.float32)
    gparr = []

    for i, w in enumerate(ww):
        
        print("generating gp model {i}")
        train_y = torch.tensor(w, dtype=torch.float32)

        # Create likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(noise))
        gp_model = ExactGPModel(train_x, train_y, likelihood, kerneltype=kerneltype, smooth_lengths=smooth_lengths, sigma_l=sigma_l, sigma_0=sigma_0)

        if optimize:
            print("training gp model {i}")
            gp_model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            training_iterations = 50
          for _ in trange(training_iterations,desc='Training GP model, step:'):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            print("training gp model {i} done!")
        gparr.append((gp_model, likelihood))

    # generate a function to predict PCA weights using the GP objects just created
    predict_weights = ft.partial(predict_weights_func, gparr=gparr, ww=ww, output_cov=output_cov)
    # using this function generate an emulator of the statistics
    emulator = ft.partial(emulator_func, mean=mean_stat, std=std_stat, pca=PC,
                          predict_new_weights=predict_weights, npcmax=npcmax, output_cov=output_cov)
   
    parnames, pararr = list(*zip([name,param] for name, param in gp_model.named_parameters()))
    outentries = [emulator, None, pararr, parnames]

    return outentries

