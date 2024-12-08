import os
import argparse
import random 
import pandas as pd
import numpy as np
import datatable as dt
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
from pyro.optim import ExponentialLR
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate

import torch
import torch.nn as nn
import torch.nn.functional as ft
from torch.utils.data import DataLoader
from torch.distributions import constraints
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from torch.distributions.transforms import SoftmaxTransform

from .utils.custom_mlp import MLP, Exp
from .utils.utils import convert_to_tensor, tensor_to_numpy
from .utils.utils import CustomDataset,CustomMultiOmicsDataset,CustomMultiOmicsDataset2


import warnings
warnings.filterwarnings("ignore")

import dill as pickle
import gzip 
from packaging.version import Version
torch_version = torch.__version__

from typing import Literal


def set_random_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA, set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups.
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for Pyro
    pyro.set_rng_seed(seed)

class SUREMO(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on single cell datasets
    :param output_size:  size of the tensor representing the class label
    :param input_size: size of the tensor representing the cell
    :param z_dim: size of the tensor representing the latent random variable z
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cude: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(self,
                 input_size1: int = 2000,
                 input_size2: int = 500,
                 undesired_size: int = 2,
                 codebook_size: int = 200,
                 z_dim: int = 50,
                 hidden_layers: list = [500],
                 hidden_layer_activation: Literal['relu','softplus','leakyrelu','linear'] = 'relu',
                 use_dirichlet: bool = True,
                 dirichlet_mass: float = 1,
                 loss_func1: Literal['negbinomial','poisson','multinomial','gaussian'] = 'negbinomial',
                 loss_func2: Literal['negbinomial','poisson','multinomial','gaussian'] = 'negbinomial',
                 inverse_dispersion: float = 10.0,
                 nn_dropout: float = 0.1,
                 zero_inflation1: Literal['exact','inexact','none'] = 'exact',
                 zero_inflation2: Literal['exact','inexact','none'] = 'exact',
                 gate_prior: float = 0.6,
                 delta: float = 0.0,
                 post_layer_fct: list = ['layernorm'],
                 post_act_fct: list = None,
                 latent_dist: Literal['normal','laplacian','cauchy','studentt'] = 'normal',
                 studentt_dof: float = 8,
                 config_enum: str = 'parallel',
                 use_cuda: bool = False,
                 dtype=torch.float32,
                 ):
        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.input_size1 = input_size1
        self.input_size2 = input_size2
        self.undesired_size = undesired_size
        self.inverse_dispersion = inverse_dispersion
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.decoder_hidden_layers = hidden_layers[::-1]
        self.use_undesired = True if self.undesired_size>0 else False
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.delta = delta
        self.loss_func1 = loss_func1
        self.loss_func2 = loss_func2
        
        self.options = None
        self.code_size=codebook_size

        self.use_studentt = False
        self.use_laplacian=False
        self.latent_dist = latent_dist
        if latent_dist.lower() in ['laplacian']:
            self.use_laplacian=True
        elif latent_dist.lower() in ['studentt','student-t','t']:
            self.use_studentt = True 
        self.dof = studentt_dof
        self.dtype = dtype

        self.dist_model = 'dmm' if use_dirichlet else 'mm'
        self.dirimulti_mass = dirichlet_mass

        self.use_zeroinflate1 = False
        self.use_zeroinflate2 = False
        self.use_exact_zeroinflate1 = False
        self.use_exact_zeroinflate2 = False
        if zero_inflation1=='Exact':
            self.use_zeroinflate1 = True
            self.use_exact_zeroinflate1 = True
        elif zero_inflation1=='Inexact':
            self.use_zeroinflate1 = True
        if zero_inflation2=='Exact':
            self.use_zeroinflate2 = True
            self.use_exact_zeroinflate2 = True
        elif zero_inflation2=='Inexact':
            self.use_zeroinflate2 = True

        if gate_prior < 1e-5:
            gate_prior = 1e-5
        elif gate_prior == 1:
            gate_prior = 1-1e-5
        self.gate_prior = np.log(gate_prior) - np.log(1-gate_prior)

        self.nn_dropout = nn_dropout
        self.post_layer_fct = post_layer_fct
        self.post_act_fct = post_act_fct
        self.hidden_layer_activation = hidden_layer_activation

        self.codebook_weights = None

        # define and instantiate the neural networks representing
        # the parameters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        decoder_hidden_sizes = self.decoder_hidden_layers

        nn_layer_norm, nn_batch_norm, nn_layer_dropout = False, False, False
        na_layer_norm, na_batch_norm, na_layer_dropout = False, False, False

        if self.post_layer_fct is not None:
            nn_layer_norm=True if ('layernorm' in self.post_layer_fct) or ('layer_norm' in self.post_layer_fct) else False
            nn_batch_norm=True if ('batchnorm' in self.post_layer_fct) or ('batch_norm' in self.post_layer_fct) else False
            nn_layer_dropout=True if 'dropout' in self.post_layer_fct else False

        if self.post_act_fct is not None:
            na_layer_norm=True if ('layernorm' in self.post_act_fct) or ('layer_norm' in self.post_act_fct) else False
            na_batch_norm=True if ('batchnorm' in self.post_act_fct) or ('batch_norm' in self.post_act_fct) else False
            na_layer_dropout=True if 'dropout' in self.post_act_fct else False

        if nn_layer_norm and nn_batch_norm and nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout),nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif nn_layer_norm and nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.LayerNorm(layer.module.out_features))
        elif nn_batch_norm and nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.BatchNorm1d(layer.module.out_features))
        elif nn_layer_norm and nn_batch_norm:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif nn_layer_norm:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.LayerNorm(layer.module.out_features)
        elif nn_batch_norm:
            post_layer_fct = lambda layer_ix, total_layers, layer:nn.BatchNorm1d(layer.module.out_features)
        elif nn_layer_dropout:
            post_layer_fct = lambda layer_ix, total_layers, layer: nn.Dropout(self.nn_dropout)
        else:
            post_layer_fct = lambda layer_ix, total_layers, layer: None

        if na_layer_norm and na_batch_norm and na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout),nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif na_layer_norm and na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.LayerNorm(layer.module.out_features))
        elif na_batch_norm and na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.Dropout(self.nn_dropout), nn.BatchNorm1d(layer.module.out_features))
        elif na_layer_norm and na_batch_norm:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Sequential(nn.BatchNorm1d(layer.module.out_features), nn.LayerNorm(layer.module.out_features))
        elif na_layer_norm:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.LayerNorm(layer.module.out_features)
        elif na_batch_norm:
            post_act_fct = lambda layer_ix, total_layers, layer:nn.BatchNorm1d(layer.module.out_features)
        elif na_layer_dropout:
            post_act_fct = lambda layer_ix, total_layers, layer: nn.Dropout(self.nn_dropout)
        else:
            post_act_fct = lambda layer_ix, total_layers, layer: None

        if self.hidden_layer_activation == 'relu':
            activate_fct = nn.ReLU
        elif self.hidden_layer_activation == 'softplus':
            activate_fct = nn.Softplus
        elif self.hidden_layer_activation == 'leakyrelu':
            activate_fct = nn.LeakyReLU
        elif self.hidden_layer_activation == 'linear':
            activate_fct = nn.Identity

        # define the neural networks used later in the model and the guide.
        self.encoder_n = MLP(
            [self.z_dim] + hidden_sizes + [self.code_size],
            activation=activate_fct,
            output_activation=None,
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        self.encoder_zn = MLP(
            [self.input_size1 + self.input_size2] + hidden_sizes + [[z_dim, z_dim]],
            activation=activate_fct,
            output_activation=[None, Exp],
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        if self.use_undesired:
            if self.loss_func1 in ['gaussian','lognormal']:
                self.decoder_concentrate_first_omics = MLP(
                    [self.undesired_size + self.z_dim] + decoder_hidden_sizes + [[self.input_size1, self.input_size1]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate_first_omics = MLP(
                    [self.undesired_size + self.z_dim] + decoder_hidden_sizes + [self.input_size1],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate_first_omics = MLP(
                [self.undesired_size + self.z_dim] + hidden_sizes + [[self.input_size1, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ) 
        else:
            if self.loss_func1 in ['gaussian','lognormal']:
                self.decoder_concentrate_first_omics = MLP(
                    [self.z_dim] + decoder_hidden_sizes + [[self.input_size1, self.input_size1]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate_first_omics = MLP(
                    [self.z_dim] + decoder_hidden_sizes + [self.input_size1],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate_first_omics = MLP(
                [self.z_dim] + hidden_sizes + [[self.input_size1, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ) 

        if self.use_undesired:
            if self.loss_func2 in ['gaussian','lognormal']:
                self.decoder_concentrate_second_omics = MLP(
                    [self.undesired_size + self.z_dim] + decoder_hidden_sizes + [[self.input_size2, self.input_size2]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate_second_omics = MLP(
                    [self.undesired_size + self.z_dim] + decoder_hidden_sizes + [self.input_size2],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate_second_omics = MLP(
                [self.undesired_size + self.z_dim] + hidden_sizes + [[self.input_size2, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ) 
        else:
            if self.loss_func2 in ['gaussian','lognormal']:
                self.decoder_concentrate_second_omics = MLP(
                    [self.z_dim] + decoder_hidden_sizes + [[self.input_size2, self.input_size2]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate_second_omics = MLP(
                    [self.z_dim] + decoder_hidden_sizes + [self.input_size2],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate_second_omics = MLP(
                [self.z_dim] + hidden_sizes + [[self.input_size2, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            ) 

        if self.latent_dist=='studentt':
            self.codebook = MLP(
                [self.code_size] + hidden_sizes + [[z_dim,z_dim,z_dim]],
                activation=activate_fct,
                output_activation=[Exp,None,Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )
        else:
            self.codebook = MLP(
                [self.code_size] + hidden_sizes + [[z_dim,z_dim]],
                activation=activate_fct,
                output_activation=[None,Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def get_device(self):
        return next(self.parameters()).device

    def cutoff(self, xs, thresh=None):
        eps = torch.finfo(xs.dtype).eps
        
        if not thresh is None:
            if eps < thresh:
                eps = thresh

        xs = xs.clamp(min=eps)

        if torch.any(torch.isnan(xs)):
            xs[torch.isnan(xs)] = eps

        return xs

    def softmax(self, xs):
        #soft_enc = nn.Softmax(dim=1)
        #xs = soft_enc(xs)
        #xs = clamp_probs(xs)
        #xs = ft.normalize(xs, 1, 1)
        xs = SoftmaxTransform()(xs)
        return xs

    def sigmoid(self, xs):
        sigm_enc = nn.Sigmoid()
        xs = sigm_enc(xs)
        xs = clamp_probs(xs)
        return xs

    def softmax_logit(self, xs):
        eps = torch.finfo(xs.dtype).eps
        xs = self.softmax(xs)
        xs = torch.logit(xs, eps=eps)
        return xs

    def logit(self, xs):
        eps = torch.finfo(xs.dtype).eps
        xs = torch.logit(xs, eps=eps)
        return xs

    def dirimulti_param(self, xs):
        xs = self.dirimulti_mass * self.sigmoid(xs)
        return xs

    def multi_param(self, xs):
        xs = self.softmax(xs)
        return xs

    def model1(self, xs, xs2):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module('suremo', self)

        total_count1 = pyro.param("inverse_dispersion1", 10.0 * xs.new_ones(self.input_size1), constraint=constraints.positive)
        total_count2 = pyro.param("inverse_dispersion2", 10.0 * xs2.new_ones(self.input_size2), constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        self.options = options

        I = torch.eye(self.code_size, **self.options)
        if self.latent_dist=='studentt':
            acs_dof,acs_loc,acs_scale = self.codebook(I)
        else:
            acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            ###############################################
            # p(zn)
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior))

            prior_loc = torch.matmul(ns, acs_loc)
            prior_scale = torch.matmul(ns, acs_scale)

            if self.latent_dist == 'studentt':
                prior_dof = torch.matmul(ns, acs_dof)
                zns = pyro.sample('zn', dist.StudentT(df=prior_dof, loc=prior_loc, scale=prior_scale).to_event(1))
            elif self.latent_dist == 'laplacian':
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            elif self.latent_dist == 'cauchy':
                zns = pyro.sample('zn', dist.Cauchy(prior_loc, prior_scale).to_event(1))
            elif self.latent_dist == 'normal':
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            ###############################################
            # p(a | zys, zks, zk2s)
            zs = zns

            if self.loss_func1 == 'gaussian':
                concentrate_omic1_loc, concentrate_omic1_scale = self.decoder_concentrate_first_omics(zs)
                concentrate_omic1 = concentrate_omic1_loc
            else:
                concentrate_omic1 = self.decoder_concentrate_first_omics(zs)

            if self.loss_func2 == 'gaussian':
                concentrate_omic2_loc, concentrate_omic2_scale = self.decoder_concentrate_second_omics(zs)
                concentrate_omic2 = concentrate_omic2_loc
            else:
                concentrate_omic2 = self.decoder_concentrate_second_omics(zs)

            if self.dist_model == 'dmm':
                concentrate_omic1 = self.dirimulti_param(concentrate_omic1)
                theta_omic1 = dist.DirichletMultinomial(total_count=1, concentration=concentrate_omic1).mean

                concentrate_omic2 = self.dirimulti_param(concentrate_omic2)
                theta_omic2 = dist.DirichletMultinomial(total_count=1, concentration=concentrate_omic2).mean
                
            elif self.dist_model == 'mm':
                probs_omic1 = self.multi_param(concentrate_omic1)
                theta_omic1 = dist.Multinomial(total_count=1, probs=probs_omic1).mean

                probs_omic2 = self.multi_param(concentrate_omic2)
                theta_omic2 = dist.Multinomial(total_count=1, probs=probs_omic2).mean

            # zero-inflation model
            if self.use_zeroinflate1:
                gate_loc1 = self.gate_prior * torch.ones(batch_size, self.input_size1, **options)
                gate_scale1 = torch.ones(batch_size, self.input_size1, **options)
                gate_logits1 = pyro.sample('gate_logit1', dist.Normal(gate_loc1, gate_scale1).to_event(1))
                gate_probs1 = self.sigmoid(gate_logits1)

                if self.use_exact_zeroinflate1:
                    if self.loss_func1 == 'multinomial':
                        theta_omic1 = probs_to_logits(theta_omic1) + probs_to_logits(1-gate_probs1)
                        theta_omic1 = logits_to_probs(theta_omic1)
                else:
                    if self.loss_func1 != 'gaussian':
                        theta_omic1 = probs_to_logits(theta_omic1) + probs_to_logits(1-gate_probs1)
                        theta_omic1 = logits_to_probs(theta_omic1)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one1', dist.Binomial(probs=1-gate_probs1).to_event(1), obs=ones)

            if self.use_zeroinflate2:
                gate_loc2 = self.gate_prior * torch.ones(batch_size, self.input_size2, **options)
                gate_scale2 = torch.ones(batch_size, self.input_size2, **options)
                gate_logits2 = pyro.sample('gate_logit2', dist.Normal(gate_loc2, gate_scale2).to_event(1))
                gate_probs2 = self.sigmoid(gate_logits2)

                if self.use_exact_zeroinflate2:
                    if self.loss_func2 == 'multinomial':
                        theta_omic2 = probs_to_logits(theta_omic2) + probs_to_logits(1-gate_probs2)
                        theta_omic2 = logits_to_probs(theta_omic2)
                else:
                    if self.loss_func2 != 'gaussian':
                        theta_omic2 = probs_to_logits(theta_omic2) + probs_to_logits(1-gate_probs2)
                        theta_omic2 = logits_to_probs(theta_omic2)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one2', dist.Binomial(probs=1-gate_probs2).to_event(1), obs=ones)

            if self.loss_func1 == 'negbinomial':
                if self.use_exact_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count1, probs=theta_omic1),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count1, probs=theta_omic1).to_event(1), obs=xs)
            elif self.loss_func1 == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta_omic1
                if self.use_exact_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func1 == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta_omic1), obs=xs)
            elif self.loss_func1 == 'gaussian':
                if self.use_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_omic1, concentrate_omic1_scale),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_omic1, concentrate_omic1_scale).to_event(1), obs=xs)
            elif self.loss_func1 == 'lognormal':
                if self.use_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_omic1, concentrate_omic1_scale),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_omic1, concentrate_omic1_scale).to_event(1), obs=xs)

            if self.loss_func2 == 'negbinomial':
                if self.use_exact_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count2, probs=theta_omic2),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.NegativeBinomial(total_count=total_count2, probs=theta_omic2).to_event(1), obs=xs2)
            elif self.loss_func2 == 'poisson':
                rate = xs2.sum(1).unsqueeze(-1) * theta_omic2
                if self.use_exact_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.Exponential(rate=rate).to_event(1), obs=xs2)
            elif self.loss_func2 == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta_omic2), obs=xs2)
            elif self.loss_func2 == 'gaussian':
                if self.use_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.Normal(concentrate_omic2, concentrate_omic2_scale),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.Normal(concentrate_omic2, concentrate_omic2_scale).to_event(1), obs=xs2)
            elif self.loss_func2 == 'lognormal':
                if self.use_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_omic2, concentrate_omic2_scale),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.LogNormal(concentrate_omic2, concentrate_omic2_scale).to_event(1), obs=xs2)

    def guide1(self, xs, xs2):
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data'):
            # q(zn | x)
            zn_loc, zn_scale = self.encoder_zn([xs,xs2])
            zn_scale = self.cutoff(zn_scale)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            # q(n | x)
            alpha_n = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha_n))

            ####################################
            # q(gate | xs)
            zs = zns

            if self.use_zeroinflate1:
                omics1_loc,omics1_scale = self.encoder_gate_first_omics(zs)
                omics1_scale = self.cutoff(omics1_scale)
                gates_logit1 = pyro.sample('gate_logit1', dist.Normal(omics1_loc,omics1_scale).to_event(1))

            if self.use_zeroinflate2:
                omics2_loc,omics2_scale = self.encoder_gate_second_omics(zs)
                omics2_scale = self.cutoff(omics2_scale)
                gates_logit2 = pyro.sample('gate_logit2', dist.Normal(omics2_loc,omics2_scale).to_event(1))

    def model2(self, xs, xs2, us):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module('suremo', self)

        total_count1 = pyro.param("inverse_dispersion1", 10.0 * xs.new_ones(self.input_size1), constraint=constraints.positive)
        total_count2 = pyro.param("inverse_dispersion2", 10.0 * xs2.new_ones(self.input_size2), constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        self.options = options

        I = torch.eye(self.code_size, **self.options)
        if self.latent_dist=='studentt':
            acs_dof,acs_loc,acs_scale = self.codebook(I)
        else:
            acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            ###############################################
            # p(zn)
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior))

            prior_loc = torch.matmul(ns, acs_loc)
            prior_scale = torch.matmul(ns, acs_scale)

            if self.latent_dist == 'studentt':
                prior_dof = torch.matmul(ns, acs_dof)
                zns = pyro.sample('zn', dist.StudentT(df=prior_dof, loc=prior_loc, scale=prior_scale).to_event(1))
            elif self.latent_dist == 'laplacian':
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            elif self.latent_dist == 'cauchy':
                zns = pyro.sample('zn', dist.Cauchy(prior_loc, prior_scale).to_event(1))
            elif self.latent_dist == 'normal':
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            ###############################################
            # p(a | zys, zks, zk2s)
            zs = [us,zns]

            if self.loss_func1 == 'gaussian':
                concentrate_omic1_loc, concentrate_omic1_scale = self.decoder_concentrate_first_omics(zs)
                concentrate_omic1 = concentrate_omic1_loc
            else:
                concentrate_omic1 = self.decoder_concentrate_first_omics(zs)

            if self.loss_func2 == 'gaussian':
                concentrate_omic2_loc, concentrate_omic2_scale = self.decoder_concentrate_second_omics(zs)
                concentrate_omic2 = concentrate_omic2_loc
            else:
                concentrate_omic2 = self.decoder_concentrate_second_omics(zs)

            if self.dist_model == 'dmm':
                concentrate_omic1 = self.dirimulti_param(concentrate_omic1)
                theta_omic1 = dist.DirichletMultinomial(total_count=1, concentration=concentrate_omic1).mean

                concentrate_omic2 = self.dirimulti_param(concentrate_omic2)
                theta_omic2 = dist.DirichletMultinomial(total_count=1, concentration=concentrate_omic2).mean
                
            elif self.dist_model == 'mm':
                probs_omic1 = self.multi_param(concentrate_omic1)
                theta_omic1 = dist.Multinomial(total_count=1, probs=probs_omic1).mean

                probs_omic2 = self.multi_param(concentrate_omic2)
                theta_omic2 = dist.Multinomial(total_count=1, probs=probs_omic2).mean

            # zero-inflation model
            if self.use_zeroinflate1:
                gate_loc1 = self.gate_prior * torch.ones(batch_size, self.input_size1, **options)
                gate_scale1 = torch.ones(batch_size, self.input_size1, **options)
                gate_logits1 = pyro.sample('gate_logit1', dist.Normal(gate_loc1, gate_scale1).to_event(1))
                gate_probs1 = self.sigmoid(gate_logits1)

                if self.use_exact_zeroinflate1:
                    if self.loss_func1 == 'multinomial':
                        theta_omic1 = probs_to_logits(theta_omic1) + probs_to_logits(1-gate_probs1)
                        theta_omic1 = logits_to_probs(theta_omic1)
                else:
                    if self.loss_func1 != 'gaussian':
                        theta_omic1 = probs_to_logits(theta_omic1) + probs_to_logits(1-gate_probs1)
                        theta_omic1 = logits_to_probs(theta_omic1)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one1', dist.Binomial(probs=1-gate_probs1).to_event(1), obs=ones)

            if self.use_zeroinflate2:
                gate_loc2 = self.gate_prior * torch.ones(batch_size, self.input_size2, **options)
                gate_scale2 = torch.ones(batch_size, self.input_size2, **options)
                gate_logits2 = pyro.sample('gate_logit2', dist.Normal(gate_loc2, gate_scale2).to_event(1))
                gate_probs2 = self.sigmoid(gate_logits2)

                if self.use_exact_zeroinflate2:
                    if self.loss_func2 == 'multinomial':
                        theta_omic2 = probs_to_logits(theta_omic2) + probs_to_logits(1-gate_probs2)
                        theta_omic2 = logits_to_probs(theta_omic2)
                else:
                    if self.loss_func2 != 'gaussian':
                        theta_omic2 = probs_to_logits(theta_omic2) + probs_to_logits(1-gate_probs2)
                        theta_omic2 = logits_to_probs(theta_omic2)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one2', dist.Binomial(probs=1-gate_probs2).to_event(1), obs=ones)

            if self.loss_func1 == 'negbinomial':
                if self.use_exact_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count1, probs=theta_omic1),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count1, probs=theta_omic1).to_event(1), obs=xs)
            elif self.loss_func1 == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta_omic1
                if self.use_exact_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func1 == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta_omic1), obs=xs)
            elif self.loss_func1 == 'gaussian':
                if self.use_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_omic1, concentrate_omic1_scale),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_omic1, concentrate_omic1_scale).to_event(1), obs=xs)
            elif self.loss_func1 == 'lognormal':
                if self.use_zeroinflate1:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_omic1, concentrate_omic1_scale),gate_logits=gate_logits1).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_omic1, concentrate_omic1_scale).to_event(1), obs=xs)

            if self.loss_func2 == 'negbinomial':
                if self.use_exact_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count2, probs=theta_omic2),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.NegativeBinomial(total_count=total_count2, probs=theta_omic2).to_event(1), obs=xs2)
            elif self.loss_func2 == 'poisson':
                rate = xs2.sum(1).unsqueeze(-1) * theta_omic2
                if self.use_exact_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.Exponential(rate=rate).to_event(1), obs=xs2)
            elif self.loss_func2 == 'multinomial':
                pyro.sample('x2', dist.Multinomial(total_count=int(1e8), probs=theta_omic2), obs=xs2)
            elif self.loss_func2 == 'gaussian':
                if self.use_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.Normal(concentrate_omic2, concentrate_omic2_scale),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.Normal(concentrate_omic2, concentrate_omic2_scale).to_event(1), obs=xs2)
            elif self.loss_func2 == 'lognormal':
                if self.use_zeroinflate2:
                    pyro.sample('x2', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_omic2, concentrate_omic2_scale),gate_logits=gate_logits2).to_event(1), obs=xs2)
                else:
                    pyro.sample('x2', dist.LogNormal(concentrate_omic2, concentrate_omic2_scale).to_event(1), obs=xs2)

    def guide2(self, xs, xs2, us=None):
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data'):
            # q(zn | x)
            zn_loc, zn_scale = self.encoder_zn([xs,xs2])
            zn_scale = self.cutoff(zn_scale)
            if self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(zn_loc, zn_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            # q(n | x)
            alpha_n = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha_n))

            ####################################
            # q(gate | xs)
            if self.use_undesired:
                zs = [us,zns]
            else:
                zs = zns

            if self.use_zeroinflate1:
                omics1_loc,omics1_scale = self.encoder_gate_first_omics(zs)
                omics1_scale = self.cutoff(omics1_scale)
                gates_logit1 = pyro.sample('gate_logit1', dist.Normal(omics1_loc,omics1_scale).to_event(1))

            if self.use_zeroinflate2:
                omics2_loc,omics2_scale = self.encoder_gate_second_omics(zs)
                omics2_scale = self.cutoff(omics2_scale)
                gates_logit2 = pyro.sample('gate_logit2', dist.Normal(omics2_loc,omics2_scale).to_event(1))

    def _get_metacell_coordinates(self):
        I = torch.eye(self.code_size, **self.options)
        if self.latent_dist=='studentt':
            _,cb,_ = self.codebook(I)
        else:
            cb,_ = self.codebook(I)
        return cb
    
    def get_metacell_coordinates(self):
        cb = self._get_metacell_coordinates()
        cb = tensor_to_numpy(cb)
        return cb
    
    def _get_metacell_expressions(self):
        cbs = self._get_metacell_coordinates()
        concentrate1, concentrate2 = self._expression(cbs)
        return concentrate1, concentrate2
    
    def get_metacell_expressions(self):
        concentrate1, concentrate2 = self._get_metacell_expressions()
        concentrate1 = tensor_to_numpy(concentrate1)
        concentrate2 = tensor_to_numpy(concentrate2)
        return concentrate1, concentrate2
    
    def get_metacell_counts(self, total_count=1e3, total_counts_per_item=1e4, use_sampler=False, sample_method='nb'):
        concentrate1, concentrate2 = self._get_metacell_expressions()
        if use_sampler:
            if sample_method.lower() == 'nb':
                counts1, counts2 = self._count_sample(concentrate1, concentrate2, total_count=total_count)
            elif sample_method.lower() == 'poisson':
                counts1, counts2 = self._count_sample_poisson(concentrate1, concentrate2, total_counts_per_cell=total_counts_per_item)
        else:
            counts1, counts2 = self._count(concentrate1, concentrate2, total_counts_per_cell=total_counts_per_item)
        counts1 = tensor_to_numpy(counts1)
        counts2 = tensor_to_numpy(counts2)
        return counts1,counts2
    
    def _get_cell_coordinates(self, xs, xs2, use_decoder=False, soft_assign=False):
        if use_decoder:
            cb = self._get_metacell_coordinates()
            if soft_assign:
                A = self._soft_assignments(xs,xs2)
            else:
                A = self._hard_assignments(xs,xs2)
            zs = torch.matmul(A, cb)
        else:
            zs,_ = self.encoder_zn([xs,xs2])
        return zs
    
    def get_cell_coordinates(self, xs, xs2, batch_size=1024, use_decoder=False, soft_assign=False):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs, xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        Z = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                zns = self._get_cell_coordinates(batch_x1, batch_x2, use_decoder=use_decoder, soft_assign=soft_assign)
                Z.append(tensor_to_numpy(zns))
                pbar.update(1)

        Z = np.concatenate(Z)
        return Z
    
    def _get_codebook(self):
        I = torch.eye(self.code_size, **self.options)
        if self.latent_dist=='studentt':
            _,cb_loc,cb_scale = self.codebook(I)
        else:
            cb_loc,cb_scale = self.codebook(I)
        return cb_loc,cb_scale
    
    def get_codebook(self):
        cb_loc,cb_scale = self._get_codebook()
        return tensor_to_numpy(cb_loc),tensor_to_numpy(cb_scale)
    
    def _code(self, xs, xs2):
        zns,_ = self.encoder_zn([xs,xs2])
        alpha = self.encoder_n(zns)
        return alpha
    
    def _soft_assignments(self, xs, xs2):
        alpha = self._code(xs, xs2)
        alpha = self.softmax(alpha)
        return alpha
    
    def soft_assignments(self, xs, xs2, batch_size=1024):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs, xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                a = self._soft_assignments(batch_x1, batch_x2)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A

    def _hard_assignments(self, xs, xs2):
        alpha = self._code(xs, xs2)
        res, ind = torch.topk(alpha, 1)
        # convert the digit(s) to one-hot tensor(s)
        ns = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ns
    
    def hard_assignments(self, xs, xs2, batch_size=1024):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs, xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                a = self._hard_assignments(batch_x1, batch_x2)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A
    
    def _expression(self, zns):    
        if self.use_undesired:
            us = torch.zeros(zns.shape[0], self.undesired_size, **self.options)
            zs=[us,zns]
        else:
            zs=zns

        if not (self.loss_func1 in ['gaussian','lognormal']):
            concentrate1 = self.decoder_concentrate_first_omics(zs)
        else:
            concentrate1,_ = self.decoder_concentrate_first_omics(zs)

        if not (self.loss_func2 in ['gaussian','lognormal']):
            concentrate2 = self.decoder_concentrate_second_omics(zs)
        else:
            concentrate2,_ = self.decoder_concentrate_second_omics(zs)

        return concentrate1,concentrate2

    def _count(self,concentrate1,concentrate2,total_counts_per_cell=1e6):
        if self.dist_model == 'dmm':
            concentrate1 = self.dirimulti_param(concentrate1)
            theta1 = dist.DirichletMultinomial(total_count=1, concentration=concentrate1).mean
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate1)
            theta1 = dist.Multinomial(total_count=1, probs=probs).mean

        if self.dist_model == 'dmm':
            concentrate2 = self.dirimulti_param(concentrate2)
            theta2 = dist.DirichletMultinomial(total_count=1, concentration=concentrate2).mean
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate2)
            theta2 = dist.Multinomial(total_count=1, probs=probs).mean

        counts1 = theta1 * total_counts_per_cell
        counts2 = theta2 * total_counts_per_cell
        return counts1,counts2
    
    def _count_sample_bk(self,concentrate1,concentrate2,total_counts_per_cell=1e6):
        if self.dist_model == 'dmm':
            concentrate1 = self.dirimulti_param(concentrate1)
            counts1 = dist.DirichletMultinomial(total_count=total_counts_per_cell, concentration=concentrate1).sample()
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate1)
            counts1 = dist.Multinomial(total_count=total_counts_per_cell, probs=probs).sample()

        if self.dist_model == 'dmm':
            concentrate2 = self.dirimulti_param(concentrate2)
            counts2 = dist.DirichletMultinomial(total_count=total_counts_per_cell, concentration=concentrate2).sample()
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate2)
            counts2 = dist.Multinomial(total_count=total_counts_per_cell, probs=probs).sample()

        return counts1,counts2
    
    def _count_sample_poisson(self,concentrate1,concentrate2,total_counts_per_cell=1e4):
        counts1,counts2 = self._count(concentrate1=concentrate1, concentrate2=concentrate2, total_counts_per_cell=total_counts_per_cell)
        counts1 = dist.Poisson(rate=counts1).to_event(1).sample()
        counts2 = dist.Poisson(rate=counts2).to_event(1).sample()
        return counts1,counts2
    
    def _count_sample(self,concentrate1,concentrate2,total_count=1e3):
        theta1,theta2 = self._count(concentrate1=concentrate1, concentrate2=concentrate2, total_counts_per_cell=1)
        counts1 = dist.NegativeBinomial(total_count=int(total_count), probs=theta1).to_event(1).sample()
        counts2 = dist.NegativeBinomial(total_count=int(total_count), probs=theta2).to_event(1).sample()
        return counts1,counts2
    
    def get_cell_expressions(self, xs, xs2, batch_size=1024, use_decoder=False, soft_assign=True):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs,xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E1,E2 = [],[]
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                zns = self._get_cell_coordinates(batch_x1, batch_x2, use_decoder=use_decoder, soft_assign=soft_assign)
                concentrate1,concentrate2 = self._expression(zns)
                E1.append(tensor_to_numpy(concentrate1))
                E2.append(tensor_to_numpy(concentrate2))
                pbar.update(1)
        
        E1 = np.concatenate(E1)
        E2 = np.concatenate(E2)
        return E1,E2

    def get_cell_counts(self, xs, xs2, total_count=1e3, total_counts_per_cell=1e4, batch_size=1024, use_decoder=False, soft_assign=True, use_sampler=False, sample_method='nb'):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs, xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        E1,E2 = [],[]
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                zns = self._get_cell_coordinates(batch_x1, batch_x2, use_decoder=use_decoder, soft_assign=soft_assign)
                concentrate1,concentrate2 = self._expression(zns)
                if use_sampler:
                    if sample_method.lower() == 'nb':
                        counts1,counts2 = self._count_sample(concentrate1,concentrate2,total_count)
                    elif sample_method.lower() == 'poisson':
                        counts1,counts2 = self._count_sample_poisson(concentrate1, concentrate2,total_counts_per_cell)
                else:
                    counts1,counts2 = self._count(concentrate1,concentrate2,total_counts_per_cell)
                E1.append(tensor_to_numpy(counts1))
                E2.append(tensor_to_numpy(counts2))
                pbar.update(1)
        
        E1 = np.concatenate(E1)
        E2 = np.concatenate(E2)
        return E1,E2
    
    def scale_data(self, xs, xs2, batch_size=1024):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs,xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E1,E2 = [],[]
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                zns = self._get_cell_coordinates(batch_x1, batch_x2, use_decoder=False, soft_assign=False)
                concentrate1,concentrate2 = self._expression(zns)
                E1.append(tensor_to_numpy(concentrate1))
                E2.append(tensor_to_numpy(concentrate2))
                pbar.update(1)
        
        E1 = np.concatenate(E1)
        E2 = np.concatenate(E2)
        return E1,E2
    
    def generate_scaled_data(self, zs, batch_size=1024):
        zs = convert_to_tensor(zs, device=self.get_device())
        dataset = CustomDataset(zs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E1,E2 = [],[]
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_z, _ in dataloader:
                concentrate1,concentrate2 = self._expression(batch_z)
                E1.append(tensor_to_numpy(concentrate1))
                E2.append(tensor_to_numpy(concentrate2))
                pbar.update(1)
        
        E1 = np.concatenate(E1)
        E2 = np.concatenate(E2)
        return E1,E2
    
    def generate_count_data(self, zs, batch_size=1024, total_counts_per_cell=1e4, sample_method='nb'):
        zs = convert_to_tensor(zs, device=self.get_device())
        dataset = CustomDataset(zs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E1,E2 = [],[]
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_z, _ in dataloader:
                concentrate1,concentrate2 = self._expression(batch_z)
                if sample_method.lower() == 'nb':
                    counts1,counts2 = self._count_sample(concentrate1, concentrate2)
                elif sample_method.lower() == 'poisson':
                    counts1,counts2 = self._count_sample_poisson(concentrate1, concentrate2, total_counts_per_cell)
                E1.append(tensor_to_numpy(counts1))
                E2.append(tensor_to_numpy(counts2))
                pbar.update(1)
        
        E1 = np.concatenate(E1)
        E2 = np.concatenate(E2)
        return E1,E2
    
    def log_prob(self, xs, xs2, batch_size=1024):
        xs = convert_to_tensor(xs, device=self.get_device())
        xs2 = convert_to_tensor(xs2, device=self.get_device())
        dataset = CustomMultiOmicsDataset(xs, xs2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        codebook_loc,codebook_scale = self._get_codebook()

        log_prob_sum = 0.0
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x1, batch_x2, _ in dataloader:
                z_q = self._get_cell_coordinates(batch_x1, batch_x2)
                z_a = self._soft_assignments(batch_x1, batch_x2)
                z_p_loc = torch.matmul(z_a, codebook_loc)
                z_p_scale = torch.matmul(z_a, codebook_scale)
                log_prob_sum += dist.Normal(z_p_loc, z_p_scale).to_event(1).log_prob(z_q).sum()
                pbar.update(1)
        
        return tensor_to_numpy(log_prob_sum)
    
    def latent_log_prob(self, zs, batch_size=1024):
        zs = convert_to_tensor(zs, device=self.get_device())
        dataset = CustomDataset(zs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        codebook_loc,codebook_scale = self._get_codebook()

        log_prob_sum = 0.0
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_z, _ in dataloader:
                z_q = batch_z
                z_a = self.encoder_n(z_q)
                z_a = self.softmax(z_a)
                z_p_loc = torch.matmul(z_a, codebook_loc)
                z_p_scale = torch.matmul(z_a, codebook_scale)
                log_prob_sum += dist.Normal(z_p_loc, z_p_scale).to_event(1).log_prob(z_q).sum()
                pbar.update(1)
        
        return tensor_to_numpy(log_prob_sum)
    
    def fit(self, xs, xs2, us=None,
            num_epochs=200, learning_rate=0.0001, batch_size=256, 
            algo='adam', beta_1=0.9, weight_decay=0.005, decay_rate=0.9,
            config_enum = 'parallel',
            use_jax = False):
        
        xs = convert_to_tensor(xs, dtype=self.dtype, device=self.get_device())
        xs2 = convert_to_tensor(xs2, dtype=self.dtype, device=self.get_device())
        if us is not None:
            us = convert_to_tensor(us, dtype=self.dtype, device=self.get_device())

        dataset = CustomMultiOmicsDataset2(xs, xs2, us)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # setup the optimizer
        optim_params = {'lr': learning_rate, 'betas': (beta_1, 0.999), 'weight_decay': weight_decay}

        if algo.lower()=='rmsprop':
            optimizer = torch.optim.RMSprop
        elif algo.lower()=='adam':
            optimizer = torch.optim.Adam
        elif algo.lower() == 'adamw':
            optimizer = torch.optim.AdamW
        else:
            raise ValueError("An optimization algorithm must be specified.")
        scheduler = ExponentialLR({'optimizer': optimizer, 'optim_args': optim_params, 'gamma': decay_rate})

        pyro.clear_param_store()

        # set up the loss(es) for inference, wrapping the guide in config_enumerate builds the loss as a sum
        # by enumerating each class label form the sampled discrete categorical distribution in the model
        Elbo = JitTraceEnum_ELBO if use_jax else TraceEnum_ELBO
        elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
        if us is None:
            guide = config_enumerate(self.guide1, config_enum, expand=True)
            loss_basic = SVI(self.model1, guide, scheduler, loss=elbo)
        else:
            guide = config_enumerate(self.guide2, config_enum, expand=True)
            loss_basic = SVI(self.model2, guide, scheduler, loss=elbo)

        # build a list of all losses considered
        losses = [loss_basic]
        num_losses = len(losses)

        with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
            for epoch in range(num_epochs):
                epoch_losses = [0.0] * num_losses
                for batch_x1, batch_x2, batch_u, _ in dataloader:
                    if us is None:
                        batch_u = None

                    for loss_id in range(num_losses):
                        if batch_u is None:
                            new_loss = losses[loss_id].step(batch_x1, batch_x2)
                        else:
                            new_loss = losses[loss_id].step(batch_x1, batch_x2, batch_u)
                        epoch_losses[loss_id] += new_loss

                avg_epoch_losses_ = map(lambda v: v / len(dataloader), epoch_losses)
                avg_epoch_losses = map(lambda v: "{:.4f}".format(v), avg_epoch_losses_)

                # store the loss
                str_loss = " ".join(map(str, avg_epoch_losses))

                # Update progress bar
                pbar.set_postfix({'loss': str_loss})
                pbar.update(1)
        
        assigns = self.soft_assignments(xs, xs2)
        assigns = convert_to_tensor(assigns, dtype=self.dtype, device=self.get_device())
        self.codebook_weights = torch.sum(assigns, dim=0)
        self.codebook_weights = self.codebook_weights / torch.sum(self.codebook_weights)
        #self.inverse_dispersion = pyro.param('inverse_dispersion').item()
        #self.dof = pyro.param('dof').item()

    @classmethod
    def save_model(cls, model, file_path, compression=False):
        """Save the model to the specified file path."""
        file_path = os.path.abspath(file_path)

        model.eval()
        if compression:
            with gzip.open(file_path, 'wb') as pickle_file:
                pickle.dump(model, pickle_file)
        else:
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(model, pickle_file)

        print(f'Model saved to {file_path}')

    @classmethod
    def load_model(cls, file_path):
        """Load the model from the specified file path and return an instance."""
        print(f'Model loaded from {file_path}')

        file_path = os.path.abspath(file_path)
        if file_path.endswith('gz'):
            with gzip.open(file_path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        else:
            with open(file_path, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        
        return model


EXAMPLE_RUN = (
    "example run: SUREMO --help"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SUREMO\n{}".format(EXAMPLE_RUN))

    parser.add_argument(
        "--cuda", action="store_true", help="use GPU(s) to speed up training"
    )
    parser.add_argument(
        "--jit", action="store_true", help="use PyTorch jit to speed up training"
    )
    parser.add_argument(
        "-n", "--num-epochs", default=200, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "-enum",
        "--enum-discrete",
        default="parallel",
        help="parallel, sequential or none. uses parallel enumeration by default",
    )
    parser.add_argument(
        "--omic1-file",
        default=None,
        type=str,
        help="the data file of first omics",
    )
    parser.add_argument(
        "--omic2-file",
        default=None,
        type=str,
        help="the data file for second omics",
    )
    parser.add_argument(
        "-undesired",
        "--undesired-factor-file",
        default=None,
        type=str,
        help="the file for the record of undesired factors",
    )
    parser.add_argument(
        "-delta",
        "--delta",
        default=0.0,
        type=float,
        help="penalty weight for zero inflation loss",
    )
    parser.add_argument(
        "-64",
        "--float64",
        action="store_true",
        help="use double float precision",
    )
    parser.add_argument(
        "--z-dist",
        default='normal',
        type=str,
        choices=['normal','laplacian','studentt','cauchy'],
        help="distribution model for latent representation",
    )
    parser.add_argument(
        "-dof",
        "--degree-of-freedom",
        default=8,
        type=int,
        help="degree of freedom for Student-t distribution",
    )
    parser.add_argument(
        "-cs",
        "--codebook-size",
        default=100,
        type=int,
        help="size of vector quantization codebook",
    )
    parser.add_argument(
        "-zd",
        "--z-dim",
        default=10,
        type=int,
        help="size of the tensor representing the latent variable z variable",
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        nargs="+",
        default=[500],
        type=int,
        help="a tuple (or list) of MLP layers to be used in the neural networks "
        "representing the parameters of the distributions in our model",
    )
    parser.add_argument(
        "-hla",
        "--hidden-layer-activation",
        default='relu',
        type=str,
        choices=['relu','softplus','leakyrelu','linear'],
        help="activation function for hidden layers",
    )
    parser.add_argument(
        "-plf",
        "--post-layer-function",
        nargs="+",
        default=['layernorm'],
        type=str,
        help="post functions for hidden layers, could be none, dropout, layernorm, batchnorm, or combination, default is 'dropout layernorm'",
    )
    parser.add_argument(
        "-paf",
        "--post-activation-function",
        nargs="+",
        default=['none'],
        type=str,
        help="post functions for activation layers, could be none or dropout, default is 'none'",
    )
    parser.add_argument(
        "-id",
        "--inverse-dispersion",
        default=10.0,
        type=float,
        help="inverse dispersion prior for negative binomial",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.0001,
        type=float,
        help="learning rate for Adam optimizer",
    )
    parser.add_argument(
        "-dr",
        "--decay-rate",
        default=0.9,
        type=float,
        help="decay rate for Adam optimizer",
    )
    parser.add_argument(
        "--layer-dropout-rate",
        default=0.1,
        type=float,
        help="droput rate for neural networks",
    )
    parser.add_argument(
        "-b1",
        "--beta-1",
        default=0.95,
        type=float,
        help="beta-1 parameter for Adam optimizer",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=1000,
        type=int,
        help="number of cells to be considered in a batch",
    )
    parser.add_argument(
        "-gp",
        "--gate-prior",
        default=0.6,
        type=float,
        help="gate prior for zero-inflated model",
    )
    parser.add_argument(
        "-likeli1",
        "--likelihood1",
        default='negbinomial',
        type=str,
        choices=['negbinomial','multinomial','poisson', 'gaussian','lognormal'],
        help="specify the distribution likelihood function for first omics",
    )
    parser.add_argument(
        "-likeli2",
        "--likelihood2",
        default='negbinomial',
        type=str,
        choices=['negbinomial','multinomial','poisson', 'gaussian','lognormal'],
        help="specify the distribution likelihood function for second omics",
    )
    parser.add_argument(
        "-dirichlet",
        "--use-dirichlet",
        action="store_true",
        help="use Dirichlet distribution over gene frequency",
    )
    parser.add_argument(
        "-mass",
        "--dirichlet-mass",
        default=5,
        type=float,
        help="mass param for dirichlet model",
    )
    parser.add_argument(
        "-zi1",
        "--zero-inflation1",
        default='exact',
        type=str,
        choices=['none','exact','inexact'],
        help="use zero-inflated estimation",
    )
    parser.add_argument(
        "-zi2",
        "--zero-inflation2",
        default='exact',
        type=str,
        choices=['none','exact','inexact'],
        help="use zero-inflated estimation",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for controlling randomness in this example",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        type=str,
        help="path to save model for prediction",
    )
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    assert (
        (args.omic1_file is not None) and (os.path.exists(args.omic1_file)) and \
            (args.omic2_file is not None) and (os.path.exists(args.omic2_file))
    ), "omic1_file and mic2_file must be provided"

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.float64:
        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        dtype = torch.float32
        torch.set_default_dtype(torch.float32)

    # prepare dataloaders
    xs1 = dt.fread(file=args.omic1_file, header=True).to_numpy()
    xs2 = dt.fread(file=args.omic2_file, header=True).to_numpy()
    us = None 
    if args.undesired_factor_file is not None:
        us = dt.fread(file=args.undesired_factor_file, header=True).to_numpy()

    undesired_size =  0 if us is None else us.shape[1]
    input_size1 = xs1.shape[1]
    input_size2 = xs2.shape[1]

    latent_dist = args.z_dist

    #######################################
    suremo = SUREMO(
        input_size1=input_size1,
        input_size2=input_size2,
        undesired_size=undesired_size,
        inverse_dispersion=args.inverse_dispersion,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_layers,
        hidden_layer_activation=args.hidden_layer_activation,
        use_cuda=args.cuda,
        config_enum=args.enum_discrete,
        use_dirichlet=args.use_dirichlet,
        zero_inflation1=args.zero_inflation1,
        zero_inflation2=args.zero_inflation2,
        gate_prior=args.gate_prior,
        delta=args.delta,
        loss_func1=args.likelihood1,
        loss_func2=args.likelihood2,
        dirichlet_mass=args.dirichlet_mass,
        nn_dropout=args.layer_dropout_rate,
        post_layer_fct=args.post_layer_function,
        post_act_fct=args.post_activation_function,
        codebook_size=args.codebook_size,
        latent_dist = latent_dist,
        studentt_dof = args.degree_of_freedom,
        dtype=dtype,
    )

    suremo.fit(xs1, xs2, us, 
               num_epochs=args.num_epochs,
               learning_rate=args.learning_rate,
               batch_size=args.batch_size,
               beta_1=args.beta_1,
               decay_rate=args.decay_rate,
               use_jax=args.jit,
               config_enum=args.enum_discrete,
             )

    if args.save_model is not None:
        SUREMO.save_model(suremo, args.save_model)




if __name__ == "__main__":
    main()
