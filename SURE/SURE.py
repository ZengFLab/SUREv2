import pyro
import pyro.distributions as dist
from pyro.optim import ExponentialLR
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from torch.distributions import constraints
from torch.distributions.transforms import SoftmaxTransform

from .utils.custom_mlp import MLP, Exp
from .utils.utils import CustomDataset, CustomDataset3, tensor_to_numpy, convert_to_tensor


import os
import argparse
import random
import numpy as np
import datatable as dt
from tqdm import tqdm

from typing import Literal

import warnings
warnings.filterwarnings("ignore")

import dill as pickle
import gzip
from packaging.version import Version
torch_version = torch.__version__


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

class SURE(nn.Module):
    """SUccinct REpresentation of single-omics cells

    Parameters
    ----------
    inpute_size
        Number of features (e.g., genes, peaks, proteins, etc.) per cell.
    undesired_size
        Number of undesired factors. It would be used to adjust for undesired variations like batch effect.
    codebook_size
        Number of metacells.
    z_dim
        Dimensionality of latent states and metacells. 
    hidden_layers
        A list give the numbers of neurons for each hidden layer.
    loss_func
        The likelihood model for single-cell data generation. 
        
        One of the following: 
        * ``'negbinomial'`` -  negative binomial distribution (default)
        * ``'poisson'`` - poisson distribution
        * ``'multinomial'`` - multinomial distribution
    user_dirichlet
        A boolean option. If toggled on, SURE characterizes single-cell data using a hierarchical model, such as
        dirichlet-negative binomial.
    latent_dist
        The distribution model for latent states. 
        
        One of the following:
        * ``'normal'`` - normal distribution
        * ``'laplacian'`` - Laplacian distribution
        * ``'studentt'`` - Student-t distribution. 
    use_cuda
        A boolean option for switching on cuda device. 

    Examples
    --------
    >>>
    >>>
    >>>

    """
    def __init__(self,
                 input_size: int = 2000,
                 undesired_size: int = 2,
                 codebook_size: int = 200,
                 z_dim: int = 10,
                 hidden_layers: list = [500],
                 hidden_layer_activation: Literal['relu','softplus','leakyrelu','linear'] = 'relu',
                 use_dirichlet: bool = True,
                 dirichlet_mass: float = 1.0,
                 loss_func: Literal['negbinomial','poisson','multinomial','gaussian'] = 'negbinomial',
                 inverse_dispersion: float = 10.0,
                 nn_dropout: float = 0.1,
                 zero_inflation: Literal['exact','inexact','none'] = 'exact',
                 gate_prior: float = 0.6,
                 delta: float = 0.0,
                 post_layer_fct: list = ['layernorm'],
                 post_act_fct: list = None,
                 latent_dist: Literal['normal','studentt','laplacian'] = 'normal',
                 studentt_dof: float = 8,
                 config_enum: str = 'parallel',
                 use_cuda: bool = False,
                 dtype: torch.float32 = torch.float32, # type: ignore
                 ):
        super().__init__()

        self.input_size = input_size
        self.undesired_size = undesired_size
        self.inverse_dispersion = inverse_dispersion
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.decoder_hidden_layers = hidden_layers[::-1]
        self.use_undesired = True if self.undesired_size>0 else False
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.delta = delta
        self.loss_func = loss_func
        self.dirimulti_mass = dirichlet_mass
        self.options = None
        self.code_size=codebook_size

        self.use_studentt = False
        self.use_laplacian=False
        if latent_dist.lower() in ['laplacian']:
            self.use_laplacian=True
        elif latent_dist.lower() in ['studentt','student-t','t']:
            self.use_studentt = True 
        self.dof = studentt_dof
        self.dtype = dtype

        self.dist_model = 'dmm' if use_dirichlet else 'mm'

        self.use_zeroinflate=False
        self.use_exact_zeroinflate=False
        if zero_inflation=='exact':
            self.use_exact_zeroinflate=True
            self.use_zeroinflate=True
        elif zero_inflation=='inexact':
            self.use_zeroinflate=True

        if self.loss_func.lower() in ['negbinom','nb','negb']:
            self.loss_func = 'negbinomial'

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

        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

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
            [self.input_size] + hidden_sizes + [[z_dim, z_dim]],
            activation=activate_fct,
            output_activation=[None, Exp],
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        if self.use_undesired:
            if self.loss_func in ['gaussian','lognormal']:
                self.decoder_concentrate = MLP(
                    [self.undesired_size + self.z_dim] + self.decoder_hidden_layers + [[self.input_size, self.input_size]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate = MLP(
                    [self.undesired_size + self.z_dim] + self.decoder_hidden_layers + [self.input_size],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate = MLP(
                [self.undesired_size + self.z_dim] + hidden_sizes + [[self.input_size, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )
        else:
            if self.loss_func in ['gaussian','lognormal']:
                self.decoder_concentrate = MLP(
                    [self.z_dim] + self.decoder_hidden_layers + [[self.input_size, self.input_size]],
                    activation=activate_fct,
                    output_activation=[None, Exp],
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )
            else:
                self.decoder_concentrate = MLP(
                    [self.z_dim] + self.decoder_hidden_layers + [self.input_size],
                    activation=activate_fct,
                    output_activation=None,
                    post_layer_fct=post_layer_fct,
                    post_act_fct=post_act_fct,
                    allow_broadcast=self.allow_broadcast,
                    use_cuda=self.use_cuda,
                )

            self.encoder_gate = MLP(
                [self.z_dim] + hidden_sizes + [[self.input_size, 1]],
                activation=activate_fct,
                output_activation=[None, Exp],
                post_layer_fct=post_layer_fct,
                post_act_fct=post_act_fct,
                allow_broadcast=self.allow_broadcast,
                use_cuda=self.use_cuda,
            )

        self.codebook = MLP(
            [self.code_size] + hidden_sizes + [[z_dim,z_dim]],
            activation=activate_fct,
            output_activation=[None,Exp],
            post_layer_fct=post_layer_fct,
            post_act_fct=post_act_fct,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

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

    def model1(self, xs):
        pyro.module('sure', self)

        total_count = pyro.param("inverse_dispersion", self.inverse_dispersion *
                                 xs.new_ones(self.input_size), constraint=constraints.positive)
        
        if self.use_studentt:
            dof = pyro.param("dof", self.dof * xs.new_ones(self.z_dim), 
                             constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        self.options = dict(dtype=xs.dtype, device=xs.device)

        I = torch.eye(self.code_size)
        acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior))

            prior_loc = torch.matmul(ns,acs_loc)
            prior_scale = torch.matmul(ns,acs_scale)

            if self.use_studentt:
                zns = pyro.sample('zn', dist.StudentT(df=dof, loc=prior_loc, scale=prior_scale).to_event(1))
            elif self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            zs = zns

            if self.loss_func == 'gaussian':
                concentrate_loc, concentrate_scale = self.decoder_concentrate(zs)
                concentrate = concentrate_loc
            else:
                concentrate = self.decoder_concentrate(zs)

            if self.dist_model == 'dmm':
                concentrate = self.dirimulti_param(concentrate)
                theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
            elif self.dist_model == 'mm':
                probs = self.multi_param(concentrate)
                theta = dist.Multinomial(total_count=1, probs=probs).mean

            if self.use_zeroinflate:
                gate_loc = self.gate_prior * torch.ones(batch_size, self.input_size, **self.options)
                gate_scale = torch.ones(batch_size, self.input_size, **self.options)
                gate_logits = pyro.sample('gate_logit', dist.Normal(gate_loc, gate_scale).to_event(1))
                gate_probs = self.sigmoid(gate_logits)

                if self.use_exact_zeroinflate:
                    if self.loss_func == 'multinomial':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)
                else:
                    if self.loss_func != 'gaussian':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **self.options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one', dist.Binomial(probs=1-gate_probs).to_event(1), obs=ones)

            if self.loss_func == 'negbinomial':
                theta = self.cutoff(theta, thresh=eps)
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count, probs=theta),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count, probs=theta).to_event(1), obs=xs)
            elif self.loss_func == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta), obs=xs)
            elif self.loss_func == 'gaussian':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)
            elif self.loss_func == 'lognormal':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)

    def guide1(self, xs):
        with pyro.plate('data'):
            zn_loc, zn_scale = self.encoder_zn(xs)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            alpha = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha))
            
            if self.use_zeroinflate:
                zs=zns
                loc, scale = self.encoder_gate(zs)
                scale = self.cutoff(scale)
                gates_logit = pyro.sample('gate_logit', dist.Normal(loc, scale).to_event(1))

    def model2(self, xs, us=None):
        pyro.module('sure', self)

        total_count = pyro.param("inverse_dispersion", self.inverse_dispersion *
                                 xs.new_ones(self.input_size), constraint=constraints.positive)
        
        if self.use_studentt:
            dof = pyro.param("dof", self.dof * xs.new_ones(self.z_dim), 
                             constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        self.options = dict(dtype=xs.dtype, device=xs.device)

        I = torch.eye(self.code_size)
        acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior))

            prior_loc = torch.matmul(ns,acs_loc)
            prior_scale = torch.matmul(ns,acs_scale)

            if self.use_studentt:
                zns = pyro.sample('zn', dist.StudentT(df=dof, loc=prior_loc, scale=prior_scale).to_event(1))
            elif self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            if self.use_undesired:
                zs = [us, zns]
            else:
                zs = zns

            if self.loss_func == 'gaussian':
                concentrate_loc, concentrate_scale = self.decoder_concentrate(zs)
                concentrate = concentrate_loc
            else:
                concentrate = self.decoder_concentrate(zs)

            if self.dist_model == 'dmm':
                concentrate = self.dirimulti_param(concentrate)
                theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
            elif self.dist_model == 'mm':
                probs = self.multi_param(concentrate)
                theta = dist.Multinomial(total_count=1, probs=probs).mean

            if self.use_zeroinflate:
                gate_loc = self.gate_prior * torch.ones(batch_size, self.input_size, **self.options)
                gate_scale = torch.ones(batch_size, self.input_size, **self.options)
                gate_logits = pyro.sample('gate_logit', dist.Normal(gate_loc, gate_scale).to_event(1))
                gate_probs = self.sigmoid(gate_logits)

                if self.use_exact_zeroinflate:
                    if self.loss_func == 'multinomial':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)
                else:
                    if self.loss_func != 'gaussian':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **self.options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one', dist.Binomial(probs=1-gate_probs).to_event(1), obs=ones)

            if self.loss_func == 'negbinomial':
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count, probs=theta),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count, probs=theta).to_event(1), obs=xs)
            elif self.loss_func == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta), obs=xs)
            elif self.loss_func == 'gaussian':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)
            elif self.loss_func == 'lognormal':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)

    def guide2(self, xs, us=None):
        with pyro.plate('data'):
            zn_loc, zn_scale = self.encoder_zn(xs)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            alpha = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha))
            
            if self.use_zeroinflate:
                if self.use_undesired:
                    zs=[us,zns]
                else:
                    zs=zns
                loc, scale = self.encoder_gate(zs)
                scale = self.cutoff(scale)
                gates_logit = pyro.sample('gate_logit', dist.Normal(loc, scale).to_event(1))

    def model3(self, xs, ys):
        pyro.module('sure', self)

        total_count = pyro.param("inverse_dispersion", self.inverse_dispersion *
                                 xs.new_ones(self.input_size), constraint=constraints.positive)
        
        if self.use_studentt:
            dof = pyro.param("dof", self.dof * xs.new_ones(self.z_dim), 
                             constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        self.options = dict(dtype=xs.dtype, device=xs.device)

        I = torch.eye(self.code_size)
        acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior), obs=ys)

            prior_loc = torch.matmul(ns,acs_loc)
            prior_scale = torch.matmul(ns,acs_scale)

            if self.use_studentt:
                zns = pyro.sample('zn', dist.StudentT(df=dof, loc=prior_loc, scale=prior_scale).to_event(1))
            elif self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            zs = zns

            if self.loss_func == 'gaussian':
                concentrate_loc, concentrate_scale = self.decoder_concentrate(zs)
                concentrate = concentrate_loc
            else:
                concentrate = self.decoder_concentrate(zs)

            if self.dist_model == 'dmm':
                concentrate = self.dirimulti_param(concentrate)
                theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
            elif self.dist_model == 'mm':
                probs = self.multi_param(concentrate)
                theta = dist.Multinomial(total_count=1, probs=probs).mean

            if self.use_zeroinflate:
                gate_loc = self.gate_prior * torch.ones(batch_size, self.input_size, **self.options)
                gate_scale = torch.ones(batch_size, self.input_size, **self.options)
                gate_logits = pyro.sample('gate_logit', dist.Normal(gate_loc, gate_scale).to_event(1))
                gate_probs = self.sigmoid(gate_logits)

                if self.use_exact_zeroinflate:
                    if self.loss_func == 'multinomial':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)
                else:
                    if self.loss_func != 'gaussian':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **self.options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one', dist.Binomial(probs=1-gate_probs).to_event(1), obs=ones)

            if self.loss_func == 'negbinomial':
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count, probs=theta),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count, probs=theta).to_event(1), obs=xs)
            elif self.loss_func == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta), obs=xs)
            elif self.loss_func == 'gaussian':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)
            elif self.loss_func == 'lognormal':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)

    def guide3(self, xs, ys):
        with pyro.plate('data'):
            zn_loc, zn_scale = self.encoder_zn(xs)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            alpha = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha))
            
            if self.use_zeroinflate:
                zs=zns
                loc, scale = self.encoder_gate(zs)
                scale = self.cutoff(scale)
                gates_logit = pyro.sample('gate_logit', dist.Normal(loc, scale).to_event(1))

    def model4(self, xs, us, ys):
        pyro.module('sure', self)

        total_count = pyro.param("inverse_dispersion", self.inverse_dispersion *
                                 xs.new_ones(self.input_size), constraint=constraints.positive)
        
        if self.use_studentt:
            dof = pyro.param("dof", self.dof * xs.new_ones(self.z_dim), 
                             constraint=constraints.positive)

        eps = torch.finfo(xs.dtype).eps
        batch_size = xs.size(0)
        self.options = dict(dtype=xs.dtype, device=xs.device)

        I = torch.eye(self.code_size)
        acs_loc,acs_scale = self.codebook(I)

        with pyro.plate('data'):
            prior = torch.zeros(batch_size, self.code_size, **self.options)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=prior), obs=ys)

            prior_loc = torch.matmul(ns,acs_loc)
            prior_scale = torch.matmul(ns,acs_scale)

            if self.use_studentt:
                zns = pyro.sample('zn', dist.StudentT(df=dof, loc=prior_loc, scale=prior_scale).to_event(1))
            elif self.use_laplacian:
                zns = pyro.sample('zn', dist.Laplace(prior_loc, prior_scale).to_event(1))
            else:
                zns = pyro.sample('zn', dist.Normal(prior_loc, prior_scale).to_event(1))

            zs = [us, zns]

            if self.loss_func == 'gaussian':
                concentrate_loc, concentrate_scale = self.decoder_concentrate(zs)
                concentrate = concentrate_loc
            else:
                concentrate = self.decoder_concentrate(zs)

            if self.dist_model == 'dmm':
                concentrate = self.dirimulti_param(concentrate)
                theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
            elif self.dist_model == 'mm':
                probs = self.multi_param(concentrate)
                theta = dist.Multinomial(total_count=1, probs=probs).mean

            if self.use_zeroinflate:
                gate_loc = self.gate_prior * torch.ones(batch_size, self.input_size, **self.options)
                gate_scale = torch.ones(batch_size, self.input_size, **self.options)
                gate_logits = pyro.sample('gate_logit', dist.Normal(gate_loc, gate_scale).to_event(1))
                gate_probs = self.sigmoid(gate_logits)

                if self.use_exact_zeroinflate:
                    if self.loss_func == 'multinomial':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)
                else:
                    if self.loss_func != 'gaussian':
                        theta = probs_to_logits(theta) + probs_to_logits(1-gate_probs)
                        theta = logits_to_probs(theta)

                if self.delta > 0:
                    ones = torch.zeros(batch_size, self.input_size, **self.options)
                    ones[xs > 0] = 1
                    with pyro.poutine.scale(scale=self.delta):
                        ones = pyro.sample('one', dist.Binomial(probs=1-gate_probs).to_event(1), obs=ones)

            if self.loss_func == 'negbinomial':
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.NegativeBinomial(total_count=total_count, probs=theta),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.NegativeBinomial(total_count=total_count, probs=theta).to_event(1), obs=xs)
            elif self.loss_func == 'poisson':
                rate = xs.sum(1).unsqueeze(-1) * theta
                if self.use_exact_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Exponential(rate=rate),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Exponential(rate=rate).to_event(1), obs=xs)
            elif self.loss_func == 'multinomial':
                pyro.sample('x', dist.Multinomial(total_count=int(1e8), probs=theta), obs=xs)
            elif self.loss_func == 'gaussian':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.Normal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.Normal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)
            elif self.loss_func == 'lognormal':
                if self.use_zeroinflate:
                    pyro.sample('x', dist.ZeroInflatedDistribution(dist.LogNormal(concentrate_loc, concentrate_scale),gate_logits=gate_logits).to_event(1), obs=xs)
                else:
                    pyro.sample('x', dist.LogNormal(concentrate_loc, concentrate_scale).to_event(1), obs=xs)

    def guide4(self, xs, us, ys):
        with pyro.plate('data'):
            zn_loc, zn_scale = self.encoder_zn(xs)
            zns = pyro.sample('zn', dist.Normal(zn_loc, zn_scale).to_event(1))

            alpha = self.encoder_n(zns)
            ns = pyro.sample('n', dist.OneHotCategorical(logits=alpha))
            
            if self.use_zeroinflate:
                zs=[us,zns]
                loc, scale = self.encoder_gate(zs)
                scale = self.cutoff(scale)
                gates_logit = pyro.sample('gate_logit', dist.Normal(loc, scale).to_event(1))

    def _get_metacell_coordinates(self):
        I = torch.eye(self.code_size, **self.options)
        cb,_ = self.codebook(I)
        return cb
    
    def get_metacell_coordinates(self):
        """
        Return the mean part of metacell codebook
        """
        cb = self._get_metacell_coordinates()
        cb = tensor_to_numpy(cb)
        return cb
    
    def _get_metacell_expressions(self):
        cbs = self._get_metacell_coordinates()
        concentrate = self._expression(cbs)
        return concentrate
    
    def get_metacell_expressions(self):
        """
        Return the scaled expression data of metacells
        """
        concentrate = self._get_metacell_expressions()
        concentrate = tensor_to_numpy(concentrate)
        return concentrate
    
    def get_metacell_counts(self, total_count=1e3, total_counts_per_item=1e6, use_sampler=False, sample_method='nb'):
        """
        Return the simulated count data of metacells
        """
        concentrate = self._get_metacell_expressions()
        if use_sampler:
            if sample_method.lower() == 'nb':
                counts = self._count_sample(concentrate, total_count=total_count)
            elif sample_method.lower() == 'poisson':
                counts = self._count_sample_poisson(concentrate, total_counts_per_item=total_counts_per_item)
        else:
            counts = self._count(concentrate, total_counts_per_cell=total_counts_per_item)
        counts = tensor_to_numpy(counts)
        return counts

    def _get_cell_coordinates(self, xs, use_decoder=False, soft_assign=False):
        if use_decoder:
            cb = self._get_metacell_coordinates()
            if soft_assign:
                A = self._soft_assignments(xs)
            else:
                A = self._hard_assignments(xs)
            zns = torch.matmul(A, cb)
        else:
            zns, _ = self.encoder_zn(xs)
        return zns 
    
    def get_cell_coordinates(self, 
                             xs, 
                             batch_size: int = 1024, 
                             use_decoder: bool = False, 
                             soft_assign: bool = False):
        """
        Return cells' latent representations

        Parameters
        ----------
        xs
            Single-cell expression matrix. It should be a Numpy array or a Pytorch Tensor.
        batch_size
            Size of batch processing.
        use_decoder
            If toggled on, the latent representations will be reconstructed from the metacell codebook
        soft_assign
            If toggled on, the assignments of cells will use probabilistic values.
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        Z = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                zns = self._get_cell_coordinates(X_batch, use_decoder=use_decoder, soft_assign=soft_assign)
                Z.append(tensor_to_numpy(zns))
                pbar.update(1)

        Z = np.concatenate(Z)
        return Z
    
    def _get_codebook(self):
        I = torch.eye(self.code_size, **self.options)
        cb_loc,cb_scale = self.codebook(I)
        return cb_loc,cb_scale
    
    def get_codebook(self):
        """
        Return the entire metacell codebook
        """
        cb_loc,cb_scale = self._get_codebook()
        return tensor_to_numpy(cb_loc),tensor_to_numpy(cb_scale)
    
    def _code(self, xs):
        zns,_ = self.encoder_zn(xs)
        alpha = self.encoder_n(zns)
        return alpha
    
    def _soft_assignments(self, xs):
        alpha = self._code(xs)
        alpha = self.softmax(alpha)
        return alpha
    
    def soft_assignments(self, xs, batch_size=1024):
        """
        Map cells to metacells and return the probabilistic values of metacell assignments
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                a = self._soft_assignments(X_batch)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A
    
    def _hard_assignments(self, xs):
        alpha = self._code(xs)
        res, ind = torch.topk(alpha, 1)
        ns = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ns
    
    def hard_assignments(self, xs, batch_size=1024):
        """
        Map cells to metacells and return the assigned metacell identities.
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        A = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                a = self._hard_assignments(X_batch)
                A.append(tensor_to_numpy(a))
                pbar.update(1)

        A = np.concatenate(A)
        return A
    
    def _expression(self,zns):
        if self.use_undesired:
            ks2 = torch.zeros(zns.shape[0], self.undesired_size, **self.options)
            zs=[ks2,zns]
        else:
            zs=zns

        if not (self.loss_func in ['gaussian','lognormal']):
            concentrate = self.decoder_concentrate(zs)
        else:
            concentrate,_ = self.decoder_concentrate(zs)

        return concentrate
    
    def _count(self,concentrate,total_counts_per_cell=1e6):
        if self.dist_model == 'dmm':
            concentrate = self.dirimulti_param(concentrate)
            theta = dist.DirichletMultinomial(total_count=1, concentration=concentrate).mean
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate)
            theta = dist.Multinomial(total_count=int(1), probs=probs).mean

        counts = theta * total_counts_per_cell
        return counts
    
    def _count_sample_bk(self,concentrate,total_counts_per_cell=1e6):
        if self.dist_model == 'dmm':
            concentrate = self.dirimulti_param(concentrate)
            counts = dist.DirichletMultinomial(total_count=total_counts_per_cell, concentration=concentrate).sample()
        elif self.dist_model == 'mm':
            probs = self.multi_param(concentrate)
            counts = dist.Multinomial(total_count=int(total_counts_per_cell), probs=probs).sample()

        return counts
    
    def _count_sample_poisson(self,concentrate,total_counts_per_cell=1e4):
        counts = self._count(concentrate=concentrate, total_counts_per_cell=total_counts_per_cell)
        counts = dist.Poisson(rate=counts).to_event(1).sample()
        return counts
    
    def _count_sample(self,concentrate,total_count=1e3):
        #theta = self._count(concentrate=concentrate, total_counts_per_cell=1)
        theta = self.sigmoid(concentrate)
        counts = dist.NegativeBinomial(total_count=int(total_count), probs=theta).to_event(1).sample()
        return counts
    
    def _count_sample_nb(self,concentrate,total_count=1e3):
        return self._count_sample(concentrate=concentrate, total_count=total_count)

    def get_cell_expressions(self, xs, 
                             batch_size: int = 1024, 
                             use_decoder: bool = False, 
                             soft_assign: bool = True):
        """
        Return the scaled expression data of input cells.

        Parameters
        ----------
        xs
            Single-cell expression matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are features.
        batch_size
            Size of batch processing
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                zns = self._get_cell_coordinates(X_batch, use_decoder=use_decoder, soft_assign=soft_assign)
                concentrate = self._expression(zns)
                E.append(tensor_to_numpy(concentrate))
                pbar.update(1)
        
        E = np.concatenate(E)
        return E
    
    def get_cell_counts(self, xs, 
                        total_count: float = 1e3, 
                        total_counts_per_cell: float = 1e6, 
                        batch_size: int = 1024, 
                        use_decoder: bool = False, 
                        soft_assign: bool = True, 
                        use_sampler: bool = False, 
                        sample_method: Literal['nb','negbinomial','poisson'] = 'nb'):
        """
        Return the simulated count data of input cells.

        Parameters
        ----------
        xs
            Single-cell expression matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are features.
        total_count
            Paramter for negative binomial distribution.
        total_counts_per_cell
            Parameter for poisson distribution.
        batch_size
            Size of batch processing.
        use_decoder
            If toggled on, it will use latent representations reconstructed from metacells.
        soft_assign
            If toggled on, soft metacell assignments will be used instead of hard assignments.
        use_sampler
            If toggled on, data generation is done by sampling from the distribution.
        sample_method
            Method for sampling data. It will draw samples from negative binomial distribution or poisson distribution.
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        E = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                zns = self._get_cell_coordinates(X_batch, use_decoder=use_decoder, soft_assign=soft_assign)
                concentrate = self._expression(zns)
                if use_sampler:
                    if sample_method.lower() in ['nb','negbinomial']:
                        counts = self._count_sample(concentrate,total_count)
                    elif sample_method.lower() in ['poisson']:
                        counts = self._count_sample_poisson(concentrate, total_counts_per_cell)
                else:
                    counts = self._count(concentrate,total_counts_per_cell)
                E.append(tensor_to_numpy(counts))
                pbar.update(1)
        
        E = np.concatenate(E)
        return E
    
    def scale_data(self, xs, 
                   batch_size: int = 1024):
        """
        Return the scaled expression data of input data.

        Parameters
        ----------
        xs
            Single-cell experssion matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are features.
        batch_size
            Size of batch processing.
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for X_batch, _ in dataloader:
                zns = self._get_cell_coordinates(X_batch, use_decoder=False, soft_assign=False)
                concentrate = self._expression(zns)
                E.append(tensor_to_numpy(concentrate))
                pbar.update(1)
        
        E = np.concatenate(E)
        return E
    
    def generate_scaled_data(self, zs, 
                             batch_size: int = 1024):
        """
        Return the scaled data of input latent representations.

        Parameters
        ----------
        zs
            Input latent representations. It should be a Numpy array or a Pytorch Tensor. Rows are cells.
        batch_size
            Size of batch processing.
        """
        zs = convert_to_tensor(zs, device=self.get_device())
        dataset = CustomDataset(zs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_z, _ in dataloader:
                concentrate = self._expression(batch_z)
                E.append(tensor_to_numpy(concentrate))
                pbar.update(1)
        
        E = np.concatenate(E)
        return E
    
    def generate_count_data(self, zs, 
                            batch_size: int = 1024, 
                            total_count: float = 1e3, 
                            total_counts_per_cell: float = 1e4, 
                            sample_method: Literal['nb','negbinomial','poisson'] = 'nb'):
        """
        Return the simulated data of input latent representations.

        Parameters
        ----------
        zs
            Input latent representations. It should be a Numpy array or a Pytorch Tensor. Rows are cells.
        batch_size
            Size of batch processing.
        total_count
            Parameter for negative binomial generator.
        total_counts_per_cell
            Parameter for poisson generator.
        sample_method
            Method for sampling data. It will draw samples from negative binomial distribution or poisson distribution.
        """
        zs = convert_to_tensor(zs, device=self.get_device())
        dataset = CustomDataset(zs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        E = []
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_z, _ in dataloader:
                concentrate = self._expression(batch_z)
                if sample_method.lower() == 'nb':
                    counts = self._count_sample(concentrate, total_count)
                elif sample_method.lower() == 'poisson':
                    counts = self._count_sample_poisson(concentrate, total_counts_per_cell)
                E.append(tensor_to_numpy(counts))
                pbar.update(1)
        
        E = np.concatenate(E)
        return E
    
    def log_prob(self, xs, 
                 batch_size: int = 1024):
        """
        Return probabilistic score of input data.

        Parameters
        ----------
        xs
            Single-cell experssion matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are features.
        batch_size
            Size of batch processing.
        """
        xs = convert_to_tensor(xs, device=self.get_device())
        dataset = CustomDataset(xs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        codebook_loc,codebook_scale = self._get_codebook()

        log_prob_sum = 0.0
        with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
            for batch_x, _ in dataloader:
                z_q = self._get_cell_coordinates(batch_x)
                z_a = self._soft_assignments(batch_x)
                z_p_loc = torch.matmul(z_a, codebook_loc)
                z_p_scale = torch.matmul(z_a, codebook_scale)
                log_prob_sum += dist.Normal(z_p_loc, z_p_scale).to_event(1).log_prob(z_q).sum()
                pbar.update(1)
        
        return tensor_to_numpy(log_prob_sum)
    
    def latent_log_prob(self, zs, 
                        batch_size: int = 1024):
        """
        Return probabilistic score of input latent representations.

        Parameters
        ----------
        zs
            Input latent representations. It should be a Numpy array or a Pytorch Tensor. Rows are cells.
        batch_size
            Size of batch processing.
        """
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
    
    def fit(self, xs, 
            us = None, 
            ys = None,
            num_epochs: int = 200, 
            learning_rate: float = 0.0001, 
            batch_size: int = 256, 
            algo: Literal['adam','rmsprop','adamw'] = 'adam', 
            beta_1: float = 0.9, 
            weight_decay: float = 0.005, 
            decay_rate: float = 0.9,
            config_enum: str = 'parallel',
            use_jax: bool = False):
        """
        Train the SURE model.

        Parameters
        ----------
        xs
            Single-cell experssion matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are features.
        us
            Undesired factor matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are undesired factors.
        ys
            Desired factor matrix. It should be a Numpy array or a Pytorch Tensor. Rows are cells and columns are desired factors.
        num_epochs
            Number of training epochs.
        learning_rate
            Parameter for training.
        batch_size
            Size of batch processing.
        algo
            Optimization algorithm.
        beta_1
            Parameter for optimization.
        weight_decay
            Parameter for optimization.
        decay_rate 
            Parameter for optimization.
        use_jax
            If toggled on, Jax will be used for speeding up. CAUTION: This will raise errors because of unknown reasons when it is called in
            the Python script or Jupyter notebook. It is OK if it is used when runing SURE in the shell command.
        """
        
        xs = convert_to_tensor(xs, dtype=self.dtype, device=self.get_device())
        if us is not None:
            us = convert_to_tensor(us, dtype=self.dtype, device=self.get_device())
        if ys is not None:
            ys = convert_to_tensor(ys, dtype=self.dtype, device=self.get_device())

        dataset = CustomDataset3(xs, us, ys)
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
            if ys is None:
                guide = config_enumerate(self.guide1, config_enum, expand=True)
                loss_basic = SVI(self.model1, guide, scheduler, loss=elbo)
            else:
                guide = config_enumerate(self.guide3, config_enum, expand=True)
                loss_basic = SVI(self.model3, guide, scheduler, loss=elbo)
        else:
            if ys is None:
                guide = config_enumerate(self.guide2, config_enum, expand=True)
                loss_basic = SVI(self.model2, guide, scheduler, loss=elbo)
            else:
                guide = config_enumerate(self.guide4, config_enum, expand=True)
                loss_basic = SVI(self.model4, guide, scheduler, loss=elbo)

        # build a list of all losses considered
        losses = [loss_basic]
        num_losses = len(losses)

        with tqdm(total=num_epochs, desc='Training', unit='epoch') as pbar:
            for epoch in range(num_epochs):
                epoch_losses = [0.0] * num_losses
                for batch_x, batch_u, batch_y, _ in dataloader:
                    if us is None:
                        batch_u = None
                    if ys is None:
                        batch_y = None

                    for loss_id in range(num_losses):
                        if batch_u is None:
                            if batch_y is None:
                                new_loss = losses[loss_id].step(batch_x)
                            else:
                                new_loss = losses[loss_id].step(batch_x, batch_y)
                        else:
                            if batch_y is None:
                                new_loss = losses[loss_id].step(batch_x, batch_u)
                            else:
                                new_loss = losses[loss_id].step(batch_x, batch_u, batch_y)
                        epoch_losses[loss_id] += new_loss

                avg_epoch_losses_ = map(lambda v: v / len(dataloader), epoch_losses)
                avg_epoch_losses = map(lambda v: "{:.4f}".format(v), avg_epoch_losses_)

                # store the loss
                str_loss = " ".join(map(str, avg_epoch_losses))

                # Update progress bar
                pbar.set_postfix({'loss': str_loss})
                pbar.update(1)
        
        assigns = self.soft_assignments(xs)
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
    "example run: SURE --help"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="SURE\n{}".format(EXAMPLE_RUN))

    parser.add_argument(
        "--cuda", action="store_true", help="use GPU(s) to speed up training"
    )
    parser.add_argument(
        "--jit", action="store_true", help="use PyTorch jit to speed up training"
    )
    parser.add_argument(
        "-n", "--num-epochs", default=40, type=int, help="number of epochs to run"
    )
    parser.add_argument(
        "-enum",
        "--enum-discrete",
        default="parallel",
        help="parallel, sequential or none. uses parallel enumeration by default",
    )
    parser.add_argument(
        "-data",
        "--data-file",
        default=None,
        type=str,
        help="the data file",
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
        "-la",
        "--laplace",
        action="store_true",
        help="use laplace distribution for latent representation",
    )
    parser.add_argument(
        "-st",
        "--student-t",
        action="store_true",
        help="use Student-t distribution for latent representation",
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
        "-likeli",
        "--likelihood",
        default='negbinomial',
        type=str,
        choices=['negbinomial', 'multinomial', 'poisson', 'gaussian','lognormal'],
        help="specify the distribution likelihood function",
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
        default=1,
        type=float,
        help="mass param for dirichlet model",
    )
    parser.add_argument(
        "-zi",
        "--zero-inflation",
        default='none',
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
        (args.data_file is not None) and (
            os.path.exists(args.data_file))
    ), "data file must be provided"

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.float64:
        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        dtype = torch.float32
        torch.set_default_dtype(torch.float32)

    xs = dt.fread(file=args.data_file, header=True).to_numpy()
    us = None 
    if args.undesired_factor_file is not None:
        us = dt.fread(file=args.undesired_factor_file, header=True).to_numpy()

    input_size = xs.shape[1]
    undesired_size = 0 if us is None else us.shape[1]

    latent_dist = 'normal'
    if args.laplace:
        latent_dist='laplacian'
    if args.student_t:
        latent_dist='studentt'

    ###########################################
    sure = SURE(
        input_size=input_size,
        undesired_size=undesired_size,
        inverse_dispersion=args.inverse_dispersion,
        z_dim=args.z_dim,
        hidden_layers=args.hidden_layers,
        hidden_layer_activation=args.hidden_layer_activation,
        use_cuda=args.cuda,
        config_enum=args.enum_discrete,
        use_dirichlet=args.use_dirichlet,
        zero_inflation=args.zero_inflation,
        gate_prior=args.gate_prior,
        delta=args.delta,
        loss_func=args.likelihood,
        dirichlet_mass=args.dirichlet_mass,
        nn_dropout=args.layer_dropout_rate,
        post_layer_fct=args.post_layer_function,
        post_act_fct=args.post_activation_function,
        codebook_size=args.codebook_size,
        latent_dist = latent_dist,
        studentt_dof = args.degree_of_freedom,
        dtype=dtype,
    )

    sure.fit(xs, us, 
             num_epochs=args.num_epochs,
             learning_rate=args.learning_rate,
             batch_size=args.batch_size,
             beta_1=args.beta_1,
             decay_rate=args.decay_rate,
             use_jax=args.jit,
             config_enum=args.enum_discrete,
             )

    if args.save_model is not None:
        SURE.save_model(sure, args.save_model)

    


if __name__ == "__main__":

    main()