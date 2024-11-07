import torch
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist

import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

from tqdm import tqdm

from ..utils import convert_to_tensor, tensor_to_numpy
from ..utils import CustomDataset2

def codebook_predict(codebook, z):
    codebook = convert_to_tensor(codebook)
    z = convert_to_tensor(z)
    distances = torch.cdist(z, codebook.unsqueeze(0))
    cluster_assignments = torch.argmin(distances.squeeze(0), dim=1)
    return tensor_to_numpy(cluster_assignments), tensor_to_numpy(distances)


def codebook_compress(sure_model, xs):
    '''return the logits of codebook assignments'''
    xs = convert_to_tensor(xs, dtype=sure_model.dtype, device=sure_model.get_device())
    return tensor_to_numpy(sure_model._code(xs))


def codebook_recover(sure_model, codes):
    '''return the latent representation of the recovered data'''
    codes = convert_to_tensor(codes, dtype=sure_model.dtype, device=sure_model.get_device())
    ns = dist.OneHotCategorical(logits=codes).sample()

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    return tensor_to_numpy(zs)

def codebook_generate(sure_model, n_samples):
    code_weights = convert_to_tensor(sure_model.codebook_weights, dtype=sure_model.dtype, device=sure_model.get_device())
    ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    return tensor_to_numpy(zs), tensor_to_numpy(ns)


def codebook_sample(sure_model, xs, n_samples):
    xs = convert_to_tensor(xs, dtype=sure_model.dtype, device=sure_model.get_device())
    assigns = sure_model.soft_assignments(xs)
    code_weights = codebook_weights(assigns)

    code_weights = convert_to_tensor(code_weights, dtype=sure_model.dtype, device=sure_model.get_device())
    ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()

    xs_zs = sure_model.get_cell_coordinates(xs)
    xs_zs = convert_to_tensor(xs_zs, dtype=sure_model.dtype, device=sure_model.get_device())

    #xs_dist = torch.cdist(zs, xs_zs)
    #idx = xs_dist.argmin(dim=1)

    nbrs = NearestNeighbors(n_jobs=-1, n_neighbors=1)
    nbrs.fit(tensor_to_numpy(xs_zs))
    idx = nbrs.kneighbors(tensor_to_numpy(zs), return_distance=False)
    idx = idx.flatten()

    return tensor_to_numpy(xs[idx]), tensor_to_numpy(idx)


def codebook_sketch(sure_model, xs, n_samples):
    return codebook_sample(sure_model, xs, n_samples)

def codebook_boost_sketch(sure_model, xs, n_samples, n_neighbors=10, aggregate_means='mean', pval=1e-12):
    xs = convert_to_tensor(xs, dtype=sure_model.dtype, device=sure_model.get_device())
    xs_zs = sure_model.get_cell_coordinates(xs)
    xs_zs = tensor_to_numpy(xs_zs)

    # generate samples that follow the metacell distribution of the given data
    assigns = sure_model.soft_assignments(xs)
    code_weights = codebook_weights(assigns)

    code_weights = convert_to_tensor(code_weights, dtype=sure_model.dtype, device=sure_model.get_device())
    ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

    codebook_loc, codebook_scale = sure_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=sure_model.dtype, device=sure_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=sure_model.dtype, device=sure_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    zs = tensor_to_numpy(zs)

    # find the neighbors of sample data in the real data space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(xs_zs)
        
    xs_list = []
    distances, ids = nbrs.kneighbors(zs, return_distance=True)
    dist_pdf = gaussian_kde(distances.flatten())

    xs = tensor_to_numpy(xs)    
    sketch_cells = dict()
    with tqdm(total=n_samples, desc='Sketching', unit='sketch') as pbar:
        for i in np.arange(n_samples):
            cells_i = ids[i, dist_pdf(distances[i]) > pval]

            xs_i = xs[cells_i]
            if aggregate_means == 'mean':
                xs_i = np.mean(xs_i, axis=0, keepdims=True)
            elif aggregate_means == 'median':
                xs_i = np.median(xs_i, axis=0, keepdims=True)
            elif aggregate_means == 'sum':
                xs_i = np.sum(xs_i, axis=0, keepdims=True)

            xs_list.append(xs_i)
            sketch_cells[i] = cells_i 

            pbar.update(1)

    return np.vstack(xs_list),tensor_to_numpy(ns),sketch_cells

def multiomics_codebook_sample(suremo_model, xs1, xs2, n_samples):
    xs1 = convert_to_tensor(xs1, dtype=suremo_model.dtype, device=suremo_model.get_device())
    xs2 = convert_to_tensor(xs2, dtype=suremo_model.dtype, device=suremo_model.get_device())
    assigns = suremo_model.soft_assignments(xs1, xs2)
    code_weights = codebook_weights(assigns)

    code_weights = convert_to_tensor(code_weights, dtype=suremo_model.dtype, device=suremo_model.get_device())
    ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

    codebook_loc, codebook_scale = suremo_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=suremo_model.dtype, device=suremo_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=suremo_model.dtype, device=suremo_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()

    xs_zs = suremo_model.get_cell_coordinates(xs1,xs2)
    xs_zs = convert_to_tensor(xs_zs, dtype=suremo_model.dtype, device=suremo_model.get_device())

    #xs_dist = torch.cdist(zs, xs_zs)
    #idx = xs_dist.argmin(dim=1)

    nbrs = NearestNeighbors(n_jobs=-1, n_neighbors=1)
    nbrs.fit(tensor_to_numpy(xs_zs))
    idx = nbrs.kneighbors(tensor_to_numpy(zs), return_distance=False)
    idx = idx.flatten()

    return tensor_to_numpy(xs1[idx]), tensor_to_numpy(xs2[idx]), tensor_to_numpy(idx)

def multiomics_codebook_sketch(suremo_model, xs1, xs2, n_samples):
    return multiomics_codebook_sample(suremo_model, xs1, xs2, n_samples)

def multiomics_codebook_boost_sketch(suremo_model, xs1, xs2, n_samples, n_neighbors=10, aggregate_means='mean', pval=1e-12):
    xs1 = convert_to_tensor(xs1, dtype=suremo_model.dtype, device=suremo_model.get_device())
    xs2 = convert_to_tensor(xs2, dtype=suremo_model.dtype, device=suremo_model.get_device())
    xs_zs = suremo_model.get_cell_coordinates(xs1, xs2)
    xs_zs = tensor_to_numpy(xs_zs)

    # generate samples that follow the metacell distribution of the given data
    assigns = suremo_model.soft_assignments(xs1,xs2)
    code_weights = codebook_weights(assigns)

    code_weights = convert_to_tensor(code_weights, dtype=suremo_model.dtype, device=suremo_model.get_device())
    ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

    codebook_loc, codebook_scale = suremo_model.get_codebook()
    codebook_loc = convert_to_tensor(codebook_loc, dtype=suremo_model.dtype, device=suremo_model.get_device())
    codebook_scale = convert_to_tensor(codebook_scale, dtype=suremo_model.dtype, device=suremo_model.get_device())

    loc = torch.matmul(ns, codebook_loc)
    scale = torch.matmul(ns, codebook_scale)
    zs = dist.Normal(loc, scale).to_event(1).sample()
    zs = tensor_to_numpy(zs)

    # find the neighbors of sample data in the real data space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(xs_zs)
        
    xs1_list,xs2_list = [],[]
    distances, ids = nbrs.kneighbors(zs, return_distance=True)
    dist_pdf = gaussian_kde(distances.flatten())
        
    xs1 = tensor_to_numpy(xs1)
    xs2 = tensor_to_numpy(xs2)
    sketch_cells = dict()
    with tqdm(total=n_samples, desc='Sketching', unit='sketch') as pbar:
        for i in np.arange(n_samples):
            cells_i = ids[i, dist_pdf(distances[i]) > pval]

            xs1_i = xs1[cells_i]
            xs2_i = xs2[cells_i]
            if aggregate_means == 'mean':
                xs1_i = np.mean(xs1_i, axis=0, keepdims=True)
                xs2_i = np.mean(xs2_i, axis=0, keepdims=True)
            elif aggregate_means == 'median':
                xs1_i = np.median(xs1_i, axis=0, keepdims=True)
                xs2_i = np.median(xs2_i, axis=0, keepdims=True)
            elif aggregate_means == 'sum':
                xs1_i = np.sum(xs1_i, axis=0, keepdims=True)
                xs2_i = np.sum(xs2_i, axis=0, keepdims=True)

            xs1_list.append(xs1_i)
            xs2_list.append(xs2_i)

            sketch_cells[i] = cells_i

            pbar.update(1)

    return np.vstack(xs1_list),np.vstack(xs2_list),tensor_to_numpy(ns),sketch_cells

def codebook_summarize_(assigns, xs):
    assigns = convert_to_tensor(assigns)
    xs = convert_to_tensor(xs)
    results = torch.matmul(assigns.T, xs)
    results = results / torch.sum(assigns.T, dim=1, keepdim=True)
    return tensor_to_numpy(results)


def codebook_summarize(assigns, xs, batch_size=1024):
    assigns = convert_to_tensor(assigns)
    xs = convert_to_tensor(xs)

    dataset = CustomDataset2(assigns, xs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    R = None
    W = None
    with tqdm(total=len(dataloader), desc='', unit='batch') as pbar:
        for A_batch, X_batch, _ in dataloader:
            r = torch.matmul(A_batch.T, X_batch)
            w = torch.sum(A_batch.T, dim=1, keepdim=True)
            if R is None:
                R = r 
                W = w 
            else:
                R += r 
                W += w 
            pbar.update(1)

    results = R / W
    return tensor_to_numpy(results)



def codebook_weights(assigns):
    assigns = convert_to_tensor(assigns)
    results = torch.sum(assigns, dim=0)
    results = results / torch.sum(results)
    return tensor_to_numpy(results)


def codebook_alignment(sure_model, codebook_pos, 
                       another_codebook_expr,
                       another_codebook_pos):
    pass