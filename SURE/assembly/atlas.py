import os 
import tempfile
import subprocess 
import scanpy as sc
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
import pandas as pd
import datatable as dt
from tqdm import tqdm
import umap 
from sklearn.neighbors import NearestNeighbors
import faiss 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyro.distributions as dist

from ..SURE import SURE
from .assembly import assembly,get_data,get_subdata,batch_encoding,get_uns
from ..codebook import codebook_summarize_,codebook_generate
from ..utils import convert_to_tensor, tensor_to_numpy
from ..utils import CustomDataset
from ..utils import pretty_print, Colors
from ..utils import PriorityQueue

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import dill as pickle
import gzip 
from packaging.version import Version
torch_version = torch.__version__

from typing import Literal

import warnings
warnings.filterwarnings("ignore")

class SingleOmicsAtlas(nn.Module):
    """
    Compressed Cell Atlas

    Parameters
    ----------
    atlas_name
        Name of the built atlas.
    hvgs
        Highly variable genes.
    eps
        Low bound.
    """
    def __init__(self, 
                 atlas_name: str = 'Atlas', 
                 hvgs: list = None, 
                 eps: float = 1e-12):
        super().__init__()
        self.atlas_name = atlas_name
        self.model = None 
        self.sure_models_list = None 
        self.hvgs = hvgs 
        self.adata = None
        self.sample_adata = None
        self.layer = None
        self.n_sure_models = None
        self.umap_metric='euclidean'
        self.umap = None
        self.adj = None
        self.subatlas_list = None 
        self.n_subatlas = 0
        self.pheno_keys = None
        self.nearest_neighbor_engine = None
        self.knn_k = 5
        self.network = None
        self.network_pos = None
        self.sample_network = None 
        self.sample_network_pos = None
        self.eps=eps

    def fit(self, adata_list, 
            batch_key: str = None, 
            pheno_keys: list = None, 
            preprocessing: bool = True, 
            hvgs: list = None,
            n_top_genes: int = 5000, 
            hvg_method: Literal['seurat','seurat_v3','cell_ranger'] ='seurat', 
            layer: str = 'counts', 
            cuda_id: int = 0, 
            use_jax: bool = False,
            codebook_size: int = 500, 
            codebook_size_per_adata: int = 500, 
            learning_rate: float = 0.0001,
            batch_size: int = 512, 
            batch_size_per_adata: int = 512, 
            n_epochs: int = 200, 
            latent_dist: Literal['normal','laplacian','studentt'] = 'normal',
            use_dirichlet: bool = True, 
            use_dirichlet_per_adata: bool = True, 
            zero_inflation: bool = True, 
            zero_inflation_per_adata: bool = True,
            likelihood: Literal['negbinomial','poisson','multinomial','gaussian'] = 'negbinomial', 
            likelihood_per_adata: Literal['negbinomial','poisson','multinomial','gaussian'] = 'negbinomial', 
            n_samples_per_adata: int = 10000, 
            total_samples: int = 300000, 
            sketching: bool = False, 
            boost_sketch: bool = False, 
            n_sketch_neighbors: int = 10, 
            edge_thresh: float = 0.001,
            metric: Literal['euclidean','correlation','cosine'] ='euclidean',
            knn_k: int = 30):
        """
        Fit the input list of AnnData datasets.

        Parameters
        ----------
        adata_list
            A list of AnnData datasets.
        batch_key
            Undesired factor. 
        pheno_keys
            A list of phenotype factors, of which the information should be retained in the built atlas.
        preprocessing
            If toggled on, the input datasets will go through the standard Scanpy proprocessing steps including normalization and log1p transformation.
        hvgs
            If a list of highly variable genes is given, the subsequent steps will rely on these genes.
        n_top_genes
            Parameter for Scanpy's highly_variable_genes
        hvg_method
            Parameter for Scanpy's highly_variable_genes
        layer
            Data used for building the atlas.
        cuda_id
            Cuda device.
        use_jax
            If toggled on, Jax will be used for speeding.
        codebook_size
            Size of metacells in the built atlas.
        codebook_size_per_adata
            Size of metacells for each adata.
        learning_rate
            Parameter for optimization.
        batch_size
            Parameter for building the atlas.
        batch_size_per_adata
            Parameter for calling metacells within each adata.
        n_epochs
            Number of epochs.
        latent_dist
            Distribution for latent representations.
        use_dirichlet
            Use Dirichlet model for building the atlas.
        use_dirichlet_per_adata
            Use Dirichlet model for calling metacells within each adata.
        zero_inflation
            Use zero-inflated model for building the atlas.
        zero_inflation_per_adata
            Use zero-inflated model for calling metacells within each adata.
        likelihood
            Data generation model for building the atlas.
        likelihood_per_adata
            Data generation model for calling metacells within each adata.
        n_samples_per_adata
            Number of samples drawn from each adata for building the atlas.
        total_samples
            Total number of samples for building the atlas.
        sketching
            If toggled on, sketched cells will be used for building the atlas.
        boost_sketch
            If toggled on, boosted sketching will be used instead of simple sketching.
        n_sketch_neighbors
            Parameter for boosted sketching.
        edge_thresh
            Parameter for building network.
        metric
            Parameter for UMAP.
        knn_k
            Parameter for K-nearest-neighbor machine.
        """
        
        print(Colors.YELLOW + 'Create A Distribution-Preserved Single-Cell Omics Atlas' + Colors.RESET)
        n_adatas = len(adata_list)
        self.layer = layer
        self.n_sure_models = n_adatas
        self.umap_metric = metric
        self.pheno_keys = pheno_keys
        zero_inflation = True if sketching else zero_inflation

        # assembly
        print(f'{n_adatas} adata datasets are given')
        self.model,_,self.hvgs = assembly(adata_list, batch_key, 
                 preprocessing, hvgs, n_top_genes, hvg_method, layer, cuda_id, use_jax,
                 codebook_size, codebook_size_per_adata, learning_rate,
                 batch_size, batch_size_per_adata, n_epochs, latent_dist,
                 use_dirichlet, use_dirichlet_per_adata,
                 zero_inflation, zero_inflation_per_adata,
                 likelihood, likelihood_per_adata,
                 n_samples_per_adata, total_samples, 
                 sketching, boost_sketch, n_sketch_neighbors)
        
        # summarize expression
        X,W,adj = None,None,None
        for i in np.arange(n_adatas):
            print(f'Adata {i+1} / {n_adatas}: Summarize data in {layer}')
            adata_i = adata_list[i][:,self.hvgs].copy()
            adata_i_ = adata_list[i].copy()

            xs_i = get_data(adata_i, layer).values
            xs_i_ = get_data(adata_i_, layer).values
            ws_i_sup = self.model.soft_assignments(xs_i)
            xs_i_sup = codebook_summarize_(ws_i_sup, xs_i_)

            if X is None:
                X = xs_i_sup
                W = np.sum(ws_i_sup.T, axis=1, keepdims=True)

                a = convert_to_tensor(ws_i_sup)
                a_t = a.T / torch.sum(a.T, dim=1, keepdim=True)
                adj = torch.matmul(a_t, a)
            else:
                X += xs_i_sup
                W += np.sum(ws_i_sup.T, axis=1, keepdims=True)

                a = convert_to_tensor(ws_i_sup)
                a_t = a.T / torch.sum(a.T, dim=1, keepdim=True)
                adj += torch.matmul(a_t, a)
        X = X / W 
        self.adata = sc.AnnData(X)
        self.adata.var_names = adata_i_.var_names

        adj = tensor_to_numpy(adj) / self.n_sure_models
        self.adj = (adj + adj.T) / 2
        n_nodes = adj.shape[0]
        self.adj[np.arange(n_nodes), np.arange(n_nodes)] = 0

        # summarize phenotypes
        if pheno_keys is not None:
            self._summarize_phenotypes_from_adatas(adata_list, pheno_keys)

        # compute visualization position for the atlas
        print('Compute the reference position of the atlas')
        n_samples = np.max([n_samples_per_adata * self.n_sure_models, 50000])
        n_samples = np.min([n_samples, total_samples])
        self.instantiation(n_samples)

        # create nearest neighbor indexing
        self.build_nearest_neighbor_engine(knn_k)
        self.knn_k = knn_k

        self.build_network(edge_thresh=edge_thresh)

        print(Colors.YELLOW + f'A distribution-preserved atlas has been built from {n_adatas} adata datasets.' + Colors.RESET)

    def map(self, adata_query, 
            batch_size: int = 1024):
        """
        Map query data to the atlas.

        Parameters
        ----------
        adata_query
            Query data. It should be an AnnData object.
        batch_size
            Size of batch processing.
        """
        adata_query = adata_query.copy()
        X_query = get_subdata(adata_query, self.hvgs, self.layer).values

        X_map = self.model.get_cell_counts(X_query, batch_size=batch_size)
        Z_map = self.model.get_cell_coordinates(X_query, batch_size=batch_size)
        A_map = self.model.soft_assignments(X_query)

        return X_map, Z_map, A_map
    
    def sample(self, 
               n_samples: int = 5000):
        """
        Return samples drawn from the atlas.

        Parameters
        ----------
        n_samples
            Number of samples.
        """
        xs,_ = self.sample_with_origin(n_samples)
        return xs
    
    def sample_with_origin(self, n_samples=5000):
        zs,ns = codebook_generate(self.model, n_samples)
        xs = self.model.generate_count_data(zs)
        return xs,ns
    
    def sample_by_adata(self, adata, 
                        n_samples: int = 5000):
        """
        Return samples drawn from the atlas according to the distribution of input adata within the atlas.

        Parameters
        ----------
        adata
            Input data. It should be an AnnData object.
        n_samples
            Number of samples.
        """
        _,_,A = self.map(adata)
        code_weights = A.sum(axis=0)
        code_weights /= np.sum(code_weights)
        code_weights = convert_to_tensor(code_weights, dtype=self.model.dtype, device=self.model.get_device())
        ns = dist.OneHotCategorical(probs=code_weights).sample([n_samples])

        codebook_loc, codebook_scale = self.model.get_codebook()
        codebook_loc = convert_to_tensor(codebook_loc, dtype=self.model.dtype, device=self.model.get_device())
        codebook_scale = convert_to_tensor(codebook_scale, dtype=self.model.dtype, device=self.model.get_device())

        loc = torch.matmul(ns, codebook_loc)
        scale = torch.matmul(ns, codebook_scale)
        zs = dist.Normal(loc, scale).to_event(1).sample()

        xs = self.model.generate_count_data(zs)
        return xs, tensor_to_numpy(ns)

    def sketch_cells(self, adata, 
                     n_sketches: int = 5000):
        """
        Simple reference-based sketching. 
        It returns the identity of sketched cells.

        Parameters
        ----------
        adata
            Input adata for sketching. It should be an AnnData object.
        n_sketches
            Number of sketched cells.
        """
        xs_sample, xs_assign = self.sample_with_origin(n_sketches)
        zs_sample = self.model.get_cell_coordinates(xs_sample)

        xs = get_subdata(adata, self.hvgs, layer=self.layer)
        zs = self.model.get_cell_coordinates(xs.values)

        nbrs = FaissKNeighbors(n_neighbors=1)
        nbrs.fit(zs)

        sketch_cells = []
        _,ids = nbrs.kneighbors(zs_sample)
        sketch_cells = ids.flatten()

        return sketch_cells, np.argmax(xs_assign, axis=1)
    
    def sketch_cells_bk(self, adata, 
                     n_sketches: int = 5000):
        """
        Simple reference-based sketching. 
        It returns the identity of sketched cells.

        Parameters
        ----------
        adata
            Input adata for sketching. It should be an AnnData object.
        n_sketches
            Number of sketched cells.
        """
        xs_sample, xs_assign = self.sample_with_origin(n_sketches)
        zs_sample = self.model.get_cell_coordinates(xs_sample)

        xs = get_subdata(adata, self.hvgs, layer=self.layer)
        zs = self.model.get_cell_coordinates(xs.values)

        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        nbrs.fit(zs)

        sketch_cells = []
        ids = nbrs.kneighbors(zs_sample, return_distance=False)
        sketch_cells = ids.flatten()

        return sketch_cells, np.argmax(xs_assign, axis=1)
    
    def boost_sketch_adata(self, adata, 
                           n_sketches: int = 5000, 
                           n_neighbors: int = 10, 
                           aggregate_means: Literal['mean','sum','median'] = 'mean', 
                           pheno_keys: list = None, 
                           pval: float = 1e-12):     
        """
        Reference-based boosted sketching. It returns an AnnData object of sketched cells.

        Parameters
        ----------
        adata
            Input data for sketching. It should be an AnnData object.
        n_sketches
            Number of sketched cells.
        n_neighbors
            Number of nearest neighbors for building sketched cells.
        aggregate_means
            Method for aggregating data.
        pheno_keys
            Phenotype factors that will be retained for sketched cells.
        pval
            Distance threshold for defining nearest neighbors.
        """
        # samples drawn from metacells, and compute their embeddings
        xs_sample, xs_assigns = self.sample_with_origin(n_sketches)
        zs_sample = self.model.get_cell_coordinates(xs_sample)

        # real data and their embeddings
        xs = get_subdata(adata, self.hvgs, layer=self.layer)
        zs = self.model.get_cell_coordinates(xs.values)

        # search engine for real data
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nbrs.fit(zs)
        
        # generate sketch data by averaging over local neighborhoods of samples
        sketch_adata_list = []
        distances, ids = nbrs.kneighbors(zs_sample, return_distance=True)
        dist_pdf = gaussian_kde(distances.flatten())
        
        with tqdm(total=n_sketches, desc='Sketching', unit='sketch') as pbar:
            for i in np.arange(n_sketches):
                cells_i = ids[i, dist_pdf(distances[i]) > pval]

                adata_ = adata[cells_i]
                xs = get_data(adata_, self.layer).values
                if aggregate_means == 'mean':
                    xs = np.mean(xs, axis=0, keepdims=True)
                elif aggregate_means == 'median':
                    xs = np.median(xs, axis=0, keepdims=True)
                elif aggregate_means == 'sum':
                    xs = np.sum(xs, axis=0, keepdims=True)

                adata_i = sc.AnnData(sparse.csr_matrix(xs))
                adata_i.obs_names = [f'sketch_{i}']
                adata_i.var_names = adata.var_names
                adata_i.obs['metacell'] = np.argmax(xs_assigns[i])
                adata_i.uns['aggregate_cells'] = cells_i
                sketch_adata_list.append(adata_i)

                if pheno_keys:
                    for pheno in pheno_keys:
                        adata_i.obs[pheno] = adata_.obs[pheno].mode()[0]

                pbar.update(1)

        return sc.concat(sketch_adata_list)
    
    def summarize_phenotypes(self, adata_list=None, atlas_list=None, pheno_keys=None):
        if adata_list is not None:
            self._summarize_phenotypes_from_adatas(adata_list, pheno_keys)

        if atlas_list is not None:
            self._summarize_phenotypes_from_atlases(atlas_list, pheno_keys)

    def _summarize_phenotypes_from_adatas(self, adata_list, pheno_keys):
        n_adatas = len(adata_list)
        for pheno in pheno_keys:
            Y = list()
            for i in np.arange(n_adatas):
                if pheno in adata_list[i].obs.columns:
                    print(f'Adata {i+1} / {n_adatas}: Summarize data in {pheno}')
                    adata_i = adata_list[i][:,self.hvgs].copy()

                    xs_i = get_data(adata_i, self.layer).values
                    ws_i_sup = self.model.soft_assignments(xs_i)
                    ys_i = batch_encoding(adata_i, pheno)
                    columns_i = ys_i.columns.tolist()
                    ys_i = codebook_summarize_(ws_i_sup, ys_i.values)

                    Y.append(pd.DataFrame(ys_i, columns=columns_i))

            Y_df = aggregate_dataframes(Y)
            
            #Y = Y_df.values
            #Y[Y<self.eps] = 0
            #Y = Y / Y.sum(axis=1, keepdims=True)

            self.adata.uns[pheno] = Y_df
            #self.adata.uns[f'{pheno}_columns'] = Y_df.columns
            self.adata.obs[pheno] = Y_df.idxmax(axis=1).tolist()

    def _summarize_phenotypes_from_atlases(self, atlas_list, pheno_keys):
        n_atlases = len(atlas_list)
        for pheno in pheno_keys:
            Y = list()
            for i in np.arange(n_atlases):
                if pheno in atlas_list[i].model.pheno_keys:
                    print(f'Atlas {atlas_list[i].atlas_name} / {n_atlases}: Summarize data in {pheno}')
                    atlas_i = atlas_list[i]
                    xs = atlas_i.sample_adata.X
                    ws_i = atlas_i.model.soft_assignments(xs)
                    ys_i = get_uns(atlas_i.adata, pheno)
                    #if sparse.issparse(atlas_i.uns[pheno]):
                    #    ys_i = pd.DataFrame(atlas_i.uns[pheno].toarray(), 
                    #                        columns=atlas_i.uns[f'{pheno}_columns'])
                    #else:
                    #    ys_i = atlas_i.uns[pheno]
                    columns_i = ys_i.columns
                    ys_i = matrix_dotprod(ws_i, ys_i.values)

                    ws = self.model.soft_assignments(xs)
                    ws = ws.T / np.sum(ws.T, axis=1, keepdims=True)
                    ys_i = matrix_dotprod(ws, ys_i)
                    Y.append(pd.DataFrame(ys_i, columns=columns_i))

            Y_df = aggregate_dataframes(Y)

            #Y = Y_df.values 
            #Y[Y<self.eps] = 0
            #Y = Y / Y.sum(axis=1, keepdims=True)

            self.adata.uns[pheno] = Y_df
            #self.adata.uns[f'{pheno}_columns'] = Y_df.columns
            self.adata.obs[pheno] = Y_df.idxmax(axis=1).tolist()

    def instantiation(self, n_samples=50000):
        zs,ws = codebook_generate(self.model, n_samples)
        xs = self.model.generate_count_data(zs)
        ws = self.model.soft_assignments(xs)
        zs_mc = self.model.get_metacell_coordinates()
        zs_all = np.vstack([zs_mc,zs])
        self.umap = umap.UMAP(metric=self.umap_metric).fit(zs_all)
        umap_pos = self.umap.transform(zs_all)

        self.adata.obsm['X_umap'] = umap_pos[:zs_mc.shape[0]]
        self.adata.obsm['X_sure'] = zs_mc

        self.sample_adata = sc.AnnData(xs)
        self.sample_adata.var_names = self.hvgs
        self.sample_adata.obsm['weight'] = ws 
        self.sample_adata.obsm['X_umap'] = umap_pos[zs_mc.shape[0]:] 
        self.sample_adata.obsm['X_sure'] = zs 

        if self.layer.lower() != 'x':
            self.sample_adata.layers[self.layer] = xs

        if self.pheno_keys is not None:
            for pheno in self.pheno_keys:
                ys = pd.DataFrame(matrix_dotprod(ws, self.adata.uns[pheno].values), columns=self.adata.uns[pheno].columns)
                self.sample_adata.obs[pheno] = ys.idxmax(axis=1).tolist()

    def phenotype_predict(self, adata_query, pheno_key, batch_size=1024):
        _,_,ws = self.map(adata_query, batch_size)
        A = matrix_dotprod(ws, self.adata.uns[pheno_key].values)
        A = pd.DataFrame(A, columns=self.adata.uns[pheno_key].columns)
        return A.idxmax(axis=1).tolist()
    
    def build_metacell_nearest_neighbor_engine(self, k=30, use_latent=True):
        self.nearest_neighbor_engine = NearestNeighbors(n_neighbors=k)
        if use_latent:
            self.nearest_neighbor_engine.fit(self.adata.obsm['X_sure'])
        else:
            xs = self.model.generate_count_data(self.adata.obsm['X_sure'])
            self.nearest_neighbor_engine.fit(xs)
        self.knn_k = k

    def build_nearest_neighbor_engine(self, k=100, use_latent=True):
        self.nearest_neighbor_engine = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        if use_latent:
            self.nearest_neighbor_engine.fit(self.sample_adata.obsm['X_sure'])
        else:
            xs = self.model.generate_count_data(self.sample_adata.obsm['X_sure'])
            self.nearest_neighbor_engine.fit(xs)
        self.knn_k = k

    def position_of_query_bk(self, query_atlas, thresh=0.01, k=50, knn_k=15, n_samples=None, optimize=True, lr=0.2, max_epoch=200, algo='adam'):
        n_refs = self.adata.shape[0]
        n_querys = query_atlas.adata.shape[0]

        # reference positional shift
        if n_samples:
            xs_sketch = self.sample(n_samples)
            sketch_adata = sc.AnnData(xs_sketch)
            sketch_adata.var_names = self.hvgs
            if self.layer.lower() != 'x':
                sketch_adata.layers[self.layer] = xs_sketch.copy()
            _,zs_sketch,ws = self.map(sketch_adata)
            wst = ws.T / np.sum(ws.T, axis=1, keepdims=True)

            # reference positional shift
            if query_atlas.layer.lower() != 'x':
                sketch_adata.layers[query_atlas.layer] = xs_sketch.copy()
            _, zs, s2q_scores = query_atlas.map(sketch_adata)

            ref_sketch_pos = self.umap.transform(zs_sketch)
            ref_sketch_pos_in_query = query_atlas.umap.transform(zs)
            ref_sketch_pos_delta = ref_sketch_pos - ref_sketch_pos_in_query

            r2q_scores = wst @ s2q_scores
            ref_pos_delta = wst @ ref_sketch_pos_delta
        else:
            if query_atlas.layer.lower() != 'x':
                self.sample_adata.layers[query_atlas.layer] = self.sample_adata.X.copy()
            _, zs, s2q_scores = query_atlas.map(self.sample_adata)
            ws = self.sample_adata.obsm['weight']
            wst = ws.T / np.sum(ws.T, axis=1, keepdims=True)
            ref_sketch_pos_in_query = query_atlas.umap.transform(zs)
            ref_sketch_pos = self.sample_adata.obsm['X_umap']
            ref_sketch_pos_delta = ref_sketch_pos - ref_sketch_pos_in_query

            r2q_scores = wst @ s2q_scores
            ref_pos_delta = wst @ ref_sketch_pos_delta
        
        nbrs = NearestNeighbors(n_neighbors=knn_k)
        nbrs.fit(query_atlas.sample_adata.obsm['X_sure'])

        # find the anchor points between reference atlas and query atlas
        query_to_ref_pairs = dict()
        q_anchors = dict()
        n_anchors = 0
        for i in np.arange(n_querys):
            for j in np.arange(n_refs):
                score_i = r2q_scores[j,i]
                if score_i > thresh:
                    if i in q_anchors:
                        q_anchors[i].append(j)
                    else:
                        q_anchors[i] = [j]
                    n_anchors += 1
                    query_to_ref_pairs[(i,j)] = score_i
        
        query_anchor_points = q_anchors.keys()
        ir_idx = list(q_anchors.keys())

        ref_anchor_points = []
        for ref_anchors_i in q_anchors.values():
            ref_anchor_points.extend(ref_anchors_i)
        ref_anchor_points = list(set(ref_anchor_points))
        print(f'Find {n_anchors} anchor pairs between {len(ref_anchor_points)} reference metacells and {len(query_anchor_points)} query metacells.')

        # sort neighbors of query anchors and select top k neighbors
        for i in q_anchors.keys():
            js_i = q_anchors[i]
            if len(js_i) > k:
                q2r_scores_i = r2q_scores[js_i,i]
                sorted_values_with_indices = sorted(enumerate(q2r_scores_i), key=lambda x: -x[1])
                sorted_indices, _ = zip(*sorted_values_with_indices)
                q_anchors[i] = [js_i[u] for u in sorted_indices[:k]]

        # compute the positional shift of query anchor
        q_delta = dict()
        for i in q_anchors.keys():
            js_i = q_anchors[i]

            q_delta_i, q_delta_w = [],[]
            for j in js_i:
                q_delta_i.append(ref_pos_delta[j] * r2q_scores[j,i])
                q_delta_w.append(r2q_scores[j,i])
            q_delta[i] = np.sum(q_delta_i, axis=0) / np.sum(q_delta_w)
        
        # infer the positional shift for points that are not selected as anchors
        queue = PriorityQueue()
        network = query_atlas.network.copy()
        q_anchor_nbrs = {}
        max_dist = 0
        for i in network.nodes():
            if i not in q_anchors:
                i_neighbors = network.neighbors(i)
                i_anchor_nbrs = [j for j in i_neighbors if j in q_anchors]
                #queue.put(i, priority=-len(i_anchor_nbrs))

                if i_anchor_nbrs:
                    i_pos = query_atlas.adata.obsm['X_sure'][i].reshape(1, -1)
                    nbrs_pos = query_atlas.adata.obsm['X_sure'][i_anchor_nbrs]
                    i_to_nbrs_dist = sp.spatial.distance.cdist(i_pos, nbrs_pos)
                    queue.put(i, priority=i_to_nbrs_dist.mean())

                    i_max = i_to_nbrs_dist.max()
                    max_dist = 10*i_max if i_max>max_dist else max_dist
                else:
                    queue.put(i, priority=max_dist)

                q_anchor_nbrs[i] = i_anchor_nbrs
        
        while queue.peek() is not None:
            # get the point that has the most neighbors with positional shifts
            i = queue.get()
            i_anchor_nbrs = q_anchor_nbrs[i]

            if i_anchor_nbrs:
                # infer its shift
                i_delta = []
                i_weight = []
                for j in i_anchor_nbrs:
                    i_delta.append(q_delta[j])
                    i_weight.append(network[i][j]['weight'])

                sorted_weights_with_indices = sorted(enumerate(i_weight), key=lambda x: -x[1])
                sorted_indices, _ = zip(*sorted_weights_with_indices)
                sorted_indices = list(sorted_indices)

                ids = sorted_indices
                if len(i_delta) > k:
                    ids = sorted_indices[:k]
                i_delta = [i_delta[j]*i_weight[j] for j in ids]
                i_weight = [i_weight[j] for j in ids]
                q_delta[i] = np.sum(i_delta, axis=0) / np.sum(i_weight)
                #i_delta = [i_delta[j] for j in ids]
                #i_weight = [i_weight[j] for j in ids]
                #q_delta[i] = np.sum(i_delta, axis=0) / len(ids)

                # update anchor points
                q_anchors[i] = i_anchor_nbrs

                # update info for its neighbors
                i_neighbors = set(network.neighbors(i))
                i_neighbors.difference_update(set(q_anchors.keys()))
                for j in i_neighbors:
                    q_anchor_nbrs[j].append(i)
                    #queue.update(j, -len(q_anchor_nbrs[j]))
                    
                    j_pos = query_atlas.adata.obsm['X_sure'][j].reshape(1,-1)
                    nbrs_pos = query_atlas.adata.obsm['X_sure'][q_anchor_nbrs[j]]
                    j_to_nbrs_dist = sp.spatial.distance.cdist(j_pos, nbrs_pos)
                    queue.update(j, j_to_nbrs_dist.mean())
            else:
                query_pos_i = query_atlas.adata.obsm['X_sure'][i].reshape(1, -1)
                distances, i_knn_neighbors = nbrs.kneighbors(query_pos_i, return_distance=True)
                sd = distances.sum(axis=1, keepdims=True)
                distances = distances / sd 
                similarities = np.exp(-distances)
                similarities = similarities / similarities.sum(axis=1, keepdims=True)

                # infer its shift
                i_delta = []
                i_weight = []
                i_anchor_nbrs = []
                for u in np.arange(knn_k):
                    j = i_knn_neighbors[0][u]
                    if j in q_anchors:
                        i_anchor_nbrs.append(j)
                        i_delta.append(q_delta[j])
                        i_weight.append(similarities[0][u])

                if i_weight:
                    sorted_weights_with_indices = sorted(enumerate(i_weight), key=lambda x: -x[1])
                    sorted_indices, _ = zip(*sorted_weights_with_indices)
                    sorted_indices = list(sorted_indices)

                    ids = sorted_indices
                    if len(i_delta) > k:
                        ids = sorted_indices[:k]
                    i_delta = [i_delta[j]*i_weight[j] for j in ids]
                    i_weight = [i_weight[j] for j in ids]
                    q_delta[i] = np.sum(i_delta, axis=0) / np.sum(i_weight)
                    #i_delta = [i_delta[j] for j in ids]
                    #q_delta[i] = np.sum(i_delta, axis=0) / len(ids)
                else:
                    mean_delta = 0 
                    for q in q_delta.keys():
                        mean_delta += q_delta[q]
                    q_delta[i] = mean_delta / len(q_delta)

                # update anchor points
                q_anchors[i] = i_anchor_nbrs

                # update info for its neighbors
                i_neighbors = set(network.neighbors(i))
                i_neighbors.difference_update(set(q_anchors.keys()))
                for j in i_neighbors:
                    q_anchor_nbrs[j].append(i)
                    #queue.update(j, -len(q_anchor_nbrs[j]))

                    j_pos = query_atlas.adata.obsm['X_sure'][j].reshape(1,-1)
                    nbrs_pos = query_atlas.adata.obsm['X_sure'][q_anchor_nbrs[j]]
                    j_to_nbrs_dist = sp.spatial.distance.cdist(j_pos, nbrs_pos)
                    queue.update(j, j_to_nbrs_dist.mean())
                
        delta_pos = np.vstack([q_delta[j] for j in query_atlas.network.nodes()])
        new_query_pos = query_atlas.adata.obsm['X_umap'] + delta_pos

        # test
        oor_idx = []
        if optimize:
            oor_idx = [i for i in np.arange(n_querys) if i not in ir_idx]

        if len(oor_idx) > 0:
            print(f'Optimize the position of {len(oor_idx)} out-of-reference cells')

            OOR_new = convert_to_tensor(new_query_pos[oor_idx])
            IR_new = convert_to_tensor(new_query_pos[ir_idx])
            ref_new = convert_to_tensor(ref_sketch_pos)

            old_query_pos = query_atlas.adata.obsm['X_umap']
            OOR_old = convert_to_tensor(old_query_pos[oor_idx])
            IR_old = convert_to_tensor(old_query_pos[ir_idx])
            ref_old = convert_to_tensor(ref_sketch_pos_in_query)

            X = nn.Parameter(OOR_new)

            OOR_IR_old = torch.cdist(OOR_old,IR_old) 
            OOR_Ref_old = torch.cdist(OOR_old, ref_old)
            OOR_OOR_old = torch.cdist(OOR_old, OOR_old)

            if algo.lower() == 'adamW':
                optimizer = torch.optim.AdamW([X], lr=lr)
                optimizer2 = torch.optim.AdamW([X], lr=lr)
                optimizer3 = torch.optim.AdamW([X], lr=lr)
            elif algo.lower() == 'rmsprop':
                optimizer = torch.optim.RMSprop([X], lr=lr)
                optimizer2 = torch.optim.RMSprop([X], lr=lr)
                optimizer3 = torch.optim.RMSprop([X], lr=lr)
            else:
                optimizer = torch.optim.Adam([X], lr=lr)
                optimizer2 = torch.optim.Adam([X], lr=lr)
                optimizer3 = torch.optim.Adam([X], lr=lr)

            criterion = nn.MSELoss() 

            with tqdm(total=max_epoch, desc='Training', unit='epoch') as pbar:
                for epoch in np.arange(max_epoch):
                    # 3
                    OOR_OOR_new = torch.cdist(X,X)
                    loss3 = criterion(OOR_OOR_new, OOR_OOR_old)

                    optimizer3.zero_grad()   # Zero the gradients
                    loss3.backward()         # Backpropagation
                    optimizer3.step()        # Update weights

                    # 1
                    OOR_IR_new = torch.cdist(X,IR_new) 
                    loss = criterion(OOR_IR_new, OOR_IR_old)
                    
                    optimizer.zero_grad()   # Zero the gradients
                    loss.backward()         # Backpropagation
                    optimizer.step()        # Update weights

                    # 2
                    OOR_Ref_new = torch.cdist(X,ref_new)
                    loss2 = criterion(OOR_Ref_new, OOR_Ref_old)

                    optimizer2.zero_grad()   # Zero the gradients
                    loss2.backward()         # Backpropagation
                    optimizer2.step()        # Update weights

                    pbar.set_postfix({'IR loss': loss.item(), 'Ref loss': loss2.item(), 'OOR loss': loss3.item()})
                    pbar.update(1)

            new_query_pos[oor_idx] = tensor_to_numpy(X)

        return new_query_pos
    
    def position_of_query(self, query_atlas, 
                          knn_k: int = 15, 
                          n_samples: int = 20000, 
                          pval: float = 1e-5, 
                          eps: float = 1e-12,
                          optimize: bool = True, 
                          lr: float = 0.2, 
                          max_epoch: int = 200, 
                          algo: Literal['adam','adamw','rmsprop'] = 'adam'):
        """
        Reference-based positioning of query data.

        Parameters
        ----------
        query_atlas
            Query data. It should be a SingleOmicsAtlas object.
        knn_k
            Number of nearest neighbors.
        n_samples
            Number of samples drawn from the reference atlas.
        pval
            Distance threshold.
        eps
            Low bound.
        optimize
            If toggled on, optimization algorithm will be used to search for the optimized positions.
        lr
            Learning rate for optimization algorithm.
        max_epoch
            Epochs for training optimization algorithm.
        algo
            Optimization algorithm.
        """
        n_queries = query_atlas.adata.shape[0]

        # find out-of-reference query cells
        self.out_of_reference(query_atlas, n_samples=n_samples, eps=eps)
        oor_thresh = -np.log10(pval)
        oor_cells = np.arange(n_queries)[query_atlas.adata.obs['out_of_ref'] > oor_thresh]
        ir_cells = np.arange(n_queries)[query_atlas.adata.obs['out_of_ref'] < oor_thresh]

        # map reference samples to query space
        if query_atlas.layer.lower() != 'x':
            self.sample_adata.layers[query_atlas.layer] = self.sample_adata.X.copy()
        _, zs, _ = query_atlas.map(self.sample_adata)
        ref_sketch_pos_in_query = query_atlas.umap.transform(zs)
        ref_sketch_pos = self.sample_adata.obsm['X_umap']
        ref_sketch_pos_delta = ref_sketch_pos - ref_sketch_pos_in_query

        ref_nbrs = NearestNeighbors(n_neighbors=1)
        ref_nbrs.fit(ref_sketch_pos_in_query)
        query_ref_nbrs = ref_nbrs.kneighbors(query_atlas.adata.obsm['X_umap'], return_distance=False)

        # find the anchor points between reference atlas and query atlas
        q_anchors = dict()
        n_anchors = 0
        for u in np.arange(len(ir_cells)):
            i = ir_cells[u]
            j = query_ref_nbrs[i][0]
            q_anchors[i] = [j]
            n_anchors += 1

        query_anchor_points = q_anchors.keys()

        ref_anchor_points = []
        for ref_anchors_i in q_anchors.values():
            ref_anchor_points.extend(ref_anchors_i)
        ref_anchor_points = list(set(ref_anchor_points))
        print(f'Find {n_anchors} anchor pairs between {len(ref_anchor_points)} reference metacells and {len(query_anchor_points)} query metacells.')
        
        # compute the positional shift of query anchor
        q_delta = dict()
        for i in q_anchors.keys():
            js_i = q_anchors[i]

            q_delta_i = []
            for j in js_i:
                q_delta_i.append(ref_sketch_pos_delta[j])
            q_delta[i] = np.sum(q_delta_i, axis=0) / len(js_i)
        
        # infer the positional shift for points that are not selected as anchors
        queue = PriorityQueue()
        network = query_atlas.network.copy()
        q_anchor_nbrs = {}
        max_dist = 0
        for i in network.nodes():
            if i not in q_anchors:
                i_neighbors = network.neighbors(i)
                i_anchor_nbrs = [j for j in i_neighbors if j in q_anchors]
                #queue.put(i, priority=-len(i_anchor_nbrs))

                if i_anchor_nbrs:
                    i_pos = query_atlas.adata.obsm['X_sure'][i].reshape(1, -1)
                    nbrs_pos = query_atlas.adata.obsm['X_sure'][i_anchor_nbrs]
                    i_to_nbrs_dist = sp.spatial.distance.cdist(i_pos, nbrs_pos)
                    queue.put(i, priority=i_to_nbrs_dist.mean())

                    i_max = i_to_nbrs_dist.max()
                    max_dist = 10*i_max if i_max>max_dist else max_dist
                else:
                    queue.put(i, priority=max_dist)

                q_anchor_nbrs[i] = i_anchor_nbrs
        
        # a knn search engine for query cells
        nbrs = NearestNeighbors(n_neighbors=knn_k)
        nbrs.fit(query_atlas.adata.obsm['X_sure'])

        while queue.peek() is not None:
            # get the point that has the most neighbors with positional shifts
            i = queue.get()
            i_anchor_nbrs = q_anchor_nbrs[i]

            if i_anchor_nbrs:
                # infer its shift
                i_delta = []
                i_weight = []
                for j in i_anchor_nbrs:
                    i_delta.append(q_delta[j])
                    i_weight.append(network[i][j]['weight'])

                sorted_weights_with_indices = sorted(enumerate(i_weight), key=lambda x: -x[1])
                sorted_indices, _ = zip(*sorted_weights_with_indices)
                sorted_indices = list(sorted_indices)

                ids = sorted_indices
                if len(i_delta) > knn_k:
                    ids = sorted_indices[:knn_k]
                i_delta = [i_delta[j]*i_weight[j] for j in ids]
                i_weight = [i_weight[j] for j in ids]
                q_delta[i] = np.sum(i_delta, axis=0) / np.sum(i_weight)
                #i_delta = [i_delta[j] for j in ids]
                #i_weight = [i_weight[j] for j in ids]
                #q_delta[i] = np.sum(i_delta, axis=0) / len(ids)

                # update anchor points
                q_anchors[i] = i_anchor_nbrs

                # update info for its neighbors
                i_neighbors = set(network.neighbors(i))
                i_neighbors.difference_update(set(q_anchors.keys()))
                for j in i_neighbors:
                    q_anchor_nbrs[j].append(i)
                    #queue.update(j, -len(q_anchor_nbrs[j]))
                    
                    j_pos = query_atlas.adata.obsm['X_sure'][j].reshape(1,-1)
                    nbrs_pos = query_atlas.adata.obsm['X_sure'][q_anchor_nbrs[j]]
                    j_to_nbrs_dist = sp.spatial.distance.cdist(j_pos, nbrs_pos)
                    queue.update(j, j_to_nbrs_dist.mean())
            else:
                query_pos_i = query_atlas.adata.obsm['X_sure'][i].reshape(1, -1)
                distances, i_knn_neighbors = nbrs.kneighbors(query_pos_i, return_distance=True)
                sd = distances.sum(axis=1, keepdims=True)
                distances = distances / sd 
                similarities = np.exp(-distances)
                similarities = similarities / similarities.sum(axis=1, keepdims=True)

                # infer its shift
                i_delta = []
                i_weight = []
                i_anchor_nbrs = []
                for u in np.arange(knn_k):
                    j = i_knn_neighbors[0][u]
                    if j in q_anchors:
                        i_anchor_nbrs.append(j)
                        i_delta.append(q_delta[j])
                        i_weight.append(similarities[0][u])

                if i_weight:
                    sorted_weights_with_indices = sorted(enumerate(i_weight), key=lambda x: -x[1])
                    sorted_indices, _ = zip(*sorted_weights_with_indices)
                    sorted_indices = list(sorted_indices)

                    ids = sorted_indices
                    if len(i_delta) > knn_k:
                        ids = sorted_indices[:knn_k]
                    i_delta = [i_delta[j]*i_weight[j] for j in ids]
                    i_weight = [i_weight[j] for j in ids]
                    q_delta[i] = np.sum(i_delta, axis=0) / np.sum(i_weight)
                    #i_delta = [i_delta[j] for j in ids]
                    #q_delta[i] = np.sum(i_delta, axis=0) / len(ids)
                else:
                    mean_delta = 0 
                    for q in q_delta.keys():
                        mean_delta += q_delta[q]
                    q_delta[i] = mean_delta / len(q_delta)

                # update anchor points
                q_anchors[i] = i_anchor_nbrs

                # update info for its neighbors
                i_neighbors = set(network.neighbors(i))
                i_neighbors.difference_update(set(q_anchors.keys()))
                for j in i_neighbors:
                    q_anchor_nbrs[j].append(i)
                    #queue.update(j, -len(q_anchor_nbrs[j]))

                    j_pos = query_atlas.adata.obsm['X_sure'][j].reshape(1,-1)
                    nbrs_pos = query_atlas.adata.obsm['X_sure'][q_anchor_nbrs[j]]
                    j_to_nbrs_dist = sp.spatial.distance.cdist(j_pos, nbrs_pos)
                    queue.update(j, j_to_nbrs_dist.mean())
                
        delta_pos = np.vstack([q_delta[j] for j in query_atlas.network.nodes()])
        new_query_pos = query_atlas.adata.obsm['X_umap'] + delta_pos

        # test
        oor_idx = oor_cells if optimize else []
        ir_idx = ir_cells

        if len(oor_idx) > 0:
            print(f'Optimize the position of {len(oor_idx)} out-of-reference cells')

            OOR_new = convert_to_tensor(new_query_pos[oor_idx])
            IR_new = convert_to_tensor(new_query_pos[ir_idx])
            ref_new = convert_to_tensor(ref_sketch_pos)

            old_query_pos = query_atlas.adata.obsm['X_umap']
            OOR_old = convert_to_tensor(old_query_pos[oor_idx])
            IR_old = convert_to_tensor(old_query_pos[ir_idx])
            ref_old = convert_to_tensor(ref_sketch_pos_in_query)

            X = nn.Parameter(OOR_new)

            OOR_IR_old = torch.cdist(OOR_old,IR_old) 
            OOR_Ref_old = torch.cdist(OOR_old, ref_old)
            OOR_OOR_old = torch.cdist(OOR_old, OOR_old)

            if algo.lower() == 'adamw':
                optimizer = torch.optim.AdamW([X], lr=lr)
                optimizer2 = torch.optim.AdamW([X], lr=lr)
                optimizer3 = torch.optim.AdamW([X], lr=lr)
            elif algo.lower() == 'rmsprop':
                optimizer = torch.optim.RMSprop([X], lr=lr)
                optimizer2 = torch.optim.RMSprop([X], lr=lr)
                optimizer3 = torch.optim.RMSprop([X], lr=lr)
            else:
                optimizer = torch.optim.Adam([X], lr=lr)
                optimizer2 = torch.optim.Adam([X], lr=lr)
                optimizer3 = torch.optim.Adam([X], lr=lr)

            criterion = nn.MSELoss() 

            with tqdm(total=max_epoch, desc='Training', unit='epoch') as pbar:
                for epoch in np.arange(max_epoch):
                    # 3
                    OOR_OOR_new = torch.cdist(X,X)
                    loss3 = criterion(OOR_OOR_new, OOR_OOR_old)

                    optimizer3.zero_grad()   # Zero the gradients
                    loss3.backward()         # Backpropagation
                    optimizer3.step()        # Update weights

                    # 1
                    OOR_IR_new = torch.cdist(X,IR_new) 
                    loss = criterion(OOR_IR_new, OOR_IR_old)
                    
                    optimizer.zero_grad()   # Zero the gradients
                    loss.backward()         # Backpropagation
                    optimizer.step()        # Update weights

                    # 2
                    OOR_Ref_new = torch.cdist(X,ref_new)
                    loss2 = criterion(OOR_Ref_new, OOR_Ref_old)

                    optimizer2.zero_grad()   # Zero the gradients
                    loss2.backward()         # Backpropagation
                    optimizer2.step()        # Update weights

                    pbar.set_postfix({'IR loss': loss.item(), 'Ref loss': loss2.item(), 'OOR loss': loss3.item()})
                    pbar.update(1)

            new_query_pos[oor_idx] = tensor_to_numpy(X)

        return new_query_pos
    
    def out_of_reference(self, query_atlas, 
                         n_samples: int = 20000, 
                         smooth: bool = False, 
                         edge_thresh: float = 0.001, 
                         k: int = 3, 
                         use_cdf: bool = False, 
                         eps: float = 1e-12):
        """
        Calculate out-of-reference score.

        Parameters
        ----------
        query_atlas
            Query data. It should be a SingleOmicsAtlas object.
        n_samples
            Number of samples drawn from the reference atlas.
        smooth
            If toggled on, OOR scores will be smoothed according to nearest neighbors.
        edge_thresh
            Parameter for smoothing.
        k
            Number of nearest neighbors.
        use_cdf
            If toggled on, CDF will be used instead of PDF.
        eps
            Low bound.
        """
        scores1,_ = self.out_of_reference_(query_atlas, n_samples, smooth, edge_thresh, k, True, use_cdf, eps)
        scores2,_ = self.out_of_reference_(query_atlas, n_samples, smooth, edge_thresh, k, False, use_cdf, eps)
        scores = (scores1 + scores2) / 2
        query_atlas.adata.obs['out_of_ref'] = scores

    def out_of_reference_(self, query_atlas, n_samples=10000, smooth=False, edge_thresh=0.001, k=3, use_umap=True, use_cdf=False, eps=1e-12):
        n_refs = self.adata.shape[0]
        n_querys = query_atlas.adata.shape[0]

        eps = eps if eps else self.eps 

        score_list = []
        if True:
            # reference positional shift
            xs_sketch = self.sample(n_samples)
            sketch_adata = sc.AnnData(xs_sketch)
            sketch_adata.var_names = self.hvgs
            if self.layer.lower() != 'x':
                sketch_adata.layers[self.layer] = xs_sketch.copy()

            # reference positional shift
            if query_atlas.layer.lower() != 'x':
                sketch_adata.layers[query_atlas.layer] = xs_sketch.copy()
            zs = query_atlas.model.get_cell_coordinates(get_subdata(sketch_adata, query_atlas.hvgs, query_atlas.layer).values)
            
            if use_umap:
                zs = query_atlas.umap.transform(zs)
            ref_pos_in_query = zs

            # create a nearest neighbor engine
            ref_nbrs = NearestNeighbors(n_neighbors=2)
            ref_nbrs.fit(ref_pos_in_query)

            # compute the background distribution
            distances_, _ = ref_nbrs.kneighbors(ref_pos_in_query)
            distances_bkg = []
            for i in np.arange(n_refs):
                distances_bkg.append(np.max(distances_[i]))
            bkg_dist = gaussian_kde(distances_bkg)
            mean_dist = np.mean(distances_bkg)
            sd_dist = np.std(distances_bkg)

            # compute the distance between query and reference
            if use_umap:
                query_pos = query_atlas.adata.obsm['X_umap']
            else:
                query_pos = query_atlas.adata.obsm['X_sure']
            distances_, _ = ref_nbrs.kneighbors(query_pos)
            distances = []
            for i in np.arange(n_querys):
                distances.append(np.mean(distances_[i]))
            #density = bkg_dist(distances)

            # test
            if use_cdf:
                distances_all = distances_bkg + distances
                distances_all.sort()
                density_all = bkg_dist(distances_all)
                CDF = cdf(density_all, distances_all)
                nbrs_all = NearestNeighbors(n_neighbors=1)
                nbrs_all.fit(np.array(distances_all).reshape(-1, 1))
                ids = nbrs_all.kneighbors(np.array(distances).reshape(-1, 1), return_distance=False)
                density = 1 - CDF[ids]
            else:
                density = bkg_dist(distances)
            # end of test

            density[density<eps] = eps
            score = -np.log10(density)
            score[score<0] = 0
            score[np.isnan(score)] = 0
            score[distances<mean_dist] = 0
            if smooth:
                score = self.smoothing(score, edge_thresh, k)

            score_list.append(score)

        count = 0
        result = None
        for score in score_list:
            if count>0:
                #query_atlas.adata.obs['out_of_ref'] += score
                result += score
            else:
                #query_atlas.adata.obs['out_of_ref'] = score
                result = score
            count += 1

        #query_atlas.adata.obs['out_of_ref'] /= count
        return result / count, mean_dist / sd_dist

    def smoothing(self, xs, edge_thresh=0.001, k=5):
        ys = np.zeros_like(xs)

        queue = PriorityQueue()
        for i in self.network.nodes():
            for j in self.network.neighbors(i):
                queue.put(j, -self.network[i][j]['weight'])

            score_i = 0
            count_i = 0
            while queue.peek() is not None:
                j = queue.get()
                wij = (self.network[i][j]['weight'] + self.network[j][i]['weight']) / 2
                if (count_i < k) and (wij > edge_thresh):
                    score_i += wij * xs[j]
                    count_i += 1
                    
            ys[i] = (xs[i] + score_i) / (1 + count_i)
        
        return ys

    def join(self, another_atlas, n_samples=5000):
        pass 

    def build_network(self, edge_thresh=0.001):
        # metacells
        adj = self.adj.copy()
        adj[adj<edge_thresh] = 0

        self.network = nx.from_numpy_array(adj)
        for i, j in self.network.edges():
            self.network[i][j]['weight'] = adj[i, j]

        initial_pos = dict()
        for i in self.network.nodes():
            initial_pos[i] = self.adata.obsm['X_umap'][i]

        self.network_pos = self.adata.obsm['X_umap'].copy()
        self.adata.obsm['X_graph'] = self.adata.obsm['X_umap'].copy()

    def display_network(self, pheno_key, draw_node_id=False, node_size=15, font_size=15,
                        edge_width=2, legend_loc='best'):
        for i in self.network.nodes():
            self.network.add_node(i, category=self.adata.obs[pheno_key][i])
        
        categories = list(set(nx.get_node_attributes(self.network, 'category').values()))
        num_categories = len(categories)

        color_map = {}
        colors = plt.cm.tab20(np.linspace(0, 1, num_categories))
        colors = ListedColormap(colors)
        for i, category in enumerate(categories):
            color_map[category] = colors(i)

        node_colors = [color_map[self.network.nodes[node]['category']] for node in self.network.nodes()]

        nx.draw(self.network, self.network_pos, with_labels=draw_node_id, node_color=node_colors, 
                width=edge_width, node_size=node_size, font_size=font_size, font_color='black')
        
        if legend_loc is not None:
            handles = [mpatches.Patch(color=color_map[cat], label=cat) for cat in categories]
            plt.legend(handles=handles, title=pheno_key, loc=legend_loc, bbox_to_anchor=(1, 1))

    @classmethod
    def assemble(cls, atlas_list, n_samples=20000, codebook_size=500, 
               latent_dist='normal', likelihood='negbinomial', cuda_id=0, use_jax=True, 
               batch_size=1000, n_epochs=200, learning_rate=0.0001):
        return assemble_single_omics_atlases(atlas_list, n_samples, codebook_size, cuda_id, 
                                             batch_size, n_epochs, latent_dist, likelihood, 
                                             learning_rate, use_jax)

    @classmethod
    def save_model(cls, atlas, file_path, compression=False):
        """Save the model to the specified file path."""
        file_path = os.path.abspath(file_path)

        atlas.sample_adata = None
        atlas.eval()

        if compression:
            with gzip.open(file_path, 'wb') as pickle_file:
                pickle.dump(atlas, pickle_file)
        else:
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(atlas, pickle_file)

        print(f'Model saved to {file_path}')

    @classmethod
    def load_model(cls, file_path, n_samples=10000):
        """Load the model from the specified file path and return an instance."""
        print(f'Model loaded from {file_path}')

        file_path = os.path.abspath(file_path)
        if file_path.endswith('gz'):
            with gzip.open(file_path, 'rb') as pickle_file:
                atlas = pickle.load(pickle_file)
        else:
            with open(file_path, 'rb') as pickle_file:
                atlas = pickle.load(pickle_file)
        
        xs = atlas.sample(n_samples)
        atlas.sample_adata = sc.AnnData(xs)
        atlas.sample_adata.var_names = atlas.hvgs

        zs = atlas.model.get_cell_coordinates(xs)
        ws = atlas.model.soft_assignments(xs)
        atlas.sample_adata.obsm['X_umap'] = atlas.umap.transform(zs)
        atlas.sample_adata.obsm['X_sure'] = zs 
        atlas.sample_adata.obsm['weight'] = ws

        return atlas



def assemble_single_omics_atlases(atlas_list, n_samples=5000, codebook_size=500, cuda_id=0, batch_size=1000, 
                                  n_epochs=200, latent_dist='normal', use_dirichlet=True, zero_inflation=False,
                                  likelihood='negbinomial', learning_rate=0.0001, use_jax=True):
    n_atlases = len(atlas_list)

    hvgs = None
    for i in np.arange(n_atlases):
        if hvgs is None:
            hvgs = set(atlas_list[i].hvgs)
        else:
            hvgs = hvgs & set(atlas_list[i].hvgs)
    hvgs = list(hvgs)

    jit='--jit' if use_jax else ''

    with tempfile.TemporaryDirectory() as temp_dir:
        # generate samples from the learned distributions for assembly
        adatas_to_assembly=[]
        for i in np.arange(n_atlases):
            print(f'Generate {n_samples} samples from SingleOmicsAtlas {i+1} / {n_atlases} ')
            xs_i = atlas_list[i].sample(n_samples)
            df_i = pd.DataFrame(xs_i, columns=atlas_list[i].hvgs)
            xs_i = df_i[hvgs].values
            adata_i = sc.AnnData(xs_i)
            adata_i.obs['atlas_id'] = i
            adatas_to_assembly.append(adata_i)

        # assembly
        adata_to_assembly = sc.concat(adatas_to_assembly)
        temp_count_file = os.path.join(temp_dir, f'temp_counts.txt.gz')
        temp_uwv_file = os.path.join(temp_dir, f'temp_uwv.txt.gz')
        temp_model_file = os.path.join(temp_dir, f'temp_model.pth')

        X = get_data(adata_to_assembly, layer='X')
        U = batch_encoding(adata_to_assembly, batch_key='atlas_id')
        dt.Frame(X).to_csv(temp_count_file)
        dt.Frame(U).to_csv(temp_uwv_file)

        if latent_dist == 'lapacian':
            latent_dist_param='-la'
        elif latent_dist == 'studentt':
            latent_dist_param='-st'
        else:
            latent_dist_param=''
        
        dirichlet = '-dirichlet' if use_dirichlet else ''
        zi = '-zi exact' if zero_inflation else ''

        print(f'Create distribution-preserved atlas with {codebook_size} metacells from {n_samples * n_atlases} samples')
        cmd = f'CUDA_VISIBLE_DEVICES={cuda_id}  SURE --data-file "{temp_count_file}" \
                        --undesired-factor-file "{temp_uwv_file}" \
                        --seed 0 \
                        --cuda {jit} \
                        -lr {learning_rate} \
                        -n {n_epochs} \
                        -bs {batch_size} \
                        -cs {codebook_size} \
                        -likeli {likelihood} {latent_dist_param} {dirichlet} {zi} \
                        --save-model "{temp_model_file}" '
        pretty_print(cmd)
        subprocess.call(f'{cmd}', shell=True)

        sure_model = SURE.load_model(temp_model_file)

        model = SingleOmicsAtlas()
        model.model = sure_model
        model.sure_models_list = []
        model.n_sure_models = 0
        model.hvgs = hvgs
        model.subatlas_list = atlas_list
        model.n_subatlas = len(atlas_list)
        for i in np.arange(n_atlases):
            print(f'Integrate SURE models from SingleOmicsAtlas {i+1} / {n_atlases} ')
            if atlas_list[i].sure_models_list:
                model.sure_models_list += atlas_list[i].sure_models_list
            model.n_sure_models += atlas_list[i].n_sure_models
            model.layer = model.layer

        pheno_keys = set(atlas_list[0].pheno_keys)
        for i in np.arange(n_atlases-1):
            pheno_keys = pheno_keys.union(set(atlas_list[i+1].pheno_keys))
        model.pheno_keys = list(pheno_keys)
        model.summarize_phenotypes(atlas_list=atlas_list, pheno_keys=pheno_keys)

    return model

def aggregate_dataframes(df_list):
    n_dfs = len(df_list)
    all_columns = set(df_list[0].columns)
    for i in np.arange(n_dfs-1):
        all_columns = all_columns.union(set(df_list[i+1].columns))

    for col in all_columns:
        for i in np.arange(n_dfs):
            if col not in df_list[i]:
                df_list[i][col] = 0  

    df = df_list[0]
    for i in np.arange(n_dfs-1):
        df += df_list[i+1]

    df /= n_dfs
    return df

def smooth_y_over_x(xs, ys, knn_k):
    n = xs.shape[0]
    nbrs = NearestNeighbors(n_neighbors=knn_k, n_jobs=-1)
    nbrs.fit(xs)
    ids = nbrs.kneighbors(xs, return_distance=False)
    ys_smooth = np.zeros_like(ys)
    for i in np.arange(knn_k):
        ys_smooth += ys[ids[:,i]]
    ys_smooth -= ys
    ys_smooth /= knn_k-1
    return ys_smooth

def matrix_dotprod(A, B, dtype=torch.float32):
    A = convert_to_tensor(A, dtype=dtype)
    B = convert_to_tensor(B, dtype=dtype)
    AB = torch.matmul(A, B)
    return tensor_to_numpy(AB)

def matrix_elemprod(A, B):
    A = convert_to_tensor(A)
    B = convert_to_tensor(B)
    AB = A * B
    return tensor_to_numpy(AB)

def cdf(density, xs, initial=0):
    CDF = cumtrapz(density, xs, initial=initial)
    CDF /= CDF[-1]
    return CDF



class FaissKNeighbors:
    def __init__(self, n_neighbors=5):
        self.index = None
        self.k = n_neighbors

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        return distances, indices