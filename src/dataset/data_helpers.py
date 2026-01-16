import os

import open3d as o3d
import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data

EPS = 0.0001 # epsilon for float imprecision error

def load_mesh(filename):
    """replaces load_unstructured_grid_data"""
    mesh = o3d.io.read_triangle_mesh(filename) 
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def get_edges_mesh(mesh: o3d.geometry.TriangleMesh, surf_points):
    """generates list of adjacent points (aka edges) from mesh"""
    # lists are not hashable
    adj = mesh.compute_adjacency_list().adjacency_list
    edges_inds = set()
    for vecind, adjinds in enumerate(adj):
        for i in adjinds:
            edges_inds.add((vecind, i))

    # each triangle has three edges, & we consider all orderings as unique
    assert len(edges_inds) == 3 * len(mesh.triangles)

    # get actual pointwise coords
    edges = [[], []]
    for i, j in edges_inds:
        edges[0].append(surf_points[i])
        edges[1].append(surf_points[j])

    return np.array(edges)

def get_edges_sdf(ext_points):
    """
    creates edges for every point in the sdf grid.
    same behavior as the shapenetcar example.
    """
    edges = [[], []]
    for p in ext_points:
        for q in ext_points:
            if check_point(p, q):
                edges[0].append(p)  
                edges[1].append(q)
    return np.array(edges)

def check_point(p, q):
    """
    checks if point is in 1 of 6 immediate 
    positions surrounding central point
    """
    dists = [0.013634, 0.007745, 0.007642]
    v = np.abs(p - q)
    aligned = 0
    neighbor = 0
    for e, i in enumerate(v):
        if EPS < i and i < dists[e] + EPS:
            neighbor += 1
        elif i < EPS:
            aligned += 1
    return aligned == 2 and neighbor == 1

def get_edge_indices(pos, edges):
    """
    there may be a more streamlined version of this,
    but i wanted to keep it close to the example.
    """
    indices = {tuple(i): e for e, i in enumerate(pos)}
    edgeinds = set()
    for i in range(edges.shape[1]):
        edgeinds.add((indices[edges[0][i]], indices[edges[1][i]]))
    edge_index = np.array(list(edgeinds)).T
    return edge_index

def get_datalist(
        root, samples,
        savedir=None, 
        load_existing=False
    ):
    dataset = []
    for s in samples:
        if load_existing and savedir is not None:
            save_path = savedir / s
            if not save_path.exists():
                continue
            x = np.load(save_path / 'x.npy')
            y = np.load(save_path / 'y.npy')
            pos = np.load(save_path / 'pos.npy')
            surf = np.load(save_path / 'surf.npy')
            edge_index = np.load(save_path / 'edge_index.npy')
        else:
            meshf = (root / "mesh-4k" / s).with_suffix(".obj")
            sdff = (root / "sdf-4k-h64" / s).with_suffix(".csv")
            
            mesh = load_mesh(meshf)
            sdfs = pd.read_csv(sdff).to_numpy()
            surf_points = np.array(mesh.vertices)
            ext_points = sdfs[:, :3]

            edges_mesh = get_edges_mesh(mesh, surf_points)
            edges_sdf = get_edges_sdf(ext_points)
            edges = np.c_[edges_sdf, edges_mesh]
            
            # prep for torch geometric
            n_ext = np.size(ext_points, axis=0)
            n_surf = np.size(surf_points, axis=0)
            pos = np.concatenate([ext_points, surf_points])
            x   = np.concatenate([ext_points, surf_points])
            y   = np.concatenate([
                sdfs[:, 3],
                np.zeros(n_surf, dtype=np.float64)
            ])
            surf = np.concatenate([
                np.zeros(n_ext, dtype=bool), 
                np.ones(n_surf, dtype=bool)
            ])
            edge_index = get_edge_indices(pos, edges)

            if savedir is not None:
                save_path = savedir / s
                if not save_path.exists():
                    os.makedirs(save_path)
                np.save(save_path / 'pos.npy', pos)
                np.save(save_path / 'x.npy', x)
                np.save(save_path / 'y.npy', y)
                np.save(save_path / 'surf.npy', surf)
                np.save(save_path / 'edge_index.npy', edge_index) 
        
        pos = torch.tensor(pos)
        x = torch.tensor(x)
        y = torch.tensor(y)
        surf = torch.tensor(surf)
        edge_index = torch.tensor(edge_index)
 
        # create data obj
        data = Data(pos=pos, x=x, y=y, surf=surf, edge_index=edge_index)
        dataset.append(data)

    return dataset
