# additional functions taken from dataset.py
import numpy as np
import random
import torch
from torch_geometric import nn as nng
from torch_geometric.data import Data, Dataset

def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def get_surf_pc(data: Data, max_n_point=8192, normalize=True):
    # data.surf is booleans; True -> surf, False -> not part of surface
    # surf_indices is corresponding indexes of those "True"'s
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    surf_pc = data.pos[surf_indices].clone()

    if normalize:
        surf_pc = pc_normalize(surf_pc)

    return surf_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    """blindly creates edges within certain radius"""
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data

class GraphDataset(Dataset):
    def __init__(self, datalist: list[Data], use_cfd_mesh=True, r=None):
        super().__init__()
        self.datalist = datalist

        # rather than using cfd mesh connectivity, 
        # just create proximity-based edges.
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_surf_pc(data)
        return data, shape

