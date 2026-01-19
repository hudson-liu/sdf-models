import torch.nn as nn
from neuralop.layers.gno_block import GNOBlock

from transolver import Transolver

class GNOTransolver(nn.Module):
    def __init__(self):
        self.transolver = Transolver(n_hidden=256, n_layers=8, space_dim=7,
                  fun_dim=0,
                  n_head=8,
                  mlp_ratio=2, out_dim=4,
                  slice_num=32,
                  unified_pos=0)
        self.gno = GNOBlock() 
