import torch.nn as nn
from neuralop.layers.gno_block import GNOBlock

from .transolver import Transolver

class GNOTransolver(nn.Module):
    """specific model for predicting SDFs from mesh geometry"""

    def __init__(self, 
            embedding_channels: int,
            gno_radius: float
        ):
        super().__init__()

        self.gno = GNOBlock(
            in_channels=0, # we are not using f_y, so this is unused
            out_channels=embedding_channels,
            coord_dim=3,
            radius=gno_radius
        ) 
        self.transolver = Transolver(
            space_dim=embedding_channels,
            fun_dim=0, # only spatial dims; we are not passing any function vals
            out_dim=1, # single prediction: sdf values
            n_hidden=256, 
            n_layers=8, 
            n_head=4,
            mlp_ratio=2,
            slice_num=32,
            unified_pos=False
        )

    class _DataWrapper:
        """transolver requires a small data wrapper"""
        def __init__(self, x):
            self.x = x

    def forward(self, y, x):
        embedding = self.gno(y, x)
        wrapped = self._DataWrapper(embedding)
        t_pred = self.transolver(wrapped)
        return t_pred
