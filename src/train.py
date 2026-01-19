import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.enable_grad()
def train_epoch(
        model: nn.Module, 
        loader: DataLoader, 
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module
    ):
    model.train()
    losses = []
    for (y, x), t in tqdm(loader):
        prep = lambda q : q.to(device).squeeze(dim=0) # prepare tensors
        y = prep(y)
        x = prep(x)
        t = prep(t)
        
        optimizer.zero_grad()

        out = model(y, x)
        loss = loss_fn(out, t)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

@torch.no_grad()
def test_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        loss_fn: nn.Module
    ):
    model.eval()
    losses = []
    for (y, x), t in tqdm(loader):
        prep = lambda q : q.to(device).squeeze(dim=0)
        y = prep(y)
        x = prep(x)
        t = prep(t)
        
        out = model(y, x)
        loss = loss_fn(out, t)

        losses.append(loss.item())

    return np.mean(losses)

