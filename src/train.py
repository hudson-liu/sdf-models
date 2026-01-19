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
        optimizer: torch.optim.Optimizer
    ):
    model.train()
    criterion_func = nn.MSELoss(reduction="none")
    losses = []
    for (y, x), t in tqdm(loader):
        y = y.to(device)
        x = x.to(device)
        t = t.to(device)
        
        optimizer.zero_grad()

        out = model(y, x)
        loss = criterion_func(out, t)
        loss.backward()
        optimizer.step()

        losses.append(loss)

    return np.mean(losses)

@torch.no_grad()
def test_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
    ):
    model.eval()
    criterion_func = nn.MSELoss(reduction='none')
    losses = []
    for (y, x), t in tqdm(loader):
        y = y.to(device)
        x = x.to(device)
        t = t.to(device)
        
        out = model(y, x)
        loss = criterion_func(out, t)

        losses.append(loss)

    return np.mean(losses)

