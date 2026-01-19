import numpy as np
import time, json, os
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader

from tqdm import tqdm

def train_epoch(
        model: nn.Module, 
        loader: DataLoader, 
        device: torch.device,
        optimizer: torch.optim.Optimizer
    ):
    model.train()
    for (y, x), t in tqdm(loader):
        y = y.to(device)
        x = x.to(device)
        t = t.to(device)
        



def step_model(model, loader, optimizer=None, scheduler=None, train=False):
    """step model, can be used for train or test"""
    if train:
        model.train()
    else:
        model.eval()

    criterion_func = nn.MSELoss(reduction='none')
    losses = []
    for data in loader:
        with torch.set_grad_enabled(train):
            if train and optimizer is not None:
                optimizer.zero_grad()

            data = data.to(device)
            out = model(data)
            targets = data.y
            loss = criterion_func(
                    out[data.surf, 0], 
                    targets[data.surf, 0]
            ).mean(dim=0)
            
            if train and None not in (optimizer, scheduler):
                loss.backward()
                optimizer.step()
                scheduler.step()

        losses.append(loss.item())

    return np.mean(losses)

def main(device, train_dataset, val_dataset, Net, hparams, path, reg=1, val_iter=1, coef_norm=[]):
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        total_steps=(len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        final_div_factor=1000.,
    )
    start = time.time()

    train_loss, val_loss = 1e5, 1e5
    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:
        train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
        train_loss = step_model(device, model, train_loader, optimizer, lr_scheduler, train=False)
        del (train_loader)

        if val_iter is not None and (epoch == hparams['nb_epochs'] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)

            val_loss = step_model(device, model, val_loader)
            del (val_loader)

            pbar_train.set_postfix(train_loss=train_loss, val_loss=val_loss)
        else:
            pbar_train.set_postfix(train_loss=train_loss)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    torch.save(model, path + os.sep + f'model_{hparams["nb_epochs"]}.pth')

    return model

