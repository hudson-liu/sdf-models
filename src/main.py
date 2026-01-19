import yaml
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from neuralop.models.fnogno import FNOGNO
from tqdm import tqdm
import json

from config import Config
from dataset import load_train_val_fold
from train import train_epoch, test_epoch
from models.gno_transolver import GNOTransolver

CONFIGPATH = "../confs/main.yaml"
with open(CONFIGPATH, "r", encoding="utf-8") as f:
    confd = yaml.safe_load(f)
cv = lambda x : Path(confd[x]).expanduser()
confd["data_dir"] = cv("data_dir")
confd["save_dir"] = cv("save_dir")
args = Config(**confd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data = load_train_val_fold(args)
train_dl = DataLoader(train_data, batch_size=1, shuffle=False)
val_dl = DataLoader(val_data, batch_size=1, shuffle=False)

match args.model:
    case "GNOTransolver":
        model = GNOTransolver(
            embedding_channels=128,
            gno_radius=0.04
        )
    case "FNOGNO":
        model = FNOGNO(
            in_channels=0,
            out_channels=1
        )
    case _:
        raise ValueError("Invalid args.model key!!")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# main training loop
HARD_LIMIT = 1000 # stops program from running too long
VAL_FREQ = 3 # validation step occurs every 3 training epochs
patience = 5 # must have n consecutive epochs with no improvement

start_time = time.time()
pbar = tqdm(total=None)
loss_t = []
loss_v = []
best_loss = float("inf")
q = 0
c = 0 # counter of num of consecutive epochs where val loss is increasing
while q < HARD_LIMIT:
    train_loss = train_epoch(model, train_dl, device, optimizer)
    
    pbar.update(1)
    q += 1

    if q % VAL_FREQ == 0:
        val_loss = test_epoch(model, val_dl, device)
        loss_v.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            c = 0
        else:
            c += 1
        if c == patience:
            break

end_time = time.time()
elapsed = end_time - start_time
print(f"Finished training, elapsed time {elapsed}")

norm = "_norm" if args.normalize else ""
savepath = f"../logs/experiment_{args.model}_{args.split}_{args.fold_id}{norm}.json"
log = {
    "model": args.model,
    "split": args.split,
    "fold_id": args.fold_id,
    "normalization": args.normalize,
    "start_time": start_time,
    "end_time": end_time,
    "time_to_train": elapsed,
    "train_errors": loss_t,
    "val_errors": loss_v
}
with open(savepath, "w") as f:
    json.dump(log, f)

