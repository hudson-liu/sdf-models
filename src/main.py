import yaml
import torch
from config import Config
from pathlib import Path

from dataset import load_train_val_fold
from models.transolver import Transolver

CONFIGPATH = "../main.yaml"
with open(CONFIGPATH, "r", encoding="utf-8") as f:
    confd = yaml.safe_load(f)
cv = lambda x : Path(confd[x]).expanduser()
confd["data_dir"] = cv("data_dir")
confd["save_dir"] = cv("save_dir")
args = Config(**confd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data = load_train_val_fold(args)

if args.cfd_model == 'Transolver':
    model = Model(n_hidden=256, n_layers=8, space_dim=7,
                  fun_dim=0,
                  n_head=8,
                  mlp_ratio=2, out_dim=4,
                  slice_num=32,
                  unified_pos=0).cuda()

path = f'metrics/{args.cfd_model}/{args.fold_id}/{args.nb_epochs}_{args.weight}'
if not os.path.exists(path):
    os.makedirs(path)

model = train.main(device, train_ds, val_ds, model, path, val_iter=args.val_iter, reg=args.weight)
