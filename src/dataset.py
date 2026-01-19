import pickle
from pathlib import Path
import math

import open3d as o3d
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from config import Config

def get_stems(root):
    """gets all stems of sdf directories"""
    meshf = root / "mesh-4k"
    sdff = root / "sdf-4k-h64"
    meshes = {i.stem for i in meshf.iterdir()}
    sdfs = {i.stem for i in sdff.iterdir()}
    shared = sorted(meshes & sdfs)
    return shared

def get_samples(args: Config):
    """finds files & generates folds"""
    root = args.data_dir
    
    foldpath = root / f"folds_{args.num_folds}.pkl"

    if foldpath.exists() and not args.generate_new_folds:
        with open(foldpath, "rb") as f:
            dirs = pickle.load(f)
    else:
        shared = get_stems(root)
        foldsize = math.ceil(len(shared) / args.num_folds)
        dirs = {0: []}
        fold = 0
        for p in shared:
            if len(dirs[fold]) < foldsize:
                dirs[fold].append(p)
            else:
                fold += 1
                dirs[fold] = [p]

        with open(foldpath, "wb") as f:
            pickle.dump(dirs, f)
    return dirs

def trainlst_split(samples, fold_id, split):
    """deterministic assuming that samples is always deterministic (which it is)"""
    train = samples.copy()
    train.pop(fold_id)
    train_s = sorted(train.items(), key=lambda x: x[0])
    splitsize = int(len(train) * split)
    trainlst = train_s[:splitsize]
    print(f"Using {split} of total training data")
    return dict(trainlst)

def load_train_val_fold(args: Config):
    """loads data"""
    samples = get_samples(args)
    trainlst = trainlst_split(samples, args.fold_id, args.split)
    vallst = {args.fold_id: samples[args.fold_id]}

    print("Loading data.")
    if args.load_existing_data:
        print("Using preprocessed data...")
    train_dataset = get_datalist_folds(args, trainlst)
    val_dataset = get_datalist_folds(args, vallst)
    print("Loaded data.")
    return train_dataset, val_dataset

def load_train_val_custom(args: Config):
    root = args.data_dir
    shared = np.array(get_stems(root))
    split_dir = root / "split-4k"
    with open(split_dir / "train.txt", "r") as f:
        spl_t = [int(i) for i in f]
    with open(split_dir / "test.txt", "r") as f:
        spl_v = [int(i) for i in f]
    trainlst = shared[spl_t]
    vallst = shared[spl_v]

    print("Loading data.")
    train_dataset = get_datalist_custom(args, trainlst, "train")
    val_dataset = get_datalist_custom(args, vallst, "test")
    print("Loaded data.")
    return train_dataset, val_dataset
 
def load_mesh(filename):
    mesh = o3d.io.read_triangle_mesh(filename) 
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def process_sample(s, args):
    root = args.data_dir
    meshf = (root / "mesh-4k" / s).with_suffix(".obj")
    sdff = (root / "sdf-4k-h64" / s).with_suffix(".csv")
    mesh = load_mesh(meshf)
    sdfs = pd.read_csv(sdff).to_numpy(dtype=np.float32)
    
    y = np.array(mesh.vertices, dtype=np.float32) # (npoints, 3)
    x = sdfs[:, :3] # (npoints, 3)
    t = sdfs[:, 3]
    
    # note y and x are on the same coordinate system
    if args.normalize:
        bbox_max = y.max(axis=0)
        bbox_min = y.min(axis=0)
        norm = (bbox_max - bbox_min) / 2 # normalization factor
        center = (bbox_max + bbox_min) / 2 # coordinate center
        y = (y - center) / norm
        x = (x - center) / norm
        t /= norm.max()

    return y, x, t

def get_datalist_custom(args: Config, split_files: np.ndarray, save_as: str):
    root = args.data_dir
    norm = "_norm" if args.normalize else ""
    save_path = args.save_dir / f"split_{save_as}{norm}.pkl"
    if args.load_existing_data:
        print("Using preprocessed data...")
        with open(save_path, "rb") as f:
            all_y, all_x, all_t = pickle.load(f)
    else:
        all_y = []
        all_x = []
        all_t = []
        for s in tqdm(split_files, desc="Samples"):
            y, x, t = process_sample(s, args)
            all_y.append(y)
            all_x.append(x)
            all_t.append(t)

        with open(save_path, "wb") as f:
             pickle.dump([all_y, all_x, all_t], f)

    dataset = SDFData(all_y, all_x, all_t)

    return dataset

def get_datalist_folds(args: Config, fold_dirs: dict):
    root = args.data_dir
    all_y = [] # all mesh points
    all_x = [] # all query points
    all_t = [] # all target values (sdf values)
    for idx, samples in tqdm(fold_dirs.items(), desc="Folds"):
        norm = "_norm" if args.normalize else ""
        save_path = args.save_dir / f"fold_{idx}{norm}.pkl"
        if args.load_existing_data:
            if not save_path.exists():
                raise FileNotFoundError(f"Fold {idx} could not be loaded!")
            with open(save_path, "rb") as f:
                fold_y, fold_x, fold_t = pickle.load(f)
        else:
            fold_y = [] # (nsamples, npoints, 3)
            fold_x = []
            fold_t = []
            for s in tqdm(samples, desc="Samples", leave=False):
                y, x, t = process_sample(s, args) 
                fold_y.append(y)
                fold_x.append(x)
                fold_t.append(t)
            
            with open(save_path, "wb") as f:
                pickle.dump([fold_y, fold_x, fold_t], f)

        all_y += fold_y
        all_x += fold_x
        all_t += fold_t

    dataset = SDFData(all_y, all_x, all_t)

    return dataset

class SDFData(Dataset):
    """basic torch dataset for sdf data"""

    def __init__(self, 
            y: list[np.ndarray], 
            x: list[np.ndarray], 
            t: list[np.ndarray]
        ):
        # y, x, t does not have to necessarily be regular
        # (can be different n_points btwn samples)
        # so we use lists instead of numpy arrays for flexibility
        assert len(y) == len(x) == len(t)

        self.y = y
        self.x = x
        self.t = t

    def __getitem__(self, idx):
        totensor = lambda x: torch.from_numpy(x)
        y = totensor(self.y[idx])
        x = totensor(self.x[idx])
        t = totensor(self.t[idx]).unsqueeze(dim=1)
        return (y, x), t
    
    def __len__(self):
        return len(self.y)

# testing
if __name__ == "__main__":
    # For testing "get_samples"
    # TODO dont use
    # ROOT = Path("/resnick/scratch/hliu9/")
    # dirs = get_samples(ROOT)

    # For running full data pipeline
    args = Config(
        data_dir=Path("/resnick/scratch/hliu9/"),
        save_dir=Path("/resnick/scratch/hliu9/savedata/"),
        normalize=True,
        generate_new_folds=False,
        load_existing_data=True,
        num_folds=21,
        fold_id=20,
        split=0.1,
        model="GNOTransolver"
    )
    train, val = load_train_val_fold(args)
