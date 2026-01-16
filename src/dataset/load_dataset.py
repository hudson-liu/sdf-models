from pathlib import Path

from dataset.data_helpers import get_datalist

NUM_FOLDS = 9

def get_samples(root):
    """finds files & generates folds"""
    meshf = root / "mesh-4k"
    sdff = root / "sdf-4k-h64"

    foldsize = len(meshf) // NUM_FOLDS + 1
    dirs = [[]]
    for i in meshf.iterdir():
        for j in sdff.iterdir():
            if i.stem == j.stem:
                if len(dirs[-1]) < foldsize:
                    dirs[-1].append(i.stem)
                else:
                    dirs.append([i.stem])
                
    return dirs

def load_train_val_fold(args, load_existing):
    """main data loading file"""
    samples = get_samples(args.data_dir)
    trainlst = [samples[i] for i in range(NUM_FOLDS) if i != args.fold_id]
    vallst = samples[args.fold_id]

    print("Loading data.")
    if load_existing:
        print("Using preprocessed data...")
    train_dataset = get_datalist(
            args.data_dir, 
            trainlst, 
            savedir=args.save_dir, 
            load_existing=load_existing
    )
    val_dataset = get_datalist(
            args.data_dir, 
            vallst, 
            savedir=args.save_dir, 
            load_existing=load_existing
    )
    print("Loaded data.")
    return train_dataset, val_dataset
 

# example args
class Args:
    data_dir = Path("/resnick/scratch/hliu9/")
    fold_id = 2 # btwn 1 to 9
    save_dir = Path("/resnick/scratch/hliu9/savedata/")

if __name__ == "__main__":
    # For testing "get_samples"
    # ROOT = Path("/resnick/scratch/hliu9/")
    # dirs = get_samples(ROOT)

    # For running full data pipeline
    args = Args()
    load_train_val_fold(args, False)
