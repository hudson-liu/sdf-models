# gets number of points of dataset

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ..src.dataset import get_samples

def same_sdf_query_points(root, samples):
    prev = None
    for s in tqdm(samples):
        file = (root / "sdf-4k-h64" / s).with_suffix(".csv")
        sdfshape = pd.read_csv(file).to_numpy().shape
        if prev is not None:
            assert sdfshape == prev
        prev = sdfshape
    print(f"All {len(samples)} SDF files each are of shape: {prev}")

if __name__ == "__main__":
    root = Path("/resnick/scratch/hliu9/")
    samples = [j for i in get_samples(root) for j in i]
    same_sdf_query_points(root, samples)
