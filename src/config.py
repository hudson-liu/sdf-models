from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """this should be updated along w main.yaml for new entries"""
    data_dir: Path
    save_dir: Path
    log_dir: Path
    normalize: bool
    generate_new_folds: bool
    load_existing_data: bool
    num_folds: int
    fold_id: int
    split: float
    model: str

