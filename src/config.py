from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """this should be updated along w main.yaml for new entries"""
    data_dir: Path
    save_dir: Path
    log_dir: Path

    normalize: bool
    load_existing_data: bool
    use_custom_splits: bool
    use_amp: bool

    model: str
    
    generate_new_folds: bool
    num_folds: int
    fold_id: int
    split: float

