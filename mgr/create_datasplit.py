import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def create_splits(folders, val_ratio: float, test_ratio: float, seed=42):
    sorted(folders)
    rnd = np.random.RandomState(seed)
    rnd.shuffle(folders)
    num = len(folders)
    test_idx = int(num - num * test_ratio)
    val_idx = int(test_idx - num * val_ratio)

    train = folders[:val_idx]
    val = folders[val_idx:test_idx]
    test = folders[test_idx:]
    return train, val, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates data splits for given folders and saves them")
    parser.add_argument("--root_dirs", type=str, required=True,
                        help="The folder where we look for data (should be regex like '../../*data*/*')")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Ratio of test data")
    parser.add_argument("--split_file", type=str, default="split.csv", help="Name of output csv")
    parser.add_argument("--random_seed", type=int, default=42, help="Which number for the random seed to take")

    args = parser.parse_args()

    #folders = [f for f in glob(args.root_dirs) if Path(f).is_dir()]
    folders = [f for f in glob(args.root_dirs + "/*") if Path(f).is_file()]
    train, val, test = create_splits(folders, args.val_ratio, args.test_ratio, args.random_seed)

    train_df = pd.DataFrame({"folders": train})
    train_df["split"] = "train"
    val_df = pd.DataFrame({"folders": val})
    val_df["split"] = "val"
    test_df = pd.DataFrame({"folders": test})
    test_df["split"] = "test"

    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    df.to_csv(args.split_file, index=False)
