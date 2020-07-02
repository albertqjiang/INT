import random
import shutil
import argparse
import os

random.seed(0)


def split(split_dir, split_percentage=0.9):
    train_dir = os.path.join(split_dir, "train")
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    all_combos = set()
    for f_name in os.listdir(split_dir):
        if f_name.startswith("steps"):
            all_combos.add(f_name.split("_")[1])
    all_combos = list(all_combos)

    train_combos = all_combos[:int(split_percentage * len(all_combos))]
    test_combos = all_combos[int(split_percentage * len(all_combos)):]
    train_combos = set(train_combos)
    test_combos = set(test_combos)

    for f_name in os.listdir(split_dir):
        if f_name.startswith("steps"):
            combo_ind = f_name.split("_")[1]
            if combo_ind in train_combos:
                shutil.move(os.path.join(split_dir, f_name), os.path.join(train_dir, f_name))
            elif combo_ind in test_combos:
                shutil.move(os.path.join(split_dir, f_name), os.path.join(test_dir, f_name))
            else:
                raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-sp", "--split_percentage", type=float, required=False,
                    default=0.9)
    ap.add_argument("--s_dir", type=str, required=False,
                    default="/u/ajiang/Projects/ineqSolver/Inequality/data/renaming_dataset/1/")
    args = ap.parse_args()

    split_dir = args.s_dir
    split_percentage = args.split_percentage
    split(split_dir, split_percentage)
