import glob
import os
import random
import shutil

random.seed(123)
cur_dirs = glob.glob("*/")
split_percentage = 0.9
problem_per_combo = 1

for cdir in cur_dirs:
    # for cdir in ["1/"]:
    data_dirs = glob.glob(os.path.join(cdir, "*.p"))
    num_combo = int(len(data_dirs) / problem_per_combo)
    combo_ind = list(range(num_combo))
    random.shuffle(combo_ind)
    train_dest = os.path.join(cdir, "train")
    if not os.path.exists(train_dest):
        os.mkdir(train_dest)
    test_dest = os.path.join(cdir, "test")
    if not os.path.exists(test_dest):
        os.mkdir(test_dest)
    # train_combos = combo_ind[:num_combo*split_percentage]
    # test_combos = combo_ind[num_combo*split_percentage:]
    for i, combo in enumerate(combo_ind):
        for data_dir in glob.glob(os.path.join(cdir, "steps_{}_*".format(combo))):
            print(data_dir)
            if cdir.split("/")[0] == "1":
                if not os.path.exists(os.path.join(train_dest, data_dir.split("/")[-1])):
                    shutil.copy(data_dir, os.path.join(train_dest, "."))
                if not os.path.exists(os.path.join(test_dest, data_dir.split("/")[-1])):
                    shutil.copy(data_dir, os.path.join(test_dest, "."))
            else:
                if i < num_combo * split_percentage:
                    if not os.path.exists(os.path.join(train_dest, data_dir.split("/")[-1])):
                        shutil.move(data_dir, os.path.join(train_dest, "."))
                else:
                    if not os.path.exists(os.path.join(test_dest, data_dir.split("/")[-1])):
                        shutil.move(data_dir, os.path.join(test_dest, "."))
