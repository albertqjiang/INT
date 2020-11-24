import os
import argparse
import random


random.seed(213)

parser = argparse.ArgumentParser(description='Mode generator')
parser.add_argument('--path', '-p', default="/scratch/hdd001/home/ajiang/data/INT/seq")
parser.add_argument('--output_path', '-op', default="/scratch/hdd001/home/ajiang/data/INT/seq/preprocessed")
args = parser.parse_args()

if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)


def remove_spaces_and_split(input_string):
    output_string = input_string.replace("  ", " ")
    output_string = output_string.replace(r"\geq ", ">=")
    output_string = output_string.replace(r"\frac", "/")
    output_string = output_string.replace(r"ø", "emp")
    return output_string


sources_to_targets = dict()

for kl_dir in sorted(os.listdir(args.path)):
    if kl_dir.startswith("pre") or kl_dir.startswith("chared"):
        continue
    with open(os.path.join(args.path, kl_dir, "all.src")) as f_src:
        with open(os.path.join(args.path, kl_dir, "all.tgt")) as f_tgt:
            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()
            for src, tgt in zip(src_lines, tgt_lines):
                src_stripped, tgt_stripped = remove_spaces_and_split(src.strip()), remove_spaces_and_split(tgt.strip())
                if src_stripped in sources_to_targets:
                    continue
                sources_to_targets[src_stripped] = tgt_stripped

source_keys_randomised = list(sources_to_targets.keys())
random.shuffle(source_keys_randomised)

train_keys = source_keys_randomised[:-2000]
valid_keys = source_keys_randomised[-2000:-1000]
test_keys = source_keys_randomised[-1000:]

with open(os.path.join(args.output_path, "train.src"), "w") as train_src:
    with open(os.path.join(args.output_path, "train.tgt"), "w") as train_tgt:
        for train_key in train_keys:
            train_src.write(train_key)
            train_src.write("\n")
            train_tgt.write(sources_to_targets[train_key])
            train_tgt.write("\n")

with open(os.path.join(args.output_path, "valid.src"), "w") as valid_src:
    with open(os.path.join(args.output_path, "valid.tgt"), "w") as valid_tgt:
        for valid_key in valid_keys:
            valid_src.write(valid_key)
            valid_src.write("\n")
            valid_tgt.write(sources_to_targets[valid_key])
            valid_tgt.write("\n")

with open(os.path.join(args.output_path, "test.src"), "w") as test_src:
    with open(os.path.join(args.output_path, "test.tgt"), "w") as test_tgt:
        for test_key in test_keys:
            test_src.write(test_key)
            test_src.write("\n")
            test_tgt.write(sources_to_targets[test_key])
            test_tgt.write("\n")
