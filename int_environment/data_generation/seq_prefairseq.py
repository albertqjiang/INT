import os.path as osp
import os
import pickle
import collections
from io import open
import random
import argparse

from data_generation.gen_seq2seq import generate_multiple_seq2seq


def control_symbols(string):
    if not string:
        return []
    else:
        symbs = string.strip().split(",")
        return [sym.encode("ascii") for sym in symbs]


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, _ = list(zip(*pairs))

    with open(name, "wb") as f:
        for word in words:
            if isinstance(word, str):
                f.write(str.encode(word))
            else:
                f.write(word)
            f.write("\n".encode("ascii"))


class Chars:
    def __init__(self, special_words):
        self.special_words = special_words
        self.list = []

    def update(self, string):
        if len(string) > 1 and string not in self.special_words:
            self.update_seq(string)
        else:
            if string == " ":
                self.list.append("<space>")
            else:
                self.list.append(string)

    def update_seq(self, seq_string):
        for s in seq_string:
            self.update(s)


def filter_arrow(string, control_syms="<consequences> <consequences_others> <used_local_facts> <SEP>"):
    arrow_find = False
    special_word = ""
    chars = Chars(control_syms)
    for s in string:
        if s == "<":
            if not arrow_find:
                arrow_find = True
            else:
                chars.update_seq(special_word)
                special_word = ""
        if arrow_find:
            special_word += s
            if s == ">":
                arrow_find = False
                chars.update(special_word)
                special_word = ""
        else:
            chars.update(s)
    return chars.list


def chared_to_string(chared_list):
    # convert a character-level sequence back to the executable string
    seq_list = list()
    for element in chared_list:
        if element == "<space>":
            seq_list.append(" ")
        else:
            seq_list.append(element)
    return "".join(seq_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accurarcy.')
    parser.add_argument('--input', type=str,
                        help='The input path')
    parser.add_argument('--out', type=str,
                        help='The output path')

    args = parser.parse_args()

    if not osp.isdir(args.out):
        os.makedirs(args.out)

    random.seed(213)

    all_problems = list()

    for file_name in sorted(os.listdir(args.input)):
        if file_name.startswith("problems_part"):
            all_problems.extend(pickle.load(open(osp.join(args.input, file_name), "rb")))

    distinct_problems = list()
    distinct_goals = set()
    for problem in all_problems:
        if problem[0]["observation"]["objectives"][0].name not in distinct_goals:
            distinct_goals.add(problem[0]["observation"]["objectives"][0].name)
            distinct_problems.append(problem)

    random.shuffle(distinct_problems)
    train_problems, valid_problems, test_problems = \
        distinct_problems[:-2000], distinct_problems[-2000:-1000], distinct_problems[-1000:]
    pickle.dump(train_problems, open(osp.join(args.input, "train_problems.pkl"), "wb"))
    pickle.dump(valid_problems, open(osp.join(args.input, "valid_problems.pkl"), "wb"))
    pickle.dump(test_problems, open(osp.join(args.input, "test_problems.pkl"), "wb"))

    train_sources_to_targets = generate_multiple_seq2seq(train_problems)
    valid_sources_to_targets = generate_multiple_seq2seq(valid_problems)
    test_sources_to_targets = generate_multiple_seq2seq(test_problems)

    randomised_keys = list(train_sources_to_targets.keys())
    random.shuffle(randomised_keys)
    with open(os.path.join(args.input, "train.src"), "w") as src_out:
        with open(os.path.join(args.input, "train.tgt"), "w") as tgt_out:
            for key in randomised_keys:
                src_out.write(key)
                src_out.write("\n")
                tgt_out.write(train_sources_to_targets[key])
                tgt_out.write("\n")

    randomised_keys = list(valid_sources_to_targets.keys())
    random.shuffle(randomised_keys)
    with open(os.path.join(args.input, "valid.src"), "w") as src_out:
        with open(os.path.join(args.input, "valid.tgt"), "w") as tgt_out:
            for key in randomised_keys:
                src_out.write(key)
                src_out.write("\n")
                tgt_out.write(valid_sources_to_targets[key])
                tgt_out.write("\n")

    randomised_keys = list(test_sources_to_targets.keys())
    random.shuffle(randomised_keys)
    with open(os.path.join(args.input, "test.src"), "w") as src_out:
        with open(os.path.join(args.input, "test.tgt"), "w") as tgt_out:
            for key in randomised_keys:
                src_out.write(key)
                src_out.write("\n")
                tgt_out.write(test_sources_to_targets[key])
                tgt_out.write("\n")

    file_names = ["train.src", "train.tgt", "valid.src", "valid.tgt", "test.src", "test.tgt"]
    counter = collections.Counter()
    for file_name in file_names:
        with open(osp.join(args.input, file_name), 'r', encoding='utf-8') as f:
            data = f.read()

            print('Length of {}: {}'.format(file_name, len(data)))

            print('- Tokenizing...')
            # Change space ' ' to underscore '_'
            str_list = filter_arrow(data)
            counter.update(str_list)
            print('- Writing...')
            open(osp.join(args.out, file_name), 'w').write(" ".join(str_list))

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    chars, counts = list(zip(*count_pairs))

    vocab = {}
    count = 0
    ctl_symbols = control_symbols("<pad>,<eos>,<unk>")
    for sym in ctl_symbols:
        vocab[sym] = len(vocab)

    for char, freq in zip(chars, counts):
        if char in vocab:
            print("Warning: found duplicate token %s, ignored" % char)
            continue

        vocab[char] = len(vocab)
        count += freq
    save_vocab(osp.join(args.out, "vocab.txt"), vocab)

    print("Total words: %d" % sum(counts))
    print("Unique words: %d" % len(chars))
    print("Vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))
