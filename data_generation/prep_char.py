#!/usr/bin/env python
# coding=utf-8

import os
import sys
import zipfile
import os.path as osp
import collections
from io import open
import tqdm
import argparse

parser = argparse.ArgumentParser(description='Calculate accurarcy.')
parser.add_argument('--input', type=str, default='data',
                    help='The input path')
parser.add_argument('--out', type=str,
                    default='', help='The output path')

args = parser.parse_args()


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
        if len(string)>1 and string not in self.special_words:
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
        f = open(osp.join(args.out, file_name), 'w').write(" ".join(str_list))

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
