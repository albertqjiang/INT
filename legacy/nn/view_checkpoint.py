import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from legacy.nn.autoencoder import *
import argparse
from legacy.logic.end2endQ import *


def view_and_save(file_name):
    state = torch.load("../pt_models/" + file_name)
    with open("../pt_models/" + file_name + "_examine.txt", "w") as outh:
        outh.write("EPOCH: " + str(state['epoch']))
        if 'iter' in state:
            outh.write("ITERATION: " + str(state['iter']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="file name of the examined checkpoint")
    args = parser.parse_args()
    view_and_save(args.name)
