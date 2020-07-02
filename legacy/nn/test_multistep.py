import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from legacy.nn.multi_step_transformation_seq2seq import prepareData, evaluate_to_file

encoder = torch.load("../pt_models/multi_step_encoder{}.pt".format(device))
decoder = torch.load("../pt_models/multi_step_decoder{}.pt".format(device))
output_path = "../test_results/multi-step-all{}.txt".format(device)

input_lang, output_lang, pairs = prepareData('exp', 'tra')
evaluate_to_file(encoder, decoder, pairs, input_lang, output_lang,
                 output_path=output_path)
