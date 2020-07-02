import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from legacy.nn.autoencoder import *


def evaluate_to_file(encoder, decoder, lang, pairs, n=1000):
    with open("../test_results/logic_statements_autoencoder{}.txt".format(n), "w") as fh:
        for _ in range(n):
            pair = random.choice(pairs)
            fh.write('>' + pair[0] + '\n')
            fh.write('=' + pair[1] + '\n')
            output_words, attentions = evaluate(encoder, decoder, lang, pair[0])
            output_sentence = ' '.join(output_words)
            fh.write('<' + output_sentence + '\n')
            fh.write('' + '\n')


if __name__ == "__main__":
    encoder = torch.load("../pt_models/logic_statements_autoencoder_encodercuda.pt")
    decoder = torch.load("../pt_models/logic_statements_autoencoder_decodercuda.pt")

    lang, pairs = prepareData('Math')
    evaluate_to_file(encoder, decoder, lang, pairs)
