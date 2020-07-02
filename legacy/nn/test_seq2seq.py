import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

encoder = torch.load("encoder.pt")
decoder = torch.load("decoder.pt")

lang, pairs = prepareData('Math')
evaluateRandomly(encoder, decoder, lang, pairs)
