import json
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import math
import seaborn as sns
import utils
import argparse

sns.set()
sns.set_style("darkgrid")

parser = argparse.ArgumentParser(description='curves')
parser.add_argument("--log_dir", required=True)
args = parser.parse_args()
utils.plot_stuff(args.log_dir)
