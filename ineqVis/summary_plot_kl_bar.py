import utils
import argparse

parser = argparse.ArgumentParser(description='kl-barchart')
parser.add_argument("--log_dir", required=True)
parser.add_argument("--model_dir", required=True)
args = parser.parse_args()
utils.plot_against_kl(args.model_dir, args.log_dir)
