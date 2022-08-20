import os
if os.path.basename(os.getcwd())=='tests':
    os.chdir("./..")
if os.path.basename(os.getcwd())=='experiments':
    os.chdir("./..")
import sys
sys.path.append(os.getcwd())
from model.hierarchical_size_model_new import *
from utils.experiment import *
from utils.runs_and_configs import read_configs_from_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str, )
parser.add_argument('--group', type=str, default="set_parameters_as_constants")
parser.add_argument('--notes', type=str, default=None)
args = parser.parse_args()
notes = args.notes
group = args.group

configs = read_configs_from_file(args.config_file)

# run experiments
print(f"running experiments for {len(configs)} configs from file {args.config_file}")
for config in configs:
    run_experiment(config, group=group, notes=notes)