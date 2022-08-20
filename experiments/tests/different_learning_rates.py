import os
if os.path.basename(os.getcwd())=='tests':
    os.chdir("./..")
if os.path.basename(os.getcwd())=='experiments':
    os.chdir("./..")
import sys
sys.path.append(os.getcwd())
from model.hierarchical_size_model_new import *
from utils.experiment import *
from utils.runs_and_configs import read_configs_from_file, update_learning_rate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=str, )
parser.add_argument('--group', type=str, default="set_parameters_as_constants")
parser.add_argument('--notes', type=str, default=None)
parser.add_argument('--learning_rates', type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.2, 0.1])
args = parser.parse_args()
additional_notes = args.notes
group = args.group
configs = read_configs_from_file(args.config_file)
learning_rates = args.learning_rates


print(f"running experiments for {len(configs)} configs from file {args.config_file} with learning rates {learning_rates}")

for config in configs:
    for lr in learning_rates:
        modified_config = config.copy()
        update_learning_rate(modified_config, lr)
        notes = f"lr={lr}" + (f", {additional_notes}" if additional_notes is not None else "")
        run_experiment(modified_config, group=group, notes=notes)
