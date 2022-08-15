import os
if os.path.basename(os.getcwd())=='tests':
    os.chdir("./..")
if os.path.basename(os.getcwd())=='experiments':
    os.chdir("./..")
import sys
sys.path.append(os.getcwd())
import wandb
from model.hierarchical_size_model_new import *
from utils.experiment import *
import itertools
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default="set_update_order")
parser.add_argument('--notes', type=str, default=None)
args = parser.parse_args()
additional_notes = args.notes
group = args.group

# create configs for different learning parameters
configs = []
for element in itertools.permutations(["mu_c", "mu_a", "sigma_c", "eta_r"]):
    config = default_config(); config.update({"update_order": element})
    configs.append( config )

# run experiments
print("Running experiments with different update order")
for config in configs:
    run_experiment(config, group=group, notes=additional_notes)