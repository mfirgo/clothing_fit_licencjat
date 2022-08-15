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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default="base_tests")
parser.add_argument('--notes', type=str, default=None)
args = parser.parse_args()
additional_notes = args.notes
group = args.group

# default config, 100 iterations
print(f"Running experiment with config {default_config()}")
notes = "default configuration" + (", "+additional_notes) if additional_notes is not None else ""
run_experiment(default_config(), notes=notes, group=group)

# default config, 1000 iterations
config = default_config()
config["max_iter"] = 10000
config["evaluation_iterations"] = [1, 100, 200, 500, 1000, 2000, 5000, 10000] 
print(f"Running experiment with config {config}")
notes = "default configuration, 10000 iterations" + (", "+additional_notes) if additional_notes is not None else ""
run_experiment(default_config(), notes=notes, group=group)