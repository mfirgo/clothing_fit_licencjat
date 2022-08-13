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

# default config, 100 iterations
print(f"Running experiment with config {default_config()}")
run_experiment(default_config(), notes="default configuration", group="base_tests")

# default config, 1000 iterations
config = default_config()
config["max_iter"] = 10000
config["evaluation_iterations"] = [1, 100, 200, 500, 1000, 2000, 5000, 10000] 
print(f"Running experiment with config {config}")
run_experiment(default_config(), notes="default configuration, 10000 iterations", group="base_tests")