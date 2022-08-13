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
import random

####################
# Helper functions #
####################
def create_config(mu_c, mu_a, sigma_c, eta_r, config = None):
    if config is None:
        config = default_config()
    else:
        config = config.copy()
    config["mean_mu_c_isconstant"], config["variance_mu_c_isconstant"] = mu_c, mu_c
    config["mean_mu_a_isconstant"], config["variance_mu_a_isconstant"] = mu_a, mu_a
    config["mean_sigma_c_isconstant"], config["variance_sigma_c_isconstant"] = sigma_c, sigma_c
    config["mean_eta_small_isconstant"], config["variance_eta_small_isconstant"] = eta_r, eta_r
    config["mean_eta_big_isconstant"], config["variance_eta_big_isconstant"] = eta_r, eta_r
    return config

# eta kept is learned
config = default_config(); config.update({"mean_eta_kept_isconstant":False, "variance_eta_kept_isconstant":False})
print(f"Running experiment with config {config}")
run_experiment(config, notes="eta kept learned", group="different learning parameters")

# create configs for different learning parameters
configs = []
for element in itertools.product(*[[True, False] for _ in range(4)]):
    configs.append(create_config(*element))
random.shuffle(configs)

# run experiments
print("Running experiments with different learning parameters")
for config in configs:
    run_experiment(config, group="different learning parameters")