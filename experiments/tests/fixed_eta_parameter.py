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
from utils.run_recreation import get_run_directory, safe_run_config_to_file, safe_run_parameters, safe_run_id_to_file
import itertools
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default="fixed eta")
parser.add_argument('--notes', type=str, default=None)
parser.add_argument('--mean_small', type=float, default=-0.6)
parser.add_argument('--mean_big', type=float, default=0.54)
parser.add_argument('--var_small', type=float, default=0.0001)
parser.add_argument('--var_big', type=float, default=0.0001)
parser.add_argument('--eta_lr', type=float, default=None)
args = parser.parse_args()
additional_notes = args.notes
group = args.group

print(args.mean_small)
####################
# Helper functions #
####################
def create_config(mu_c, mu_a, sigma_c, config = None):
    if config is None:
        config = default_config()
    else:
        config = config.copy()
    config["mean_mu_c_isconstant"], config["variance_mu_c_isconstant"] = mu_c, mu_c
    config["mean_mu_a_isconstant"], config["variance_mu_a_isconstant"] = mu_a, mu_a
    config["beta_sigma_c_isconstant"] = sigma_c
    return config, "".join(['c' if var else 'v' for var in [mu_c, mu_a, sigma_c]])

def update_config_with_eta(config):
    config["mean_eta_small_value"] = args.mean_small
    config["mean_eta_big_value"] = args.mean_big
    config["variance_eta_small_value"] = args.var_small
    config["variance_eta_big_value"] = args.var_big
    if args.eta_lr is not None:
        config["mean_eta_small_learning_rate"] = args.eta_lr
        config["mean_eta_big_learning_rate"] = args.eta_lr
        config["variance_eta_small_learning_rate"] = args.eta_lr
        config["variance_eta_big_learning_rate"] = args.eta_lr
    else:
        config["mean_eta_small_isconstant"] = True
        config["mean_eta_big_isconstant"] = True
        config["variance_eta_small_isconstant"] = True
        config["variance_eta_big_isconstant"] = True

# create configs for different learning parameters
configs = []
for element in itertools.product(*[[True, False] for _ in range(3)]):
    config, shortened_config = create_config(*element)
    update_config_with_eta(config)
    configs.append((config, shortened_config))
#random.shuffle(configs)

# run experiments
print("Running experiments with different learning parameters and fixed eta")
for config, shortened_config in configs:
    notes = shortened_config + ((", "+additional_notes) if additional_notes is not None else "") + ("" if args.eta_lr is None else f", eta_lr={args.eta_lr}")
    print(notes)
    run, model, acc, target_prob = run_experiment(config, group=group, notes=notes, return_run_and_model=True, wandb_finish=False)
    run_name = run.name
    if acc>0.1 or target_prob>0.05:
        os.makedirs(get_run_directory(run_name))
        #safe_run_config_to_file(run)
        safe_run_parameters(run_name, model)
        safe_run_id_to_file(run)
    wandb.finish()
