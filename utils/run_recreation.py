RUN_DIRECTORY = "saved-runs"
from utils.runs_and_configs import *
from utils.data_preparation import get_data_from_config
from model.hierarchical_size_model_new import *
import os
from os.path import exists
if not exists(RUN_DIRECTORY):
    raise OSError(2, f"Directory does not exist")

def get_run_directory(run_name):
    return f"{RUN_DIRECTORY}/{run_name}"

def safe_run_config_to_file(run):
    with open(get_run_directory(run.name)+"/config.json", "w") as outfile:
        json.dump(run.config, outfile)
    
def safe_run_id_to_file(run):
    with open(get_run_directory(run.name)+"/id", "w") as outfile:
        json.dump(run.id, outfile)

def safe_run_parameters(run_name, model):
    with open(get_run_directory(run_name)+"/parameters", "wb") as file:
            pickle.dump(model.get_parameters(), file)

def get_run_parameters(run_name):
    with open(get_run_directory(run_name)+"/parameters", "rb") as file:
        return pickle.load(file)

def get_run_id_from_file(run_name):
    with open(get_run_directory(run_name)+"/id", "r") as infile:
        id = json.load(infile)
    return id

def get_run_config_from_file(run_name):
    with open(get_run_directory(run_name)+"/config.json", "r") as infile:
        config = json.load(infile)
    return config

def get_run(run_name):
    try:
        id = get_run_id_from_file(run_name)
        run = get_run_by_id(id)
    except:
        run = get_run_by_name(run_name)
        safe_run_id_to_file(run)
    return run

def rerun_run(run_name):
    config = get_run_config_from_file(run_name)  
    if "evaluation_iterations" in config: del config["evaluation_iterations"]
    train = get_data_from_config(config["data_info"]["train"])
    model = HierarchicalSize(train, config["default_learning_rate"], config=config)
    for i in tqdm(range(config["max_iter"])):
        model.update()
        if model.all_converged():
            break
    safe_run_parameters(run_name, model)
    return model

def get_model(run_name):
    run_directory = get_run_directory(run_name)
    if not exists(run_directory):
        os.makedirs(get_run_directory(run_name))
        run = get_run(run_name)
        safe_run_config_to_file(run)
    if not exists(get_run_directory(run_name)+"/config.json"):
        run = get_run(run_name)
        safe_run_config_to_file(run)
    if not exists(get_run_directory(run_name)+"/parameters"):
        model = rerun_run(run_name)
    else:
        config = get_run_config_from_file(run_name)
        parameters = get_run_parameters(run_name)
        train = get_data_from_config(config["data_info"]["train"])
        model = HierarchicalSize(train, config["default_learning_rate"], config=config)
        model.load_parameters(parameters.items())
    return model