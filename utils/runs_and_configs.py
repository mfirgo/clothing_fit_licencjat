import wandb
import pandas as pd
import json

############
# Get runs #
############
WANDB_ENTITY = "clothing-fit-licencjat"
WANDB_PROJECT = "clothing-fit"

def get_runs(entity=WANDB_ENTITY, project=WANDB_PROJECT):
    api = wandb.Api()
    return api.runs(entity+"/"+project)

def get_runs_by_id(run_ids, entity=WANDB_ENTITY, project=WANDB_PROJECT):
    api = wandb.Api()
    runs = []
    for id in run_ids:
        runs.append(api.run(entity+"/"+project+"/"+id))
    return runs

def get_run_by_id(run_id, entity=WANDB_ENTITY, project=WANDB_PROJECT):
    api = wandb.Api()
    return api.run(entity+"/"+project+"/"+run_id)

def get_run_by_name(run_name, entity=WANDB_ENTITY, project=WANDB_PROJECT):
    api = wandb.Api()
    for run in get_runs(entity, project):
        if run.name == run_name:
            return run

def filter_runs_by_name(runs, run_names):
    result =[]
    for run in runs:
        if run.name in run_names:
            result.append(run)
    return result

def get_runs_by_name(run_names, entity=WANDB_ENTITY, project=WANDB_PROJECT):
    return filter_runs_by_name(get_runs(entity, project), run_names)
###############
# select runs #
###############
def get_run_stats(run, summary_fields=["accuracy", "mean_target_probability","RMSE", "_step"], config_fields=["default_learning_rate"]):
    stats = {
        "name": run.name,
        "id": run.id,
        "group": run.group,
        "time": run.created_at,
        "tags": run.tags
    }
    for field in summary_fields:
        stats[field] = run.summary[field] if field in run.summary else None
    for field in config_fields:
        stats[field] = run.config[field] if field in run.config else None
    return stats

def get_runs_df(runs, summary_fields=["accuracy", "mean_target_probability","RMSE", "_step"], config_fields=["default_learning_rate"]):
    runs_stats = []
    for run in runs:
        runs_stats.append(get_run_stats(run, summary_fields, config_fields))
        runs_stats[-1]["notes"] = run.notes
    return pd.DataFrame(runs_stats)

def get_runs_configs(runs):
    run_configs = []
    for run in runs:
        run_configs.append(run.config.copy())
        run_configs[-1].update({"parent_run":run.name})
    return run_configs

def filter_runs_by_summary(runs, field, value, over=True):
    filtered_runs = []
    for run in runs:
        if field in run.summary and (run.summary[field] >= value if over else run.summary[field] <= value):
            filtered_runs.append(run)
    return filtered_runs

def filter_runs_by_config(runs, field, value):
    filtered_runs = []
    for run in runs:
        if field in run.config and (run.config[field] == value):
            filtered_runs.append(run)
    return filtered_runs


def filter_runs_by_tags(runs, tag, no_tag=False):
    tag_runs = []
    no_tag_runs = []
    for run in runs:
        if tag in run.tags:
            tag_runs.append(run)
        else:
            no_tag_runs.append(run)
    if no_tag:
        return no_tag_runs
    else:
        return tag_runs

def filter_runs_by_data_config(runs, field, value, part="train"):
    filtered_runs = []
    for run in runs:
        if "data_info" in run.config and run.config["data_info"][part][field] == value:
            filtered_runs.append(run)
    return filtered_runs

def filter_runs_by_group(runs, groups, included=False):
    included_runs = []
    excluded_runs = []
    for run in runs:
        if run.group in groups:
            included_runs.append(run)
        else:
            excluded_runs.append(run)
    if included:
        return included_runs
    else:
        return excluded_runs

#################
# Modify config #
#################
def delete_parameter_variables(config, suffixes=("_value", "_init"), debug=True):
    to_delete = []
    for key, value in config.items():
        if key.endswith(suffixes): to_delete.append(key)
    if debug: print("following keys will be deleted:", to_delete)
    for key in to_delete:
        del config[key]

def update_learning_rate(config, learning_rate, all_variables=True, debug=False):
    if all_variables:
        delete_parameter_variables(config, ("_learning_rate",), debug=debug)
        config["default_learning_rate"]=learning_rate
    else:
        previous_default = config["default_learning_rate"]
        for key in config.keys():
            if key.endswith("_learning_rate") and config[key]==previous_default:
                config[key] = learning_rate

def clean_config(config, default_learning_rate=1, new_data_info=None, debug=True):
    delete_parameter_variables(config, debug=debug)
    update_learning_rate(config, learning_rate=default_learning_rate, all_variables=True)
    if "model_name" in config: del config["model_name"]
    if new_data_info is not None:
        set_data_info(config, new_data_info)

def set_data_info(config, new_data_info):
    config["data_info"] = new_data_info

def set_evaluation(config, max_iter=100, evaluation_iterations=None, every=None):
    config["max_iter"] = max_iter
    if every is not None:
        evaluation_iterations = [i for i in range(1, max_iter+1, every)]
    if evaluation_iterations is None:
        evaluation_iterations = [1,2,5,10,20,50,100]
    config["evaluation_iterations"] = evaluation_iterations

def stratified_data_info(dataset=None, random_state=None, test_size=None):
    test_info = {"dataset_type": "test-stratified"}
    train_info = {"dataset_type": "train-stratified"}
    if dataset is not None: train_info["dataset"], test_info["dataset"] = dataset, dataset
    if random_state is not None : train_info["random_state"], test_info["random_state"] = random_state, random_state
    if test_size is not None :train_info["test_size"], test_info["test_size"] = test_size, test_size
    return {"train": train_info, "test": test_info}

##############################
# Saving and reading configs #
##############################
def safe_configs_to_file(configs, filename):
    with open(f"./experiments/tests/configs/{filename}.json", "w") as outfile:
        json.dump(configs, outfile)

def read_configs_from_file(filename):
    with open(f"./experiments/tests/configs/{filename}.json", "r") as infile:
        configs = json.load(infile)
    return configs