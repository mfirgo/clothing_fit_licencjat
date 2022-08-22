from selectors import EpollSelector
import wandb
from utils.data_preparation import *
from model.hierarchical_size_model_new import *
from utils.evaluation import *

def evaluation_iterations(max_iter, strategy="linear", value=10, base=10):
    if strategy=="linear":
        result = [i+1 for i in range(max_iter) if (i+1)%value==0]
    if strategy=="exp":
        x=1
        result=[x]
        while x<=max_iter:
            x = x*value
            result.append(int(x))
    if result[-1]!=max_iter:
        result.append(max_iter)
    return result 

def default_config():
    return{
        "default_learning_rate":1,
        "max_iter":100,
        "evaluation_iterations":[1,2,5,10,20,50,100],
        "data_info":{
            "train":{
                "dataset_type":"train-stratified"
                },
            "test":{
                "dataset_type":"test-stratified"
                }
            }
        }

def run_experiment(config=None, notes=None, project="clothing-fit", group=None, finish_if_converged=True, tags=None, entity=None, log_target_prob_every=None, return_run_and_model=False):
    if config is None: config=default_config()
    run = wandb.init(project=project, config=config, notes=notes, group=group, tags=tags, entity=entity)
    if "data_info" not in wandb.config:
        data_info = {"train":{}, "test":{}}
        train = get_processed_renttherunway_data(data_info=data_info["train"])
        test = get_test_runttherunway_data(data_info=data_info["test"])
        wandb.config.update({"data_info": data_info})
    else:
        data_info = wandb.config.data_info
        train = get_data_from_config(data_info["train"])
        test = get_data_from_config(data_info["test"])
    model = HierarchicalSize(train, wandb.config.default_learning_rate, config=wandb.config)
    run.tags = run.tags + model.tags
    wandb.config.update(model.get_model_config())
    improving_target_prob, improving_accuracy, target_prob, accuracy = False, False, 0.03, 0.04
    for i in tqdm(range(wandb.config["max_iter"])):
        model.update()
        log_values = model.get_parameters_stats()
        if model.iterations in wandb.config["evaluation_iterations"]:
            results = model.predict(test)
            log_values.update(result_stats_size_model(results))
            log_values.update(categorical_mean_prob_stats_size_model(results))
            improving_target_prob = log_values["mean_target_probability"]>target_prob; improving_accuracy = log_values["accuracy"] > accuracy
            target_prob, accuracy = log_values["mean_target_probability"], log_values["accuracy"]
        elif log_target_prob_every is not None and model.iterations%log_target_prob_every==0:
            results = model.size_prob(test)
            log_values.update({"mean_target_probability": results["size_prob"].mean(),
                                "mean_log_target_probability":np.log(results["size_prob"]).mean()})
        wandb.log(log_values, step=model.iterations)
        if finish_if_converged and model.all_converged():
            break
    if "mean_target_probability" not in log_values:
        results = model.predict(test)
        log_values.update(result_stats_size_model(results))
        wandb.log(log_values, step=model.iterations)
    run.tags = run.tags + ("model_"+("converged" if model.all_converged() else "not_converged"),)
    run.tags = run.tags + ("accuracy_"+("improving" if improving_accuracy else "not_improving"),)
    run.tags = run.tags + ("target_prob_"+("improving" if improving_target_prob else "not_improving"),)
    wandb.finish()
    if return_run_and_model:
        return run, model
    

    