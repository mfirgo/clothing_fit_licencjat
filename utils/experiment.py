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
        "evaluation_iterations":[1,2,5,10,20,50,100]
        }

def run_experiment(config=default_config(), notes=None, project="clothing-fit", group=None):
    wandb.init(project=project, config=config, notes=notes, group=group)
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
    wandb.config.update(model.get_model_config())
    evaluations=0
    for i in tqdm(range(wandb.config["max_iter"])):
        model.update()
        log_values = model.get_parameters_stats()
        if i == wandb.config["evaluation_iterations"][evaluations]:
            evaluations+=1
            results = model.predict(test)
            log_values.update(result_stats_size_model(results))
        wandb.log(log_values, step=model.iterations)
    wandb.finish()
    

    