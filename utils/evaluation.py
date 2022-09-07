import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import sklearn.metrics

def create_confusion_matrix(result_df,target_column, prediction_column):
    labels = result_df[target_column].unique()
    labels.sort()
    confusion_m = pd.DataFrame(confusion_matrix(result_df[target_column], result_df[prediction_column], labels=labels), 
                                index=['true_'+str(l) for l in labels], 
                                columns=['predicted_'+str(l) for l in labels])
    return confusion_m
    
def show_confusion_matrix(results,target_column, prediction_column):
    confusion_m = create_confusion_matrix(results,target_column, prediction_column)
    sns.heatmap(confusion_m, annot=True, fmt='d')

# def compute_mean_target_probability(results):
#     return results.apply(lambda row: row[row['result']], axis=1).mean()

# def compute_mean_log_probability(results):
#     #return np.log(results.apply(lambda row: row[row['result']], axis=1)).mean()
#     return results.apply(lambda row: np.log(row[row['result']]), axis=1).mean()

def result_stats_size_model(results, target_size_col="size", predicted_size_col="predicted_size", target_probability_col="size_prob", predicted_probability_col="predicted_prob"):
    result_stats = {
        "mean_target_probability" : results[target_probability_col].mean(),
        "mean_log_target_probability" : np.log(results[target_probability_col]).mean(),
        "RMSE" : np.sqrt(((results[target_size_col]-results[predicted_size_col])**2).mean()),
        "accuracy": sklearn.metrics.accuracy_score(results[target_size_col], results[predicted_size_col]),
        "f1_micro": sklearn.metrics.f1_score(results[target_size_col], results[predicted_size_col], average="micro"),
        "f1_macro": sklearn.metrics.f1_score(results[target_size_col], results[predicted_size_col], average="macro"),
        "f1_weighted": sklearn.metrics.f1_score(results[target_size_col], results[predicted_size_col], average="weighted"),
    }
    return result_stats

def categorical_mean_prob_stats_size_model(results, all_size_results="all_sizes_results", target_probability_col="size_prob"):
    categorical_prob = results[target_probability_col]/(results[all_size_results].apply(sum))
    return {
        "mean_target_categorical_probability": categorical_prob.mean(),
        "mean_log_target_categorical_probability": np.log(categorical_prob).mean()
    }

def target_prob_stats(full_result_just_target_prob):
    print(f'mean target prob: {full_result_just_target_prob["target_prob"].mean()}, std: {full_result_just_target_prob["target_prob"].std()}')
    print(f'mean log target prob: {np.log(full_result_just_target_prob["target_prob"]).replace({-np.inf:np.nan}).mean()}, std: {np.log(full_result_just_target_prob["target_prob"]).replace({-np.inf:np.nan}).std()}')
    print(f'mean log target prob no replacement: {np.log(full_result_just_target_prob["target_prob"]).mean()}, std: {np.log(full_result_just_target_prob["target_prob"]).std()}')
    print("number of ommited values (all, target_prob, size_prob, return_status_prob) :", (np.log(full_result_just_target_prob["target_prob"])==-np.inf).sum(), (full_result_just_target_prob["target_prob"]==0).sum(), (full_result_just_target_prob["size_prob"]==0).sum(),(full_result_just_target_prob["return_status_prob"]==0).sum())

def full_model_accuracy_size(full_result, real_size="size", size_model_prediction="predicted_size_fit", full_model_prediction="predicted_pair_size"):
    print("full model, size: ", sklearn.metrics.accuracy_score(full_result[real_size], full_result[full_model_prediction]))
    print("size model, on fit: ", sklearn.metrics.accuracy_score(full_result[real_size], full_result[size_model_prediction]))

def full_model_accuracy_return_status(full_result, real_status="result_original", return_model_prediction="", full_model_prediction="predicted_pair_return_status"):
    print("full model, status: ", sklearn.metrics.accuracy_score(full_result[real_status], full_result[full_model_prediction]))
    #print("status model: ", sklearn.metrics.accuracy_score(full_result[real_status], full_result["result"]))

def size_rmse(full_result, real_size="size", size_model_prediction="predicted_size_fit", full_model_prediction="predicted_pair_size"):
    print("full model, size: ", sklearn.metrics.mean_squared_error(full_result[real_size], full_result[full_model_prediction], squared=False))
    print("size model, on fit: ", sklearn.metrics.mean_squared_error(full_result[real_size], full_result[size_model_prediction], squared=False))

def full_model_stats(full_result, prob_column='target_categorical_prob'):
    if prob_column in full_result.columns:
        print("-- Target probability --")
        print(f'mean target prob: {full_result[prob_column].mean()}, std: {full_result[prob_column].std()}')
        print(f'mean log target prob: {np.log(full_result[prob_column]).replace({-np.inf:np.nan}).mean()}, std: {np.log(full_result[prob_column]).replace({-np.inf:np.nan}).std()}')
        print(f'mean log target prob no replacement: {np.log(full_result[prob_column]).mean()}, std: {np.log(full_result[prob_column]).std()}')
        print("number of ommited values (all, target_prob, size_prob, return_status_prob) :", (np.log(full_result[prob_column])==-np.inf).sum(), (full_result[prob_column]==0).sum())
    #target_prob_stats(full_result)
    print("-- accuracy --")
    full_model_accuracy_size(full_result)
    full_model_accuracy_return_status(full_result)
    print("-- rmse --")
    size_rmse(full_result)
    #print("-- size model stats --")
    #result_stats_size_model(full_result)

def accuracy_vs_coverage(results, prediction_col, target_col, prediction_probability_column, nsamples = 50):
    new_results = results[["user_id", "item_id",target_col, prediction_col, prediction_probability_column]].sort_values(prediction_probability_column)
    records = results.shape[0]
    partition_size = int(np.ceil(records/nsamples))
    acc_vs_coverage = []
    for i in range(1,nsamples+1):
        split_size = min(i*partition_size, records)
        data_for_accuracy = new_results.tail(split_size)
        acc_vs_coverage.append({
            "coverage": split_size/records,
            "accuracy": (data_for_accuracy[target_col]==data_for_accuracy[prediction_col]).mean()
        })
    return acc_vs_coverage