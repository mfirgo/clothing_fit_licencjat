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