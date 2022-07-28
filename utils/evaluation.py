import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

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
