import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def create_confusion_matrix(results):
    labels = ["fit", "small", "large"]
    confusion_m = pd.DataFrame(confusion_matrix(results['result'], results['predicted_result'], labels=labels), 
                                index=['true_'+l for l in labels], 
                                columns=['predicted_'+l for l in labels])
    return confusion_m
def show_confusion_matrix(results):
    confusion_m = create_confusion_matrix(results)
    sns.heatmap(confusion_m, annot=True, fmt='d')

def compute_mean_target_probability(results):
    return results.apply(lambda row: row[row['result']], axis=1).mean()

def compute_mean_log_probability(results):
    #return np.log(results.apply(lambda row: row[row['result']], axis=1)).mean()
    return results.apply(lambda row: np.log(row[row['result']]), axis=1).mean()
