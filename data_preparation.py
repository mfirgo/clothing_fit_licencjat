import opendatasets as od
import kaggle
from os.path import exists
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ORIGINAL_DATA_DIRECTORY = 'clothing-fit-dataset-for-size-recommendation'
DATA_DIRECTORY = 'data'
ORIGINAL_MODCLOTH_FILE = 'modcloth_final_data.json'
ORIGINAL_RENTTHERUNWAY_FILE = 'renttherunway_final_data.json'
PROCESSED_MODCLOTH_FILE = 'modcloth.csv'
PROCESSED_RENTTHERUNWAY_FILE = 'renttherunway.csv'

def download_dataset():
    od.download('https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation?select=modcloth_final_data.json')

def is_downloaded(dataset = 'both'):
    if dataset == 'modcloth':
        return exists(f"{ORIGINAL_DATA_DIRECTORY}/{ORIGINAL_MODCLOTH_FILE}")
    if dataset == 'renttherunway':
        return exists(f"{ORIGINAL_DATA_DIRECTORY}/{ORIGINAL_RENTTHERUNWAY_FILE}")
    if dataset == 'both':
        return is_downloaded('modcloth') and is_downloaded('renttherunway')
    else:
        raise ValueError(f"Unknown dataset {dataset}")

def load_original_data(dataset = 'renttherunway'):
    if dataset == 'renttherunway':
        return pd.read_json(f"{ORIGINAL_DATA_DIRECTORY}/{ORIGINAL_RENTTHERUNWAY_FILE}", lines=True)
    if dataset == 'modcloth':
        return pd.read_json(f"{ORIGINAL_DATA_DIRECTORY}/{ORIGINAL_MODCLOTH_FILE}", lines=True)
    else:
        raise ValueError("unknown dataset {dataset}")

def reindex_column(df, column_name):
    result_df = df.sort_values(column_name)
    result_column = []
    current_new, current_old = 0, result_df[column_name][0]
    for i, row in enumerate(result_df[column_name]):
        if row != current_old:
            current_old = row
            if i != 0:
                current_new+=1
        result_column.append(current_new)
    result_df = result_df.rename(columns = {column_name: column_name+"_old"})
    result_df[column_name]= result_column
    return result_df

def prepare_renttherunway_df(original_df):
    chosen_columns = ["result", "user_id", "item_id", "size", "review_date", "category"]
    df = original_df.rename(columns={"fit":"result"})[chosen_columns]
    df = reindex_column(df, "user_id")
    df = reindex_column(df, "item_id")
    df = reindex_column(df, "result")
    return df.sort_index()

def preprocess_renttherunway_data():
    if not is_downloaded('renttherunway'):
        download_dataset()
    df = load_original_data('renttherunway')
    df = prepare_renttherunway_df(df)
    df.to_csv(f"{DATA_DIRECTORY}/{PROCESSED_RENTTHERUNWAY_FILE}", index=False)
    
def get_processed_renttherunway_data():
    datapath = f"{DATA_DIRECTORY}/{PROCESSED_RENTTHERUNWAY_FILE}"
    if not exists(datapath):
        preprocess_renttherunway_data()
    return pd.read_csv(datapath)
        
def split_renttherunway_data():
    df = get_processed_renttherunway_data()
    train, test = train_test_split(df, test_size=0.10, random_state=2022)
    train.to_csv(f"{DATA_DIRECTORY}/train_{PROCESSED_RENTTHERUNWAY_FILE}", index=False)
    test.to_csv(f"{DATA_DIRECTORY}/test_{PROCESSED_RENTTHERUNWAY_FILE}", index=False)

def get_test_runttherunway_data():
    datapath = f"{DATA_DIRECTORY}/test_{PROCESSED_RENTTHERUNWAY_FILE}"
    if not exists(datapath):
        split_renttherunway_data()
    return pd.read_csv(datapath)

def get_train_runttherunway_data():
    datapath = f"{DATA_DIRECTORY}/train_{PROCESSED_RENTTHERUNWAY_FILE}"
    if not exists(datapath):
        split_renttherunway_data()
    return pd.read_csv(datapath)
