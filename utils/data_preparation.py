import opendatasets as od
import kaggle
from os.path import exists
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings

ORIGINAL_DATA_DIRECTORY = 'clothing-fit-dataset-for-size-recommendation'
DATA_DIRECTORY = 'data'
ORIGINAL_MODCLOTH_FILE = 'modcloth_final_data.json'
ORIGINAL_RENTTHERUNWAY_FILE = 'renttherunway_final_data.json'
PROCESSED_MODCLOTH_FILE = 'modcloth.csv'
PROCESSED_RENTTHERUNWAY_FILE = 'renttherunway.csv'

DEFAULT_RANDOM_STATE=2022
DEFAULT_TEST_SIZE=0.1

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
    result_df = result_df.rename(columns = {column_name: column_name+"_original"})
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
    
def get_processed_renttherunway_data(data_info=None):
    datapath = f"{DATA_DIRECTORY}/{PROCESSED_RENTTHERUNWAY_FILE}"
    if not exists(datapath):
        preprocess_renttherunway_data()
    if data_info is not None:
        data_info.update({"datapath":datapath, "dataset":"renttherunway", "dataset_type": "full"})
    return pd.read_csv(datapath)

def get_renttherunway_dataset_filepath(dataset_type="full", test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, dataset="renttherunway"):
    if dataset=="renttherunway":
        base_file = PROCESSED_RENTTHERUNWAY_FILE
    else:
        base_file = dataset+".csv"
    if dataset_type=='full':
        return f"{DATA_DIRECTORY}/{base_file}"
    else:
        return f"{DATA_DIRECTORY}/{dataset_type}_tsize={test_size}_rand={random_state}_{base_file}"

def split_renttherunway_data(test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    df = get_processed_renttherunway_data()
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(get_renttherunway_dataset_filepath("train", test_size, random_state), index=False)
    test.to_csv(get_renttherunway_dataset_filepath("test", test_size, random_state), index=False)

def split_data(dataset="renttherunway", test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    df = get_data_from_config({"dataset":dataset, "dataset_type":"full"})
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train.to_csv(get_datapath_from_config({"dataset_type":"train","dataset":dataset, "test_size":test_size, "random_state":random_state}), index=False)
    test.to_csv(get_datapath_from_config({"dataset_type":"test","dataset":dataset, "test_size":test_size, "random_state":random_state}), index=False)

def fix_train_test_split(train, test, column):
    isin_mask = test[column].isin(train[column])
    add_to_train = test[~isin_mask]
    train = train.append(add_to_train)
    test = test.drop(test[~isin_mask].index)
    return train, test

def stratified_split_renttherunway_data(test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    df = get_processed_renttherunway_data()
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, test = fix_train_test_split(train, test, "user_id")
    train, test = fix_train_test_split(train, test, "item_id")
    train.to_csv(get_renttherunway_dataset_filepath("train-stratified", test_size, random_state), index=False)
    test.to_csv(get_renttherunway_dataset_filepath("test-stratified", test_size, random_state), index=False)

def stratified_split_data(dataset="renttherunway", test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE):
    df = get_data_from_config({"dataset":dataset, "dataset_type":"full"})
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, test = fix_train_test_split(train, test, "user_id")
    train, test = fix_train_test_split(train, test, "item_id")
    train.to_csv(get_datapath_from_config({"dataset_type":"train-stratified","dataset":dataset, "test_size":test_size, "random_state":random_state}), index=False)
    test.to_csv(get_datapath_from_config({"dataset_type":"test-stratified","dataset":dataset, "test_size":test_size, "random_state":random_state}), index=False)

def get_test_runttherunway_data(test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, data_info=None):
    datapath = get_renttherunway_dataset_filepath("test", test_size, random_state)
    if not exists(datapath):
        split_renttherunway_data(test_size=test_size, random_state=random_state)
    if data_info is not None:
        data_info.update({"datapath":datapath, "dataset":"renttherunway", "dataset_type":"test", "test_size":test_size, "random_state":random_state})
    return pd.read_csv(datapath)

def get_train_runttherunway_data(test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, data_info=None):
    datapath = get_renttherunway_dataset_filepath("train", test_size, random_state)
    if not exists(datapath):
        split_renttherunway_data(test_size=test_size, random_state=random_state)
    if data_info is not None:
        data_info.update({"datapath":datapath, "dataset":"renttherunway", "dataset_type":"train", "test_size":test_size, "random_state":random_state})
    return pd.read_csv(datapath)

def get_config_from_filepath(filepath):
    config = {"datapath":filepath}
    split_filepath = filepath.split("/")[-1].split("_")
    if len(split_filepath)==4:
        config["dataset_type"] = split_filepath[0]
        config["test_size"] = float(split_filepath[1].split("=")[1])
        config["random_state"] = int(split_filepath[2].split("=")[1])
        config["dataset"] = split_filepath[-1].split(".")[0]
    elif len(split_filepath) == 1:
        config["dataset_type"] = "full"
        config["dataset"] = split_filepath[-1].split(".")[0]
    else:
        warnings.warn(f"Can't parse filepath {filepath}")
    return config

def fill_data_config_with_default(config):
    if "dataset" not in config:
        config["dataset"] = "renttherunway"
    if "dataset_type" not in config:
        config["dataset_type"]="full"
    if config["dataset_type"] != "full":
        if "test_size" not in config:
            config["test_size"]=DEFAULT_TEST_SIZE
        if "random_state" not in config:
            config["random_state"]=DEFAULT_RANDOM_STATE
        
def get_datapath_from_config(config):
    dataset_type = config["dataset_type"] if "dataset_type" in config else "full"
    test_size = config["test_size"] if "test_size" in config else DEFAULT_TEST_SIZE
    random_state = config["random_state"] if "random_state" in config else DEFAULT_RANDOM_STATE
    dataset = config["dataset"] if "dataset" in config else "renttherunway"
    fill_config_with_default(config)
    return get_renttherunway_dataset_filepath(dataset_type, test_size, random_state, dataset)

def fill_config_with_default(config):
    config["dataset"] = config["dataset"] if "dataset" in config else "renttherunway"
    config["dataset_type"] = config["dataset_type"] if "dataset_type" in config else "full"
    if config["dataset_type"] != "full":
        config["test_size"] = config["test_size"] if "test_size" in config else DEFAULT_TEST_SIZE
        config["random_state"] = config["random_state"] if "random_state" in config else DEFAULT_RANDOM_STATE

def create_data_from_config(config):
    fill_config_with_default(config)
    if config["dataset_type"] in ["train", "test"]:
        split_data(config["dataset"], config["test_size"], config["random_state"])
    elif config["dataset_type"] in ["train-stratified", "test-stratified"]:
        stratified_split_data(config["dataset"], config["test_size"], config["random_state"])
    else:
        raise ValueError(f"This function does not support creating dataset_type {config['dataset_type']}")

def get_data_from_config(config):
    if "datapath" in config:
        datapath = config["datapath"]
        config_from_filepath = get_config_from_filepath(datapath); config_from_filepath.update(config)
        new_datapath = get_datapath_from_config(config_from_filepath)
        if new_datapath != datapath:
            warnings.warn(f"Datapaths {datapath} and {new_datapath} are different")
            config["datapath"] = new_datapath
    else:
        config["datapath"] = get_datapath_from_config(config)
    datapath = config["datapath"]
    if not exists(datapath):
        create_data_from_config(config)
        #raise ValueError(f"File {datapath} does not exist")
    return pd.read_csv(datapath)

KEPT_STRING = 'fit'
FIT_STRING = KEPT_STRING
LARGE_STRING = 'large'
BIG_STRING = LARGE_STRING
SMALL_STRING = 'small'

FIT_LABEL = 0
LARGE_LABEL = 1
BIG_LABEL = LARGE_LABEL
SMALL_LABEL = 2

status_to_number = {FIT_STRING: 0, LARGE_STRING: 1, SMALL_STRING: 2}
number_to_status = {value:key for key, value in status_to_number.items()}