import time
import pickle
from os.path import exists
from os import listdir, remove
import re

def question(question_text):
    answer = input(question_text+ "[y/n] ").lower()
    if answer in ["yes", "y", "ye"]:
        return True
    if answer in ["no", "n"]:
        return False
    else:
        print("Please enter yes or no.")
        return question(question_text)

def current_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def save_model(model, filename, dir = "models/", add_date=True, extension = "model", prevent_overwrite=True):
    if model is not None:
        if add_date:
            filename += current_timestamp()
        filepath = f"{dir}{filename}.{extension}"
        if exists(filepath) and prevent_overwrite:
            if not question(f"file {filepath} already exists. Do you want to overwrite it?"):
                print("Exiting. Model was not saved.")
                return
            else:
                print("Model will be overwritten.")
        with open(filepath, "wb") as file:
            pickle.dump(model, file)

def get_matching_models(filename, dir ="models/", extension = "model"):
    return [f for f in listdir("models/") if re.match(filename+"\d{8}-\d{6}\."+extension, f)]

def delete_files(file_list, dir = ""):
    file_list = [dir+f for f in file_list]
    if question(f"Following files will be deleted : {file_list}. Continue?"):
        for filename in file_list:
            remove(filename)

def delete_models(filename, leave="none" ,dir="models/", extension = "model"):
    models_to_delete = get_matching_models(filename, dir, extension)
    if leave in ["latest", "last", "newest", "new"]:
        models_to_delete.remove(max(models_to_delete))
    if leave in ["old", "oldest", "first"]:
        models_to_delete.remove(min(models_to_delete))
    if leave not in ["no_date"] and exists(f"{dir}{filename}.{extension}"):
        models_to_delete.append(f"{filename}.{extension}")
    delete_files(models_to_delete, dir)


def load_model(filename, dir ="models/", select = "latest", extension = "model"):
    models_with_date = get_matching_models(filename, dir, extension)
    if select is None or select == "none":
        filepath = f"{dir}{filename}.{extension}"
    elif select in ["latest", "last", "newest", "new"]:
        filepath = f"{dir}{max(models_with_date)}"
    elif select in ["first", "oldest", "old"]:
        filepath = f"{dir}{min(models_with_date)}"
    else:
        raise ValueError(f"Unrecognised select value {select}. Select can be: 'none', 'latest', 'first'")
    with open(filepath, "rb") as file:
        print(f"Loading model {filepath}")
        return pickle.load(file)
