import json
import os


def is_valid_folder(path):
    return os.path.exists(path)

def clean_name(name):
    name = name.replace('<', '')
    name = name.replace('>', '')
    name = name.replace("'''", '')
    name = name.strip()
    return name


def dump_json(filename: str, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")


def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def dump_txt(filename: str, in_data: str):
    if not filename.endswith('.txt'):
        filename += '.txt'

    with open(filename, 'w') as fbj:
        fbj.write(in_data)


def load_txt(filename: str):
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as fbj:
        return fbj.read()



