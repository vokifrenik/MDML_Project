import json
import pandas as pd


def load_data():
    # load a json file
    train = pd.read_json('data/processed/train.json')
    test = pd.read_json('data/processed/test.json')

    print('Data loaded successfully')
    return train, test