import pandas as pd

def load_dataset(data_path, label_path):
    X = pd.read_csv(data_path, index_col = 0).T
    Y = pd.read_csv(label_path)
    return X, Y
    