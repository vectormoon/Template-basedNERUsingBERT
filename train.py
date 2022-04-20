import config
import pandas as pd


def read_data(path):
    tmp = pd.read_csv(path, sep=",").keys()
    data = pd.read_csv(path, sep=",").values().tolist()
    df = pd.DataFrame(data, columns=["input_text", "target_text"])
    return data, df


if __name__ == '__main__':
    train_data, train_df = read_data(config.path)
    dev_data, dev_df = read_data(config.path)