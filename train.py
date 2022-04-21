import config
import pandas as pd
from data_loader import PromptDataset
from torch.utils.data import DataLoader


def read_data(path):
    tmp = pd.read_csv(path, sep=",").keys()
    data = pd.read_csv(path, sep=",").values.tolist()
    df = pd.DataFrame(data, columns=["input_text", "target_text"])
    return data, df


if __name__ == '__main__':
    train_data, train_df = read_data(config.data_dir + "/train.csv")
    dev_data, dev_df = read_data(config.data_dir + "/dev.csv")

    train_dataset = PromptDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                  collate_fn=train_dataset.collate_fn)

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_dataloader):
        print(input_ids)

