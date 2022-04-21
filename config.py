import os
import torch

data_dir = os.getcwd() + "/conll2003"
max_length =50
batch_size = 8

bert_model = "bert-base-uncased"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
