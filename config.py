import os
import torch

data_dir = os.getcwd() + "/conll2003"

bert_model = "bert-base-cased"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
