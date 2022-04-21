import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import config


class PromptDataset(Dataset):
    def __init__(self, data):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model)
        self.data = data
        # self.dataset = self.process(data)
        self.device = config.device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, data):
        ndata = [tuple(d) for d in data]
        data = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=ndata,
                                                truncation=True,
                                                padding='max_length',
                                                max_length=config.max_length,
                                                return_tensors='pt'
                                                )
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']

        mask_idx = []
        for nd in ndata:
            source_sentence = nd[0].split()
            target_sentence = nd[1].split()
            # 刚刚好不用特殊处理，因为下标从0开始-1，又是倒数第二个词需要mask，与cls和sep位置抵消
            mask_idx.append(len(source_sentence)+len(target_sentence))

        for i in range(config.batch_size):
            label = input_ids[i, mask_idx[i]].reshape(-1).clone()
            if i == 0:
                labels = label
            else:
                labels = torch.cat((labels, label), dim=0)
            input_ids[i, mask_idx[i]] = self.tokenizer.get_vocab()[self.tokenizer.mask_token]

        return input_ids, attention_mask, token_type_ids, labels


