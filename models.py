from typing import List
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification


class BERT(nn.Module):
    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 num_labels: int = 2,
                 max_length: int = 64,
                 device='cpu'
                 ):
        super(BERT, self).__init__()

        # general
        self.device = device

        # models and their parameters
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self._bert_finetune_last_layers()  # Only train the final layers of bert. Freeze all the others

    def forward(self, **batch):
        # Bert
        titles_tokenized = self.tokenize(batch['titles'])
        batch = {k: v.to(self.device) for k, v in titles_tokenized.items() if k not in ['titles']}
        return self.bert(**batch)

    def tokenize(self, titles: List[str]):
        return self.tokenizer(titles, padding='max_length', max_length=self.max_length, truncation=True,
                              return_tensors='pt')

    def _bert_finetune_last_layers(self):
        # this will make only the last encoding layers to be learned
        # set the other layers to be frozen
        layers_to_learn = ["classifier.", "pooler", "encoder.layer.11"]
        for name, param in self.bert.named_parameters():
            to_update = [True if layer in name else False for layer in layers_to_learn]
            if any(to_update):
                param.requires_grad = True
            else:
                param.requires_grad = False


class tBERT(nn.Module):
    pass