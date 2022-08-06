from typing import List
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from sklearn.decomposition import LatentDirichletAllocation


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
        layers_to_learn = ["classifier.", "pooler", "encoder.layer.11", "encoder.layer.10"]
        for name, param in self.bert.named_parameters():
            to_update = [True if layer in name else False for layer in layers_to_learn]
            if any(to_update):
                param.requires_grad = True
            else:
                param.requires_grad = False


class tBERT(nn.Module):
    def __init__(self,
                 corpus: List[str],
                 model_name: str = 'bert-base-uncased',
                 num_labels: int = 2,
                 n_topics: int = 40,
                 alpha: float = None,
                 max_length=64,
                 device='cpu'
                 ):
        super(tBERT, self).__init__()

        # general
        self.device = device

        # models and their parameters
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.bert = BertModel.from_pretrained(model_name)
        self._bert_finetune_last_layers()  # Only train the final layers of bert. Freeze all the others

        self.lda = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=alpha)
        embedded_corpus = self.tokenize(corpus)['input_ids']
        assert len(corpus) == embedded_corpus.shape[0], 'error with the embedded_sentences'
        self.lda.fit(embedded_corpus)

        self.classifier = nn.Sequential(
            nn.Linear(768 + n_topics, 500, bias=True),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(500, num_labels, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, titles: List[str], **kwargs):
        # LDA
        titles_tokenized = self.tokenize(titles)
        outputs_lda = self.lda.transform(titles_tokenized['input_ids'].cpu().detach().numpy())
        outputs_lda = torch.from_numpy(outputs_lda).to(self.device)


        # Bert
        batch_for_bert = {k: v.to(self.device) for k, v in titles_tokenized.items() if k not in ['labels']}
        outputs_bert = self.bert(**batch_for_bert)["last_hidden_state"][:, 0, :]

        # classifier
        outputs_bert_lda = torch.cat([outputs_lda, outputs_bert], axis=1).float().to(self.device)
        return self.classifier(outputs_bert_lda)

    def tokenize(self, sentences: List[str]):
        return self.tokenizer(sentences, padding='max_length', max_length=self.max_length, truncation=True,
                              return_tensors='pt')

    def _bert_finetune_last_layers(self):
        # this will make only the last encoding layers to be learned
        # set the other layers to be frozen
        layers_to_learn = ["classifier.", "encoder.layer.11"]
        for name, param in self.bert.named_parameters():
            to_update = [True if layer in name else False for layer in layers_to_learn]
            if any(to_update):
                param.requires_grad = True
            else:
                param.requires_grad = False
