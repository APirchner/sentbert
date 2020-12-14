from argparse import ArgumentParser

import torch.nn.functional as F
from transformers import BertForSequenceClassification, AdamW
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import F1

class SentBert(pl.LightningModule):
    def __init__(self, out_classes: int = 3, lr: float = 1e-5, weight_decay: float = 1e-2):
        super(SentBert, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                  num_labels=out_classes, return_dict=True)
        self.f1 = F1(num_classes=out_classes)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        argparser = ArgumentParser(parents=[parent_parser], add_help=False)
        argparser.add_argument('--learning_rate', type=float, default='1e-5',
                               help='The learning rate')
        argparser.add_argument('--weight_decay', type=float, default='1e-2',
                               help='The weight decay')
        return argparser

    def forward(self, batch):
        return self.bert(batch['input_ids'], batch['attention_mask'])

    def training_step(self, batch, *args, **kwargs):
        pred = self.bert(batch['input_ids'], batch['attention_mask'], labels=batch['label'])
        loss = pred['loss']
        self.log('train_ce', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        pred = self.bert(batch['input_ids'], batch['attention_mask'], labels=batch['label'])
        self.f1(pred.logits, batch['label'])
        self.log('val_f1', self.f1, on_step=False, on_epoch=True)
        self.log('val_ce', pred['loss'], on_step=False, on_epoch=True)

    def test_step(self, batch):
        pass

    def configure_optimizers(self):
        return {
            'optimizer': AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        }
