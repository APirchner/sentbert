from argparse import ArgumentParser

import torch
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import F1, Accuracy, Precision, Recall


class SentBert(pl.LightningModule):
    def __init__(self, out_classes: int = 3, lr_bert: float = 1e-5, lr_class: float = 1e-4,
                 weight_decay: float = 1e-2, freeze_base: bool = False, train_steps: int = 100):
        super(SentBert, self).__init__()
        self.lr_bert = lr_bert
        self.lr_class = lr_class
        self.weight_decay = weight_decay
        self.train_steps = train_steps
        self.save_hyperparameters()
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=out_classes, return_dict=True)
        if freeze_base:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False
        self.f1 = F1(num_classes=out_classes, average='macro')
        self.acc = Accuracy()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        argparser = ArgumentParser(parents=[parent_parser], add_help=False)
        argparser.add_argument('--lr_bert', type=float, default=1e-5,
                               help='The learning rate for BERT')
        argparser.add_argument('--lr_class', type=float, default=1e-4,
                               help='The learning rate for the classification head')
        argparser.add_argument('--weight_decay', type=float, default=1e-2,
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
        self.acc(torch.argmax(pred['logits'], dim=1), batch['label'])
        self.f1(torch.argmax(pred['logits'], dim=1), batch['label'])
        self.log('val_ce', pred['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_f1', self.f1, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, *args, **kwargs):
        pred = self.bert(batch['input_ids'], batch['attention_mask'], labels=batch['label'])
        self.acc(torch.argmax(pred['logits'], dim=1), batch['label'])
        self.f1(torch.argmax(pred['logits'], dim=1), batch['label'])
        self.log('test_acc', self.acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_f1', self.f1, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.bert.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.lr_bert,
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.bert.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.lr_bert,
                'weight_decay': 0.0,
            },
            {
                'params': [p for n, p in self.bert.classifier.named_parameters() if 'bias' not in n],
                'lr': self.lr_class,
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.bert.classifier.named_parameters() if 'bias' in n],
                'lr': self.lr_class,
                'weight_decay': 0,
            }
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_steps // 100, num_training_steps=self.train_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
