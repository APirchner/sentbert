from argparse import ArgumentParser
from typing import Dict, Tuple, Union, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from utils import encode_label


class SemEvalDataModule(pl.LightningDataModule):
    def __init__(self, path_train: str, path_val: str, path_test: str,
                 batch_size: int = 32, num_workers: int = 8):
        super(SemEvalDataModule, self).__init__()
        self._path_train = path_train
        self._path_val = path_val
        self._path_test = path_test
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        argparser = ArgumentParser(parents=[parent_parser], add_help=False)
        argparser.add_argument('--path_train', type=str, required=True, help='Path to training set tsv')
        argparser.add_argument('--path_val', type=str, required=True, help='Path to validation set tsv')
        argparser.add_argument('--path_test', type=str, required=True, help='Path to test set tsv')
        argparser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        argparser.add_argument('--workers', type=int, default=8, help='Number of dataset worker threads')
        return argparser

    def setup(self, stage: str = None) -> None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if stage == 'fit' or stage is None:
            # build torch datasets
            self.data_train = SemEvalData(self._path_train, tokenizer)
            self.data_val = SemEvalData(self._path_val, tokenizer)
        if stage == 'test' or stage is None:
            self.data_test = SemEvalData(self._path_test, tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=self.batch_size,
                          shuffle=True, drop_last=True,
                          pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.data_train, batch_size=self.batch_size,
                          shuffle=False, drop_last=False,
                          pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.data_test, batch_size=self.batch_size,
                          shuffle=False, drop_last=False,
                          pin_memory=torch.cuda.is_available(), num_workers=self.num_workers)


class SemEvalData(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer):
        super(SemEvalData, self).__init__()
        self.tokenizer = tokenizer
        labels, tweets = SemEvalData._read_clean_data(path)
        tweets = self.tokenizer(tweets, padding=True, truncation=True,
                                max_length=142, return_tensors='pt')
        self.data = {'labels': labels, **tweets}

    @staticmethod
    def _read_clean_data(path: str) -> Tuple[List[str], List[str]]:
        data = pd.read_csv(path, sep='\t',
                           header=None, index_col=0, names=['index', 'label', 'tweet'])
        data['tweet'] = data['tweet'].str.strip()  # strip leading/training whitespace
        data['tweet'] = data['tweet'].str.replace(r'http\S+|www.\S+', '', case=False)  # remove links
        data = data[data['tweet'].str.len() < 141]  # restrict to 140 characters
        return data['label'].tolist(), data['tweet'].tolist()

    def __len__(self) -> int:
        return len(self.data['labels'])

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        return {'label': torch.tensor(encode_label(self.data['labels'][item])),
                'input_ids': self.data['input_ids'][item],
                'token_type_ids': self.data['token_type_ids'][item],
                'attention_mask': self.data['attention_mask'][item],
                }
