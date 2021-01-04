from argparse import ArgumentParser, Namespace

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

from dataset import SemEvalData
from sentbert.model import SentBert

def main(args: Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_test = DataLoader(SemEvalData(
        path=args.path_test,
        tokenizer=tokenizer),
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    model = SentBert.load_from_checkpoint(checkpoint_path=args.checkpoint)
    trainer = pl.Trainer.from_argparse_args(args, logger=False)
    trainer.test(model, test_dataloaders=data_test)


if __name__ == '__main__':
    argparser = ArgumentParser(add_help=True)
    argparser.add_argument('--checkpoint', type=str, help='The model checkpoint')
    argparser.add_argument('--path_test', type=str, help='Path to test set txt')
    argparser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    argparser.add_argument('--workers', type=int, default=8, help='Number of dataset worker threads')
    argparser = pl.Trainer.add_argparse_args(argparser)
    args = argparser.parse_args()
    main(args)
