from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from dataset import SemEvalDataModule
from model import SentBert

def main(args: Namespace) -> None:
    data = SemEvalDataModule(
        path_train=args.path_train,
        path_val=args.path_val,
        path_test=args.path_test,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    model = SentBert(out_classes=3, lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = pl.Trainer.from_argparse_args(
        args,
        check_val_every_n_epoch=2,
        accelerator='ddp',
        auto_select_gpus=True,
        default_root_dir=args.log_dir,
        num_sanity_val_steps=2,
        profiler='simple'
    )
    trainer.fit(model=model, datamodule=data)

if __name__ == '__main__':
    argparser = ArgumentParser(add_help=True)
    argparser.add_argument('--log_dir', type=str,
                           help='The base log directory')
    argparser.add_argument('--seed', type=int, default=1234,
                           help='The seed for sampling etc')
    argparser = SemEvalDataModule.add_argparse_args(argparser)
    argparser = SentBert.add_argparse_args(argparser)
    argparser = pl.Trainer.add_argparse_args(argparser)
    args = argparser.parse_args()
    pl.seed_everything(args.seed)
    main(args)