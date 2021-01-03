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
    data.prepare_data()
    data.setup('fit')
    total_steps = args.max_epochs * len(data.data_train)
    effective_steps = total_steps // (args.gpus * args.num_nodes * args.accumulate_grad_batches)
    model = SentBert(out_classes=3, lr=args.learning_rate,
                     weight_decay=args.weight_decay, train_steps=effective_steps)
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator='ddp',
        auto_select_gpus=True,
        default_root_dir=args.log_dir,
        num_sanity_val_steps=2,
        profiler='simple'
    )
    trainer.fit(model=model, datamodule=data)
    data.setup('test')
    trainer.test()


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
