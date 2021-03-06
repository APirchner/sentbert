import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from dataset import SemEvalDataModule
from model import SentBert


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def main(args: Namespace):
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(storage=os.path.join('sqlite:///{}'.format(args.log_dir), 'optuna.db'),
                                study_name=args.study_name,
                                load_if_exists=True,
                                direction='minimize', pruner=pruner
                                )

    def objective(trial: optuna.Trial):
        log_dir = os.path.join(args.log_dir, 'trial_{}'.format(trial.number))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=log_dir,
            monitor='val_ce',
            mode='min'
        )

        data = SemEvalDataModule(
            path_train=args.path_train,
            path_val=args.path_val,
            batch_size=trial.suggest_categorical('batch_size', choices=[16, 32, 64]),
            num_workers=args.workers
        )
        data.prepare_data()
        data.setup('fit')

        epochs = trial.suggest_categorical('epochs', choices=[3, 4, 5])
        lr_bert = trial.suggest_loguniform('lr_bert', 1e-6, 1e-4)
        lr_class = trial.suggest_loguniform('lr_class', 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1)

        total_steps = epochs * len(data.data_train)
        effective_steps = total_steps // (min(args.gpus, 1) * args.num_nodes * args.accumulate_grad_batches)

        model = SentBert(out_classes=3, lr_bert=lr_bert, lr_class=lr_class,
                         weight_decay=weight_decay, train_steps=effective_steps)
        metrics_callback = MetricsCallback()
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_ce')
        trainer = pl.Trainer.from_argparse_args(
            args,
            default_root_dir=args.log_dir,
            max_epochs=epochs,
            checkpoint_callback=True,
            accelerator='ddp',
            auto_select_gpus=True,
            num_sanity_val_steps=0,
            profiler='simple',
            callbacks=[checkpoint_callback, metrics_callback, pruning_callback]
        )
        trainer.fit(model=model, datamodule=data)
        return metrics_callback.metrics[-1]['val_ce'].item()

    study.optimize(objective, n_trials=10, n_jobs=1)

    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    argparser = ArgumentParser(add_help=True)
    argparser.add_argument('--log_dir', type=str,
                           help='The base log directory')
    argparser.add_argument('--study_name', type=str,
                           help='The name of the Optuna study')
    argparser.add_argument('--seed', type=int, default=1234,
                           help='The seed for sampling etc')
    argparser = SemEvalDataModule.add_argparse_args(argparser)
    argparser = SentBert.add_argparse_args(argparser)
    argparser = pl.Trainer.add_argparse_args(argparser)
    args = argparser.parse_args()
    pl.seed_everything(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)
