from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from core import BertNSP, STCDataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    args = parser.parse_args()

    model = BertNSP(args)

    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_on_train_epoch_end=False,
        save_top_k=1,
    )

    val_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='val-epoch={epoch:02d}-setp={step}-val_loss={val_loss:.5f}',
        save_last=True,
        save_on_train_epoch_end=False,
        save_top_k=3
    )

    early_stopping = EarlyStopping(
        'val_loss',
        mode='min',
        patience=5,
        check_on_train_epoch_end=False
    )

    trainer = Trainer.from_argparse_args(args, callbacks=[
                                         best_checkpoint_callback, val_checkpoint_callback, early_stopping])
    datamodule = STCDataModule(batch_size=args.batch_size)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)
