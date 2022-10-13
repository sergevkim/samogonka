from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import ResNet18Model
from samogonka.modules import ClassificationModule


def main(args):
    seed_everything(9, workers=True)

    model = ResNet18Model()
    module = ClassificationModule(model=model)
    datamodule = CIFAR10DataModule(batch_size=512)
    logger = WandbLogger(project='samogonka')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='val_accuracy',
        mode='max',
        dirpath='checkpoints',
        filename='student-nodis-{epoch:02d}-{val_loss:.2f}',
    )
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=10,
        accelerator='gpu',
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
