from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import ResNet18Model
from samogonka.modules import ClassificationModule


def main(args):
    seed_everything(9, workers=True)

    #model = ResNet18Model()
    from torchvision.models import resnet18
    import torch.nn as nn
    model = resnet18()
    model.fc = nn.Linear(512, 10, bias=True)

    module = ClassificationModule(model=model, learning_rate=0.001)
    datamodule = CIFAR10DataModule(batch_size=2048)
    logger = WandbLogger(project='samogonka')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='val_accuracy',
        mode='max',
        dirpath='checkpoints',
        filename='student-nodis-{epoch:02d}-{val_accuracy:.2f}',
    )
    trainer = Trainer.from_argparse_args(
        args,
        max_epochs=40,
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
