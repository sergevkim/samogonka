from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import ResNet18Model
from samogonka.modules import ClassificationModule


def main(args):
    model = ResNet18Model()
    module = ClassificationModule(model=model)
    datamodule = CIFAR10DataModule()
    logger = WandbLogger(project='samogonka')
    trainer = Trainer.from_argparse_args(args, accelerator='cpu', logger=logger)
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
