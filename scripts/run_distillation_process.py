from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import ResNet18Model, ResNet50Model
from samogonka.modules import DistillationModule
from samogonka.utils.lightning import process_lightning_state_dict


def main(args):
    student = ResNet18Model()

    teacher = ResNet50Model()
    ckpt_filename = '/Users/sergevkim/Downloads/teacher-cifar10-epoch=03-val_loss=0.73.ckpt'
    teacher_ckpt = torch.load(
        ckpt_filename,
        map_location='cpu',
    )
    teacher.load_state_dict(process_lightning_state_dict(teacher_ckpt['state_dict']))

    module = DistillationModule(student=student, teacher=teacher)
    datamodule = CIFAR10DataModule()
    logger = WandbLogger(project='samogonka')
    trainer = Trainer.from_argparse_args(args, accelerator='cpu', logger=logger)
    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
