from argparse import ArgumentParser
from tabnanny import check

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import ResNet18Model, ResNet50Model
from samogonka.modules import DistillationModule
from samogonka.utils.lightning import process_lightning_state_dict


def main(args):
    seed_everything(9, workers=True)

    student = ResNet18Model()
    teacher = ResNet50Model()
    ckpt_filename = '/content/samogonka/checkpoints/teacher-epoch=08-val_loss=0.72.ckpt'
    teacher_state_dict = torch.load(ckpt_filename)['state_dict']
    teacher.load_state_dict(process_lightning_state_dict(teacher_state_dict))

    module = DistillationModule(student=student, teacher=teacher)
    datamodule = CIFAR10DataModule()
    logger = WandbLogger(project='samogonka')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='val_accuracy',
        mode='max',
        dirpath='checkpoints',
        filename='student-dis-{epoch:02d}-{val_loss:.2f}',
    )
    trainer = Trainer.from_argparse_args(
        args,
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
