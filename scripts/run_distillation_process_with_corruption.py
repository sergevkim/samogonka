from argparse import ArgumentParser
from tabnanny import check

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import ResNet18Model, ResNet50Model, ProjectorLayer
from samogonka.modules import DistillationModule
from samogonka.utils.corruptors import MaskCorruptor
from samogonka.utils.lightning import process_lightning_state_dict


def main(args):
    seed_everything(9, workers=True)

    student = ResNet18Model()
    teacher = ResNet50Model()
    ckpt_filename = '/content/samogonka/checkpoints/teacher-epoch=34-val_accuracy=0.84.ckpt'
    teacher_state_dict = torch.load(ckpt_filename)['state_dict']
    teacher.load_state_dict(process_lightning_state_dict(teacher_state_dict))

    corruptor = MaskCorruptor()
    generator = ProjectorLayer(channels_num=512, blocks_num=2)

    module = DistillationModule(
        student=student,
        teacher=teacher,
        corruptor=corruptor,
        generator=generator,
        alpha_coef=0.1,
        learning_rate=0.001,
    )
    datamodule = CIFAR10DataModule(batch_size=2048)
    logger = WandbLogger(project='samogonka')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='val_accuracy',
        mode='max',
        dirpath='checkpoints',
        filename='student-dis-{epoch:02d}-{val_accuracy:.2f}',
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
