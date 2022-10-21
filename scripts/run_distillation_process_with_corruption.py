import hydra
import torch
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from samogonka.utils.lightning import process_lightning_state_dict

CONFIG_NAME = 'student_with_distillation'


@hydra.main(version_base=None, config_path='../conf', config_name=CONFIG_NAME)
def main(cfg):
    seed_everything(9, workers=True)

    student = instantiate(cfg.student)
    teacher = instantiate(cfg.teacher)

    ckpt_filename = '/content/samogonka/checkpoints/teacher-epoch=34-val_accuracy=0.84.ckpt'
    teacher_state_dict = torch.load(ckpt_filename)['state_dict']
    teacher.load_state_dict(process_lightning_state_dict(teacher_state_dict))

    corruptor = instantiate(cfg.corruptor)
    generator = instantiate(cfg.generator)

    module = instantiate(
        cfg.module,
        student=student,
        teacher=teacher,
        corruptor=corruptor,
        generator=generator,
    )
    datamodule = instantiate(cfg.datamodule)
    logger = instantiate(cfg.logger)
    callbacks = [
        ModelCheckpoint(
            save_top_k=3,
            monitor='val_accuracy',
            mode='max',
            dirpath='checkpoints',
            filename='student-dis-{epoch:02d}-{val_accuracy:.2f}',
        )
    ]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    main()
