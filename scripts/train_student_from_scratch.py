import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


CONFIG_NAME = 'student_from_scratch'


@hydra.main(version_base=None, config_path='../conf', config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    seed_everything(9, workers=True)

    student = instantiate(cfg.student)
    module = instantiate(cfg.module, model=student)
    datamodule = instantiate(cfg.datamodule)
    logger = None#instantiate(cfg.logger)
    callbacks = [
        ModelCheckpoint(
            save_top_k=3,
            monitor='val_accuracy',
            mode='max',
            dirpath='checkpoints',
            filename='student-nodis-{epoch:02d}-{val_accuracy:.2f}',
        )
    ]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    main()
