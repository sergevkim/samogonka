from argparse import ArgumentParser

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(version_base=None, config_path='../conf', config_name='student_from_scratch')
def main(cfg: DictConfig) -> None:
    seed_everything(9, workers=True)

    model = instantiate(cfg.model)
    module = instantiate(cfg.module, model=model)
    datamodule = instantiate(cfg.datamodule)
    logger = instantiate(cfg.logger)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='val_accuracy',
        mode='max',
        dirpath='checkpoints',
        filename='student-nodis-{epoch:02d}-{val_accuracy:.2f}',
    )
    trainer = Trainer(
        max_epochs=40,
        accelerator='gpu',
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    main()
