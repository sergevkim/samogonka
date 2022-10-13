import os

import hydra
from hydra.utils import instantiate
import pytorch_lightning as ptl
from datetime import datetime


@hydra.main(config_path='.', config_name="config")
def main(cfg):
    print(cfg)
    data_module = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
    os.makedirs(f"/root/hse_distil/ckpts/{dt_string}")
    print(f"/root/hse_distil/ckpts/{dt_string}")

    trainer = ptl.Trainer(
        strategy="ddp",
        accelerator="gpu", devices=1,
        amp_backend='native',
        check_val_every_n_epoch=1,
        logger=ptl.loggers.wandb.WandbLogger(
            save_dir=f"/root/hse_distil/ckpts/",
            name=cfg.exp_name,
            project="distil resnet",
            log_model="all",
        ),
        callbacks=[
            ptl.callbacks.ModelCheckpoint(
                dirpath=f"/root/hse_distil/ckpts/{dt_string}",
                verbose=True,
                save_last=True,
                every_n_epochs=1,
                save_top_k=3,
                monitor="val accuracy",
                mode="max",
            ),
        ],
        max_epochs=100,
        profiler=None,
        replace_sampler_ddp=True,
        sync_batchnorm=True,
        enable_model_summary=True,
        multiple_trainloader_mode='max_size_cycle',
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
