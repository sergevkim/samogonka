from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from samogonka.datamodules import CIFAR10DataModule
from samogonka.models import resnet18, resnet34
from samogonka.modules import SelfDistillationModule


def main(args):
    model = resnet18(num_classes=10) if args.model == 'resnet18' else resnet34(num_classes=10)
    module = SelfDistillationModule(model=model, temp=2)
    datamodule = CIFAR10DataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    logger = WandbLogger(project='samogonka', name=args.exp_name)
    trainer = Trainer.from_argparse_args(args, accelerator='gpu', logger=logger, max_epochs=100)
    trainer.fit(
        module,
        datamodule=datamodule
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--exp_name', default='self_distill-resnet18', type=str)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
