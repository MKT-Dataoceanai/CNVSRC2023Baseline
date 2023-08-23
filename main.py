import logging
import os

import hydra
import torch
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from lightning import ModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from omegaconf import DictConfig, OmegaConf

# Set environment variables and logger level
os.environ["WANDB_SILENT"] = "true"
logging.basicConfig(level=logging.WARNING)


@hydra.main(config_path="conf", config_name="test_multi-speaker")
def main(cfg: DictConfig) -> None:
    seed_everything(42, workers=True)
    # cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]
    if not os.path.exists(cfg.data_root_dir):
        print('cfg.data_root_dir doesn\'t exist!')
        raise RuntimeError
    if not os.path.exists(cfg.code_root_dir):
        print('should set cfg.code_root_dir before running')
        raise RuntimeError
    if os.path.dirname(__file__)+'/' != cfg.code_root_dir \
        and \
        os.path.dirname(__file__) != cfg.code_root_dir:
        print('should set cfg.code_root_dir as current path')
        print(os.path.dirname(__file__))
        raise RuntimeError
    
    cfg.gpus = torch.cuda.device_count()

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename="{epoch}",
        save_top_k=cfg.checkpoint.save_top_k,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Configure logger
    logger = TensorBoardLogger(save_dir='tblog', name=cfg.logger.name)

    # Set modules and trainer
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False) if cfg.gpus > 1 else None
    )

    # Training and testing
    if cfg.train:
        trainer.fit(model=modelmodule, datamodule=datamodule)

        # only 1 process should save the checkpoint and compute WER
        if cfg.gpus > 1:
            torch.distributed.destroy_process_group()

        if trainer.is_global_zero:
            cfg.ckpt_path = ensemble(cfg)
            cfg.transfer_frontend = False
            cfg.gpus = cfg.trainer.gpus = cfg.trainer.num_nodes = 1
            trainer = Trainer(**cfg.trainer, logger=logger, strategy=None)
            modelmodule.model.load_state_dict(
                torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
            )
            trainer.test(model=modelmodule, datamodule=datamodule)
    else:
        modelmodule.model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=lambda storage, loc: storage)
        )
        trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
