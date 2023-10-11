import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import (
    set_seed,
)


from train import (
    default_hyper_train,
)

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    args = OmegaConf.create()   # cfg seems to be read-only
    args = OmegaConf.merge(args, cfg.task_args, cfg.model_args, cfg.training_args)
    args = SimpleNamespace(**args)
    args.output_dir = HydraConfig.get().run.dir
    writer = SummaryWriter(log_dir=args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(le5velname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(logging.INFO)

    set_seed(args.seed)

    default_hyper_train(args, writer = writer)

if __name__ == "__main__":
    main()