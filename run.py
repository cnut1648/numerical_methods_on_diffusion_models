import hydra
import numpy as np
import torch
from omegaconf import DictConfig



@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    from src.trainer.trainer import Trainer

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = hydra.utils.instantiate(cfg.model).to(device)
    scheduler = hydra.utils.instantiate(cfg.scheduler, method=cfg.method)
    trainer = Trainer(model, scheduler, diffusion_step=cfg.scheduler.diffusion_step, sample_speed=cfg.sample_speed)
    if cfg.get("train"):
        trainer.train(dataset_cfg=cfg.dataset, **cfg.train)
    else:
        trainer.sample_fid(dataset_cfg=cfg.dataset, **cfg.sample)

if __name__ == "__main__":
    main()