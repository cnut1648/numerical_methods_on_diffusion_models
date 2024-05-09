import os
import time

import torch
import torch as th
import torch.utils.data as data
import torch.utils.tensorboard as tb
import torchvision.utils as tvu
from omegaconf import DictConfig
from scipy import integrate
from tqdm.auto import tqdm

from ..model.ema import EMAHelper
from ..utils import get_dataset, inverse_data_transform
from tqdm import tqdm


def get_optim(params, config):
    if config['optimizer'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'],
                            betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                            eps=config['eps'])
    elif config['optimizer'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'], momentum=0.9)
    else:
        raise NotImplementedError

    return optim

class Trainer:
    def __init__(self, model, schedule, sample_speed, diffusion_step):
        self.diffusion_step = diffusion_step
        self.sample_speed = sample_speed
        self.schedule = schedule
        self.model = model
        self.device = next(model.parameters()).device

    def train(self,
              dataset_cfg: DictConfig, batch_size, num_workers,
              optim_cfg: DictConfig,
              ema_rate, use_ema: bool, epochs: int,
              save_path):
        dataset, _ = get_dataset(dataset_cfg["dataset"], dataset_cfg["image_size"])
        schedule = self.schedule
        model = self.model
        model = th.nn.DataParallel(model)

        optim = get_optim(model.parameters(), optim_cfg)

        train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        
        total_steps = len(train_loader) * epochs
        pbar = tqdm(total=total_steps)

        ema = None
        if use_ema:
            ema = EMAHelper(mu=ema_rate)
            ema.register(model)

        tb_logger = tb.SummaryWriter(f'temp/tensorboard/{time.strftime("%m%d-%H%M")}')
        epoch, step = 0, 0

        for epoch in range(epochs):
            for i, (img, y) in enumerate(train_loader):
                n = img.shape[0]
                model.train()
                step += 1
                pbar.update(1)
                t = th.randint(low=0, high=self.diffusion_step, size=(n // 2 + 1,))
                t = th.cat([t, self.diffusion_step - t - 1], dim=0)[:n].to(self.device)
                img = img.to(self.device) * 2.0 - 1.0

                img_n, noise = schedule.diffusion(img, t)
                noise_p = model(img_n, t)

                loss = (noise_p - noise).abs().sum(dim=(1, 2, 3)).mean(dim=0)

                optim.zero_grad()
                loss.backward()
                try:
                    th.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg['grad_clip'])
                except Exception:
                    pass
                optim.step()

                if ema is not None:
                    ema.update(model)

                if step % 10 == 0:
                    tb_logger.add_scalar('loss', loss, global_step=step)
                if step % 50 == 0:
                    print(step, loss.item())
                if step % 500 == 0:
                    model.eval()
                    skip = self.diffusion_step // self.sample_speed
                    seq = range(0, self.diffusion_step, skip)
                    noise = th.randn(16, dataset_cfg['channels'], dataset_cfg['image_size'],
                                     dataset_cfg['image_size'], device=self.device)
                    img = self.sample_image(noise, seq, model)
                    img = th.clamp(img * 0.5 + 0.5, 0.0, 1.0)
                    tb_logger.add_images('sample', img, global_step=step)
                    model.train()

                if step % 500 == 0:
                    train_state = [model.state_dict(), optim.state_dict(), epoch, step]
                    os.makedirs(save_path, exist_ok=True)
                    th.save(train_state, os.path.join(save_path, 'train.ckpt'))
                    if ema is not None:
                        th.save(ema.state_dict(), os.path.join(save_path, 'ema.ckpt'))

    def sample_fid(self,
                   method, model_path,
                   batch_size, total_num, dataset_cfg: DictConfig,
                   image_output_path
                   ):
        model = self.model
        pflow = True if method == 'PF' else False
        
        statedicts = th.load(model_path, map_location=self.device)[0]
        statedicts = {
            k.replace("module.", ""): v for k, v in statedicts.items()
        }
            

        model.load_state_dict(statedicts, strict=True)
        model.eval()

        n = batch_size

        skip = self.diffusion_step // self.sample_speed
        seq = range(0, self.diffusion_step, skip)
        image_num = 0

        my_iter = tqdm(range(total_num // n + 1), ncols=120)
        
        os.makedirs(image_output_path, exist_ok=True)

        for _ in my_iter:
            noise = th.randn(n, dataset_cfg['channels'], dataset_cfg['image_size'],
                             dataset_cfg['image_size'], device=self.device)

            img = self.sample_image(noise, seq, model, pflow)

            img = inverse_data_transform(img)
            for i in range(img.shape[0]):
                if image_num+i > total_num:
                    break
                tvu.save_image(img[i], os.path.join(image_output_path, f"{image_num+i}.png"))

            image_num += n

    def sample_image(self, noise, seq, model, pflow=False):
        with th.no_grad():
            if pflow:
                shape = noise.shape
                device = self.device
                tol = 1e-5

                def drift_func(t, x):
                    x = th.from_numpy(x.reshape(shape)).to(device).type(th.float32)
                    drift = self.schedule.denoising(x, None, t, model, pflow=pflow)
                    drift = drift.cpu().numpy().reshape((-1,))
                    return drift

                solution = integrate.solve_ivp(drift_func, (1, 1e-3), noise.cpu().numpy().reshape((-1,)),
                                               rtol=tol, atol=tol, method='RK45')
                img = th.tensor(solution.y[:, -1]).reshape(shape).type(th.float32)

            else:
                imgs = [noise]
                seq_next = [-1] + list(seq[:-1])

                start = True
                n = noise.shape[0]

                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = (th.ones(n) * i).to(self.device)
                    t_next = (th.ones(n) * j).to(self.device)

                    img_t = imgs[-1].to(self.device)
                    img_next = self.schedule.denoising(img_t, t_next, t, model, start, pflow)
                    start = False

                    imgs.append(img_next.to('cpu'))

                img = imgs[-1]

            return img
