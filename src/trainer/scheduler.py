import numpy as np
import torch
import torch as th

from .decode import choose_method


class Schedule():
    def __init__(self, method, beta_start, beta_end, diffusion_step):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        betas = np.linspace(beta_start, beta_end, diffusion_step)
        betas = torch.from_numpy(betas).float()
        alphas = 1.0 - betas
        alphas_cump = alphas.cumprod(dim=0)
        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = diffusion_step

        self.method = choose_method(method) # decoding method
        self.ets = None

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model, first_step=False, pflow=False):
        if pflow:
            drift = self.method(img_n, t_start, t_end, model, self.betas, self.total_step)
            return drift
        if first_step:
            self.ets = []
        img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)

        return img_next