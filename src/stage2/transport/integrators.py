import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm
import torch

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        time_dist_shift,
        project_fn=None, 
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.time_dist_shift = time_dist_shift
        self.project_fn = project_fn

    def __Euler_Maruyama_step(self, x, mean_x, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t_curr
        dw = w_cur * th.sqrt(t_curr - t_next)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x - drift * (t_curr - t_next)
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t_curr, t_next, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(t_curr - t_next)
        diffusion = self.diffusion(x, th.ones(x.size(0)).to(x) * t_curr)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(
            xhat, th.ones(x.size(0)).to(x) * t_curr, model, **model_kwargs
        )
        xp = xhat - (t_curr - t_next) * K1
        K2 = self.drift(
            xp, th.ones(x.size(0)).to(x) * t_next, model, **model_kwargs
        )
        return xhat - 0.5 * (t_curr - t_next) * (K1 + K2), xhat

    def __forward_fn(self):
        sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }
        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Sampler type not implemented.")
        return sampler

    def sample(self, init, model, **model_kwargs) -> tuple[th.Tensor]:
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for t_curr, t_next in zip(self.t[:-1], self.t[1:]):
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, t_curr, t_next, model, **model_kwargs)
                
                # Apply Projection if defined
                if self.project_fn is not None:
                    x = self.project_fn(x)
                    mean_x = self.project_fn(mean_x)
                
                samples.append(x)

        return samples

class ode:
    """ODE solver class"""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        time_dist_shift,
        project_fn=None, 
    ):
        assert t0 < t1, "ODE sampler has to be in forward time"

        self.drift = drift
        self.t = 1 - th.linspace(t0, t1, num_steps)
        self.t = time_dist_shift * self.t / (1 + (time_dist_shift - 1) * self.t)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type
        self.project_fn = project_fn

    def sample(self, x, model, **model_kwargs) -> th.Tensor:
            device = x[0].device if isinstance(x, tuple) else x.device
            
            if self.project_fn is not None:
                curr_x = self.project_fn(x)
            else:
                curr_x = x

            samples = []
            
            for i in range(len(self.t) - 1):
                t_curr = self.t[i].to(device)
                t_next = self.t[i+1].to(device)
                dt = t_next - t_curr 

                t_in = th.ones(curr_x.size(0)).to(device) * t_curr
                
                v = self.drift(curr_x, t_in, model, **model_kwargs)
                
                if self.project_fn is not None:
                    v_dot = (v * curr_x).sum(dim=1, keepdim=True)
                    v = v - v_dot * curr_x 
                if self.project_fn is not None:
                    v_norm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
                    
                    angle = v_norm * dt
                    
                    next_x = torch.cos(angle) * curr_x + torch.sin(angle) * (v / v_norm)
                    
                    next_x = self.project_fn(next_x)
                else:
                    next_x = curr_x + v * dt

                samples.append(next_x)
                curr_x = next_x
            
            return th.stack(samples)
