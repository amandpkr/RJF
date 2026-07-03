import torch as th
import torch.nn.functional as F
import numpy as np
import logging
import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()

class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()
    SPHERICAL = enum.auto() 

class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(shape, mu, sigma, low=0.0, high=1.0):
    mu   = th.as_tensor(mu)
    sigma= th.as_tensor(sigma)
    low  = th.as_tensor(low)
    high = th.as_tensor(high)
    z_low  = th.logit(low)
    z_high = th.logit(high)
    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low  - mu) / sigma
    beta  = (z_high - mu) / sigma
    cdf_alpha = base.cdf(alpha)
    cdf_beta  = base.cdf(beta)
    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)
    Z = mu + sigma * base.icdf(U)
    X = th.sigmoid(Z)
    return X.clamp(low, high)


class Transport:
    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
            PathType.SPHERICAL: path.SphericalICPlan, 
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_type = path_type 
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0, "time distribution shift must be >= 1.0."
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    
    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan, path.SphericalICPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1):
        """Sampling x0 & t based on shape of x1"""

        x0 = th.randn_like(x1)
        
        if self.path_type == PathType.SPHERICAL:
            x0 = F.normalize(x0, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)

        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if dist_options[0] == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        elif dist_options[0] == "logit-normal":
            assert len(dist_options) == 3
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            t = truncated_logitnormal_sample(
                (x1.shape[0],), mu=mu, sigma=sigma, low=t0, high=t1
            )
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")
        
        t = t.to(x1.device)
        
        # self.time_dist_shift = 1 
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)

        return t, x0, x1
    
    def training_losses(self, model, x1, model_kwargs=None):
        """
        x1 is expected to be [B, C, H, W] (e.g., [B, 768, 16, 16])
        No permuting/reshaping is done here.
        """
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1_target = self.sample(x1)

        t, xt, ut = self.path_sampler.plan(t, x0, x1_target)
        scale_factor = xt.shape[1] ** 0.5 
        model_input = xt * scale_factor

        model_output = model(model_input, t, **model_kwargs)

        terms = {}

        if self.path_type == PathType.SPHERICAL and self.model_type == ModelType.VELOCITY:
            dot_prod = (model_output * xt).sum(dim=1, keepdim=True)
            model_output = model_output - dot_prod * xt

        terms['pred'] = model_output

        if self.model_type == ModelType.VELOCITY:
            
            if self.path_type == PathType.SPHERICAL:
                dot_omega = (x0 * x1_target).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                omega = th.acos(dot_omega)
                t_expanded = path.expand_t_like_x(t, x0)
                dist_rem = (1.0 - t_expanded) * omega
                eps = 1e-6
                jacobi_factor = th.sin(dist_rem) / (dist_rem + eps)
                jacobi_factor = th.where(dist_rem < 1e-4, th.ones_like(jacobi_factor), jacobi_factor)
                weight = jacobi_factor ** 2
                terms['loss'] = mean_flat(weight * ((model_output - ut) ** 2))
            else:
                terms['loss'] = mean_flat(((model_output - ut) ** 2))

        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
        return terms

    def get_drift(self):
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) 
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            scale_factor = x.shape[1] ** 0.5  # sqrt(C)
            model_input = x * scale_factor
            model_output = model(model_input, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            return model_output

        return body_fn
    
    def get_score(self):
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        return score_fn


class Sampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def sde_diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        def sde_drift_fn(x, t, model, **kwargs):
            drift_mean = self.drift(x, t, model, **kwargs) - sde_diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
            return drift_mean
        return sde_drift_fn, sde_diffusion_fn
    
    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = lambda x, t, model, **model_kwargs: x - sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t 
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = lambda x, t, model, **model_kwargs: x - self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()
        return last_step_fn

    def sample_sde(self, *, sampling_method="Euler", diffusion_form="SBDM", diffusion_norm=1.0, last_step="Mean", last_step_size=0.04, num_steps=250):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        project_fn = None
        if self.transport.path_type == PathType.SPHERICAL:
            project_fn = lambda x: F.normalize(x, p=2, dim=1)

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
            project_fn=project_fn 
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            
        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * (1 - t1)
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            
            # Project final step too if spherical
            if project_fn is not None:
                x = project_fn(x)
            
            xs.append(x)
            assert len(xs) == num_steps, "Samples does not match the number of steps"
            return xs
        return _sample
    
    def sample_ode(self, *, sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False):
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        project_fn = None
        if self.transport.path_type == PathType.SPHERICAL:
            project_fn = lambda x: F.normalize(x, p=2, dim=1)
            
            if sampling_method not in ["euler", "heun"]:
                print("Warning: Spherical Flow Matching requires manual stepping for projection. Switching to 'euler'.")
                sampling_method = "euler"

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            # time_dist_shift=1,
            time_dist_shift=self.transport.time_dist_shift,
            project_fn=project_fn 
        )
        
        return _ode.sample