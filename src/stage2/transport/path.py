import torch as th
import torch.nn.functional as F
import numpy as np
from functools import partial

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    For [B, C, H, W], this returns [B, 1, 1, 1]
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

#################### Coupling Plans ####################

class ICPlan:
    """Linear Coupling Plan"""
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return 1 - t, -1
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return t, 1
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return -1 / (1 - t)

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * t,
            "decreasing": 0.25 * (norm * th.cos(np.pi * (1 - t)) + 1) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * (1 - t)) ** 2,
        }
        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0
    
    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt
    
    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0
    
    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class SphericalICPlan(ICPlan):
    """
    Riemannian Flow Matching on the Sphere.
    x0: Noise (Uniform on Sphere)
    x1: Data (Normalized)
    Interpolation: SLERP (Geodesic)
    Vector Field: Derivative of SLERP projected on Tangent Space
    OPERATES ON [B, C, H, W] where C (dim=1) is the feature dimension.
    """
    def __init__(self, sigma=0.0):
        super().__init__(sigma)

    def compute_xt_ut_spherical(self, t, x0, x1):
        t = expand_t_like_x(t, x0)
        
        dot = (x0 * x1).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = th.acos(dot)
        sin_theta = th.sin(theta)

        epsilon = 1e-5
        is_small_angle = sin_theta < epsilon
        
        sin_theta_safe = th.where(is_small_angle, th.ones_like(sin_theta), sin_theta)
        
        scale_data = th.sin((1.0 - t) * theta) / sin_theta_safe # Coeff for x1
        scale_noise = th.sin(t * theta) / sin_theta_safe       # Coeff for x0

        xt = scale_data * x1 + scale_noise * x0

        
        coeff_data = -theta * th.cos((1.0 - t) * theta) / sin_theta_safe
        coeff_noise = theta * th.cos(t * theta) / sin_theta_safe
    

        ut = coeff_data * x1 + coeff_noise * x0
        u_dot = (ut * xt).sum(dim=1, keepdim=True)
        ut = ut - u_dot * xt

        return xt, ut

    def plan(self, t, x0, x1):
        xt, ut = self.compute_xt_ut_spherical(t, x0, x1)
        return t, xt, ut
    
    def compute_xt(self, t, x0, x1):
        xt, _ = self.compute_xt_ut_spherical(t, x0, x1)
        return xt
        
    def compute_ut(self, t, x0, x1, xt):
        _, ut = self.compute_xt_ut_spherical(t, x0, x1)
        return ut


class VPCPlan(ICPlan):
    """class for VP path flow matching"""
    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min 
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t):
        p_alpha_t = 2 * self.log_mean_coeff(t)
        alpha_t = th.sqrt(1 - th.exp(p_alpha_t))
        d_alpha_t = th.exp(p_alpha_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * alpha_t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = self.log_mean_coeff(t)
        sigma_t = th.exp(sigma_t)
        d_sigma_t = sigma_t * self.d_log_mean_coeff(t)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2

class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
    
    def compute_alpha_t(self, t):
        alpha_t = th.cos(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        sigma_t = th.sin(t * np.pi / 2)
        d_sigma_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        return -np.pi / 2 * th.tan(t * np.pi / 2)