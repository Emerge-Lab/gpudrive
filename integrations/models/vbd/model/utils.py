import torch
import logging
import pickle
import glob
import random
import numpy as np
from .model_utils import *
from torch.utils.data import Dataset
from torch.nn import functional as F


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_beta_schedule(variant, timesteps):
    if variant == "cosine":
        return betas_for_alpha_bar(timesteps)
    elif variant == "linear":
        return linear_beta_schedule(timesteps)
    else:
        raise NotImplemented


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02

    return torch.linspace(beta_start, beta_end, timesteps)


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    """

    def alpha_bar(time_step):
        # ! Hard code to shift the schedule
        # return np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2
        return (
            np.cos((time_step + 0.008) / 1.008 * np.pi / 2) ** 2
        ) * 0.98 + 0.02

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))

    return torch.tensor(betas, dtype=torch.float32)


class DDPM_Sampler(torch.nn.Module):
    def __init__(self, steps=100, schedule="cosine", clamp_val: float = 5.0):
        super().__init__()
        self.num_steps = steps
        self.schedule = schedule
        self.clamp_val = clamp_val

        self.register_buffer(
            "betas", get_beta_schedule(self.schedule, self.num_steps)
        )
        self.register_buffer("betas_sqrt", self.betas.sqrt())
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, 0))

    @torch.no_grad()
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ):

        assert (timesteps < self.num_steps).all()

        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(
            original_samples.shape
        ):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noised_samples = (
            sqrt_alpha_prod * original_samples
            + sqrt_one_minus_alpha_prod * noise
        )

        return noised_samples

    def set_timesteps(self, num_inference_steps=None, device=None):

        timesteps = (
            np.linspace(0, self.num_steps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        # Compute predicted previous sample µ_t-1
        pred_prev_sample_mean = self.q_mean(
            model_output, timestep, sample, prediction_type=prediction_type
        )
        # 6. Add noise
        device = model_output.device
        variance_noise = torch.randn(
            model_output.shape, device=device, dtype=model_output.dtype
        )

        variance = (self.q_variance(timestep) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample_mean + variance

        return pred_prev_sample

    def q_mean(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """
        if type(timestep) == int:
            t = timestep
        else:
            t = timestep[0][0]
        prev_t = t - 1

        # 1. Compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Compute predicted original sample from predicted noise also called "predicted x_0"
        if prediction_type == "sample":
            pred_original_sample = model_output
        elif prediction_type == "error":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif prediction_type == "v":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise NotImplementedError

        # 3. Clip or threshold "predicted x_0"
        pred_original_sample = pred_original_sample.clamp(
            -self.clamp_val, self.clamp_val
        )
        # samxple = sample.clamp(-self.clamp_val, self.clamp_val)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = (
            current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t
        )

        # 5. Compute predicted previous sample µ_t
        pred_prev_sample_mean = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )
        return pred_prev_sample_mean

    def q_x0(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        prediction_type: str = "sample",
    ):
        """
        Predict the denoised x0 from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
        """

        # 2. Compute predicted original sample from predicted noise also called "predicted x_0"
        if prediction_type == "sample":
            pred_original_sample = model_output
        elif prediction_type == "error":
            alpha_prod_t = self.alphas_cumprod[timestep]
            for _ in range(len(sample.shape) - len(alpha_prod_t.shape)):
                alpha_prod_t = alpha_prod_t[..., None]
            beta_prod_t = 1 - alpha_prod_t

            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        # elif prediction_type == "v":
        #     pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise NotImplementedError

        return pred_original_sample

    def q_variance(self, t):
        if t == 0:
            return 0
        prev_t = t - 1
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        variance = beta_prod_t_prev / beta_prod_t * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance
