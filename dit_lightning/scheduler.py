from dataclasses import dataclass
import torch


def _extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape):
    # ensure arr on same device as indices
    if arr.device != timesteps.device:
        arr = arr.to(timesteps.device)
    res = arr[timesteps]
    while res.ndim < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res


@dataclass
class GaussianDiffusion:
    diffusion_timesteps: int

    def __post_init__(self):
        device = torch.device('cpu')
        num_diffusion_timesteps = self.diffusion_timesteps
        scale = 1000.0 / num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 2e-2
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float32, device=device)
        self.register(betas)

    def register(self, betas: torch.Tensor):
        # Standard precomputations
        self.betas = betas
        self.sqrt_betas = torch.sqrt(betas)
        self.num_timesteps = betas.shape[0]
        alphas = 1.0 - betas
        self.alphas = alphas
        self.sqrt_alphas = torch.sqrt(alphas)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype, device=alphas.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.zeros(1, dtype=alphas.dtype, device=alphas.device)], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        if self.posterior_variance.shape[0] > 1:
            self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0))
        else:
            self.posterior_log_variance_clipped = torch.zeros_like(self.posterior_variance)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        ##track device
        # self._device = betas.device
        
    # def to(self, device: torch.device):
    #     if getattr(self, "_device", None) == device:
    #         return self
    #     self.register(self.betas.to(device))
    #     return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
               _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * eps

    def q_sample_step(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        return _extract_into_tensor(self.sqrt_alphas, t, x.shape) * x + \
               _extract_into_tensor(self.sqrt_betas, t, x.shape) * noise

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        return _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor):
        return (x_t - _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x0) / \
               _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, eps_or_x0: torch.Tensor, x_prediction: bool = False, clip: bool = False, clamp_x0_fn=None):
        pred_xstart = eps_or_x0 if x_prediction else self.predict_xstart_from_eps(x_t, t, eps_or_x0)
        if clip:
            pred_xstart = pred_xstart.clamp(-1.0, 1.0)
        if clamp_x0_fn is not None:
            pred_xstart = clamp_x0_fn(pred_xstart)
        model_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart + \
                     _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return model_mean, model_variance, model_log_variance

    def p_ddim(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        at = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        at_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
        c2 = torch.sqrt(1 - at_prev)
        x0_t = (x_t - eps * torch.sqrt(1 - at)) / torch.sqrt(at)
        xt_next = torch.sqrt(at_prev) * x0_t + c2 * eps
        return xt_next
