import torch as th
from collections import OrderedDict

import scipy.stats as stats
import numpy as np

eps = 1e-3
@th.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)
    
def sample_t(num_samples, exponential_pdf = ExponentialPDF(a=0, b=1, name='ExponentialPDF'), a=4):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = th.from_numpy(t).float()
    t = th.cat([t, 1 - t], dim=0)
    t = t[th.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def logit_normal_sample(batch_size, mean=0.0, std=1.0):
    # 標準正規分布からサンプリング
    u = th.randn(batch_size) * std + mean
    # シグモイド関数を適用
    t = th.sigmoid(u)
    return t
