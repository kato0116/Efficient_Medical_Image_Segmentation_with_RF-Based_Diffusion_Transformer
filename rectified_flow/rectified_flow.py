"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import torch as th
import numpy as np
from utils import logit_normal_sample, sample_t 


eps=1e-3
# original code
class RectifiedFlow():
    def __init__(
        self, 
        init_type='gaussian', 
        time_sampler="uniform",
        noise_scale=1.0, 
        use_ode_sampler='rk45', 
        sigma_var=0.0, 
        ode_tol=1e-5, 
        sample_N=3,
        euler_ensemble=0,
        sampler_type='euler',
        ):
      if sample_N is not None:
        self.sample_N = sample_N
        print('Number of sampling steps:', self.sample_N)
      self.init_type = init_type
      self.time_sampler    = time_sampler
      self.sampler_type    = sampler_type
      self.euler_ensemble  = euler_ensemble
      self.noise_scale     = noise_scale
      self.use_ode_sampler = use_ode_sampler
      self.ode_tol = ode_tol
      self.sigma_t = lambda t: (1. - t) * sigma_var
      print('Sampler Type:', self.sampler_type)
      print('Init. Distribution Variance:', self.noise_scale)
      print('ODE Tolerence:', self.ode_tol)
            
    @property
    def T(self):
      return 1.
    
    def train_losses(self, model, z0, z1, model_kwargs=None):
        # z1: ターゲット
        # z0: ノイズ
        device = z0.device
        if z0 is None:
          z0 = th.randn_like(z1).to(device) 
        if self.time_sampler == "uniform":
          t = th.rand(z1.shape[0], device=device) * (1 - eps) + eps
        elif self.time_sampler == "logit_norm":
          t = logit_normal_sample(z1.shape[0]).to(device) * (1 - eps) + eps
        elif self.time_sampler == "exponential":
          t = sample_t(z1.shape[0]).to(device) * (1 - eps) + eps
        t_expand = t.view(-1, 1, 1, 1).repeat(
                1, z1.shape[1], z1.shape[2], z1.shape[3]
            )
        z_t = t_expand * z1 + (1 - t_expand) * z0
        target = z1 - z0
        score  = model(z_t,t*999, **model_kwargs)
        losses = th.square(score - target)
        losses = th.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        loss   = th.mean(losses)
        return loss

    @torch.no_grad()
    def euler_ode(self, init_input, model, reverse=False, N=100, model_kwargs=None):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      eps=1e-3
      dt = -1. / N if reverse else 1. / N

      # Initial sample
      x = init_input.detach().clone()

      model_fn = model.eval()
      shape  = init_input.shape
      device = init_input.device
      
      for i in range(N):
          if reverse:
              # 逆方向
              num_t = (1 - i / N) * (self.T - eps) + eps
          else:
              # 順方向
              num_t = i / N * (self.T - eps) + eps

          t = torch.ones(shape[0], device=device) * num_t
          pred = model_fn(x, t*999, **model_kwargs)  # 時間スケールをモデル用に変換
          x = x.detach().clone() + pred * dt  # 更新ステップ        

      return x

    def get_z0(self, batch, train=True):
      n,c,h,w = batch.shape 

      if self.init_type == 'gaussian':
          ### standard gaussian #+ 0.5
          cur_shape = (n, c, h, w)
          return torch.randn(cur_shape)*self.noise_scale
      else:
          raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 

    @torch.no_grad()
    def sampler(self, model, shape, device, model_kwargs=None):
      N = self.sample_N # Euler方のステップ数
      if self.sampler_type == "euler":
        return self.euler_sampler(model, shape, N, device, model_kwargs)
      elif self.sampler_type == "ensemble_euler":
        return self.ensemble_euler_sampler(model, shape, N, device, S=self.euler_ensemble, model_kwargs=model_kwargs)

    @torch.no_grad()
    def euler_sampler(self, model, shape, N, device, model_kwargs=None):
      model.eval()
      with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        z  = z0.detach().clone()
        
        dt = 1./N
        for i in range(N):
          num_t = i / N * (1 - eps) + eps
          t = torch.ones(shape[0],device=device) * num_t
          pred = model(z, t*999, **model_kwargs)
          z = z.detach().clone() + pred * dt
        
        return z

    @torch.no_grad()
    def ensemble_euler_sampler(self, model, shape, N, device, S=10, model_kwargs=None):
      model.eval()
      with torch.no_grad():
        dt = 1./N
        z1_master = torch.randn(shape, device=device) # z1を初期化
        for i in range(N):
          num_t = i / N * (1 - eps) + eps
          t     = torch.ones(shape[0],device=device) * num_t
          ensemble_list = []
          z_list        = []
          for _ in range(S):
            z0 = torch.randn(shape, device=device)
            z  = t*z1_master + (1-t)*z0        # z1とnoiseによるzt     
            z_list.append(z)       
            pred = model(z, t*999, **model_kwargs) # verocityを計算
            ensemble_list.append(pred)
          pred_mean = torch.stack(ensemble_list).mean(dim=0)             # 平均verocityを計算 (main branchのverocity)
          z_mean    = torch.stack(z_list).mean(dim=0)
          z1_master = z_mean.detach().clone() + pred_mean * dt * (N-i) # zt → z1 への更新
        return z1_master

# rectified_flow = RectifiedFlow()
# x = torch.randn(1, 1, 256, 256)
# z0 = rectified_flow.get_z0(x)
# from unet import VelocityModel
# model = VelocityModel()
# print(z0.shape)  
# z1 = rectified_flow.sample_ode(z0,model, N=100)    
# print(z1.shape)