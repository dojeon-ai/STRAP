import numpy as np
import math
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.optim.lr_scheduler import _LRScheduler
from einops import rearrange


def set_global_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
######################
# init & norm
def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        gain = 1.0
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0)
    
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0)
        
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)    
        nn.init.constant_(m.bias, 0)
    
    return m

def xavier_uniform_init(m):
    if isinstance(m, nn.Linear):
        gain = 1.0
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
        
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)    
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    return m

def init_normalization(channels, norm_type="bn", one_d=False):
    assert norm_type in ["bn", "bn_nt", "ln", "ln_nt", None]
    if norm_type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=True)
        else:
            return nn.BatchNorm2d(channels, affine=True)
        
    elif norm_type == "bn_nt":
        if one_d:
            return nn.BatchNorm1d(channels, affine=False)
        else:
            return nn.BatchNorm2d(channels, affine=False)
        
    elif norm_type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=True)
        else:
            return nn.GroupNorm(1, channels, affine=True)
    
    elif norm_type == "ln_nt":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=False)
        else:
            return nn.GroupNorm(1, channels, affine=False)
    
    elif norm_type is None:
        return nn.Identity()


# SASRec + 위의 xavier_uniform_init
class NormInitializer():
    def __init__(self, hidden_dim):
        # init_scale: d_model^-0.5 from https://tunz.kr/post/4
        self.init_scale = 1 / np.sqrt(hidden_dim)
        
    def __call__(self, m):
        if isinstance(m, nn.Linear):
            gain = 1.0
            nn.init.orthogonal_(m.weight.data, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
        elif isinstance(m, nn.Conv2d):
            gain = nn.init.calculate_gain('relu')
            nn.init.orthogonal_(m.weight.data, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
            
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)    
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.init_scale)


######################
# scheduler
class LinearScheduler(object):
    # linear scheduler to control epsilon rate
    def __init__(self, initial_value, final_value, step_size):
        """
        Linear Interpolation between initial_value to the final_value
        [params] initial_value (float) initial output value
        [params] final_value (float) final output value
        [params] step_size (int) number of timesteps to lineary anneal initial value to the final value
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        self.step_size = step_size
        self.step = 0
        
    def get_value(self):
        """
        Return the scheduled value
        """
        self.step += 1
        interval = (self.initial_value - self.final_value) / self.step_size
        # After the schedule_timesteps, final value is returned
        if self.final_value < self.initial_value:
            return max(self.initial_value - interval * self.step, self.final_value)
        else:
            return min(self.initial_value - interval * self.step, self.final_value)
        

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_ratio : float = 0.2,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = int(warmup_ratio * first_cycle_steps) # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
    
########################
# optimization
def get_grad_norm_stats(model):
    grad_norm = []
    stats = {}
    for p in model.parameters():
        if p.grad is not None:
            grad_norm.append(p.grad.detach().data.norm(2))
    grad_norm = torch.stack(grad_norm)
    stats['min_grad_norm'] = torch.min(grad_norm).item()
    stats['mean_grad_norm'] = torch.mean(grad_norm).item()
    stats['max_grad_norm'] = torch.max(grad_norm).item()

    return stats

class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""

    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of 
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None


############ DRQ-v2-DMC ###############
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train_mode(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train_mode(state)
        return False
