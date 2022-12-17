from __future__ import annotations
from typing import Dict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from nflows.transforms import splines
from sklearn.cluster import KMeans

class UnivariateFlowMixture(nn.Module):
    def __init__(self, n_comp, bins, layers) -> None:
        super().__init__()
        self.n_comp = n_comp
        self.bins = bins
        self.layers = layers
        self.w = nn.Parameter(torch.randn((layers,n_comp,bins)), requires_grad=True)
        self.h = nn.Parameter(torch.randn((layers,n_comp,bins)), requires_grad=True)
        self.s = nn.Parameter(torch.randn((layers,n_comp,bins-1)), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(layers+1,n_comp), requires_grad=True)
        self.log_scale = nn.Parameter(torch.randn(layers+1,n_comp), requires_grad=True)
    
    def apply_forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x[:,None].expand(-1, self.n_comp)
        log_j_det = 0
        for step in range(self.layers):
            x = (x - self.bias[step]) * torch.exp(-self.log_scale[step])
            w = self.w[step][None,:,:].expand(batch_size,-1,-1)
            h = self.h[step][None,:,:].expand(batch_size,-1,-1)
            s = self.s[step][None,:,:].expand(batch_size,-1,-1)
            x, step_log_j_det = splines.unconstrained_rational_quadratic_spline(x, w, h, s, tail_bound=3)
            log_j_det = log_j_det + step_log_j_det - self.log_scale[step]
        x = (x - self.bias[self.layers]) * torch.exp(-self.log_scale[self.layers])
        return x, log_j_det - self.log_scale[self.layers]
    
    def apply_backward(self, z):
        batch_size = z.shape[0]
        z = z[:,None].expand(-1, self.n_comp)
        z = z * torch.exp(self.log_scale[self.layers]) + self.bias[self.layers]
        for step in reversed(range(self.layers)):
            w = self.w[step][None,:,:].expand(batch_size,-1,-1)
            h = self.h[step][None,:,:].expand(batch_size,-1,-1)
            s = self.s[step][None,:,:].expand(batch_size,-1,-1)
            z, _ = splines.unconstrained_rational_quadratic_spline(z, w, h, s, inverse=True, tail_bound=3)
            z = z * torch.exp(self.log_scale[step]) + self.bias[step]
        return z

class MultivariateFlowMixture(nn.Module):
    def __init__(self, variables, n_comp, bins, layers) -> None:
        super().__init__()
        self.n_comp = n_comp
        self.flows = nn.ModuleList([UnivariateFlowMixture(n_comp, bins, layers) for _ in range(variables)])
        self.comp_logits = nn.Parameter(torch.ones(n_comp)/n_comp, requires_grad=True)
        self.register_buffer('mu', torch.zeros((1,)), persistent=False)
        self.register_buffer('std', torch.ones((1,)), persistent=False)
    
    def apply_forward(self, x: torch.Tensor, dist_target: torch.distributions.Distribution):
        #print(x)
        zs, log_j_dets = zip(*[flow.apply_forward(x[:,i]) for i, flow in enumerate(self.flows)])
        #print(zs)
        log_ps = [dist_target.log_prob(z) + log_j_det for z, log_j_det in zip(zs, log_j_dets)]
        p = F.softmax(sum(log_ps).detach() + self.comp_logits.detach(), dim=1)
        log_ps = [(p * log_p).sum(1) for log_p in log_ps]
        return sum(log_p.mean() for log_p in log_ps), F.cross_entropy(self.comp_logits[None,:], p.mean(0, keepdim=True))
    
    def apply_backward(self, z: torch.Tensor):
        batch_size = z.shape[0]
        comp = torch.distributions.Categorical(logits = self.comp_logits).sample((batch_size,))
        xs = [flow.apply_backward(z[:,i]).gather(1, comp[:,None])[:,0] for i, flow in enumerate(self.flows)]
        return torch.stack(xs, dim=1), comp
    
    def initialize(self, data: torch.Tensor):
        device = next(self.parameters()).device
        k_means = KMeans(self.n_comp).fit(data.numpy())
        centers = torch.from_numpy(k_means.cluster_centers_).to(device)
        labels = torch.from_numpy(k_means.labels_).to(device)
        for var, flow in enumerate(self.flows):
            flow.w.data.zero_()
            flow.h.data.zero_()
            flow.s.data.fill_(1.0)
            
            flow.bias.data.zero_()
            var_centers = centers[:, var] 
            flow.bias.data[0] = var_centers
            
            flow.log_scale.data.zero_()
            stds = torch.tensor([torch.std(data[:,var][labels == comp].cpu() - var_centers[comp].cpu()).item() for comp in range(self.n_comp)], device=device) + 1e-2
            flow.log_scale.data[0] = torch.log(stds)
        
        self.comp_logits.data = torch.log(1e-31 + (labels[:,None] == torch.arange(0,self.n_comp,1).to(device)).float().mean(0))
    
    def sample(self, batch_size: int):
        dist = torch.distributions.Normal(self.mu, self.std)
        z = dist.sample((batch_size,len(self.flows)))[:,:,0]
        return self.apply_backward(z)

    def sample_given(self, given: Dict[int, torch.Tensor]):
        dist = torch.distributions.Normal(self.mu, self.std)
        zs, log_j_dets = zip(*[self.flows[i].apply_forward(x) for i, x in given.items()])
        log_ps = sum(dist.log_prob(z) + log_j_det for z, log_j_det in zip(zs, log_j_dets)) + self.comp_logits
        comp = torch.distributions.Categorical(logits=log_ps).sample()
        batch_size = comp.shape[0]
        z = dist.sample((batch_size,len(self.flows)))[:,:,0]
        xs = []
        for i, flow in enumerate(self.flows):
            if (x:=given.get(i)) is None:
                x = flow.apply_backward(z[:,i]).gather(1, comp[:,None])[:,0]
            xs.append(x)
        return torch.stack(xs, dim=1)
    
    def fit(self, data: torch.Tensor, epochs = 2, batch_size = 128, noise = 0, lr = 1e-3) -> MultivariateFlowMixture:
        self.initialize(data)
        dist = torch.distributions.Normal(self.mu, self.std)
        opt = optim.RMSprop(self.parameters(), lr=lr)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size, shuffle=True)
        device = self.comp_logits.device
        self.train()
        for _ in range(epochs):
            for (x,) in loader:
                opt.zero_grad()

                x = (x + (torch.rand_like(x) - 0.5)*noise).to(device)
                log_p, comp_loss = self.apply_forward(x, dist)
                loss = -log_p + comp_loss
                loss.backward()

                opt.step()
        self.eval()
        return self