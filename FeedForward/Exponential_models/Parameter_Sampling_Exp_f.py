import torch
from torch.distributions import Normal
import torch.optim as opt

import torch.nn as nn


# TRY: introduce a learable param at each step for the sign!!!


class ExpF_agent(nn.Module):

    def __init__(self, n_exps, t_steps, std = 1, ln_rate = 0.01):

        super().__init__()
        miu = torch.rand((n_exps,))
        miu /= sum(miu)
        miu[::2] *= 1
        miu[1::2] *= -1  # start form 1 element select everything with step = 2 //i.e odd indexes

        self.mean_miu = nn.Parameter(miu)

        self.mean_decay_rate = nn.Parameter(torch.rand(n_exps).view(-1,n_exps)) # if use randn also get negative values, changing sign of the exponential

        self.n_exps = n_exps

        self.t_steps = torch.arange(t_steps, dtype=torch.float).view(t_steps,-1)

        self.t_steps /= sum(self.t_steps)

        #self.t_steps[1:] = torch.log(self.t_steps[1:])

        #print(self.t_steps)

        self.exp_f = lambda t,e: torch.exp(-t/e)

        self.std = std

        self.mean_angles = nn.Parameter(torch.randn(t_steps, 1))

        self.optimiser = opt.Adam(self.parameters(),ln_rate)



        #self.optimiser = opt.SGD(self.parameters(), ln_rate, momentum=0.9,nesterov=True)


    def sample_actions(self):


        d_1 = Normal(self.mean_miu, self.std)
        d_2 = Normal(self.mean_decay_rate, self.std)
        d_3 = Normal(self.mean_angles, self.std)

        sampled_miu = d_1.sample()
        sampled_decay = d_2.sample()
        sampled_angle = d_3.sample()

        self.log_ps = torch.sum(d_1.log_prob(sampled_miu) + d_2.log_prob(sampled_decay)) + sum(d_3.log_prob(sampled_angle))

        #print("prob",self.log_ps)

        with torch.no_grad():
            actions = torch.matmul(self.exp_f(self.t_steps, sampled_decay), sampled_miu.view(self.n_exps, -1)) * sampled_angle #torch.sin(sampled_angle) , much slower using sin

        return actions


    def update(self,rwds):

        loss = sum(-self.log_ps.view(-1) * rwds)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def compute_loss(self, rwds):

        return sum(-self.log_ps.view(-1) * rwds)

    def deterministic_actions(self):

        with torch.no_grad():
            actions = torch.matmul(self.exp_f(self.t_steps, self.mean_decay_rate), self.mean_miu.view(self.n_exps, -1)) * self.mean_angles # sampled_angle, much slower using sin

        return actions

    def update_std(self):

        self.std *= 0.9