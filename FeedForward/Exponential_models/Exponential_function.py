import torch
from torch.distributions import Normal
import torch.optim as opt

import torch.nn as nn


class ExpF_agent(nn.Module):

    def __init__(self, n_exps, t_steps, std = 1, ln_rate = 0.01):

        super().__init__()

        miu = torch.rand((n_exps,))
        miu /= sum(miu)
        miu[::2] *= 1#0.1
        miu[1::2] *= - 1#0.09 #*= -1  # start form 1 element select everything with step = 2 //i.e odd indexes


        self.miu = nn.Parameter(miu)



        self.decay_rate = nn.Parameter(torch.rand(n_exps).view(-1,n_exps) ) #* 0.001) # if use randn also get negative values, changing sign of the exponential

        self.n_exps = n_exps

        self.t_steps = torch.arange(t_steps, dtype=torch.float).view(t_steps,-1)

        self.t_steps /= sum(self.t_steps)

        #self.t_steps *=10

        self.angles = nn.Parameter(torch.randn(t_steps, 1))


        #self.t_steps[1:] = torch.log(self.t_steps[1:])





        self.exp_f = lambda t,e: torch.exp(-t/e) #* torch.sin(o)

        self.std = std

        self.optimiser = opt.Adam(self.parameters(),ln_rate)





    def sample_actions(self):


        mean_value = torch.matmul(self.exp_f(self.t_steps,self.decay_rate), self.miu.view(self.n_exps, -1)) * torch.sin(self.angles) #* self.angles #


        d = Normal(mean_value, self.std)

        actions = d.sample()

        self.log_ps = d.log_prob(actions)

        return actions #, mean_value


    def update(self,rwds):


        loss = (-self.log_ps.view(-1) * rwds).mean() # + torch.norm(self.miu) #rwds
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss

    def deterministic_actions(self):

        with torch.no_grad():
            actions = torch.matmul(self.exp_f(self.t_steps, self.decay_rate),  self.miu.view(self.n_exps, -1)) * self.angles # sampled_angle, much slower using sin

        return actions


    def compute_returns(self, rwds, discounts):

        discounts **= (torch.FloatTensor(range(len(rwds))))

        return torch.flip(torch.cumsum(torch.flip(discounts * rwds, dims=(0,)), dim=0), dims=(0,)) / discounts



    def compute_returns_2(self, rwds, discount):
        cum_rwd = 0

        returns = []
        for rwd in reversed(rwds):
            cum_rwd = rwd + cum_rwd * discount

            returns.insert(0, cum_rwd)

        return torch.stack(returns)






