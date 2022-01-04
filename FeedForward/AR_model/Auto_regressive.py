import torch
import torch.optim as opt
from torch.distributions import Normal
import torch.nn as nn

class AR_model(nn.Module):

    def __init__(self, t_steps, max_t_steps = 200, std = 1,ln_rate = 0.01): # 0.01

        super().__init__()

        self.t_steps = t_steps

        #self.max_t_steps = max_t_steps

        self.std = std

        params = torch.rand(t_steps)

        norm_param = params / sum(torch.abs(params))

        #norm_param = params

        norm_param[::2] *= 1 #0.1
        norm_param[1::2] *= - 1

        # if initialise from gaussian (0,1) then normalise by variance. so that variance of all = 1, and of each 1/d

        self.params = nn.Parameter(norm_param)   #norm_param

        self.bias = nn.Parameter(torch.randn(1))

        self.optimiser = opt.Adam(self.parameters(),ln_rate)



    def sample_actions(self, inputs):


    # store as a list of tensors
        for i in range(self.t_steps, self.t_steps *2 ): #self.max_t_steps +


            inputs = torch.cat([inputs,torch.dot(inputs[torch.arange(i -1,i - self.t_steps -1 ,-1)], self.params) + self.bias ])





        d = Normal(inputs[self.t_steps:],self.std)

        actions = d.sample()

        log_p = d.log_prob(actions) # only works for 200 backwards step - need to return logp store them and then pass them to update

        return actions, log_p


    def update(self,rwds , log_ps):


        loss = (-log_ps * rwds).mean()


        self.optimiser.zero_grad()

        loss.backward()

        self.optimiser.step()

        return loss

    def update_std(self):

        self.std *= 0.9


    def compute_returns(self, rwds, discounts):

        discounts **= (torch.FloatTensor(range(len(rwds))))

        return torch.flip(torch.cumsum(torch.flip(discounts * rwds, dims=(0,)), dim=0), dims=(0,)) / discounts




    # def compute_returns_2(self, rwds, discount):
    #     cum_rwd = 0
    #
    #     returns = []
    #     for rwd in reversed(rwds):
    #         cum_rwd = rwd + cum_rwd * discount
    #
    #         returns.insert(0, cum_rwd)
    #
    #     return torch.stack(returns)

