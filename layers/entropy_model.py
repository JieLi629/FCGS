import os
import torch
import torch.nn as nn 
from typing import Tuple,List,Union
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import List
from torch import Tensor
import torch.nn.functional as tnf

class Factorized_Gaussian_Model(nn.Module):

    def __init__(self,channels:int,tail_mass=1e-9,init_scale=10,filters:Tuple[int,...]=(3,3,3),Q:float = 1):
        super().__init__()
        self.channels = int(channels)
        self.tail_mass = float(tail_mass)
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)
        
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])


        #creat parameters
        filters = (1,) + self.filters +(1,)
        scale = self.init_scale ** (1/len(self.filters) + 1)
        channels = self.channels
        self.Q = Q
        for i in range(len(self.filters)+1):
            init = np.log(np.expm1(1/scale/filters[i+1]))# expm1(x) i.e, exp(x)-1
            self.matrix = nn.Parameter(torch.FloatTensor(channels,filters[i+1],filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(torch.FloatTensor(channels,filters[i+1],1))
            nn.init.uniform_(self.bias,-0.5,0.5)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(torch.FloatTensor(channels,filters[i+1],1))
                nn.init.zeros_(self.factor)
                self._factor.append(self.factor)

    def _logits_cumulative(self,inputs:Tensor,stop_gradient:bool):
        logits = inputs
        for i in range(len(self.filters)+1):
            if stop_gradient:
                self._matrices[i].detach()
            logits = torch.matmul(tnf.softplus(self._matrices[i]),logits)
            if stop_gradient:
                self._bias[i].deatch()
            logits = logits + self.bias[i]

            if i < len(self.filters):
                if stop_gradient:
                    self._factor[i].detach()
                logits += torch.tanh(self._factor[i]) * torch.tanh(logits)
        return logits


    def _likelihood(self,inputs:Tensor) -> Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half 
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower+upper)
        sign = sign.detach()
        likelihood = torch.abs(torch.sigmoid(sign*upper)-torch.sigmoid(sign*lower))
        
        return likelihood


    def forward(self,inputs:Tensor,*args):
        # Note the difference between reshape, view and permute method
        x = inputs.permute(1,0).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0),1,-1)
        
        likelihood = checkpoint(
            self._likelihood,
            values,
            use_reentrant=False
        )
        bound = torch.tensor(1e-9)
        likelihood_bound = Lower_Bound.apply(likelihood,bound)
        likelihood_bound = likelihood_bound.reshape(shape)
        likelihood_bound = likelihood_bound.permute(1,0)
        return likelihood_bound

def lower_bound_bwd(x:Tensor,Bound:Tensor,gradient:Tensor):
    pass_through_if = (x > Bound) | (gradient < 0)
    return pass_through_if*gradient,None

class Lower_Bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bound):
        ctx.save_for_backward(x, bound)
        x = torch.max(x, bound)
        return x
    @staticmethod
    def backward(ctx,gradient):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x,bound,gradient)