# Refer to: https://github.com/jannerm/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/models/constructor.py#L7

'''
TODO
   1. Start with implementing two simple neural networks (one for T, one for R) using PyTorch

'''

import torch
from torch import nn


class T_Model(nn.Module):

   ''' Predictive Model of P( s',r| s,a ) '''

   def __init__(self) -> None:
      super(T_Model, self).__init__()
      self.s_dim = None
      self.a_dim = None
      # How to handle inputs?
      self.fc1 = nn.Linear( self.input_dim, 20)            

      pass
   


   def forward(self):
      pass




class R_Model(nn.Module):
      pass