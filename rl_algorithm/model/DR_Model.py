# For MBRL, Refer to: 
#  https://github.com/jannerm/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/models/constructor.py#L7
#  https://github.com/natolambert/dynamicslearn/blob/master/learn/models/model.py
# 
#  --Facebook Research: Library for Model Based RL 
#  https://github.com/facebookresearch/mbrl-lib
#
#  https://github.com/opendilab/awesome-model-based-RL


# For Gym, Refer to:
#  https://github.com/openai/gym/blob/d6a3431c6041646aa8b77fbee02efba5aca9b82b/gym/spaces/space.py#L21
#  
#  Gym Spaces Ex: Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
# 

# For PyTorch, Refer to:
#  https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py#L37-L49
'''
NOTES
   [2022-03-25]
   "Improving Model-Based Deep Reinforcement Learning with "Learning Degree Networks" and Its Application in Robot Control"
   used MLP (BP-Neural Network) for model. 
   
   It also provides loss function and optimizer.
   
   **** Look for code. ****

   p.3 : The system dynamics model training uses the BP neural network "supervised learning alogrithm"....

TODO
   1. Start with implementing two simple neural networks (one for T, one for R) using PyTorch

'''

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# create custom dataset class to use torch DataLoader
class ExperienceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        y_sample = self.y[idx]
        X_sample = self.X[idx]
        sample = {"X": X_sample, "y": y_sample}
        return sample


class DynamicsRewardModel(nn.Module):

   # https://datascience.stackexchange.com/questions/46093/how-to-define-a-mlp-with-multiple-outputs

   ''' Predictive Model of P( s',r| s,a ) '''

   def __init__(self,
                env: gym.Env,  # gym-env that this DR-Model will try to model.
               ) -> None:

      print("Constructing a DyanmicsReward Model based on gym-env: {}".format( env.spec.id ) )
      
      super(DynamicsRewardModel, self).__init__()
      
      # Env - Info
      self.env = env
      self.env_observation_dim = np.prod(env.observation_space.shape) 
      self.env_action_dim = np.prod( env.action_space.shape )

      # # For OpenAI-gym Reward is a scalar [ reward (float) ]
      self.env_reward_range = env.reward_range
      
      self.input_dim = self.env_observation_dim + self.env_action_dim
      self.output_dim = self.env_observation_dim + 1   # output-dimension of obs_dim + reward_dim(1)
      
      # NN Layers ( for MLP )
      self.fc1 = nn.Linear(in_features= self.input_dim, out_features = 256)
      self.fc2 = nn.Linear(in_features= 256, out_features = 128)
      self.fc3 = nn.Linear(in_features= 128, out_features = 64)
      self.fc4 = nn.Linear(in_features= 64, out_features= self.output_dim)

      self.dp1 = nn.Dropout( p = 0.1 )  # p (float, optional) – probability of an element to be zero-ed.
      self.dp2 = nn.Dropout( p = 0.25 )
      self.dp3 = nn.Dropout( p = 0.5 )
      

   def forward(self, x: torch.Tensor):
      # MLP
      
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dp1(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.dp2(x)
      x = self.fc3(x)
      x = F.relu(x)
      x = self.dp3(x)
      out = self.fc4(x)
      return out
      

   def Train(self, samples_buffer: ReplayBuffer, epoch: int) -> None:
   
      # References:
      #            https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
      #     
      #            PyTorch Loss Functions: https://neptune.ai/blog/pytorch-loss-functions
      #            PyTorch Optimizers
      ''' Preparation for Training '''

      # (1) Set Module to "training mode" (~= model.train() ) 
      self.train()           
      
      # (2) Prepare Training Data
      # Refer to: https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
      
      train_X = np.concatenate( (samples_buffer.observations, samples_buffer.actions) , axis = -1)
      train_y = np.dstack( (samples_buffer.next_observations, samples_buffer.rewards) )
      # drop all sparse rows.
      train_X = train_X[~np.all(train_X==0, axis=2)]
      train_y = train_y[~np.all(train_y==0, axis=2)]

      ExpDatasetObj = ExperienceDataset(X= train_X, y= train_y)

      ''' There is an error here '''
      train_dataloader = DataLoader(dataset = ExpDatasetObj, batch_size = 128, shuffle=True)

      # (3) Prepare Loss-function and Optimizer
      loss_function = F.mse_loss # https://neptune.ai/blog/pytorch-loss-functions
      learning_rate = 0.01
      optimizer = optim.Adam(self.parameters(), lr= learning_rate)  

      ''' Training '''

      for (batch_idx, batch) in enumerate(train_dataloader):
         

         # Ensure 'x' is np.float32 instead of double(np.float64)
         # Source: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
         #         https://stackoverflow.com/questions/34192568/torch-how-to-change-tensor-type
         #X, y = X.to(device), y.to(device)
         X = batch["X"].to(torch.float32)
         y = batch["y"].to(torch.float32)
         optimizer.zero_grad()
         y_prime = self(X) 
         loss = loss_function(input= y_prime, target= y)
         loss.backward() # auto-grad   
         optimizer.step()

         if batch_idx % 10 == 0:
            print("epoch: {} [{}/{} ({:.0f}%)]\t training loss:{:.6f}".format( epoch, batch_idx, len(train_dataloader), batch_idx/len(train_dataloader), loss.item() ) )

      return



   def predict(self, observation: torch.Tensor, action: torch.Tensor):

      input = torch.cat((observation, action), dim=1).to(torch.float32) 
      
      #input = np.concatenate( (observation, action) , axis = -1)
      #input = torch.from_numpy(input).to(torch.float32)
      predicted = self(input) # likely calls feedforward
      return predicted


   def save_model(self, filepath):
      torch.save(self, filepath) 

   def reset(self):
      pass














