# Refer to: 
#           typing (python library): https://docs.python.org/3/library/typing.html
#           stable-baselines3("SB3") Source: https://github.com/DLR-RM/stable-baselines3
#                              SB3 Document: https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/
#           gym: https://github.com/openai/gym/tree/master/gym
#           mujoco: https://gym.openai.com/envs/#mujoco  

'''
TODOs:
   (Q1) How can 2 separate RL Algorithms communicate with single env, as our scheme?
   (A1) Perhaps look into using "train()" mem-func instead of "learn()" in SB3-RL-algorithm-impl.
        Basically, "model.learn()" corresponds to a training-loop, which is comprised of 2 steps of:
        step-1: "model.collect_rollouts()" use the current policy in the environment, fill the rollout/replay buffer.
        step-2: "model.train()" opttimize the actor/critic networks, update the target networks.
        Great Reference is in "SB3 Document :: Chapter 1 :: page 80"
        > I think it is possible to implement my scheme, 
          if I appropriately use 
          - RL agent's predict() 
          - environment's step() 
          - fill out RL agent's 
            - Rollout(for OnPolicy) <-- self.rollout_buffer (Source: https://github.com/DLR-RM/stable-baselines3/blob/e88eb1c9ca98650f802409e5845e952c39be9e76/stable_baselines3/common/on_policy_algorithm.py#L111)
            - Replay Buffer(for OffPolicy)  <-- self.replay_buffer (Source: https://github.com/DLR-RM/stable-baselines3/blob/e88eb1c9ca98650f802409e5845e952c39be9e76/stable_baselines3/common/off_policy_algorithm.py#L213 )
        
   (Q2) What kind of "Dynamics-Model" would I use?

   (Q3) What would the pseudo-code for "Train()" look like?

'''
# external imports
from stable_baselines3.ddpg import DDPG
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
import gym

from typing import NewType, Union, Optional, Dict
# internal imports
from rl_algorithm.data_buffer import DataBuffer

class hybrid_mbmf:

   def __init__(self,
                env: gym.Env,
                AC_Type: Union[DDPG, SAC, TD3], # Let's assume that AC-agent used for MBAC and MFAC are the same.
                AC_kwargs: Dict = {"policy": "MlpPolicy"},
                total_steps: int = 1000,
                ) -> None:

               # key components
               self.env = env
               # Model-Free Actor-Critic RL Algorithms that are each trained with only real-data, and mixture of real-data and virtual-data. 
               self.MFAC = AC_Type(env = self.env, **AC_kwargs)
               self.MBAC = AC_Type(env = self.env, **AC_kwargs)
               # Dyanmics-Model
               self.Model = None
               # DataBuffers that each store RealData and VirtualData.
               self.RealDataBuffer = DataBuffer() # Above AC-agents might have their own internal rollout or replay-buffers, which are different from these.
               self.VirtualDataBuffer = DataBuffer()
               
               # for training
               self.total_steps = total_steps

               return


   def MBMFAction(self):
      ''' Selects between actions suggested by MFAC and MBAC. '''

      # self.MBAC
      # self.MFAC

      pass


   def Assess(self):
      ''' MF-Critic distinguishes virtual-data that are "better than nothing" and "worse than nothing" from a 'VirtualDataBatch'. '''
      pass


   def Train(self) -> None:
      '''
         Pseudo-code

         reset env

         for t in range(total_steps):
            action = self.MBMFAction()
            self.env.step(action)
            Model
      '''

      self.env.reset()
      for t in range( self.total_steps ):
         action = self.MBMFAction()
         real_data = self.env.step( action )


         # train - trainfreq?


      return

   def Test(self) -> None:
       pass