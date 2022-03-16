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
        
   (Q2) What kind of "Dynamics-Reward Model : P_{\theta}(s',r | s,a)" would I use and how should it be trained?

   (Q3) What would the pseudo-code for "Train()" look like?
   (A3) In-progress. However, would need to determine how and when to train Model and MBAC

   (Q4) How can my scheme (training MFAC with only real-data, and training MBAC with both real and virtual data)
        harmonize with off-policy (replay-buffer) learning especially for MBAC?



         

         Need to call "model.train()" instead of "model.learn()" to gain control of the "data-collection and distribution" part
         and implement my pseudocode which requires gaining control of when and where to 'train;, 
         since "model.learn()" is for doing both data-collection and training for a single RL-Agent.

         "model.train()"
            - 'SB3 off-policy algorithm' : https://github.com/DLR-RM/stable-baselines3/blob/009bb0549ad0c9c1130309d95529a237e126578c/stable_baselines3/common/off_policy_algorithm.py#L379
               >  def train(self, gradient_steps: int, batch_size: int) -> None:
                  """
                  Sample the replay buffer and do the updates
                  (gradient descent and update target networks)
                  """
                  #NOTE: gradient_steps=-1 to perform as many gradient steps as transitions collected
                  #NOTE: ** To actually do training with model.train(), would need to manipuate "replay_buffer" member variable.
                         **   For TD3, DDPG: https://github.com/DLR-RM/stable-baselines3/blob/009bb0549ad0c9c1130309d95529a237e126578c/stable_baselines3/td3/td3.py#L148
                         **   For SAC      : https://github.com/DLR-RM/stable-baselines3/blob/009bb0549ad0c9c1130309d95529a237e126578c/stable_baselines3/sac/sac.py#L199

            - 'SB3 on-policy algorithm'  : https://github.com/DLR-RM/stable-baselines3/blob/009bb0549ad0c9c1130309d95529a237e126578c/stable_baselines3/common/on_policy_algorithm.py#L221
               >  def train(self) -> None:
                  """
                  Consume current rollout data and update policy parameters.
                  Implemented by individual algorithms.
                  """
         


'''
# external imports
from stable_baselines3.ddpg import DDPG
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
import gym

from typing import NewType, Union, Optional, Dict
from collections import defaultdict
# internal imports
from rl_algorithm.data_buffer import DataBuffer

class hybrid_mbmf:

   def __init__(self,
                env: gym.Env,
                AC_Type: Union[DDPG, SAC, TD3], # Let's assume that AC-agent used for MBAC and MFAC are the same.
                AC_kwargs: Dict = {},
                training_steps: int = 1000,
                ) -> None:

               # training environment
               self.env = env
               # Model-Free Actor-Critic RL Algorithms that are each trained with only real-data, and both real-data and virtual-data. 
               self.AC_Type = AC_Type
               self.kwargs = {"policy": "MlpPolicy"} # by default, 'MlpPolicy'
               self.kwargs.update( AC_kwargs )
               self.MFAC = self.AC_Type(env = self.env, **self.kwargs)
               self.MBAC = self.AC_Type(env = self.env, **self.kwargs)
               # Dyanmics-Model
               ''' TODO ! Need to first choose what type of Model I need to use!!! '''
               self.Model = None    

               # Following 'DataBuffers' are "external" databuffers which each store real data and virtual data.
               # Above AC-RL Aents have their own "internal" databuffers ('rollout-buffer' or 'replay-buffer' for used for their learning.
               #  whether to depends 
               self.RealDataBuffer = DataBuffer(buffer_size=int(1e4), observation_space=env.observation_space, action_space= env.action_space) 
               self.VirtualDataBuffer = DataBuffer(buffer_size=int(1e4), observation_space=env.observation_space, action_space= env.action_space)
               
               # for training
               self.training_steps = training_steps

               return


   def MBMFAction(self, current_obs, current_timestep):
      ''' Selects between actions suggested by MFAC and MBAC. '''
      # Version-1: Perhaps could favor MFAC's decision earlier and favor MBAC's decision later. 

      MB_action, _ = self.MBAC.predict(current_obs, deterministic = False)  # since training, non-deterministic
      MF_action, _ = self.MFAC.predict(current_obs, deterministic = False)

      selected_action = MF_action
      #if ( current_timestep / self.training_steps ) > 0.5:
      #   selected_action = MB_action
      
      return selected_action


   def Assess(self):
      ''' MF-Critic distinguishes virtual-data that are "better than nothing" and "worse than nothing" from a 'VirtualDataBatch'. '''
      pass


   def Train(self) -> None:
      ''' '''

      # Do things in done in model.learn() except for the data-collection part and things specific to my implementation.
      # param total_timesteps: The total number of samples (env steps) to train on
      # param eval_env: Environment to use for evaluation. 
      self.MBAC._setup_learn( total_timesteps = self.training_steps, eval_env = self.env )
      self.MFAC._setup_learn( total_timesteps = self.training_steps, eval_env = self.env )
                           

      eps= 1
      eps_rewsum = defaultdict(float)
      obs = self.env.reset()
      for step in range( self.training_steps ):
         action = self.MBMFAction(obs, step)
         s = obs
         obs, reward, done, info = self.env.step( action )
         print("[episode: {} | step: {}]\nobs: {}\naction: {}\nnext_obs: {}\nreward: {}\ndone: {}\n\n".format(eps, step, s, action, obs, reward, done))
         self.RealDataBuffer.add(obs=s, action = action, next_obs= obs, reward= reward, done = done, infos= [{'info': None}])
         #self.VirtualDataBuffer
         eps_rewsum[str(eps)]+=reward

         # train MFAC with only RealData
         self.MFAC.replay_buffer = self.RealDataBuffer  # perhaps for MFAC, could just add to it's own replay_buffer.
         if self.MFAC.replay_buffer.size() % 10 == 0:
            self.MFAC.train(gradient_steps = -1, batch_size = 1000) # gradient_steps=-1 to perform as many gradient steps as transitions collected
                                                                    # batch_size

         # train Model ( Dynamics-Reward Model: P_{\theta}(s',r | s,a) )

         # train MBAC with RealData and VirtualData


         if done:
            eps+=1
            obs = self.env.reset()
         
      return eps_rewsum

   def Test(self) -> None:
       pass




   # Gets
   def get_rl_info(self):
      ''' returns rl info in dict  '''
      return dict( rl_agent = str(self.AC_Type), rl_env= str(self.env), params= self.kwargs, training_steps= self.training_steps )

   # Save and Load
   def save_hybrid(self):
      ''' save components of hybrid_mbmf perhaps as pkl if all serializable '''
      pass

   @classmethod
   def load_hybrid(cls):
      ''' perhaps load from saved_pkl and distribute each component to member-vars '''
      pass
