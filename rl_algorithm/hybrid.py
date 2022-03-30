# Refer to: 
#           typing (python library): https://docs.python.org/3/library/typing.html
#           stable-baselines3("SB3") Source: https://github.com/DLR-RM/stable-baselines3
#                              SB3 Document: https://stable-baselines3.readthedocs.io/_/downloads/en/master/pdf/
#           gym: https://github.com/openai/gym/tree/master/gym
#           mujoco: https://gym.openai.com/envs/#mujoco  

'''
   [Questions]
   1. How can my scheme (training MFAC with only real-data, and training MBAC with both real and virtual data)
      harmonize with off-policy (replay-buffer) learning especially for MBAC?
'''
# external imports
from tabnanny import verbose
from webbrowser import Grail
from psutil import virtual_memory
from stable_baselines3.ddpg import DDPG
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
import gym
import torch

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.evaluation import evaluate_policy


from typing import NewType, Union, Optional, Dict, Any
from collections import defaultdict
import numpy as np
from torch import _linalg_inv_out_helper_
# internal imports
from rl_algorithm.data_buffer import DataBuffer
from rl_algorithm.model.DR_Model import DynamicsRewardModel

from utils.misc import Test

class hybrid_mbmf:

   def __init__(self,
                env: gym.Env,
                AC_Type: Union[DDPG, SAC, TD3], # Let's assume that AC-agent used for MBAC and MFAC are the same, and offpolicy agents.
                AC_kwargs: Dict = {},
                training_steps: int = 1000,
                learning_starts: int = 100,
                gradient_steps: int = 100,
                train_freq = (1, "episode"), # (1, "step"),
                seed = 0,
                ) -> None:

               # for training
               self.training_steps = training_steps
               self.seed = seed 
               self.learning_starts = learning_starts
               self.gradient_steps = gradient_steps
               self.train_freq = TrainFreq(train_freq[0], TrainFrequencyUnit(train_freq[1])) # for compatibility with SB3 train()

               # training environment
               self.env = DummyVecEnv([lambda: env]) # for compatiblity with SB3 train()
               # Model-Free Actor-Critic RL Algorithms that are each trained with only real-data, and both real-data and virtual-data. 
               self.AC_Type = AC_Type
               self.offpolicy_kwargs = {
                                        "policy": "MlpPolicy", 
                                        "env": self.env,
                                        "learning_starts": self.learning_starts,
                                        "gradient_steps": self.gradient_steps,
                                        "seed": self.seed,
                                        "verbose": 2
                                       } 
               self.offpolicy_kwargs.update( AC_kwargs )
               
               self.MFAC = self.AC_Type(**self.offpolicy_kwargs)
               self.MBAC = self.AC_Type(**self.offpolicy_kwargs)

               # T,R Model
               self.Model = DynamicsRewardModel( env = env )    

               # Following 'DataBuffers' are "external" databuffers which each store real data and virtual data.
               # Above AC-RL Aents have their own "internal" databuffers ('rollout-buffer' or 'replay-buffer' for used for their learning.
               #  whether to depends 
               #self.NewRealData = DataBuffer(buffer_size=int(1e4), observation_space=env.observation_space, action_space= env.action_space)
               self.RealDataBuffer = DataBuffer(buffer_size=int(1e4), observation_space=env.observation_space, action_space= env.action_space) 
               self.VirtualDataBuffer = DataBuffer(buffer_size=int(1e4), observation_space=env.observation_space, action_space= env.action_space)
               

               return


   def MBMFAction(self, current_obs, current_timestep):
      ''' 
      Selects between actions suggested by MFAC and MBAC.

      [2022-03-25]
      I think I should instance-level-override the collect_rollout() of self.MFAC object.
      This function can be used somewhere in the function-body when overriding collcet_rollout.
      Goal is to avoid complicating things and try not to mess too much of SB3 code.
      '''

      pass


   #def SortOut(self, virtual_data_batch: DataBuffer):
   def SortOut(self,
               real_obs: torch.Tensor, action: torch.Tensor, 
               virtual_nextobs: torch.Tensor, virtual_reward: torch.Tensor) -> bool: # returns bool of better-than-nothing or worse-than-nothing
      
      ''' 
      [ Google Docs "Hybrid-MBMF Algorithm Design" ] 
      Educated-Guess whether the virtual-data is better-than-nothing for training or not. '''

      # Compare 
      # Coach's [ Q_{MF}(s,a) ] ------ (1)
      # and
      # Boxer's [ r + E_{s’~Μ(s,a)} [ γ * Q_{MF}(next_s, π_{MF}(next_s) ) ] ------ (2)

      is_better_than_nothing = False



      # (1) Q_{MF}(s,a)
      #real_obs = real_obs.to(dtype=torch.float32)
      #action = torch.from_numpy(action).to(dtype=torch.float32)

      # self.MFAC.critic doc:
      '''
      # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
      '''

      '''
      Error is coming from     
      [ torch.modules.nn.flatten.py ]
            def forward(self, input: Tensor) -> Tensor:
            return input.flatten(self.start_dim, self.end_dim) <<<< here
      
      "real_obs" has to be as following;
      tensor([[-0.0615,  0.0281, -0.2272, -0.0502,  0.0960,  0.2686, -0.0184, -0.4265,
         -0.8214,  1.1791,  2.2890,  1.8804, -9.0074, -4.5512,  1.0007, -8.1646,
         -5.9016]])
      
      In other words, it should be "torch.Size([1, 17])" instead of torch.size([17]), when did tensorobj.shape
      
      
      '''

      # Returned "Coach_Q" is 2 q-values from 2 q-networks

      '''
      < From source: stable-baselines.common.policies.ContinuousCritic >
      def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)
      
      # By default, it creates two critic networks used to reduce overestimation
         thanks to clipped Q-learning (cf TD3 paper).
      
      '''

      # For some AC methods (like TD3), 
      # two critic networks are used to reduce overestimation. (a.k.a., clipped Q-learning)
      # we can just get the minimum of those two.
      Coach_Q = min( self.MFAC.critic( real_obs, action ) )# Get "Q_{MF}( real_obs, action )"      
      Coach_Q = Coach_Q.item()
   
      # (2) r + E_{s’~Μ(s,a)} [ γ * Q_{MF}(next_s, π_{MF}(next_s) )
      discount_rate = self.MFAC.gamma
      virtual_reward_scalar = virtual_reward.item()
      next_Q = min( self.MFAC.critic( virtual_nextobs, self.MFAC.actor( virtual_nextobs ) ) )
      next_Q_scalar = next_Q.item()
      Boxer_Q = virtual_reward_scalar + ( discount_rate * next_Q_scalar )

      print(f"SortOut\n\tCoach_Q: {Coach_Q}\n\tBoxer_Q: {Boxer_Q}")


      ''' Dinstinguish "better than nothing" and "worse than nothing" from a 'VirtualDataBatch'. '''
      
      return is_better_than_nothing


   def Improve(self, real_obs, action, virtual_nextobs, virtual_reward) -> Any:

      ''' Improve the worse-than-nothing virtual data '''

      pass


   def Learn_Dev(self) -> None:

      '''

         [2022-03-26]

         FIRST THINK HOW TO INCORPORATE THE MODEL VIRTUAL DATA GENERATION HERE.
         [1]
         
         Override self.MFAC.collect_rollouts() at instance-level, 
         so that we can just simply insert(incorporate) the "Action-Select" part to the
         existing collect_rollouts() so that I don't mess things up in there?
         
         [ Refer to: https://stackoverflow.com/questions/394770/override-a-method-at-instance-level ]

         OR

         Make a child-class for class of self.MFAC, and override collect_rollouts()

      '''


      '''
      [2022-03-26]
      PSEUDO-CODE for "Hybrid-MBMF"

   
      1. Initialize: 
                   (1)  MFAC { Actor, Critic, Buffer(for Only Real-Data) }                   <--- " Support-Policy "
                   (2)  MBAC { Actor, Critic, Buffer(for Both Real-Data AND Virtual-Data)}   <--- " Target-Policy "
                   (3)  MODEL { P_{θ}(s’,r | s,a) }
                   (4)  Virtual-Data-Buffer  # Basically, Real-Data-Buffer is MFAC::Buffer 
      
      2. For N epochs do:

      3.     
      4.         MFAC interacts with MDP, and collects Real-Data. { Have MFAC very explorative in the beginning. }
      5.         Train MFAC with Real-Data.
      6.         Train MODEL using Accumulated Real-Data w/ Supervised Learning.

      7.     For M steps do:

      8.         Generate Virtual-Data using MODEL. 
                     (1) Sample s_{t} from MFAC::Buffer uniformly at random.
                     (2) Apply 1 Random Action to sampled s_{t} making it the new s_{t} 
                         { Reasoning: To have Virtual-Data not starting from the state we already have as Real-Data, but still near. } 
                     (3) From new s_{t}, step MODEL using MFAC::Actor.       
      9.                 
      10.    For M steps do:
      11.         SortOut better-than-nothing from generated Virtual-Data using MFAC's Critic. { MFAC is Support-Policy }
      12.         Improve the worse-than-nothing virtual data?
      13.         Train MBAC with sorted-out/imporved data.

      
      '''

      # Setups 
      _setup_learn_args = { 
                           "total_timesteps": self.training_steps, "eval_env": self.env,
                           "callback": None, "eval_freq": -1, "n_eval_episodes": 5,
                           "log_path": None, "reset_num_timesteps": True
                          }
      total_timesteps_MBAC, callback_MBAC = self.MBAC._setup_learn( **_setup_learn_args )
      total_timesteps_MFAC, callback_MFAC = self.MFAC._setup_learn( **_setup_learn_args )


      Epochs = 30
      Interaction_Steps = 3000
      VirtualData_Steps = 3000
      
      # For N epochs do:
      for N in range(Epochs):   # Try to equate 1 "Epoch" to 1 self.Model.Train function call
         
         # MFAC interacts with MDP, and collects Real-Data. 
         #  { Have MFAC very explorative in the beginning. }
         self.MFAC.collect_rollouts(callback = callback_MFAC,
                                    env = self.env, 
                                    learning_starts = self.learning_starts,                                   
                                    train_freq = self.train_freq, 
                                    replay_buffer = self.MFAC.replay_buffer
                                 )
                                    #replay_buffer = self.RealDataBuffer)  # Could collect to our RealDataBuffer instead of "self.MFAC.replay_buffer"
                                                                        # But this would need training self.MFAC would need to put into self.MFAC.replay_buffer
                                                                        # This seems redundant.


                                    #         :param train_freq: How much experience to collect
                                    #                            by doing rollouts of current policy.
                                    #                            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
                                    #                            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
                                    #                            with ``<n>`` being an integer greater than 0.
         
         # Train MFAC with Real-Data.
         self.MFAC.train(gradient_steps = self.gradient_steps, batch_size = 100)
         # TODO: Train MODEL using Accumulated Real-Data w/ Supervised Learning
         self.Model.Train( samples_buffer = self.MFAC.replay_buffer, epoch = N ) # could also pass epoch in to here.
         
         # synchronize MBAC replaybuffer and MFAC replaybuffer
         self.MBAC.replay_buffer = self.MFAC.replay_buffer



         for M in range(VirtualData_Steps):

            # TODO: 
            #        [1] Generate Virtual-Data using MODEL.
            #           (1-1) Sample s_{t} from MFAC::Buffer uniformly at random. 
            #           (1-2) From sampled s_{t}, apply 1 Random Action with the MODEL, generating one virtual-data.
            #                => { Reasoning: To have Virtual-Data starting from the state which we already have in Real-Data but with different action.; near-distribution virtual data }            # 
            #        [2] SortOut better-than-nothing from generated Virtual-Data using MFAC's Critic. { MFAC is Support-Policy }
            #        [3] Improve the worse-than-nothing virtual data?
            #        [4] Train MBAC with sorted-out/imporved data.


            ''' Implementation '''
            #[1] Generate Virtual-Data using MODEL.
            #    (1-1) Sample s_{t} from MFAC::Buffer uniformly at random.
            #                                                            [Reference]: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py
            RealData_sample = self.MFAC.replay_buffer.sample(batch_size = 1)
            #    (1-2) From sampled s_{t}, apply 1 Random Action with the MODEL, generating one virtual-data.
            #         => { Reasoning: To have Virtual-Data starting from the state which we already have in Real-Data but with different action.; near-distribution virtual data }
            
            sampled_realobs = RealData_sample.observations.to(dtype=torch.float32)
            random_action = torch.from_numpy( self.env.action_space.sample() )
            random_action = torch.unsqueeze(random_action, axis = 0)
            #raw_virtual_data = self.Model.predict(observation= sampled_realobs[0], action= random_action)
            raw_virtual_data = self.Model.predict(observation= sampled_realobs, action= random_action)
            virtual_nextobs = raw_virtual_data.squeeze()[0:-1].unsqueeze(dim=0) # first to second-last
            virtual_reward = raw_virtual_data.squeeze()[-1].unsqueeze(dim=0) # last
            
            #[2] SortOut better-than-nothing from generated Virtual-Data using MFAC's Critic. { MFAC is Support-Policy }
            Better_than_Nothing = self.SortOut(real_obs = sampled_realobs, action = random_action, 
                                               virtual_nextobs = virtual_nextobs, virtual_reward = virtual_reward)

            if Better_than_Nothing:
                pass
            else:
                  #[3] Improve the worse-than-nothing virtual data?
                  pass
         

            # 
            # self.MBAC.replay_buffer.extend( virtual_data )
            # 

            #[4] Train MBAC with sorted-out/imporved data.
            #self.MBAC.replay_buffer.extend()    # extend: Add a new batch of transitions to the buffer             
            self.MBAC.train(gradient_steps = self.gradient_steps, batch_size = 100)


            pass




      '''
      currstep = 1
      while currstep <= self.training_steps:
      
         # [2022-03-26] Write based on the pseudo-code above.

         if currstep > 0 and currstep > self.learning_starts:
            
            # MFAC being trained with real-data
            self.MFAC.train(gradient_steps = self.gradient_steps, batch_size = 100)

            # MBAC trained by a batch sampled form a buffer that contains both virtual-data and real-data
            self.MBAC.train(gradient_steps = self.gradient_steps, batch_size = 100)


         currstep += 1
      '''

      # eval and Test
      mean_reward, std_reward = evaluate_policy(self.MFAC, env = self.env, n_eval_episodes= 10)
      print("mean_reward:{}\nstd_reward:{}".format(mean_reward, std_reward))
      
      Test( self.MFAC, self.env )

      return 


   def Learn(self) -> None:
      ''' 
      [Refer to]
   
      "MBPO w/ DeepRL" Pseudo-code (Page 6 of "When to Trust Your Model-Based Policy Optimization"; Sergey Levine, et al. NIPS 2019)
      
      1. Initialize target-policy, Model 'P_{θ}(s’,r | s,a)', real-dataset 'D_real', model-dataset 'D_virtual'.
      
      2. For N epochs do:
      3.    Train Model on D_real via maximum likelihood.

      4.    For E steps do:
      5.       Take action in environment according to target-policy; 
      7.       Add experience to D_real.

      8.       for M model-rollouts do:
      9.          Sample s_{t} uniformly from D_real
      10.         Perform k-step model-rollout starting from s_{t} using target-policy, and add to D_virtual.
      11.
      12.      for G gradient updates do:
      13.         Update target-policy parameters with model-data (D_virtual).  
      
      
      "MBPO Github Repo (by Authors)" : https://github.com/JannerM/mbpo
      --> Note that MBPO uses a ensemble of models
      
      '''

      '''
      [2022-03-24 NOTES]
         TODO:  
            (4) Model proto-type
               - Model input layer, output layer 을 input_env로부터 어떻게 받을것인지.
      '''

      #eps= 1
      #eps_rewsum = defaultdict(float)

 
      _setup_learn_args = { 
                           "total_timesteps": self.training_steps, "eval_env": self.env,
                           "callback": None, "eval_freq": -1, "n_eval_episodes": 5,
                           "log_path": None, "reset_num_timesteps": True
                          }
      total_timesteps_MBAC, callback_MBAC = self.MBAC._setup_learn( **_setup_learn_args )
      total_timesteps_MFAC, callback_MFAC = self.MFAC._setup_learn( **_setup_learn_args )      
      
      currstep = 1
      while currstep < self.training_steps:
         
         '''
         [2022-03-25]

         *** Plans ***

         [1]
         Override self.MFAC.collect_rollouts() at instance-level, 
         so that we can just simply insert(incorporate) the "Action-Select" part to the
         existing collect_rollouts() so that I don't mess things up in there?
         
         [ Refer to: https://stackoverflow.com/questions/394770/override-a-method-at-instance-level ]

         [2]
         Maybe could incorporate the "Model's virtual-data generation" also in the overriding of collect_rollouts()?
         Virtual-data will anyways be added to MBAC's replay-buffer.
         
         Question is:
             what should be the (s,a) of virtual-data?

         '''

         # MFAC interacting with the model and collecting real-data
         self.MFAC.collect_rollouts(env = self.env, learning_starts = self.learning_starts, 
                                    callback = callback_MFAC, 
                                    train_freq = self.train_freq, replay_buffer = self.MFAC.replay_buffer) 
         
         if currstep > 0 and currstep > self.learning_starts:
            # MFAC being trained with real-data
            self.MFAC.train(gradient_steps = self.gradient_steps, batch_size = 100)
   
         currstep += 1


      # eval and Test
      mean_reward, std_reward = evaluate_policy(self.MFAC, env = self.env, n_eval_episodes= 10)
      print("mean_reward:{}\nstd_reward:{}".format(mean_reward, std_reward))
      
      Test( self.MFAC, self.env )


      '''

      ********************* FOLLOWING IS OLD WORK THAT DOESN'T WORK *************************************************************

      obs = self.env.reset()
      for currstep in range( self.training_steps ):    
         
         action = self.MBMFAction(obs, currstep)[0]
         s = obs
         obs, reward, done, info = self.env.step( action )
         self.MFAC.replay_buffer.add(obs=s, action = action, next_obs= obs, reward= reward, done = done, infos= [{'info': info}] )
         print("*"*80)
         print("[episode: {} | step: {}]\n\nobs: {}\n\nnext_obs: {}\n\naction: {}\n\nreward: {}\n\ndone: {}\n\n".format(eps, currstep, s, obs, action, reward, done))
         # Perhaps store transitions o it as below than above?
         # https://github.com/DLR-RM/stable-baselines3/blob/00ac43b0a90852eddec31e6f36cac7089c235614/stable_baselines3/common/off_policy_algorithm.py#L521
         # Store data in replay buffer (normalized action and unnormalized observation)
         #self.MFAC._store_transition(self.MFAC.replay_buffer, np.array([action]), np.array([obs]), np.array([reward]), np.array([done]), np.array([info]))
         #self.VirtualDataBuffer
         eps_rewsum[str(eps)]+=reward
         # train MFAC with only RealData
         #self.MFAC.replay_buffer = self.RealDataBuffer  # perhaps for MFAC, could just add to it's own replay_buffer.
         #if self.MFAC.replay_buffer.size() > 100:
         #   print("Now we are going to train at step:{} ".format(currstep))
         
         self.MFAC.train(gradient_steps = 100, batch_size = 100)  # gradient_stpeps = 1000 은 .learn() 따라하는것임.
         # train Model ( Dynamics-Reward Model: P_{θ}(s’,r | s,a) )
         # train MBAC with RealData and VirtualData
         if done:
            eps+=1
            obs = self.env.reset() 
      '''   

      return None

      


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
