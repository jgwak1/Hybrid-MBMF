# external RL
from tabnanny import verbose
import stable_baselines3
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import gym

# internal RL
from rl_algorithm.hybrid import hybrid_mbmf

# util
from argparse import ArgumentParser
from utils.plotters import episode_reward_plotter
from utils.misc import Test

'''
[2022-03-23] 
Mujoco issue solved.
Thanks to "tadashiK commented on 30 Nov 2020": https://github.com/openai/mujoco-py/issues/523
'''


def main():
   
   parser = ArgumentParser()
   parser.add_argument('-m','--mode', nargs=1, choices=['hyb','sb3'], default=['hyb'], help="choose either 'hybrid' or 'sb3'")  
   mode = parser.parse_args().mode[0]
   
   # mujoco gym-envs
   HalfCheetahEnv = gym.make("HalfCheetah-v3")
   HalfCheetahEnv.seed(0)

   # Key-Hyperparameters for Replication
   training_steps = 1000 # 1 eps equiv
   learning_starts = 10
   seed = 0
   gradient_steps = 100
   train_freq = (1,"step")
   # 일단 현재 hyb와 sb3 는 같음 .

   if mode == 'hyb':
      #hybrid MBMF model
      hybrid = hybrid_mbmf(env = HalfCheetahEnv, AC_Type= TD3, 
                           training_steps= training_steps, 
                           learning_starts = learning_starts,
                           gradient_steps = gradient_steps,
                           seed =seed)  # 1000000 : 1000 episodes
      hybrid.Learn()
      
      #episode_rewardsum = hybrid.Learn()
      #episode_reward_plotter( episode_reward_dict = episode_rewardsum, 
      #                        rl_info_dict = hybrid.get_rl_info() )

   else:    
      # regular DDPG agent trained with its own model.learn() 
      td3 = TD3( policy='MlpPolicy', env = HalfCheetahEnv, 
                 learning_starts=learning_starts, gradient_steps = gradient_steps, seed = seed, train_freq=train_freq,
                 verbose = 2)
                 #tensorboard_log= './td3_halfcheetah_tensorboard/')

      td3.learn(total_timesteps = training_steps,
                log_interval=1)
      
      mean_reward, std_reward = evaluate_policy(td3, env = td3.get_env(), n_eval_episodes= 10)
      print("mean_reward:{}\nstd_reward:{}".format(mean_reward, std_reward))
      Test(trained_agent = td3, env=td3.get_env())


if __name__ == '__main__':
   main()