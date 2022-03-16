# external RL
from stable_baselines3.ddpg import DDPG
from stable_baselines3.td3 import TD3
from stable_baselines3.sac import SAC
import gym
# internal RL
from rl_algorithm.hybrid import hybrid_mbmf
# util
from argparse import ArgumentParser
from utils.plotters import episode_reward_plotter


def main():
   # mujoco gym-envs
   HalfCheetahEnv = gym.make("HalfCheetah-v2")
   # hybrid MBMF model
   hybrid = hybrid_mbmf(env = HalfCheetahEnv, AC_Type= TD3, training_steps= 1000000)  # 1000000 : 1000 episodes
   episode_rewardsum = hybrid.Train()
   episode_reward_plotter( episode_reward_dict = episode_rewardsum, 
                           rl_info_dict = hybrid.get_rl_info() )



   # regular DDPG agent trained with its own model.learn() 
   #ddpg = DDPG( policy='MlpPolicy', env = HalfCheetahEnv )
   #ddpg.learn(total_timesteps = 1000000)


if __name__ == '__main__':
   main()