from stable_baselines3.ddpg import DDPG
import gym
from argparse import ArgumentParser

from rl_algorithm.hybrid import hybrid_mbmf

def main():
   # mujoco gym-envs
   HalfCheetahEnv = gym.make("HalfCheetah-v2")
   # hybrid MBMF model
   hybrid = hybrid_mbmf(env = HalfCheetahEnv, AC_Type= DDPG)

if __name__ == '__main__':
   main()