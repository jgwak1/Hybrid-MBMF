''' Miscellaneous Functions '''

import gym


def Test(trained_agent, env: gym.Env) -> None:
      test_reward_sum = 0
      # Enjoy trained agent
      obs = env.reset()
      for i in range(1000):
         action, _states = trained_agent.predict(obs, deterministic=True)
         obs, rewards, dones, info = env.step(action)
         test_reward_sum += rewards[0]
         #self.env.render()
      print("test_reward_sum: {}".format(test_reward_sum))