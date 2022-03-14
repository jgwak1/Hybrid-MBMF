# stable-baselines3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
# OpenAI gym
import gym
# PyTorch
import torch
# Misc
# typing: https://docs.python.org/3/library/typing.html
from typing import Optional, List, Union, Dict, Type, Any # > 
from typing import Callable # https://pyquestions.com/what-exactly-is-python-typing-callable#:~:text=typing.Callable%20is%20the%20type%20you%20use%20to%20indicate,classmethod%20s%2C%20staticmethod%20s%2C%20bound%20methods%20and%20lambdas.


class actor_critic( ActorCriticPolicy ):

   '''
   Extends Stable-Baselines3 Actor-Critic Policy
   > Source: https://github.com/DLR-RM/stable-baselines3/blob/e88eb1c9ca98650f802409e5845e952c39be9e76/stable_baselines3/common/policies.py#L379
   > Refer to: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
               https://stable-baselines3.readthedocs.io/en/sde/guide/custom_policy.html
               https://github.com/tensorlayer/RLzoo/tree/master/rlzoo/algorithms
   Parent Mem-Funcs:

   Overrides:

   Adds:

   '''
   def __init__(self, 
                observation_space: gym.spaces.Space, # Observation space
                action_space: gym.spaces.Space,  # Action Space
                lr_schedule: Schedule, # Learning Rate Schedule (could be constant) | Schedule is equivalent to Callable[[float], float]
                                       # > 
                                       # > Refer to: https://github.com/DLR-RM/stable-baselines3/blob/e88eb1c9ca98650f802409e5845e952c39be9e76/stable_baselines3/common/type_aliases.py 
                                       #             https://github.com/hill-a/stable-baselines/issues/509

                net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, # The specification of the policy and value networks
                activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,   # Activation function
                ortho_init: bool = True, # Whether to use or not 'Orthogonal Initalization'
                                         # > 
                use_sde: bool = False,   # Whether to use 'State Dependent Exploration (SDE)' or not.
                                         # > 
                log_std_init: float = 0, # Initial value for the 'log standard deviation'
                                         # > 
                full_std: bool = True,   # Whether to use (n_features * n_actions) parameters
                                         # for the 'std' instead of only (n_features, ) when using gSDE.
                                         # > 
                sde_net_arch: Optional[List[int]] = None, # Network architecture for extracting features when using gSDE.
                                                          # If None, the latent features from the policy will be used.
                                                          # Pass an empty list to use the states as features.
                                                          # >
                use_expln: bool = False,                  # Use ``expln()`` function instead of ``exp()`` to ensure
                                                          # a positive standard deviation. 
                                                          # It allows to keep variance above ero and previent it from growing too fast.
                                                          # In practice, ``exp()`` is usally enough.
                                                          # >
                squash_output: bool = False,              # Whether to squash the output using a tanh function,
                                                          # this allows to ensure boundaries when using gSDE.
                features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, # Features extractor to use.
                features_extractor_kwargs: Optional[Dict[str, Any]] = None, # Keyword arguments to pass to the features extractor.
                normalize_images: bool = True, # Whether to normalize images or not, dividing by 255.0 (True by default.)
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam, # The optimizer to use, ``torch.optim.Adam`` by default.
                optimizer_kwargs: Optional[Dict[str, Any]] = None # Additional keyword arguments, excluding the learning rate, to pass to the optimizer.
                ):

               super().__init__(observation_space, 
                                 action_space, 
                                 lr_schedule, 
                                 net_arch, 
                                 activation_fn, 
                                 ortho_init, 
                                 use_sde, 
                                 log_std_init, 
                                 full_std, 
                                 sde_net_arch,
                                 use_expln, 
                                 squash_output,
                                 features_extractor_class, 
                                 features_extractor_kwargs, 
                                 normalize_images,
                                 optimizer_class, 
                                 optimizer_kwargs)

               return 

                  