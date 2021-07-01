from numpy.core.numeric import False_
from .base_mopo import mopo_params, deepcopy
import numpy as np

params = deepcopy(mopo_params)
params.update({
    'domain': 'hopper',
    'task': 'medium-expert-v0',
    'exp_name': 'hopper_medium_expert'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/hopper-medium-expert-v0',
    'pool_load_max_size': 2 * 10**6,
    'rollout_length': 5,
    'penalty_coeff': 5.0,
    'dice_pretrain_steps': 200,
    'dice_train_steps': 400,
    'dice_lr': 1e-4,
    'lambda_param': 0.1,
    'min_ratio': 0.01,
    'max_ratio': 10,
    'clip_enable': True,
    # 'penalty_function': lambda x: np.tanh(x) - np.tanh(1),
    # 'penalty_function': lambda x: np.tanh(x-1),
    'penalty_function': lambda x: np.log(1e-8 + x),  
    'device': "0"
})
