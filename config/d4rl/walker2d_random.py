from .base_mopo import mopo_params, deepcopy
import numpy as np

params = deepcopy(mopo_params)
params.update({
    'domain': 'walker2d',
    'task': 'random-v0',
    'exp_name': 'walker2d_random'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/walker2d-random-v0',
    'pool_load_max_size': 10**6,
    'rollout_length': 1,
    'penalty_coeff': 1.0,
    'dice_pretrain_steps': 600,
    'dice_train_steps': 600,
    'dice_lr': 1e-3,
    'lambda_param': 0.1,
    'min_ratio': 0.01,
    'max_ratio': 50,
    'clip_enable': True,
    # 'penalty_function': lambda x: np.tanh(x) - np.tanh(1),
    # 'penalty_function': lambda x: np.tanh(x-1),
    'penalty_function': lambda x: np.log(1e-8 + x),  
    'device': "0"
})
