import argparse
import importlib
import runner
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
args = parser.parse_args()

module = importlib.import_module(args.config)
config_file = args.config.split('.')[-1]
params = getattr(module, 'params')
universe, domain, task = params['universe'], params['domain'], params['task']

os.environ["CUDA_VISIBLE_DEVICES"] = params['kwargs']['device']


NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e3),
    'Hopper': int(1e3),
    'HalfCheetah': int(500),#int(3e3),
    'HalfCheetahJump': int(3e3),
    'HalfCheetahVel': int(500),#int(3e3),
    'HalfCheetahVelJump': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(1000),#int(500),#int(3e3),
    'AntAngle': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(2e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(100),
    'Point2DWallEnv': int(100),
    'Reacher': int(200),
    'Pendulum': 10,
    # new
    'hopper': int(1500),
    'halfcheetah': int(500),
    'walker2d': int(3e3),
}

NUM_INITIAL_EXPLORATION_STEPS = {
    'Swimmer': 5000,
    'Hopper': 5000,
    'HalfCheetah': 5000,
    'HalfCheetahJump': 5000,
    'HalfCheetahVel': 5000,
    'HalfCheetahVelJump': 5000,
    'Walker2d': 5000,
    'Ant': 5000,
    'AntAngle': 5000,
    'Humanoid': 5000,
    'Pusher2d': 5000,
    'HandManipulatePen': 5000,
    'HandManipulateEgg': 5000,
    'HandManipulateBlock': 5000,
    'HandReach': 5000,
    'Point2DEnv': 5000,
    'Point2DWallEnv': 5000,
    'Reacher': 5000,
    'Pendulum': 5000,
    # new
    'hopper': 5000,
    'halfcheetah': 5000,
    'walker2d': 5000,
}

# print(params['kwargs'])
params['kwargs']['n_epochs'] = NUM_EPOCHS_PER_DOMAIN[domain]
params['kwargs']['n_initial_exploration_steps'] = NUM_INITIAL_EXPLORATION_STEPS[domain]
params['kwargs']['reparameterize'] = True
params['kwargs']['lr'] = 3e-4
params['kwargs']['target_update_interval'] = 1
params['kwargs']['tau'] = 5e-3
params['kwargs']['store_extra_policy_info'] = False
params['kwargs']['action_prior'] = 'uniform'

variant_spec = {
    'environment_params': {
        'training': {
            'domain': domain,
            'task': task,
            'universe': universe,
            'kwargs': {},
        },
        'evaluation': {
            'domain': domain,
            'task': task,
            'universe': universe,
            'kwargs': {},
        },
    },
    'policy_params': {
        'type': 'GaussianPolicy',
        'kwargs': {
            'hidden_layer_sizes': (256, 256),
            'squash': True,
        },
    },
    'Q_params': {
        'type': 'double_feedforward_Q_function',
        'kwargs': {
            'hidden_layer_sizes': (256, 256),
        },
    },
    'algorithm_params': params,
    'replay_pool_params': {
        'type': 'SimpleReplayPool',
        'kwargs': {
            'max_size': int(1e6),
        },
    },
    'sampler_params': {
        'type': 'SimpleSampler',
        'kwargs': {
            'max_path_length': 1000,
            'min_pool_size': 300,
            'batch_size': 256,
        },
    },
    'run_params': {
        'checkpoint_at_end': True,
        'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN[domain] // 10,
        'checkpoint_replay_pool': False,
    },
}

exp_runner = runner.ExperimentRunner(variant_spec, config_file)
diagostics = exp_runner.train()
