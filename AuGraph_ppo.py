import numpy as np
import random
import torch
from ray import tune
import ray

from AuGraph_env import AuGraphEnv
from AuGraph_model import AuGraphModel
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.catalog import ModelCatalog
import os

# 设置随机种子
seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

# 运行ray
ray.shutdown()
ray.init()
ModelCatalog.register_custom_model('augraph_model', AuGraphModel)  # 使用自定义模型
tunerun = tune.run(
    PPOTrainer,
    # resume=True,
    config={
        # 其他
        'env': AuGraphEnv,
        'framework': 'torch',
        'seed': seed_num,
        # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # GPU
        'num_gpus': 0,   # GPU

        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 0.8,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 128,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 256,  # tune.grid_search([512, 256]),
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 64,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 1e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 20,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",

        # Deprecated keys:
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        # Use config.model.vf_share_layers instead.
        # "vf_share_layers": DEPRECATED_VALUE,

        # ========= Model ============
        # 在进入actor和critic的隐藏层之前，会先运行'model'里的参数
        # "use_state_preprocessor": True,     # 可以使用自定义model
        # 自定义模型
        'model': {
            'custom_model': 'augraph_model',
            "post_fcnet_hiddens": [1024, 2048],
            "post_fcnet_activation": 'relu',  # tune.grid_search(['relu','tanh'])
        },

        'gamma': 0.98,      # 奖励衰减
        # 'timesteps_per_iteration': 100,    # 每次迭代100个step

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 5500,  # tune.grid_search([5000, 8000]),   # Timesteps over which to anneal epsilon.
        },


        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },

        "num_workers": 0,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 1,

    },
    checkpoint_at_end=True,  # 结束时存储检查点
    checkpoint_freq=1,
    # 隔几个training_iteration存储一次(每个training_iteration包含1000个timestep，这1000个timestep到底是多少个episode就得看训练的情况了)
    # restore=path #载入检查点
    stop={
        'training_iteration': 300   # 训练轮次
    }
)

# 保存一个最大的训练好的agent
best_checkpoint = tunerun.get_best_checkpoint(
    trial=tunerun.get_best_logdir('episode_reward_mean', 'max'),
    metric='episode_reward_mean',
    mode='max'
)
print(best_checkpoint)
