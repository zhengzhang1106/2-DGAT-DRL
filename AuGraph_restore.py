import numpy as np
import random
import torch
from ray import tune
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models.catalog import ModelCatalog
from AuGraph_model import AuGraphModel
import Restore_path
import AuOdlConvert
from AuGraph_env_restore import AuGraphEnvRestore
import Database

## 设置随机种子
seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)

## 运行ray
ray.shutdown()
ray.init()
ModelCatalog.register_custom_model('augraph_model', AuGraphModel)  # 使用自定义模型

# 参数配置，把ddpg文件的粘过来就可以
config_re = {
    'env': AuGraphEnvRestore,
    'framework': 'torch',
    'seed': seed_num,
    # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),  # GPU
    'num_gpus': 0,  # GPU，需要<1

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
    "train_batch_size": 128,  # tune.grid_search([512, 128]),
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 64,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
    # Stepsize of SGD.
    "lr": 5e-4,
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
    "vf_clip_param": 10.0,
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

    'gamma': 0.98,  # 奖励衰减
    # 'timesteps_per_iteration': 100,    # 每次迭代100个step

    # === Exploration Settings ===
    "exploration_config": {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
    },
    # Switch to greedy actions in evaluation workers.
    "evaluation_config": {
        "explore": False,
    },

    "num_workers": 0,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,
}

path = Restore_path.path

# 恢复经过训练的agent
agent = PPOTrainer(config=config_re, env=AuGraphEnvRestore)
agent.restore(path)
env = AuGraphEnvRestore({})
episode_reward = 0
done = False
obs = env.reset()

count = 0  # 可以在测试的时候也多跑几次，取一个最好的，把explore开启之后是能选到训练时最好的路径
reward_max = -10000
reward_list = []  # 奖励集合
while count < 20:
    while not done:
        action = agent.compute_action(obs)
        # action = agent.compute_action(obs,explore=False)
        obs, reward, done, info = env.step(action)
        # print("Action:", action*1000)
        # print("State:", obs)
        # print("Reward:", reward)
        print('------')
        episode_reward += reward
        print("Total Reward:", episode_reward)

    if episode_reward >= reward_max:  # 只把最好结果的路径记录下来
        reward_max = episode_reward
        # print('reward_max', reward_max)
        AuOdlConvert.odl_result(AuGraphEnvRestore.au_edge_list)
    #
    #     # print("RWA", AuOdlConvert.result_rwa_phy)
    #     path_count = 0
    #     path_length = []
    #     while path_count < len(AuOdlConvert.result_rwa_phy):
    #         length = int((len(AuOdlConvert.result_rwa_phy[path_count]) - 1) / 2) + int(
    #             (len(AuOdlConvert.result_rwa_phy[path_count + 1]) - 1) / 2)
    #         path_length.append(length)
    #         # path_length.append(int((len(AuOdlConvert.result_rwa_phy[path_count])-1)/2) + int((len(AuOdlConvert.result_rwa_phy[path_count+1])-1)/2))
    #         path_count += 2
    #     print("路径长度")
    #     print("ADMIRE")
    #     for pathi in path_length:
    #         print(pathi)
    #
    #     print("累计虚拟拓扑跳数")
    #     print(virtual_hop_cumulate)
    #     print("虚拟拓扑跳数50")
    #     print("ADMIRE")
    #     index = 1
    #     while index < len(virtual_hop_cumulate_list):
    #         virtual_hop_cumulate_list[index] += virtual_hop_cumulate_list[index - 1]
    #         index += 2
    #
    #     index = 1
    #     while index < len(virtual_hop_cumulate_list):
    #         print(virtual_hop_cumulate_list[index])
    #         index += 2
    # print(Service.path1)

    # with open('rwa_phy.txt', 'w') as f:
    #     for i in range(len(AuOdlConvert.result_rwa_phy)):
    #         f.write(str(AuOdlConvert.result_rwa_phy[i]))
    #         f.write('\n')
    #     f.close()
    #
    # with open('rwa_vir.txt', 'w') as file:
    #     for i in range(len(AuOdlConvert.result_rwa_vir)):
    #         file.write(str(AuOdlConvert.result_rwa_vir[i]))
    #         file.write('\n')
    #     file.close()
    #
    # # 写物理链路
    # with open('phyLinks-GA.txt', 'w') as file:
    #     for data in Database.links_physical:
    #         np.savetxt(file, data, fmt='%.3f', delimiter='\t')
    #         # file.write('\n')
    #     file.close()
    #
    # with open('link_vir.txt', 'w') as file:
    #     for k in range(len(AuGraph.links_virtual_list)):
    #         file.write(str(AuGraph.links_virtual_list[k]))
    #         file.write('\n')
    #     file.close()
    # #
    # with open('odl_iog.txt', 'w') as file:
    #     for i in range(len(AuOdlConvert.result_odl)):
    #         file.write(str(AuOdlConvert.result_odl[i]))
    #         file.write('\n')
    #     file.close()
    #
    # linkmsg_phy = [[] for _ in range(Database.time)]
    # row, col = Database.graph_connect.shape
    # for t in range(Database.time):
    #     for i in range(row):
    #         for j in range(col):
    #             if Database.graph_connect[i][j] == 1:
    #                 linkmsg_phy[t].append({'src': i, 'dst': j, str(t)+'_hour':Database.links_physical[t*3:(t+1)*3, i, j].tolist()})
    #
    # for t in range(Database.time):
    #     with open(str(t) +'_hour_links.txt', 'w') as file:
    #         for k in range(len(linkmsg_phy[t])):
    #             file.write(str(linkmsg_phy[t][k]))
    #             file.write('\n')
    #         file.close()

    reward_list.append(episode_reward / 50)
    count += 1
    episode_reward = 0
    obs = env.reset()
    done = False

print("奖励列表", reward_list)
# print("Total Reward:", episode_reward)
# print("使用波长", Compute.compute(Database.links_physical))
