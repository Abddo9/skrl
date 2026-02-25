import torch
import torch.nn as nn 
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.encode = True
        self.debug = False

        self.num_agents = 2
        self.agent_feature_size = 5 
        self.lidar_feature_size = 10 # 5 rays  2 lidars
        self.lidar_embed_dim = 4 
        self.machine_feature_size = 2 # pos, collected
        self.atten_embed_dim = 16
        self.attention_heads = 2

        self.lidar_enc = nn.Sequential(nn.LayerNorm(self.lidar_feature_size), nn.Linear(self.lidar_feature_size, self.lidar_embed_dim), nn.Tanh(), nn.LayerNorm(self.lidar_embed_dim))
        self.agent_enc = nn.Sequential(nn.Linear(self.agent_feature_size + self.lidar_embed_dim, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.machines_enc = nn.Sequential(nn.Linear(self.machine_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.other_agents_enc = nn.Sequential(nn.Linear(self.agent_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))

        self.agents_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        
        # LayerNorm after attention
        self.agents_attn_norm = nn.LayerNorm(self.atten_embed_dim)

        size = self.atten_embed_dim*3
        if not self.encode:
            size = self.num_observations

        self.net = nn.Sequential(nn.Linear(size, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def encode_objects(self, obs):
        B = obs.shape[0]
        idx = 0
        sizes = [
            (self.agent_feature_size, 1),  # agent
            (self.lidar_feature_size, 1),  # lidar
            (self.machine_feature_size, 1),  # machines
            (self.agent_feature_size, self.num_agents - 1),  # other agents
        ]

        chunks = []
        for feature_size, num in sizes:
            total = feature_size * num
            chunks.append(obs[:, idx:idx+total].reshape(B, num, feature_size))
            idx += total

        agent_info, lidar_info, machines_info, other_agents_info = chunks
        lidar_info = self.lidar_enc(lidar_info)
        agent_info = torch.cat([agent_info, lidar_info], dim=-1)
        agent_info = self.agent_enc(agent_info)

        machines_info = self.machines_enc(machines_info)
        
        other_agents_info = self.other_agents_enc(other_agents_info)
            
        other_agents_info,_ = self.agents_attention(agent_info, other_agents_info, other_agents_info, need_weights=False)
        other_agents_info = self.agents_attn_norm(other_agents_info)
            
        env_info = torch.cat([agent_info, machines_info, other_agents_info], axis =1).reshape(obs.shape[0], -1)
        
        return env_info

    def compute(self, inputs, role):
        obs = inputs["states"]
        if self.encode:
            obs = self.encode_objects(inputs["states"])

        means = self.net(obs)
        stds = self.log_std_parameter
        
        return means, stds, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.encode = True
        self.debug = False

        self.num_agents = 2
        self.agent_feature_size = 5
        self.lidar_feature_size = 10 # 5 rays  2 lidars
        self.lidar_embed_dim = 4 
        self.machine_feature_size = 2 # pos
        self.atten_embed_dim = 18
        self.attention_heads = 3

        self.lidar_enc = nn.Sequential(nn.LayerNorm(self.lidar_feature_size),nn.Linear(self.lidar_feature_size, self.lidar_embed_dim), nn.Tanh(), nn.LayerNorm(self.lidar_embed_dim))
        self.agent_enc = nn.Sequential(nn.Linear(self.agent_feature_size + self.lidar_embed_dim, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.machines_enc = nn.Sequential(nn.Linear(self.machine_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.other_agents_enc = nn.Sequential(nn.Linear(self.agent_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))

        self.agents_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        
        # LayerNorm after attention
        self.agents_attn_norm = nn.LayerNorm(self.atten_embed_dim)

        self.env_attention = nn.MultiheadAttention(self.atten_embed_dim*3,self.attention_heads, batch_first=True)
        self.env_attn_norm = nn.LayerNorm(self.atten_embed_dim*3)
        self.env_query = nn.Parameter(torch.randn(1, 1, self.atten_embed_dim*3))

        size = self.atten_embed_dim*3
        if not self.encode:
            size = self.num_observations

        self.net = nn.Sequential(nn.Linear(size, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 1))


    def encode_objects(self, obs):
        obs_shape = obs.shape

        obs = obs.reshape(-1, self.agent_feature_size + self.lidar_feature_size + (self.num_agents-1)* self.agent_feature_size + self.machine_feature_size) 
               
        B = obs_shape[0]
        idx = 0
        sizes = [
            (self.agent_feature_size, 1),  # agent
            (self.lidar_feature_size, 1),  # lidar
            (self.machine_feature_size, 1),  # machines
            (self.agent_feature_size, self.num_agents - 1),  # other agents
        ]

        # Unpack efficiently
        chunks = []
        for feature_size, num in sizes:
            total = feature_size * num
            chunks.append(obs[:, idx:idx+total].reshape(-1, num, feature_size))
            idx += total

        agent_info, lidar_info, machines_info, other_agents_info = chunks
        lidar_info = self.lidar_enc(lidar_info)
        agent_info = torch.cat([agent_info, lidar_info], dim=-1)
        agent_info = self.agent_enc(agent_info)

        machines_info = self.machines_enc(machines_info)

        other_agents_info = self.other_agents_enc(other_agents_info)

        other_agents_info,_ = self.agents_attention(agent_info, other_agents_info, other_agents_info, need_weights=False)
        other_agents_info = self.agents_attn_norm(other_agents_info)

        agents_obs = torch.cat([agent_info, machines_info, other_agents_info], axis =1).reshape(obs_shape[0],self.num_agents, -1)
        query = self.env_query.repeat(obs_shape[0],1,1)  

        agents_obs,_ = self.env_attention(query, agents_obs, agents_obs, need_weights=False)
        agents_obs = self.env_attn_norm(agents_obs)
        
        agents_obs = agents_obs.reshape(agents_obs.shape[0],-1)
        return agents_obs

    def compute(self, inputs, role):
        obs = inputs["states"]
        if self.encode:
            obs = self.encode_objects(inputs["states"])
        output = self.net(obs)
        return output, {}


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-Collect-Direct-v0")
env = wrap_env(env)

device = env.device


# instantiate memories as rollout buffer (any memory can be used for this)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# MAPPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = Policy(env.observation_space(agent_name), env.action_space(agent_name), device)
    models[agent_name]["value"] = Value(env.state_space(agent_name), env.action_space(agent_name), device)

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/mappo.html#configuration-and-hyperparameters
cfg = MAPPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 6  # 24 * 4096 / 16384
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.001
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": device}
cfg["shared_state_preprocessor"] = RunningStandardScaler
cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 180
cfg["experiment"]["checkpoint_interval"] = 1800
cfg["experiment"]["directory"] = "runs/torch/Collect/SMAPPO"
cfg["experiment"]["experiment_name"] = "Collect_Lidar20_8"

print("Model cfg:", cfg)

agent = MAPPO(possible_agents=env.possible_agents,
              models=models,
              memories=memories,
              cfg=cfg,
              observation_spaces=env.observation_spaces,
              action_spaces=env.action_spaces,
              device=device,
              shared_observation_spaces=env.state_spaces)


# configure and instantiate the RL trainer
evaluate = False
checkpoint = '/home/wahabu/skrl/runs/torch/Collect/SMAPPO/CollectDefault/checkpoints/best_agent.pt'

if evaluate and checkpoint:
    agent.load(checkpoint)
    print(f"Loaded agent from {checkpoint}")
    # file = '/home/wahabu/skrl/runs/torch/MachineTending/SMAPPO/MorePartsUncoll.05ResetOnCollisionRev16_2/checkpoints/best_agent_r0p.pth'
    # torch.save(agent.models['robot_0'], file)

if not evaluate:
    cfg_trainer = {"timesteps": 60000000, "headless": True} #36000
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # start training
    trainer.train()

if evaluate:
    # set the agent to evaluation
    print("Starting evaluation...")
    cfg_trainer = {"timesteps": 10000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # start evaluation
    trainer.eval()
