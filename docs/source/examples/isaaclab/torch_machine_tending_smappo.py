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

        self.num_agents = 2
        self.num_machines = 2
        self.num_storage_areas = 1
        self.agent_feature_size = 7 # pos, lin_vel, ang_vel, has_part
        self.machine_feature_size = 3 # pos, collected
        self.storage_feature_size = 2 # pos
        self.atten_embed_dim = 16
        self.attention_heads = 2

        self.agent_enc = nn.Linear(self.agent_feature_size, self.atten_embed_dim)
        self.machines_enc = nn.Linear(self.machine_feature_size, self.atten_embed_dim)
        self.storages_enc = nn.Linear(self.storage_feature_size, self.atten_embed_dim)
        self.other_agents_enc = nn.Linear(self.agent_feature_size, self.atten_embed_dim)

        self.machines_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.storages_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.agents_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)

        size = self.atten_embed_dim*4
        if not self.encode:
            size = self.num_observations

        self.net = nn.Sequential(nn.Linear(size, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def encode_objects(self, obs):
        B = obs.shape[0]
        idx = 0
        sizes = [
            (self.agent_feature_size, 1),  # agent
            (self.machine_feature_size, self.num_machines),  # machines
            (self.storage_feature_size, self.num_storage_areas),  # storages
            (self.agent_feature_size, self.num_agents - 1),  # other agents
        ]

        # Unpack efficiently
        chunks = []
        for feature_size, num in sizes:
            total = feature_size * num
            chunks.append(obs[:, idx:idx+total].reshape(B, num, feature_size))
            idx += total

        agent_info, machines_info, storages_info, other_agents_info = chunks

        agent_info = F.tanh(self.agent_enc(agent_info))
        machines_info = F.tanh(self.machines_enc(machines_info))
        storages_info = F.tanh(self.storages_enc(storages_info))
        other_agents_info = F.tanh(self.other_agents_enc(other_agents_info))

        machines_info,_ = self.machines_attention(agent_info, machines_info, machines_info, need_weights=False)
        storages_info,_ = self.storages_attention(agent_info, storages_info, storages_info, need_weights=False)
        other_agents_info,_ = self.agents_attention(agent_info, other_agents_info, other_agents_info, need_weights=False)
            
        env_info = torch.cat([agent_info, machines_info, storages_info, other_agents_info], axis =1).reshape(obs.shape[0], -1)
        
        return env_info

    def compute(self, inputs, role):
        obs = inputs["states"]
        if self.encode:
            obs = self.encode_objects(inputs["states"])
        return self.net(obs), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.encode = True

        self.num_agents = 2
        self.num_machines = 2
        self.num_storage_areas = 1
        self.agent_feature_size = 7 # pos, lin_vel, ang_vel, has_part
        self.machine_feature_size = 3 # pos, collected
        self.storage_feature_size = 2 # pos
        self.atten_embed_dim = 18
        self.attention_heads = 3

        self.agent_enc = nn.Linear(self.agent_feature_size, self.atten_embed_dim)
        self.machines_enc = nn.Linear(self.machine_feature_size, self.atten_embed_dim)
        self.storages_enc = nn.Linear(self.storage_feature_size, self.atten_embed_dim)
        self.other_agents_enc = nn.Linear(self.agent_feature_size, self.atten_embed_dim)

        self.machines_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.storages_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.agents_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)

        self.env_attention = nn.MultiheadAttention(self.atten_embed_dim*4,self.attention_heads, batch_first=True)
        self.env_query = nn.Parameter(torch.randn(1, 1, self.atten_embed_dim*4))

        size = self.atten_embed_dim*4
        if not self.encode:
            size = self.num_observations

        self.net = nn.Sequential(nn.Linear(size, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def encode_objects(self, obs):
        obs_shape = obs.shape

        obs = obs.reshape(-1, self.num_agents* self.agent_feature_size + self.num_machines*self.machine_feature_size + self.num_storage_areas*self.storage_feature_size) # (number of agents, agent observation size)
               
        B = obs_shape[0]
        idx = 0
        sizes = [
            (self.agent_feature_size, 1),  # agent
            (self.machine_feature_size, self.num_machines),  # machines
            (self.storage_feature_size, self.num_storage_areas),  # storages
            (self.agent_feature_size, self.num_agents - 1),  # other agents
        ]

        # Unpack efficiently
        chunks = []
        for feature_size, num in sizes:
            total = feature_size * num
            chunks.append(obs[:, idx:idx+total].reshape(-1, num, feature_size))
            idx += total

        agent_info, machines_info, storages_info, other_agents_info = chunks
        

        agent_info = F.tanh(self.agent_enc(agent_info))
        machines_info = F.tanh(self.machines_enc(machines_info))
        storages_info = F.tanh(self.storages_enc(storages_info))
        other_agents_info = F.tanh(self.other_agents_enc(other_agents_info))
        
        machines_info,_ = self.machines_attention(agent_info, machines_info, machines_info, need_weights=False)
        storages_info,_ = self.storages_attention(agent_info, storages_info, storages_info, need_weights=False)
        other_agents_info,_ = self.agents_attention(agent_info, other_agents_info, other_agents_info, need_weights=False)

        agents_obs = torch.cat([agent_info, machines_info, storages_info, other_agents_info], axis =1).reshape(obs_shape[0],self.num_agents, -1)
        query = self.env_query.repeat(obs_shape[0],1,1)  

        agents_obs,_ = self.env_attention(query, agents_obs, agents_obs, need_weights=False)

        
        agents_obs = agents_obs.reshape(agents_obs.shape[0],-1)
        return agents_obs

    def compute(self, inputs, role):
        obs = inputs["states"]
        if self.encode:
            obs = self.encode_objects(inputs["states"])
        return self.net(obs), {}


# load and wrap the environment
env = load_isaaclab_env(task_name="Isaac-Machine-Tending-Direct-v0")
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
cfg["experiment"]["directory"] = "runs/torch/MachineTending/SMAPPO"
cfg["experiment"]["experiment_name"] = "Z_T15_Dist20_Uncoll0_LongPlsenvS42"

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
checkpoint = '/home/wahabu/skrl/runs/torch/MachineTending/SMAPPO/Z_T15_Dist20_Uncoll0/checkpoints/best_agent.pt'  

if evaluate and checkpoint:
    agent.load(checkpoint)
    print(f"Loaded agent from {checkpoint}")

if not evaluate:
    cfg_trainer = {"timesteps": 2000000, "headless": True} #36000
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
