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
        self.num_machines = 2
        self.num_storage_areas = 1
        self.agent_feature_size = 6 # pos, orientation 4 (quat), lin_vel, has_part # no angular vel
        self.lidar_feature_size = 10 # 5 rays  2 lidars
        self.lidar_embed_dim = 8 
        self.machine_feature_size = 3 # pos, collected
        self.storage_feature_size = 2 # pos
        self.atten_embed_dim = 16
        self.attention_heads = 2

        self.lidar_enc = nn.Sequential(nn.LayerNorm(self.lidar_feature_size), nn.Linear(self.lidar_feature_size, self.lidar_embed_dim), nn.Tanh(), nn.LayerNorm(self.lidar_embed_dim))
        self.agent_enc = nn.Sequential(nn.Linear(self.agent_feature_size + self.lidar_embed_dim, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.machines_enc = nn.Sequential(nn.Linear(self.machine_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.storages_enc = nn.Sequential(nn.Linear(self.storage_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.other_agents_enc = nn.Sequential(nn.Linear(self.agent_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))

        self.machines_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.storages_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.agents_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        
        # LayerNorm after attention
        self.machines_attn_norm = nn.LayerNorm(self.atten_embed_dim)
        self.storages_attn_norm = nn.LayerNorm(self.atten_embed_dim)
        self.agents_attn_norm = nn.LayerNorm(self.atten_embed_dim)

        size = self.atten_embed_dim*4
        if not self.encode:
            size = self.num_observations

        self.net = nn.Sequential(nn.Linear(size, 512),
                                 nn.Tanh(),
                                 nn.LayerNorm(512),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.LayerNorm(256),
                                 nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.LayerNorm(128),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def check_all_weights(self):
        """Check all model weights for NaN/inf and print stats"""
        has_bad_weights = False
        layers = [
            ("lidar_enc", self.lidar_enc),
            ("agent_enc", self.agent_enc),
            ("machines_enc", self.machines_enc),
            ("storages_enc", self.storages_enc),
            ("other_agents_enc", self.other_agents_enc),
            ("machines_attention", self.machines_attention),
            ("storages_attention", self.storages_attention),
            ("agents_attention", self.agents_attention),
            ("net", self.net),
        ]
        for layer_name, layer in layers:
            for name, param in layer.named_parameters():
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                if has_nan or has_inf:
                    print(f"BAD WEIGHTS in {layer_name}.{name}: has_nan={has_nan}, has_inf={has_inf}")
                    has_bad_weights = True
                else:
                    print(f"{layer_name}.{name}: min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")
        # Also check log_std_parameter
        if torch.isnan(self.log_std_parameter).any() or torch.isinf(self.log_std_parameter).any():
            print(f"BAD WEIGHTS in log_std_parameter")
            has_bad_weights = True
        return has_bad_weights

    def encode_objects(self, obs):
        B = obs.shape[0]
        idx = 0
        sizes = [
            (self.agent_feature_size, 1),  # agent
            (self.lidar_feature_size, 1),  # lidar
            (self.machine_feature_size, self.num_machines),  # machines
            (self.storage_feature_size, self.num_storage_areas),  # storages
            (self.agent_feature_size, self.num_agents - 1),  # other agents
        ]

        if self.debug:
            if torch.any(torch.isnan(obs)) or torch.any(torch.isinf(obs)):
                print("encode_objects NaN or inf detected in obs")
                print( "obs", obs.shape)
                print("obs", obs)
                print("sizes", sizes)
                quit()

        chunks = []
        for feature_size, num in sizes:
            total = feature_size * num
            chunks.append(obs[:, idx:idx+total].reshape(B, num, feature_size))
            idx += total

        agent_info, lidar_info, machines_info, storages_info, other_agents_info = chunks
        
        should_quit = False
        if self.debug:
            for i, chunk in enumerate(chunks):
                if torch.any(torch.isnan(chunk)) or torch.any(torch.isinf(chunk)):
                    print("encode_objects NaN or inf detected in chunk")
                    print("chunk", chunk.shape, "index", i)
                    print("chunk", chunk)
                    should_quit = True

        lidar_info = self.lidar_enc(lidar_info)
        
        if self.debug:
            if torch.any(torch.isnan(lidar_info)) or torch.any(torch.isinf(lidar_info)):
                print("encode_objects NaN or inf detected in lidar_info first")
                print("lidar_info", lidar_info.shape)
                print("lidar_info", lidar_info)
                for name, param in self.lidar_enc.named_parameters():
                    print(f"lidar_enc {name}: min={param.min().item():.4f}, max={param.max().item():.4f}, has_nan={torch.isnan(param).any()}")
                should_quit = True

        agent_info = torch.cat([agent_info, lidar_info], dim=-1)
        if self.debug:
            if torch.any(torch.isnan(agent_info)) or torch.any(torch.isinf(agent_info)):
                print("encode_objects NaN or inf detected in agent_info after concatenation")
                print("agent_info", agent_info.shape)
                print("agent_info", agent_info)
                should_quit = True

        agent_info = self.agent_enc(agent_info)
        if self.debug:
            if torch.any(torch.isnan(agent_info)) or torch.any(torch.isinf(agent_info)):
                print("encode_objects NaN or inf detected in agent_info after encoding")
                print("agent_info", agent_info.shape)
                print("agent_info", agent_info)
                should_quit = True

        machines_info = self.machines_enc(machines_info)
        if self.debug:
            if torch.any(torch.isnan(machines_info)) or torch.any(torch.isinf(machines_info)):
                print("encode_objects NaN or inf detected in machines_info after encoding")
                print("machines_info", machines_info.shape)
                print("machines_info", machines_info)
                should_quit = True

        storages_info = self.storages_enc(storages_info)
        if self.debug:
            if torch.any(torch.isnan(storages_info)) or torch.any(torch.isinf(storages_info)):
                print("encode_objects NaN or inf detected in storages_info after encoding")
                print("storages_info", storages_info.shape)
                print("storages_info", storages_info)
                should_quit = True
        
        other_agents_info = self.other_agents_enc(other_agents_info)
        
        if self.debug:
            if torch.any(torch.isnan(other_agents_info)) or torch.any(torch.isinf(other_agents_info)):
                print("encode_objects NaN or inf detected in other_agents_info after encoding")
                print("other_agents_info", other_agents_info.shape)
                print("other_agents_info", other_agents_info)
                should_quit = True

        machines_info,_ = self.machines_attention(agent_info, machines_info, machines_info, need_weights=False)
        machines_info = self.machines_attn_norm(machines_info)
        
        if self.debug:
            if torch.any(torch.isnan(machines_info)) or torch.any(torch.isinf(machines_info)):
                print("NaN/inf after machines_attention in Policy")
                print("machines_info shape:", machines_info.shape)
                # Check attention weights for NaN
                for name, param in self.machines_attention.named_parameters():
                    if torch.any(torch.isnan(param)) or torch.any(torch.isinf(param)):
                        print(f"NaN/inf in machines_attention param: {name}")
                should_quit = True
            
        storages_info,_ = self.storages_attention(agent_info, storages_info, storages_info, need_weights=False)
        storages_info = self.storages_attn_norm(storages_info)
        
        if self.debug:
            if torch.any(torch.isnan(storages_info)) or torch.any(torch.isinf(storages_info)):
                print("NaN/inf after storages_attention in Policy")
                should_quit = True
            
        other_agents_info,_ = self.agents_attention(agent_info, other_agents_info, other_agents_info, need_weights=False)
        other_agents_info = self.agents_attn_norm(other_agents_info)
        if self.debug:
            if torch.any(torch.isnan(other_agents_info)) or torch.any(torch.isinf(other_agents_info)):
                print("NaN/inf after agents_attention in Policy")
                should_quit = True

        if should_quit:
            print("\n=== CHECKING ALL MODEL WEIGHTS ===")
            self.check_all_weights()
            quit()   
            
        env_info = torch.cat([agent_info, machines_info, storages_info, other_agents_info], axis =1).reshape(obs.shape[0], -1)
        
        return env_info

    def compute(self, inputs, role):
        obs = inputs["states"]
        if self.encode:
            obs = self.encode_objects(inputs["states"])

        means = self.net(obs)
        stds = self.log_std_parameter
        
        if self.debug:
            if torch.any(torch.isnan(inputs["states"])) or torch.any(torch.isinf(inputs["states"])):
                print("NaN or inf detected in states")
                print("states", inputs["states"])
                quit()
        if self.debug:
            if torch.any(torch.isnan(obs)) or torch.any(torch.isinf(obs)):
                print("NaN or inf detected in obs")
                print("inputs[\"states\"]", inputs["states"].shape, "obs", obs.shape)
                print("obs", obs)
                quit()
        if self.debug:
            if torch.any(torch.isnan(means)) or torch.any(torch.isinf(means)):
                print("NaN or inf detected in means")
                print("means", means)
                quit()
        return means, stds, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.encode = True
        self.debug = False

        self.num_agents = 2
        self.num_machines = 2
        self.num_storage_areas = 1
        self.agent_feature_size = 6 # pos, orientation 4 (quat), lin_vel, has_part # no angular vel
        self.lidar_feature_size = 10 # 5 rays  2 lidars
        self.lidar_embed_dim = 8 
        self.machine_feature_size = 3 # pos, collected
        self.storage_feature_size = 2 # pos
        self.atten_embed_dim = 18
        self.attention_heads = 3

        self.lidar_enc = nn.Sequential(nn.LayerNorm(self.lidar_feature_size),nn.Linear(self.lidar_feature_size, self.lidar_embed_dim), nn.Tanh(), nn.LayerNorm(self.lidar_embed_dim))
        self.agent_enc = nn.Sequential(nn.Linear(self.agent_feature_size + self.lidar_embed_dim, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.machines_enc = nn.Sequential(nn.Linear(self.machine_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.storages_enc = nn.Sequential(nn.Linear(self.storage_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))
        self.other_agents_enc = nn.Sequential(nn.Linear(self.agent_feature_size, self.atten_embed_dim), nn.Tanh(), nn.LayerNorm(self.atten_embed_dim))

        self.machines_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.storages_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        self.agents_attention = nn.MultiheadAttention(self.atten_embed_dim, self.attention_heads, batch_first=True)
        
        # LayerNorm after attention
        self.machines_attn_norm = nn.LayerNorm(self.atten_embed_dim)
        self.storages_attn_norm = nn.LayerNorm(self.atten_embed_dim)
        self.agents_attn_norm = nn.LayerNorm(self.atten_embed_dim)

        self.env_attention = nn.MultiheadAttention(self.atten_embed_dim*4,self.attention_heads, batch_first=True)
        self.env_attn_norm = nn.LayerNorm(self.atten_embed_dim*4)
        self.env_query = nn.Parameter(torch.randn(1, 1, self.atten_embed_dim*4))

        size = self.atten_embed_dim*4
        if not self.encode:
            size = self.num_observations

        self.net = nn.Sequential(nn.Linear(size, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 1))

    def check_all_weights(self):
        """Check all model weights for NaN/inf and print stats"""
        has_bad_weights = False
        layers = [
            ("lidar_enc", self.lidar_enc),
            ("agent_enc", self.agent_enc),
            ("machines_enc", self.machines_enc),
            ("storages_enc", self.storages_enc),
            ("other_agents_enc", self.other_agents_enc),
            ("machines_attention", self.machines_attention),
            ("storages_attention", self.storages_attention),
            ("agents_attention", self.agents_attention),
            ("env_attention", self.env_attention),
            ("net", self.net),
        ]
        for layer_name, layer in layers:
            for name, param in layer.named_parameters():
                has_nan = torch.isnan(param).any().item()
                has_inf = torch.isinf(param).any().item()
                if has_nan or has_inf:
                    print(f"BAD WEIGHTS in Value.{layer_name}.{name}: has_nan={has_nan}, has_inf={has_inf}")
                    has_bad_weights = True
                else:
                    print(f"Value.{layer_name}.{name}: min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")
        # Also check env_query
        if torch.isnan(self.env_query).any() or torch.isinf(self.env_query).any():
            print(f"BAD WEIGHTS in Value.env_query")
            has_bad_weights = True
        return has_bad_weights

    def encode_objects(self, obs):
        obs_shape = obs.shape

        obs = obs.reshape(-1, self.agent_feature_size + self.lidar_feature_size + (self.num_agents-1)* self.agent_feature_size + self.num_machines*self.machine_feature_size + self.num_storage_areas*self.storage_feature_size) 
               
        B = obs_shape[0]
        idx = 0
        sizes = [
            (self.agent_feature_size, 1),  # agent
            (self.lidar_feature_size, 1),  # lidar
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

        agent_info, lidar_info, machines_info, storages_info, other_agents_info = chunks

        should_quit = False
        if self.debug:
            if torch.any(torch.isnan(obs)) or torch.any(torch.isinf(obs)):
                print("Value encode_objects NaN or inf detected in obs")
                should_quit = True

        lidar_info = self.lidar_enc(lidar_info)
        if self.debug:
            if torch.any(torch.isnan(lidar_info)) or torch.any(torch.isinf(lidar_info)):
                print("Value NaN/inf after lidar_enc")
                should_quit = True

        agent_info = torch.cat([agent_info, lidar_info], dim=-1)
        agent_info = self.agent_enc(agent_info)
        if self.debug:
            if torch.any(torch.isnan(agent_info)) or torch.any(torch.isinf(agent_info)):
                print("Value NaN/inf after agent_enc")
                should_quit = True

        machines_info = self.machines_enc(machines_info)
        if self.debug:
            if torch.any(torch.isnan(machines_info)) or torch.any(torch.isinf(machines_info)):
                print("Value NaN/inf after machines_enc")
                should_quit = True

        storages_info = self.storages_enc(storages_info)
        if self.debug:
            if torch.any(torch.isnan(storages_info)) or torch.any(torch.isinf(storages_info)):
                print("Value NaN/inf after storages_enc")
                should_quit = True

        other_agents_info = self.other_agents_enc(other_agents_info)
        if self.debug:
            if torch.any(torch.isnan(other_agents_info)) or torch.any(torch.isinf(other_agents_info)):
                print("Value NaN/inf after other_agents_enc")
                should_quit = True

        machines_info,_ = self.machines_attention(agent_info, machines_info, machines_info, need_weights=False)
        machines_info = self.machines_attn_norm(machines_info)
        if self.debug:
            if torch.any(torch.isnan(machines_info)) or torch.any(torch.isinf(machines_info)):
                print("Value NaN/inf after machines_attention")
                should_quit = True

        storages_info,_ = self.storages_attention(agent_info, storages_info, storages_info, need_weights=False)
        storages_info = self.storages_attn_norm(storages_info)
        if self.debug:
            if torch.any(torch.isnan(storages_info)) or torch.any(torch.isinf(storages_info)):
                print("Value NaN/inf after storages_attention")
                should_quit = True

        other_agents_info,_ = self.agents_attention(agent_info, other_agents_info, other_agents_info, need_weights=False)
        other_agents_info = self.agents_attn_norm(other_agents_info)
        if self.debug:
            if torch.any(torch.isnan(other_agents_info)) or torch.any(torch.isinf(other_agents_info)):
                print("Value NaN/inf after agents_attention")
                should_quit = True

        agents_obs = torch.cat([agent_info, machines_info, storages_info, other_agents_info], axis =1).reshape(obs_shape[0],self.num_agents, -1)
        query = self.env_query.repeat(obs_shape[0],1,1)  

        agents_obs,_ = self.env_attention(query, agents_obs, agents_obs, need_weights=False)
        agents_obs = self.env_attn_norm(agents_obs)
        if self.debug:
            if torch.any(torch.isnan(agents_obs)) or torch.any(torch.isinf(agents_obs)):
                print("Value NaN/inf after env_attention")
                should_quit = True

        if should_quit:
            print("\n=== CHECKING ALL VALUE MODEL WEIGHTS ===")
            self.check_all_weights()
            quit()

        agents_obs = agents_obs.reshape(agents_obs.shape[0],-1)
        return agents_obs

    def compute(self, inputs, role):
        obs = inputs["states"]
        if self.encode:
            obs = self.encode_objects(inputs["states"])
        output = self.net(obs)
        return output, {}


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
cfg["experiment"]["experiment_name"] = "NoResetOnColl16_nAngVel_Arena3_orntZ_RRNois2Lidar5Col5Cur6_1_5"

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
checkpoint = '/home/wahabu/skrl/runs/torch/MachineTending/SMAPPO/FNoResetOnColl16_nAngVel_RandArena27_orntZ/checkpoints/best_agent.pt'

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
