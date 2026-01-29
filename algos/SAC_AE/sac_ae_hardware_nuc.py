import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import intel_extension_for_pytorch as ipex
from algos.SAC_AE.encoder import make_encoder

N = 200

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, action_range
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        
        # action rescaling
        self.action_scale = torch.FloatTensor(
            (action_range[1] - action_range[0]) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_range[1] + action_range[0]) / 2.)
        
        self.features = 100.0 * np.ones((N, encoder_feature_dim), dtype=np.float32)
        self.count = 0

        self.apply(weight_init)

    def forward(
        self, obs, robot_goal_state, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        # self.features[self.count % N, :] = obs.cpu().data.numpy().flatten()
        # self.count += 1
        
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        std = log_std.exp()

        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        pi = y_t * self.action_scale + self.action_bias
        log_pi = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_pi -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_pi = log_pi.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return mu, pi, log_pi, log_std
    


class SacAeAgentHardware(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        robot_goal_state_dim, 
        action_shape,
        action_range, 
        hidden_dim=256,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        num_layers=4,
        num_filters=32
    ):

        self.actor = Actor(
            obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, action_range
        )

    def select_action(self, obs, robot_goal_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            obs = obs.unsqueeze(0)
            robot_goal_state = torch.FloatTensor(robot_goal_state)
            robot_goal_state = robot_goal_state.unsqueeze(0)
            mu, _1, _2, _3 = self.actor(obs, robot_goal_state)
            return mu.data.numpy().flatten()

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=torch.device('cpu')))
        self.actor.eval()
        # Invoke optimize function against the model object
        self.actor = ipex.optimize(self.actor, dtype=torch.float32)
            

