import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
import math

from algos.SAC_AE.utils import preprocess_obs  # soft_update_params not needed for PPO
from algos.SAC_AE.encoder import make_encoder
from algos.SAC_AE.decoder import make_decoder

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

        # Encoder to process the observation
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # MLP trunk to process the concatenated encoder output and robot goal state
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        
        # Action rescaling
        self.action_scale = torch.FloatTensor(
            (action_range[1] - action_range[0]) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_range[1] + action_range[0]) / 2.)
        
        self.features = 100.0 * np.ones((N, encoder_feature_dim), dtype=np.float32)
        self.count = 0

        self.apply(weight_init)

    def forward(self, obs, robot_goal__emotion_state, detach_encoder=False):
        # Encode the observation
        obs = self.encoder(obs, detach=detach_encoder)
        # Concatenate the encoded observation with the robot goal state
        obs = torch.cat((obs, robot_goal__emotion_state), dim=-1)
        # Pass through the MLP trunk to get mean and log standard deviation
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = log_std.exp()

        # Sample actions using the reparameterization trick
        normal = Normal(mu, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        pi = y_t * self.action_scale + self.action_bias
        log_pi = normal.log_prob(x_t)
        # Enforce action bounds
        log_pi -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_pi = log_pi.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return mu, pi, log_pi, log_std

    def get_log_prob(self, obs, robot_goal__emotion_state, actions):
        """
        Compute log probability for given actions.
        This function recomputes the distribution parameters from the observation
        and then computes the log probability of the provided actions.
        """
        obs_encoded = self.encoder(obs)
        obs_cat = torch.cat((obs_encoded, robot_goal__emotion_state), dim=-1)
        mu, log_std = self.trunk(obs_cat).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        std = log_std.exp()
        normal = Normal(mu, std)
        # Recover pre-tanh value from actions:
        # actions = tanh(x) * action_scale + action_bias  ==> tanh(x) = (actions - action_bias)/action_scale
        actions_norm = (actions - self.action_bias.to(actions.device)) / self.action_scale.to(actions.device)
        eps = 1e-6
        actions_norm = torch.clamp(actions_norm, -1 + eps, 1 - eps)
        pre_tanh = 0.5 * torch.log((1 + actions_norm) / (1 - actions_norm))
        log_prob = normal.log_prob(pre_tanh) - torch.log(self.action_scale.to(actions.device) * (1 - torch.tanh(pre_tanh).pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

class ValueNetwork(nn.Module):
    """State-value network for PPO."""
    def __init__(self, obs_shape, robot_goal_state_dim, hidden_dim, encoder_type, encoder_feature_dim, num_layers, num_filters):
        super().__init__()
        self.encoder = make_encoder(encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)
        
    def forward(self, obs, robot_goal__emotion_state, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal__emotion_state), dim=-1)
        value = self.trunk(obs)
        return value

class PPOAeAgent(object):
    """PPO + Autoencoder algorithm."""
    def __init__(
        self,
        obs_shape,
        robot_goal_state_dim, 
        action_shape,
        action_range, 
        device,
        hidden_dim=256,
        clip_epsilon=0.2,
        ppo_epochs=10,
        minibatch_size=64,
        encoder_type='pixel',
        encoder_feature_dim=50,
        num_layers=4,
        num_filters=32,
        decoder_type='pixel',
        encoder_lr=1e-3,
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0
    ):
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        # Actor initialization remains similar
        self.actor = Actor(
            obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, log_std_min=-10, log_std_max=2,
            num_layers=num_layers, num_filters=num_filters, action_range=action_range
        ).to(device)
        
        # New value network for PPO (replaces SAC's critic)
        self.value_net = ValueNetwork(
            obs_shape, robot_goal_state_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)
        
        # Optionally, tie encoder weights between actor and value network
        self.actor.encoder.copy_conv_weights_from(self.value_net.encoder)
        
        # Autoencoder (decoder) remains unchanged
        self.decoder = None
        if (decoder_type != 'identity'):
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)
            # Optimizer for value network's encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.value_net.encoder.parameters(), lr=encoder_lr
            )
            # Optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )
        
        # Optimizers for actor and value network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        
        self.train()
    
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.value_net.train(training)
        if self.decoder is not None:
            self.decoder.train(training)
    
    @property
    def discount(self):
        # Discount factor used externally for computing returns.
        return 0.99
    
    def select_action(self, obs, robot_goal__emotion_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            robot_goal__emotion_state = torch.FloatTensor(robot_goal__emotion_state).to(self.device).unsqueeze(0)
            mu, _ , _ , _ = self.actor(obs, robot_goal__emotion_state)
            return mu.cpu().data.numpy().flatten()
    
    def sample_action(self, obs, robot_goal__emotion_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            robot_goal__emotion_state = torch.FloatTensor(robot_goal__emotion_state).to(self.device).unsqueeze(0)
            _ , pi, _ , _ = self.actor(obs, robot_goal__emotion_state)
            return pi.cpu().data.numpy().flatten()
    
    def update(self, trajectories, writer, step, chaser_id=None):
        """
        Update actor and value network using PPO.
        trajectories: a dict with keys 'obs', 'robot_goal_state', 'actions',
                      'old_log_probs', 'advantages', 'returns'
        """
        #covert trajectories to np arrays first
        trajectories['obs'] = np.array(trajectories['obs'])
        trajectories['robot_goal_state'] = np.array(trajectories['robot_goal_state'])
        trajectories['actions'] = np.array(trajectories['actions'])
        trajectories['old_log_probs'] = np.array(trajectories['old_log_probs'])
        trajectories['advantages'] = np.array(trajectories['advantages'])
        trajectories['returns'] = np.array(trajectories['returns'])
        
        obs = torch.FloatTensor(trajectories['obs']).to(self.device)
        robot_goal_state = torch.FloatTensor(trajectories['robot_goal_state']).to(self.device)
        actions = torch.FloatTensor(trajectories['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectories['old_log_probs']).to(self.device)
        advantages = torch.FloatTensor(trajectories['advantages']).to(self.device)
        returns = torch.FloatTensor(trajectories['returns']).to(self.device)
        
        dataset_size = obs.size(0)
        
        for epoch in range(self.ppo_epochs):
            # Shuffle indices for minibatch sampling
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]
                
                mb_obs = obs[mb_idx]
                mb_robot_goal_state = robot_goal_state[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                
                # Compute new log probabilities for the minibatch actions
                new_log_probs = self.actor.get_log_prob(mb_obs, mb_robot_goal_state, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Value network update (critic update)
                values = self.value_net(mb_obs, mb_robot_goal_state).squeeze(-1)
                value_loss = F.mse_loss(values, mb_returns)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                if chaser_id is None:
                    writer.add_scalar('train_actor/loss', actor_loss.item(), step)
                    writer.add_scalar('train_value/loss', value_loss.item(), step)
                else:
                    writer.add_scalar(f'train_chaser_{chaser_id}_actor/loss', actor_loss.item(), step)
                    writer.add_scalar(f'train_chaser_{chaser_id}_value/loss', value_loss.item(), step)
    
    def update_decoder(self, obs, target_obs, writer, step, chaser_id=None):
        h = self.value_net.encoder(obs)
        if target_obs.dim() == 4:
            target_obs = preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()
        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        if chaser_id is None:
            writer.add_scalar('train_ae/loss', loss.item(), step)
        else:
            writer.add_scalar(f'train_chaser_{chaser_id}_ae/loss', loss.item(), step)
    
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.value_net.state_dict(), filename + '_value_net')
        if self.decoder is not None:
            torch.save(self.decoder.state_dict(), filename + '_decoder')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        torch.save(self.value_optimizer.state_dict(), filename + '_value_optimizer')
        if self.decoder is not None:
            torch.save(self.decoder_optimizer.state_dict(), filename + '_decoder_optimizer')
    
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=self.device))
        self.value_net.load_state_dict(torch.load(filename + '_value_net', map_location=self.device))
        if self.decoder is not None:
            self.decoder.load_state_dict(torch.load(filename + '_decoder', map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer', map_location=self.device))
        self.value_optimizer.load_state_dict(torch.load(filename + '_value_optimizer', map_location=self.device))
        if self.decoder is not None:
            self.decoder_optimizer.load_state_dict(torch.load(filename + '_decoder_optimizer', map_location=self.device))
    
    def load_parameters(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=self.device))
        self.value_net.load_state_dict(torch.load(filename + '_value_net', map_location=self.device))
        if self.decoder is not None:
            self.decoder.load_state_dict(torch.load(filename + '_decoder', map_location=self.device))
    
    def get_state_dict(self):
        state_dict = {
            'actor': self.actor.state_dict(),
            'value_net': self.value_net.state_dict(),
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.value_net.load_state_dict(state_dict['value_net'])
        self.actor = self.actor.to(self.device)
        self.value_net = self.value_net.to(self.device)
        if self.decoder is not None:
            self.decoder = self.decoder.to(self.device)
