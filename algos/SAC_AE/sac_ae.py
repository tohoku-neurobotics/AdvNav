import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
import math
from torch.utils.data import Dataset, DataLoader
from algos.SAC_AE.utils import preprocess_obs, soft_update_params
from algos.SAC_AE.encoder import make_encoder
from algos.SAC_AE.decoder import make_decoder

N = 200

class ReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        sample = self.replay_buffer[idx]
        obs, robot_goal_emotion_state, action, reward, next_obs, next_robot_goal_emotion_state, not_done = sample
        
        return {
            'obs': obs,
            'robot_goal_emotion_state': robot_goal_emotion_state,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'next_robot_goal_emotion_state': next_robot_goal_emotion_state,
            'not_done': not_done
        }

# Instantiate the DataLoader
def create_data_loader(replay_buffer, batch_size=256, num_workers=4):
    dataset = ReplayBufferDataset(replay_buffer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return data_loader

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

    def forward(
        self, obs, robot_goal__emotion_state, detach_encoder=False
    ):
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
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        pi = y_t * self.action_scale + self.action_bias
        log_pi = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_pi -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_pi = log_pi.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return mu, pi, log_pi, log_std
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, robot_goal_state_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + robot_goal_state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, robot_goal_state_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, robot_goal_state_dim, action_shape[0], hidden_dim
        )

        self.apply(weight_init)

    def forward(self, obs, robot_goal__emotion_state, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)

        obs = torch.cat((obs, robot_goal__emotion_state), dim=-1)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2

class SacAeAgent(object):
    """SAC+AE algorithm."""
    def __init__(
        self,
        obs_shape,
        robot_goal_state_dim, 
        action_shape,
        action_range, 
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        fixed_elo=False
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.elo_rating = 1000.0
        self.fixed_elo = fixed_elo
        self.actor = Actor(
            obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, action_range
        ).to(device)

        self.critic = Critic(
            obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if (decoder_type != 'identity'):
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def save_features(self, directory):
        file_name = directory + '/features.txt'
        np.savetxt(file_name, self.actor.features)
        self.actor.count = 0

    def select_action(self, obs, robot_goal__emotion_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            robot_goal__emotion_state = torch.FloatTensor(robot_goal__emotion_state).to(self.device)
            robot_goal__emotion_state = robot_goal__emotion_state.unsqueeze(0)
            mu, _1, _2, _3 = self.actor(obs, robot_goal__emotion_state)
            return mu.cpu().data.numpy().flatten()
        
    def output_distribution(self, obs, robot_goal__emotion_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            robot_goal__emotion_state = torch.FloatTensor(robot_goal__emotion_state).to(self.device)
            robot_goal__emotion_state = robot_goal__emotion_state.unsqueeze(0)
            mu, _1, _2, log_std = self.actor(obs, robot_goal__emotion_state)
            return mu.cpu().data.numpy().flatten(), log_std.cpu().data.numpy().flatten()

    def sample_action(self, obs, robot_goal__emotion_state):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            robot_goal__emotion_state = torch.FloatTensor(robot_goal__emotion_state).to(self.device)
            robot_goal__emotion_state = robot_goal__emotion_state.unsqueeze(0)
            _1, pi, _2, _3 = self.actor(obs, robot_goal__emotion_state)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, robot_goal__emotion_state, action, reward, next_obs, next_robot_goal_emotion_state, not_done, writer, step, chaser_id=None):
        with torch.no_grad():
            _1, policy_action, log_pi, _2 = self.actor(next_obs, next_robot_goal_emotion_state)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_robot_goal_emotion_state, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, robot_goal__emotion_state, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if chaser_id == None:
            writer.add_scalar('train_critic/loss', critic_loss, step)
        else:
            writer.add_scalar('train_chaser_{}_critic/loss'.format(chaser_id), critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, robot_goal__emotion_state, writer, step, chaser_id=None):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, robot_goal__emotion_state, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, robot_goal__emotion_state, pi, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()


        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()

        if chaser_id == None:
            writer.add_scalar('train_actor/loss', actor_loss, step)
            writer.add_scalar('train_actor/target_entropy', self.target_entropy, step)
            writer.add_scalar('train_actor/entropy', entropy.mean(), step)
            writer.add_scalar('train_alpha/loss', alpha_loss, step)
            writer.add_scalar('train_alpha/value', self.alpha, step)
        else: 
            writer.add_scalar('train_chaser_{}_actor/loss'.format(chaser_id), actor_loss, step)
            writer.add_scalar('train_chaser_{}_actor/target_entropy'.format(chaser_id), self.target_entropy, step)
            writer.add_scalar('train_chaser_{}_actor/entropy'.format(chaser_id), entropy.mean(), step)
            writer.add_scalar('train_chaser_{}_alpha/loss'.format(chaser_id), alpha_loss, step)
            writer.add_scalar('train_chaser_{}_alpha/value'.format(chaser_id), self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, target_obs, writer, step, chaser_id=None):
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        if chaser_id == None:
            writer.add_scalar('train_ae/loss', loss, step)
        else:
            writer.add_scalar('train_chaser_{}_ae/loss'.format(chaser_id), loss, step)

    def update(self, replay_buffer, writer, step,chaser_id=None):
        obs, robot_goal__emotion_state, action, reward, next_obs, next_robot_goal_emotion_state, not_done = replay_buffer.sample()
        if chaser_id == None:
            writer.add_scalar('train/batch_reward', reward.mean(), step)
        else: 
            writer.add_scalar('train/chaser_{}_batch_reward'.format(chaser_id), reward.mean(), step)

        self.update_critic(obs, robot_goal__emotion_state, action, reward, next_obs, next_robot_goal_emotion_state, not_done, writer, step, chaser_id)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, robot_goal__emotion_state, writer, step, chaser_id)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, writer, step, chaser_id)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_target.state_dict(), filename + '_critic_target')
        if self.decoder is not None:
            torch.save(self.decoder.state_dict(), filename + '_decoder')
            
        torch.save(self.actor_optimizer.state_dict(),
                   filename + '_actor_optimizer')
        torch.save(self.critic_optimizer.state_dict(),
                   filename + '_critic_optimizer')
        if self.decoder is not None:
            torch.save(self.decoder_optimizer.state_dict(),
                   filename + '_decoder_optimizer')

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + '_critic', map_location=self.device))
        self.critic_target.load_state_dict(torch.load(filename + '_critic_target', map_location=self.device))
        if self.decoder is not None:
            self.decoder.load_state_dict(torch.load(filename + '_decoder', map_location=self.device))
            
        # Move optimizer states to the correct device too
        self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer', map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + '_critic_optimizer', map_location=self.device))
        if self.decoder is not None:
            self.decoder_optimizer.load_state_dict(torch.load(filename + '_decoder_optimizer', map_location=self.device))

            
    def load_parameters(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + '_critic', map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.decoder is not None:
            self.decoder.load_state_dict(torch.load(filename + '_decoder', map_location=self.device))

    def get_state_dict(self):
        """Get the state dict of the agent for serialization to workers"""
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        # Load model parameters
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        
        # Ensure log_alpha is on the correct device
        self.log_alpha = state_dict['log_alpha'].to(self.device)
        
        # Make sure all models are on the correct device
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)
        if self.decoder is not None:
            self.decoder = self.decoder.to(self.device)


