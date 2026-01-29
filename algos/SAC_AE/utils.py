import torch
import numpy as np
import torch.nn as nn
import os
from collections import deque
import random


class eval_policy_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, robot_goal_state_dim, action_shape, 
                 capacity=200000, batch_size=32, device='cuda', obs_dtype=np.uint8):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32
        obs_dtype = obs_dtype

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.robot_goal_emotion_states = np.empty((capacity, robot_goal_state_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_robot_goal_emotion_states = np.empty((capacity, robot_goal_state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, robot_goal_state, action, reward, next_obs, next_robot_goal_state, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.robot_goal_emotion_states[self.idx], robot_goal_state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_robot_goal_emotion_states[self.idx], next_robot_goal_state)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_direct(self, batch_size=None):
        """
        Sample directly as a dictionary to support the RatioCombinedReplayBuffer.
        Returns a dictionary of tensors instead of a tuple.
        """
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        robot_goal_emotion_states = torch.as_tensor(self.robot_goal_emotion_states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        next_robot_goal_emotion_states = torch.as_tensor(self.next_robot_goal_emotion_states[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return {
            'obses': obses, 
            'robot_goal_emotion_states': robot_goal_emotion_states, 
            'actions': actions, 
            'rewards': rewards, 
            'next_obses': next_obses, 
            'next_robot_goal_emotion_states': next_robot_goal_emotion_states, 
            'not_dones': not_dones
        }

    def sample(self, batch_size=None):
        """
        Original sample method that returns a tuple of tensors.
        """
        result = self.sample_direct(batch_size)
        return (
            result['obses'], 
            result['robot_goal_emotion_states'], 
            result['actions'], 
            result['rewards'], 
            result['next_obses'], 
            result['next_robot_goal_emotion_states'], 
            result['not_dones']
        )

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.robot_goal_emotion_states[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.next_robot_goal_emotion_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path, pickle_protocol=4)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.robot_goal_emotion_states[start:end] = payload[1]
            self.next_obses[start:end] = payload[2]
            self.next_robot_goal_emotion_states[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.not_dones[start:end] = payload[6]
            self.idx = end


class CombinedReplayBuffer:
    def __init__(self, buffers, batch_size):
        self.buffers = buffers
        self.batch_size = batch_size

    def sample(self):
        num_buffers = len(self.buffers)
        per_buffer = self.batch_size // num_buffers
        remainder = self.batch_size % num_buffers
        batches = []
        for i, buf in enumerate(self.buffers):
            num = per_buffer + (1 if i < remainder else 0)
            batch = buf.sample(batch_size=num)
            batches.append(batch)
        
        # Combine each element of the tuple across all batches
        combined = []
        for i in range(len(batches[0])):
            combined.append(torch.cat([b[i] for b in batches], dim=0))
        
        return tuple(combined)

class RatioCombinedReplayBuffer:
    """
    A replay buffer that combines multiple replay buffers with specified sampling ratios.
    """
    def __init__(self, replay_buffers, batch_size, sampling_ratios=None):
        """
        Args:
            replay_buffers: List of replay buffers to sample from
            batch_size: Total batch size to sample
            sampling_ratios: List of ratios for sampling from each buffer. 
                            If None, equal sampling is used.
        """
        self.buffers = replay_buffers
        self.batch_size = batch_size
        self.num_buffers = len(replay_buffers)
        
        if sampling_ratios is None:
            # Equal sampling from each buffer if no ratios provided
            self.sampling_ratios = [1.0 / self.num_buffers] * self.num_buffers
        else:
            # Normalize ratios to sum to 1
            total = sum(sampling_ratios)
            self.sampling_ratios = [ratio / total for ratio in sampling_ratios]
            
        # Calculate samples per buffer, ensuring we get exactly batch_size samples
        self.samples_per_buffer = [int(ratio * batch_size) for ratio in self.sampling_ratios]
        # Adjust for any rounding errors
        remaining = batch_size - sum(self.samples_per_buffer)
        if remaining > 0:
            # Add remaining samples to buffers based on highest fractional parts
            fractions = [(ratio * batch_size) - int(ratio * batch_size) for ratio in self.sampling_ratios]
            indices = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)
            for i in range(remaining):
                self.samples_per_buffer[indices[i]] += 1
                
        # print(f"Buffer sampling configuration: {len(replay_buffers)} buffers")
        # print(f"Sampling ratios: {self.sampling_ratios}")
        # print(f"Samples per buffer: {self.samples_per_buffer}")

    def sample(self):
        """
        Sample from the combined buffer according to the specified ratios.
        Returns a tuple to match the original ReplayBuffer.sample() interface.
        """
        # Get samples from each buffer according to sampling ratios
        batch_samples = []
        
        for i, (buffer, num_samples) in enumerate(zip(self.buffers, self.samples_per_buffer)):
            if num_samples == 0:
                continue
                
            # Sample from this buffer
            buffer_samples = buffer.sample_direct(num_samples)
            batch_samples.append(buffer_samples)
            
        # Combine samples from all buffers
        if len(batch_samples) == 1:
            sample_dict = batch_samples[0]
        else:
            # Combine tensors from each buffer
            sample_dict = {}
            for key in batch_samples[0].keys():
                sample_dict[key] = torch.cat([batch[key] for batch in batch_samples], dim=0)
        
        # Convert dictionary to tuple to match the original interface
        return (
            sample_dict['obses'],
            sample_dict['robot_goal_emotion_states'],
            sample_dict['actions'],
            sample_dict['rewards'],
            sample_dict['next_obses'],
            sample_dict['next_robot_goal_emotion_states'],
            sample_dict['not_dones']
        )
