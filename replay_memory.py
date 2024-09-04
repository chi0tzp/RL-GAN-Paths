import random
import numpy as np
import os
import pickle

import torch

class TensorReplayMemory:
    def __init__(self, capacity, state_shape, action_shape, device='cpu'):
        self.capacity = capacity
        self.device = device
        
        # Initialize memory as PyTorch tensors
        self.state_memory = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.next_state_memory = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.action_memory = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.reward_memory = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.done_memory = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        index = self.position % self.capacity
        
        # Store experiences as tensors
        self.state_memory[index] = torch.tensor(state, device=self.device)
        self.next_state_memory[index] = torch.tensor(next_state, device=self.device)
        self.action_memory[index] = torch.tensor(action, device=self.device)
        self.reward_memory[index] = torch.tensor(reward, device=self.device)
        self.done_memory[index] = torch.tensor(done, device=self.device)
        
        self.position += 1
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, beta=None):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        state_batch = self.state_memory[indices]
        next_state_batch = self.next_state_memory[indices]
        action_batch = self.action_memory[indices]
        reward_batch = self.reward_memory[indices]
        done_batch = self.done_memory[indices]
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, None, None

    def __len__(self):
        return self.size

class PrioritizedReplayMemory:
    def __init__(self, capacity, state_shape, action_shape, device='cpu', alpha=0.6):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha

        # Initialize memory as PyTorch tensors
        self.state_memory = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.next_state_memory = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=self.device)
        self.action_memory = torch.zeros((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.reward_memory = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.done_memory = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

        # Initialize priority buffer
        self.priority_memory = torch.zeros((capacity,), dtype=torch.float32, device=self.device)

        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        index = self.position % self.capacity
        
        # Store experiences as tensors
        self.state_memory[index] = torch.tensor(state, device=self.device)
        self.next_state_memory[index] = torch.tensor(next_state, device=self.device)
        self.action_memory[index] = torch.tensor(action, device=self.device)
        self.reward_memory[index] = torch.tensor(reward, device=self.device)
        self.done_memory[index] = torch.tensor(done, device=self.device)

        # Assign max priority to new experience to ensure it is sampled at least once
        max_priority = self.priority_memory.max() if self.size > 0 else 1.0
        self.priority_memory[index] = max_priority
        
        self.position += 1
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size, beta=0.4):
        if self.size == self.capacity:
            priorities = self.priority_memory
        else:
            priorities = self.priority_memory[:self.size]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = torch.multinomial(probabilities, batch_size, replacement=False)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        state_batch = self.state_memory[indices]
        next_state_batch = self.next_state_memory[indices]
        action_batch = self.action_memory[indices]
        reward_batch = self.reward_memory[indices]
        done_batch = self.done_memory[indices]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priority_memory[idx] = priority

    def __len__(self):
        return self.size


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}.pkl".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
