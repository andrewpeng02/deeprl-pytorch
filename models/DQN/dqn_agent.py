import numpy as np
import numpy.random as random

import torch
import torch.optim as optim
import torch.nn as nn
import gym as gym
from gym.wrappers.monitor import Monitor

from models.DQN.model import DQNModel


class DQNAgent:
    def __init__(self, lr,  momentum, alpha, gamma, target_update_frequency, local_update_frequency, replay_start_size, queue_len, batch_size):
        gym.logger.set_level(40)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make('LunarLander-v2')
        self.replay_buffer = ReplayBuffer(queue_len, self.device, alpha)

        self.local_qnetwork = DQNModel().to(self.device)
        self.target_qnetwork = DQNModel().to(self.device)
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())
        self.optimizer = optim.RMSprop(self.local_qnetwork.parameters(), lr=lr, momentum=momentum)

        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.local_update_frequency = local_update_frequency
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episode_step = 0

    def agent_step(self, state, eps, beta):
        next_state, reward, done = self.env_step(state, eps)
        if len(self.replay_buffer.queue) < self.replay_start_size:
            return next_state, reward, None, done

        # Update the local q network every local_update_frequency steps
        loss = None
        if self.episode_step % self.local_update_frequency == 0:
            loss = self.qnetwork_step(beta)

        # Update the target q network every target_update_frequency steps
        if self.episode_step % self.target_update_frequency == 0:
            self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())

        self.episode_step += 1
        return next_state, reward, loss, done

    def env_step(self, state, eps):
        action = self.policy(state, eps)
        next_state, reward, done, _ = self.env.step(action)

        self.replay_buffer.put([state, action, reward, next_state, done])
        return next_state, reward, done

    def qnetwork_step(self, beta):
        states, actions, rewards, next_states, dones, indices, is_weights = self.replay_buffer.batch_get(self.batch_size, self.state_size, beta)

        # Double DQN
        next_target_actions = torch.argmax(self.local_qnetwork(next_states), dim=1).unsqueeze(1)
        next_target_rewards = self.target_qnetwork(next_states).gather(1, next_target_actions)
        target_rewards = rewards + self.gamma * next_target_rewards * (1 - dones)
        local_rewards = self.local_qnetwork(states).gather(1, actions.long())

        self.optimizer.zero_grad()
        td_error = (local_rewards - target_rewards.detach()) ** 2
        loss = torch.mean(is_weights.unsqueeze(1) * td_error)
        loss.backward()
        for param in self.local_qnetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_error.data.cpu() + 0.0001)
        return loss.item()

    def policy(self, state, eps):
        if random.random() < eps:
            # Random action
            return self.env.action_space.sample()
        else:
            # Act according to local q network
            self.local_qnetwork.eval()
            with torch.no_grad():
                out = self.local_qnetwork(torch.FloatTensor(state).to(self.device).unsqueeze(0)).cpu()
            self.local_qnetwork.train()

            return torch.argmax(out).item()

    def reset(self, record):
        self.episode_step = 0

        if record:
            self.env = Monitor(gym.make('LunarLander-v2'), "recordings", video_callable=lambda episode_id: True, force=True)
        else:
            self.env = gym.make('LunarLander-v2')

        return self.env.reset()


# Simple replay buffer class
class ReplayBuffer:
    def __init__(self, queue_len, device, alpha):
        self.queue = []
        self.priorities = np.zeros((queue_len, ))
        self.queue_len = queue_len
        self.least_recent_idx = 0

        self.device = device
        self.alpha = alpha

    # Store state, action, reward, next_state, done, priority tuples
    def put(self, experience):
        if len(self.queue) < self.queue_len:
            self.queue.append(experience)

            if len(self.queue) > 1:
                self.priorities[self.least_recent_idx] = np.max(self.priorities)
            else:
                self.priorities[0] = 1
        else:
            self.queue[self.least_recent_idx] = experience
            self.priorities[self.least_recent_idx] = np.max(self.priorities)

        self.least_recent_idx = (self.least_recent_idx + 1) % self.queue_len

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority

    def batch_get(self, batch_size, state_size, beta):
        assert len(self.queue) >= batch_size
        if len(self.queue) != self.queue_len:
            priorities = self.priorities[:len(self.queue)]
        else:
            priorities = self.priorities

        # Get the weights for all experiences
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)

        # Get the weighted experiences
        indices = random.choice(np.arange(len(self.queue)), batch_size, p=probs, replace=False)
        experiences = [self.queue[i] for i in indices]

        is_weights = (1 / (len(self.queue) * probs[indices])) ** beta
        is_weights /= is_weights.max()
        is_weights = torch.FloatTensor(is_weights)

        states, next_states = torch.zeros((batch_size, state_size)), torch.zeros((batch_size, state_size))
        actions, rewards, dones = torch.zeros((batch_size, 1)), torch.zeros((batch_size, 1)), torch.zeros((batch_size, 1))
        for i, experience in enumerate(experiences):
            states[i] = torch.FloatTensor(experience[0])
            actions[i] = experience[1]
            rewards[i] = experience[2]
            next_states[i] = torch.FloatTensor(experience[3])
            dones[i] = experience[4]
        return states.to(self.device), actions.to(self.device), rewards.to(self.device), \
               next_states.to(self.device), dones.to(self.device), indices, is_weights.to(self.device)

    def __len__(self):
        return len(self.queue)

