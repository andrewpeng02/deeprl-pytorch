import click
import os
from collections import deque

import skvideo.io

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.DQN.dqn_agent import DQNAgent


@click.command()
@click.argument('lr', type=float, default=5e-4)
@click.argument('momentum', type=float, default=0)
@click.argument('eps', type=float, default=1)
@click.argument('alpha', type=float, default=0.6)
@click.argument('beta_initial', type=float, default=0.4)
@click.argument('gamma', type=float, default=0.99)
@click.argument('target_update_frequency', type=int, default=800)
@click.argument('local_update_frequency', type=int, default=4)
@click.argument('replay_start_size', type=int, default=1e3)
@click.argument('queue_len', type=int, default=1e5)
@click.argument('batch_size', type=int, default=64)
@click.argument('num_episodes', type=int, default=1500)
def main(**kwargs):
    writer = SummaryWriter()
    writer.add_hparams(kwargs, {})
    agent = DQNAgent(kwargs['lr'], kwargs['momentum'], kwargs['alpha'], kwargs['gamma'], kwargs['target_update_frequency'],
                     kwargs['local_update_frequency'], kwargs['replay_start_size'], kwargs['queue_len'], kwargs['batch_size'])

    train(agent, writer, kwargs['eps'], kwargs['beta_initial'], kwargs['num_episodes'])


def train(agent, writer, eps_initial, beta_initial, num_episodes):
    record_every = 100
    global_step = 0

    eps = eps_initial
    beta = beta_initial
    for episode in range(num_episodes):
        record = episode % record_every == 0
        state = agent.reset(record)

        total_reward = 0
        episode_step = 0
        for i in range(1000):
            next_state, reward, loss, done = agent.agent_step(state, eps, beta)

            total_reward += reward
            if loss is not None:
                writer.add_scalar('Loss', loss, global_step)

            episode_step += 1
            global_step += 1
            state = next_state
            if done:
                break

        eps = max(eps * 0.995, 0.01)
        beta = min(beta + (1 - beta_initial) / num_episodes, 1)
        writer.add_scalar('Total reward', total_reward, global_step)
        writer.add_scalar('Episode', episode, global_step)

        # Add recording of episode to tensorboard
        if record:
            for file in os.listdir("recordings"):
                if file.endswith(".mp4"):
                    agent.env.close()

                    video = skvideo.io.vread("recordings/" + file)
                    video = torch.Tensor(video).to(torch.uint8).unsqueeze(0).permute((0, 1, 4, 2, 3))
                    writer.add_video('episode', video, global_step, fps=60)


if __name__ == "__main__":
    main()
