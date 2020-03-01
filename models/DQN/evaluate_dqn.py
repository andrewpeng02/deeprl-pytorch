import click
from collections import deque

import numpy as np

from models.DQN.dqn_agent import DQNAgent


@click.command()
@click.argument('lr', type=float, default=5e-4)
@click.argument('momentum', type=float, default=0)
@click.argument('eps', type=float, default=1)
@click.argument('gamma', type=float, default=0.99)
@click.argument('target_update_frequency', type=int, default=800)
@click.argument('local_update_frequency', type=int, default=4)
@click.argument('replay_start_size', type=int, default=1e3)
@click.argument('queue_len', type=int, default=1e5)
@click.argument('batch_size', type=int, default=64)
@click.argument('num_episodes', type=int, default=1500)
def main(**kwargs):
    episodes = []
    for i in range(50):
        agent = DQNAgent(kwargs['lr'], kwargs['momentum'], kwargs['gamma'], kwargs['target_update_frequency'],
                         kwargs['local_update_frequency'], kwargs['replay_start_size'], kwargs['queue_len'], kwargs['batch_size'])

        episode = train(agent, kwargs['eps'], kwargs['num_episodes'])

        print(f'Iteration [{i + 1} / 50] \t Episodes to completion: {episode}')
        episodes.append(episode)
    np.save('evaluate_dqn_out_vanilla', episodes)


def train(agent, eps, num_episodes):

    recent_rewards = deque(maxlen=100)
    for episode in range(num_episodes):
        state = agent.reset(False)

        total_reward = 0
        episode_step = 0
        for i in range(1000):
            next_state, reward, loss, done = agent.agent_step(state, eps)

            total_reward += reward

            episode_step += 1
            state = next_state
            if done:
                break
        eps = max(eps * 0.995, 0.01)

        recent_rewards.append(total_reward)
        if np.mean(recent_rewards) > 200 and len(recent_rewards) == 100:
            return episode
    return num_episodes


if __name__ == "__main__":
    main()
