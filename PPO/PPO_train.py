import os

import numpy as np
import torch
from model.ppo import generate_trajectory, ppo_update
from torch.nn import MSELoss
from torch.optim import Adam
from gym_torcs import TorcsEnv

from model.net import MLPPolicy

import matplotlib.pyplot as plt

def train(device):
    # hyper-parameters
    coeff_entropy = 0.007
    lr = 1e-4
    mini_batch_size = 64
    max_steps = 5000
    nupdates = 10
    episode_count = 650
    clip_value = 0.2
    train = True
    render = False
    total_steps = 0

    # initialize env
    #env = TorcsEnv(port=3101, path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    cum_rewards = []

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    policy = MLPPolicy(state_dim, action_space = action_dim)
    policy.to(device)
    if os.path.exists('policy.pth'):
        policy.load_state_dict(torch.load('policy.pth', map_location = device))
        print('Loading complete!')
    if train:
        optimizer = Adam(lr=lr, params=policy.parameters())
        mse = MSELoss()

        # start training
        for e in range(episode_count):
            # generate trajectories
            relaunch = e%10 == 0
            observations, actions, logprobs, returns, values, rewards, steps = \
                generate_trajectory(env, policy, max_steps, is_render=render,
                                    obs_fn=None, progress=True, device=device, is_relaunch = relaunch)
            print('Episode %s reward is %s' % (e, rewards.sum()))
            memory = (observations, actions, logprobs, returns[:-1], values)
            # update using ppo
            policy_loss, value_loss, dist_entropy =\
                ppo_update(
                    policy, optimizer, mini_batch_size, memory, nupdates,
                    coeff_entropy=coeff_entropy, clip_value=clip_value, device=device
                )

            total_steps += steps

            print('\nEpisode: {}'.format(e))
            print('Total reward {}'.format(rewards.sum()))
            print('Entropy', dist_entropy)
            print('Policy loss', policy_loss)
            print('Value loss', value_loss)
            print('Total steps', total_steps)
            print("")

            cum_rewards.append(rewards.sum())
            
            if np.mod(e, 3) == 0:
                print("saving model")
                torch.save(policy.state_dict(), 'policy.pth')


    env.end()  # This is for shutting down TORCS
    print("Finish.")

    np.savetxt('rewards_ppo.csv', np.array(cum_rewards), delimiter=',')

    episodes = np.arange(e + 1)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("PPO")
    plt.plot(episodes, np.array(cum_rewards))

    plt.show()


if __name__ == "__main__":
    train("cpu")
