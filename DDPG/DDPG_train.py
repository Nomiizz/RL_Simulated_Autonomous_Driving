import torch
from torch.autograd import Variable
import numpy as np
import random

from gym_torcs import TorcsEnv
import argparse
import collections

from replayBuffer import ReplayBuffer
from actorNetwork import ActorNetwork
from criticNetwork import CriticNetwork
from OU import OU
import time

import matplotlib.pyplot as plt


def train(train_indicator = 1):

    # Parameter initializations
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    vision = False

    EXPLORE = 100000.
    episode_count = 1500
    max_steps = 10000
    done = False
    step = 0
    epsilon = 1

    timeout = time.time() + 60*540   # 9 hours from now

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the actor & critic models
    actor = ActorNetwork(state_dim).to(device)
    critic = CriticNetwork(state_dim, action_dim).to(device)

    target_actor = ActorNetwork(state_dim).to(device)
    target_critic = CriticNetwork(state_dim, action_dim).to(device)

    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

     # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    # Load the actual and target models
    print("Loading models")
    try:

        actor.load_state_dict(torch.load('actormodel_nt.pth'))
        actor.eval()
        critic.load_state_dict(torch.load('criticmodel_nt.pth'))
        critic.eval()
        print("Models loaded successfully")
    except:
        print("Cannot find the models")

    target_actor.load_state_dict(actor.state_dict())
    target_actor.eval()
    target_critic.load_state_dict(critic.state_dict())
    target_critic.eval()

    # Set the optimizer and loss criterion
    criterion_critic = torch.nn.MSELoss(reduction='sum')

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


    cum_rewards = []

    print("TORCS training begins!")
    
    for ep in range(episode_count):

        if np.mod(ep, 3) == 0:
            ob = env.reset(relaunch = True)   #relaunch TORCS every 3 episodes because of the memory leak error
        else:
            ob = env.reset()

        # State variables
        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.

        for i in range(max_steps):
            critic_loss = 0
            epsilon -= 1.0 / EXPLORE  # Decaying epsilon for noise addition
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

            if torch.cuda.is_available():
                a_t_original = a_t_original.data.cpu().numpy()
            else:
                a_t_original = a_t_original.data.numpy()

            noise_t[0][0] = OU.OUnoise(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = OU.OUnoise(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] =  OU.OUnoise(a_t_original[0][2], -0.1 , 1.00, 0.05)

            # Stochastic brake
            if random.random() <= 0.1:
                print("Applying the brake")
                noise_t[0][2] = OU.OUnoise(a_t_original[0][2], 0.2, 1.00, 0.10)

            alpha = train_indicator * max(epsilon, 0)
            
            a_t[0][0] = a_t_original[0][0] + alpha * noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + alpha * noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + alpha * noise_t[0][2]

            # Perform action and get env feedback
            ob, r_t, done, info = env.step(a_t[0])

            # New state variables
            s_t_new = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            # Add to replay buffer
            buff.push(s_t, a_t[0], r_t, s_t_new, done)


            # Do the batch update
            batch = buff.sample(BATCH_SIZE)

            states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()
            actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
            rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
            new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
            dones = np.asarray([e[4] for e in batch])
            
            y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()

            # use target network to calculate target_q_value (q_prime)
            target_q_values = target_critic(new_states, target_actor(new_states))


            for k in range(len(batch)):
                if dones[k] == False:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]
                else:
                    y_t[k] = rewards[k]


            if (train_indicator):
                
                # Critic update
                q_values = critic(states, actions)
                critic_loss = criterion_critic(y_t, q_values)
                optimizer_critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                optimizer_critic.step()

                # Actor update
                # policy_loss = -torch.mean(critic(states, actor(states)))

                # optimizer_actor.zero_grad()
                # policy_loss.backward(retain_graph=True)  -------> This is leading to memory leak :(
                # optimizer_actor.step()

                a_for_grad = actor(states)
                a_for_grad.requires_grad_()    #enables the requires_grad of a_for_grad
                q_values_for_grad = critic(states, a_for_grad)
                critic.zero_grad()
                q_sum = q_values_for_grad.sum()
                q_sum.backward(retain_graph=True)

                grads = torch.autograd.grad(q_sum, a_for_grad) 

                act = actor(states)
                actor.zero_grad()
                act.backward(-grads[0])
                optimizer_actor.step()


                # update target networks 
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
                       
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


            total_reward += r_t
            s_t = s_t_new # Update the current state

            if np.mod(i, 100) == 0: 
                print("Episode", ep, "Step", step, "Action", a_t, "Reward", r_t, "Loss", critic_loss)

            step += 1
            
            if done:
                break

        if np.mod(ep, 3) == 0:
            if (train_indicator):
                print("Saving models")
                torch.save(actor.state_dict(), 'actormodel_nt.pth')
                torch.save(critic.state_dict(), 'criticmodel_nt.pth')

        cum_rewards.append(total_reward)


        print("TOTAL REWARD @ " + str(ep) +"-th Episode  : Reward " + str(total_reward))
        print("Total Steps: " + str(step))
        print("")

        if time.time() > timeout:
            break

    env.end()  # This is for shutting down TORCS
    print("Finish.")

    np.savetxt('rewards_nt.csv', np.array(cum_rewards), delimiter=',')

    episodes = np.arange(ep + 1)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DDPG (No Stochastic Braking)")
    plt.plot(episodes, np.array(cum_rewards))

    plt.show()

if __name__ == "__main__":
    train()