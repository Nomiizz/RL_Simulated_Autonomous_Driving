import torch
import numpy as np
import sys

from gym_torcs import TorcsEnv
import matplotlib.pyplot as plt

from actorNetwork import ActorNetwork


episode_count = 10
max_steps = 10000
done = False
step = 0
vision = False

action_dim = 3  #Steering/Acceleration/Brake
state_dim = 29  #of sensors input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cum_rewards = []

# Create the actor model
actor = ActorNetwork(state_dim).to(device)

# Generate a Torcs environment
env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

# Load the actor model
print("Loading actor model")
try:
    actor.load_state_dict(torch.load('actormodel_nt.pth'))
    actor.eval()
    print("Model loaded successfully")
except:
    print("Cannot find the actor model")
    env.end()  # This is for shutting down TORCS
    sys.exit()

print("TORCS experiment begins!") 
for ep in range(episode_count):

    if np.mod(ep, 3) == 0:
        ob = env.reset(relaunch = True)   #relaunch TORCS every 3 episodes because of the memory leak error
    else:
        ob = env.reset()

    # State variables
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    total_reward = 0.

    for i in range(max_steps):

        # Get actions from the actor network
        a_t = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t = a_t.data.cpu().numpy()
        else:
            a_t = a_t.data.numpy()

        # Perform action and get env feedback
        ob, r_t, done, info = env.step(a_t[0])

        # New state variables
        s_t_new = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward += r_t
        s_t = s_t_new

        if np.mod(i, 100) == 0: 
            print("Episode", ep, "Step", step, "Action", a_t, "Reward", r_t)

        step += 1
        if done: 
            break

    cum_rewards.append(total_reward)

    print("TOTAL REWARD @ " + str(ep) +"-th Episode  : Reward " + str(total_reward))
    print("Total Steps: " + str(step))
    print("")

env.end()  # This is for shutting down TORCS
print("Finish.")

print("Total reward for experiment: ", sum(cum_rewards))
print("Average reward per episode: ", sum(cum_rewards) / episode_count)