import gym
import numpy as np
from Hopper_v2_DDPG import DDPG
from torch.utils.tensorboard import SummaryWriter

env = gym.make('Hopper-v2')
env = env.unwrapped
env.seed(1)

action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
a_max = env.action_space.high
a_min = env.action_space.low
print(action_dim, obs_dim,a_max,a_min)

writer = SummaryWriter('runs/Hopper_curve/')

ddpg = DDPG(action_dim, obs_dim)
step = 0
var = 1
for episode in range(2000):
    done = False
    obs = env.reset()
    reward_sum = 0
    while not done:
        env.render()
        action = ddpg.choose_action(obs) 
        # action = np.random.normal(loc=action, scale=var)
        next_obs, reward, done, info = env.step(action)
        ddpg.store_transition(obs, action, reward, next_obs)
        if step > 5000:
            ddpg.learn()
            var *= 0.99

        reward_sum += reward
        obs = next_obs
        step += 1

    print('Episode:', episode, ' Reward:' , reward_sum)
    writer.add_scalar("Reward/Episode", reward_sum, episode)