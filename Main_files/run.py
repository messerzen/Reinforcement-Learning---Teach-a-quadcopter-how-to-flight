import sys
import numpy as np
from agent import DDPG
from task import Task

num_episodes = 1000
init_pose = np.array([0., 0., 0., 0., 0., 0.])
target_pos = np.array([0., 0., 10.])
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])

task = Task(init_pose=init_pose, target_pos=target_pos,
            init_angle_velocities=init_angle_velocities,
            init_velocities=init_velocities)
best_score = -np.inf


agent = DDPG(task)

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new
    score = 0
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        score += reward
        best_score = max(best_score, score)
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})"
                  .format(
                i_episode, score, best_score),
                end="")  # [debug]
            break
    sys.stdout.flush()