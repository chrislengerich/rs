import re

import gym
import textworld.gym
import time
import pickle

from agent import TransformerAgent, HumanAgent
from trajectory import Trajectory, Rollout, Goal

import pdb, sys

game = "grounding_game_0.z8"

# Register a text-based game as a new Gym's environment.
env_id = textworld.gym.register_game(f"tw_games/{game}",
                                     max_episode_steps=50)

env = gym.make(env_id)  # Start the environment.

agent = HumanAgent() # TransformerAgent()

max_train_epochs = 1
train_epochs = 0

def get_goal(env):
    """
    Get the current goal from the environment.
    """
    env.reset()
    obs, score, done, infos = env.step("goal")
    obs = re.sub(r"\s+", " ", obs)
    return obs.strip()

while train_epochs < max_train_epochs:
    
    delay = 0 #s
    goal = get_goal(env)
    print(f"goal is '{goal}'")
    max_actions = 4 #3
    max_rollouts = 10 #10
    rollouts = []
    import pdb
    pdb.set_trace()

    while len(rollouts) < max_rollouts:
        obs, infos = env.reset()  # Start new episode.
        # env.render()
        print(obs)
        obs = "\n".join(obs.split("\n")[-5:])
        score, actions, done = 0, 0, False
        trajectory = Trajectory()
        goal = Goal(goal)
        scores = []
        rollout = Rollout(trajectory, goal, scores)
        while not done and actions < max_actions:
            command = agent.predict(obs, rollout)
            trajectory.append([obs, goal, command])
            obs, score, done, infos = env.step(command)
            print(infos)
            scores.append(score)
            print(scores)
            env.render()
            actions += 1
        rollouts.append(rollout)
    
    with open(f"rollouts_{game}.pkl", "wb") as f:
        pickle.dump(rollouts, f)
        
    #agent.train(rollouts)
    
    # train the agent.
    # print("rollouts")
    # for r in rollouts:
    #     print(rollout)
    #     print(rollout.hindsight_trajectory())
        
    
        
    train_epochs += 1

env.close()


# print("actions: {}; score: {}".format(actions, score))