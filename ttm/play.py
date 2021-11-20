import gym
import textworld.gym
import time
import pickle

from agent import TransformerAgent, HumanAgent
from trajectory import Trajectory, Rollout, Goal

import pdb, sys

# Register a text-based game as a new Gym's environment.
env_id = textworld.gym.register_game("tw_games/tw-coin_collector-r2ipdI3j-house-mo-xEEPcbKpfN1RibBb.z8",
                                     max_episode_steps=50)

env = gym.make(env_id)  # Start the environment.

agent = HumanAgent() # TransformerAgent()

max_train_epochs = 1
train_epochs = 0

while train_epochs < max_train_epochs:
    
    delay = 0 #s
    goal = "get some coins"
    max_actions = 4 #3
    max_rollouts = 5 #10
    rollouts = []

    while len(rollouts) < max_rollouts:
        obs, infos = env.reset()  # Start new episode.
        # env.render()
        print(obs)
        obs.
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
            scores.append(score)
            print(scores)
            env.render()
            actions += 1
        rollouts.append(rollout)
    
    with open("rollouts.pkl", "wb") as f: 
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