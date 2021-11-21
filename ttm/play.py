import random
import re
import subprocess

import gym
import textworld.gym
import time
import pickle

from agent import TransformerAgent, HumanAgent
from trajectory import Trajectory, Rollout, Goal

import pdb, sys

def get_goal(env):
    """
    Get the current goal from the environment.
    """
    env.reset()
    obs, score, done, infos = env.step("goal")
    obs = re.sub(r"\s+", " ", obs)
    return obs.strip()

def setup_game(game: str= "grounding_game_0.z8"):
    # Register a text-based game as a new Gym's environment.
    env_id = textworld.gym.register_game(f"tw_games/{game}", max_episode_steps=50)
    env = gym.make(env_id)  # Start the environment.
    return env

agent_goal = "score = 10000"
agent = TransformerAgent(agent_goal, device=0)
max_train_epochs = 1
train_epochs = 0

# Meta-learning agent.
# Currently uses a fixed policy to achieve its objective.
while train_epochs < max_train_epochs:
    game = agent.build_game(1,1)
    env = setup_game(game)
    if train_epochs == 0:
        agent.load_inference("ttm/gpt2-rollout")
    else:
        agent.load_inference("ttm/gpt2-rollout-post")

    delay = 0.5 #s
    goal = get_goal(env)
    print(f"goal is '{goal}'")
    max_actions = 4 #3
    max_rollouts = 10 #10
    rollouts = []

    while len(rollouts) < max_rollouts:
        obs, infos = env.reset()  # Start new episode.
        # env.render()
        print(obs)
        obs = "\n".join(obs.split("\n")[-5:])
        score, actions, done = 0, 0, False
        trajectory = Trajectory()
        goal = Goal(goal)
        trajectory.append(["", goal, "start"])
        scores = []
        rollout = Rollout(trajectory, goal, scores)
        while not done and actions < max_actions:
            command, prediction = agent.predict(obs, rollout)
            trajectory.append([obs, goal, command])
            obs, score, done, infos = env.step(command)
            print(infos)
            scores.append(score)
            print(scores)
            env.render()
            actions += 1
        rollouts.append(rollout)
    rollout_path = agent.write_rollouts(rollouts, game)
    if train_epochs == 0:
        agent.load_model("ttm/gpt2-rollout")
    else:
        agent.load_model("ttm/gpt2-rollout-post")
    agent.train("ttm/gpt2-rollout-post", rollout_path, rollout_path)

    agent.learning_trajectories.append([f"score = {sum(scores)}", agent_goal, "ground_score"])
    rollout_path = agent.write_rollouts([Rollout(agent.learning_trajectories, agent_goal, scores)], "metalearning")

    # TODO: Train the meta-learning policy on the learning trajectories.
    # TODO: Make decisions about learning based on the learning trajectories.

    # train the agent on the data.
    train_epochs += 1

env.close()