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

def metalearn(agent, max_train_epochs=1):
    train_epochs = 0
    while train_epochs < max_train_epochs:
        rollout_path = "ttm/data/rollouts_metalearning.txt"
        print("loading model")
        if train_epochs == 0:
            agent.load_model("gpt2")
        else:
            agent.load_model("ttm/gpt2-metalearn")
        print("loaded model")
        agent.train("ttm/gpt2-metalearn", rollout_path, rollout_path)
        train_epochs += 1

def get_metalearning_action(agent, metalearn_goal):
    agent.load_inference("ttm/gpt2-metalearn")
    obs = ""
    action = agent.predict(obs, Rollout(agent.learning_trajectory, metalearn_goal, []))
    return action

agent_goal = "score = 10000"
metalearn_prefix = "metalearn: "
metalearn_goal = metalearn_prefix + agent_goal

agent = TransformerAgent(agent_goal, device=0)

max_train_epochs = 20
train_epochs = 0

# Meta-learning agent.
# Currently uses a fixed policy to achieve its objective.
while train_epochs < max_train_epochs:
    game = agent.build_game(1)
    env = setup_game(game)

    delay = 0 #.5 #s
    goal = get_goal(env)
    print(f"goal is '{goal}'")
    max_actions = 4 #3
    max_rollouts = 1 #10
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
            metalearn_action, metalearn_prediction = get_metalearning_action(agent, metalearn_goal)
            print(f"metalearn action >>>> {metalearn_action} <<<<")
            if train_epochs == 0:
                agent.load_inference("ttm/gpt2-rollout")
            else:
                agent.load_inference("ttm/gpt2-rollout-post")
            if re.match(r".*predict.*", metalearn_action):
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

    # meta-learning.
    metalearn_goal = metalearn_prefix + agent_goal
    agent.learning_trajectory.append([f"score = {sum(scores)}", metalearn_goal, "ground_score"])
    agent.reset_state(metalearn_goal, scores)
    rollout_path = agent.write_rollouts(agent.rollouts, "metalearning-post")

    # rollout_path = "ttm/data/rollouts_metalearning.txt"
    # if train_epochs == 0:
    #     agent.load_model("gpt2")
    # else:
    #     agent.load_model("ttm/gpt2-metalearn")
    # agent.train("ttm/gpt2-metalearn", rollout_path, rollout_path)

    # train the agent on the data.
    train_epochs += 1

env.close()