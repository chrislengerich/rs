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

from ttm import data


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

def build_game(world_size: float, seed: int = None) -> str:
    """Builds a text-world game at a specific difficulty level, returning the game name."""
    if not seed:
        seed = random.randint(0, 100000)
    name = f"grounding_game_{world_size}_{seed}.z8"
    subprocess.check_output(["tw-make", "custom", "--world-size", {world_size}, "--nb-objects", "4", "--quest-length",
                             "1", "--seed", seed, "--output", "tw_games", name])
# Meta-learning agent.

while train_epochs < max_train_epochs:
    game = build_game(1,1)
    env = setup_game(game)
    agent = TransformerAgent()

    max_train_epochs = 1
    train_epochs = 0

    delay = 0 #s
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

    # write the rollouts data out.
    with open(f"rollouts_{game}.pkl", "wb") as f:
        pickle.dump(rollouts, f)
    data.write_rollouts_text(f"rollouts_{game}.pkl", f"rollouts_{game}.txt")

    agent.train(self, )

    # train the agent on the data.
    train_epochs += 1

env.close()