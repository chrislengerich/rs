import argparse
import random
import re
import subprocess

import gym
import textworld.gym
import time
import pickle

from agent import TransformerAgent, HumanAgent, MetalearnedAgent
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

def run_rollouts(agent, policy: str, known_policies= ["whatcanido", "whatshouldido"], new_policy: str=""):
    """Builds a game and run rollouts in that game"""
    game = agent.build_game(1, 1000)
    env = setup_game(game)
    goal = get_goal(env)
    print(f"goal is '{goal}'")
    max_actions = 2  # 3
    max_rollouts = 1  # 10
    rollouts = []
    known_policies = sorted(known_policies, key=len, reverse=True)
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
            # metalearn_rollout = Rollout(agent.learning_trajectory, metalearn_goal, [])
            trajectory.append([obs, goal, "blank"])
            if new_policy != "" and actions == 0:
                metalearn_action = new_policy
            else:
                metalearn_action, full_action, formatted_query = agent.predict_rollout(rollout)
            # agent.get_metalearning_action(
            # HumanAgent(metalearn_goal), obs, metalearn_rollout)
            print(f"metalearn action >>>> {metalearn_action} <<<<")
            matched = False
            for p in known_policies:
                if re.match(f".*{p}.*", metalearn_action):
                    query_agent = MetalearnedAgent(metalearn_goal, path=f"ttm/data/{p}")
                    trajectory[-1][-1] = p  # compressed version of action.
                    what_can_i_do, response, formatted_query = query_agent.predict_rollout(rollout)
                    obs += " " + what_can_i_do
                    print(obs)
                    assert not matched
                    matched = True
            if not matched:
                command = metalearn_action
                trajectory[-1][-1] = command
                obs, score, done, infos = env.step(command)
                print(infos)
                scores.append(score)
                print(scores)
                env.render()
            actions += 1
        rollouts.append(rollout)
    return agent.write_rollouts(rollouts, game, policy)

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="general", help="Policy used as a suffix for the filename")
args = parser.parse_args()
rollout_path = f"ttm/data/{args.policy}/grounding_data.pkl"

agent_goal = "score = 10000"
metalearn_prefix = "metalearn: "
metalearn_goal = metalearn_prefix + agent_goal

#agent = HumanAgent(agent_goal, device=0)
agent = MetalearnedAgent(agent_goal, device=0, path="ttm/data/agent/")

max_train_epochs = 1
train_epochs = 0

# Meta-learning agent.
# Currently uses a fixed policy to achieve its objective.
while train_epochs < max_train_epochs:
    rollout_txt_path, rollout_pickle_path = run_rollouts(agent, args.policy)
    rollouts = data.read_rollouts(rollout_pickle_path)

    # metalearning loop.
    action, compressed_rollout = agent.cognitive_dissonance(rollouts)
    trajectory = compressed_rollout["trajectory"]
    new_agent = None
    while action != "predict":
        if re.match(r".*new_question_policy.*", action):
            new_question_agent = MetalearnedAgent(metalearn_goal, path="ttm/data/new_question_policy/")
            trajectory[-1][-1] = "new_question_policy"  # compressed version of the trace of the action.
            new_question, response, formatted_query = new_question_agent.predict_rollout(compressed_rollout)
            compressed_rollout["trajectory"].append([new_question, metalearn_goal, "new_prefix_policy"])
        elif re.match(r".*new_prefix_policy.*", action):
            new_prefix_agent = MetalearnedAgent(metalearn_goal, path="ttm/data/new_prefix_policy/")
            trajectory[-1][-1] = "new_prefix_policy"  # compressed version of the trace of the action.
            new_prefix, response, formatted_query = new_prefix_agent.predict_rollout(compressed_rollout)
            new_agent = MetalearnedAgent(metalearn_goal, path=None)
            # TBD: let the agent generate length and a regex parser for itself.
            new_agent.load_agent(new_prefix.split("\n"), str(compressed_rollout), [], new_question, 100)
            new_agent.save()
            compressed_rollout["trajectory"].append([f"new_prefix: {new_agent.name}", metalearn_goal, "train_grounding"])
            print(compressed_rollout)
        elif re.match(r".*train_grounding.*", action):
            run_rollouts(agent, new_agent.name, known_policies=["whatcanido", "whatshouldido"] + [new_agent.name],
                         new_policy=new_agent.name)
            compressed_rollout["trajectory"].append([f"ran rollouts", metalearn_goal, "cognitive_dissonance"])
        action = compressed_rollout["trajectory"].actions()[-1]

    # if train_epochs == 0:
    #     agent.load_model("ttm/gpt2-rollout")
    # else:
    #     agent.load_model("ttm/gpt2-rollout-post")
    # agent.train("ttm/gpt2-rollout-post", rollout_path, rollout_path)
    #
    # # meta-learning.
    # metalearn_goal = metalearn_prefix + agent_goal
    # agent.learning_trajectory.append([f"score = {sum(scores)}", metalearn_goal, "ground_score"])
    # agent.reset_state(metalearn_goal, scores)
    # rollout_path = agent.write_rollouts(agent.rollouts, "metalearning-post")

    # rollout_path = "ttm/data/rollouts_metalearning.txt"
    # if train_epochs == 0:
    #     agent.load_model("gpt2")
    # else:
    #     agent.load_model("ttm/gpt2-metalearn")
    # agent.train("ttm/gpt2-metalearn", rollout_path, rollout_path)

    # train the agent on the data.
    train_epochs += 1

env.close()