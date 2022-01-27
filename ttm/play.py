import argparse
import copy
import random
import re
import subprocess

import gym
import numpy as np
import textworld.gym
import time
import pickle
import pprint
import inspect

from jericho import FrotzEnv

from agent import TransformerAgent, HumanAgent, MetalearnedAgent, SystemAgent
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

def build_game(game: str, split: str):
    """Split is one of 'train', 'valid', 'test'.

        # Jericho games have no split associated with them.
            #game = "enchanter.z3"
            #game = "enter.z5"
            game = "gold.z5"
            #game = "dragon.z5"
            #game = "zork1.z5"
    """
    if re.match("cooking_level_1", game):
        level = [1, 1,True,False,split]
    elif re.match("cooking_level_2", game):
        level = [1, 1, True, True, split]
    elif re.match("cooking_level_3", game):
        level = [1, 9, False, False, split]
    elif re.match("cooking_level_4", game):
        level = [3, 6, True, True, split]
    else:
        env = FrotzEnv(f"jericho_games/z-machine-games-master/jericho-game-suite/{game}")
        goal = "explore, get points, don't die"

    if re.match(r"cooking.*", game):
        game = agent.build_cooking(*level)
        env = setup_game(game)
        goal = get_goal(env)

    obs, infos = env.reset()  # Start a new episode of the game.
    print(obs)
    obs = "\n".join(obs.split("\n")[-5:])

    return env, goal, obs, game

def run_rollouts(agent, policy: str, known_policies= ["whatcanido", "whatshouldido"], new_policy: str="",
                 seed: int=994, max_actions=3):
    """Builds a game and run rollouts in that game"""

    max_actions = max_actions  # 3
    max_rollouts = 1  # 10
    rollouts = []
    #known_policies = sorted(known_policies, key=len, reverse=True)
    while len(rollouts) < max_rollouts:
        trajectory = Trajectory()
        goal = Goal("get a high generalization score")
        score, actions, done = 0, 0, False
        trajectory.append([{"obs": "", "summary": "", "expectation": "", "update": "", "next_update": ""}, goal, \
                                                                                                            "start"])
        obs = "start"
        scores = []
        rollout = Rollout(trajectory, goal, scores, agent={"name": "not set" , "engine": "not set"})
        while (actions < max_actions):
            state = {"obs": obs}
            trajectory.append([state, goal, "blank"])
            action, dict_update, formatted_query = agent.predict_rollout(rollout)
            # carry through the summary.
            import pdb
            pdb.set_trace()

            if 'summary' in dict_update and dict_update['summary'] == '' and len(trajectory) > 3:
                dict_update['summary'] = trajectory[-2][0]['summary']
            state.update(dict_update)

            if done:
                break

                # rollout_copy = copy.deepcopy(rollout)
                # rollout_copy["trajectory"] = rollout.hindsight_trajectory(rollout_copy["trajectory"])
                # rollout_copy["trajectory"][-1][0]["next_update"] = rollout_copy["trajectory"][-1][0]["expectation"]
                # old_metalearn_action = metalearn_action
                # metalearn_action, _, formatted_query = aux_agent.predict_rollout(rollout_copy)
                # print(f"ACTION_UPDATE >>>> {old_metalearn_action} -> {metalearn_action}")
                # rollout["trajectory"][-1][0]["action"] = metalearn_action

            # At inference time, substitute the model expectations for the next turn's update.
            trajectory[-1][-1] = action
            if re.match(r"restore:(.*)", action):
                offset = int(re.match(r"restore:(.*)", action).group(1))
                rollouts.append(copy.deepcopy(rollout))
                env, rollout = rollout.restore(env, offset)
                trajectory = rollout["trajectory"]
                scores = rollout["scores"]
            elif re.match(r"load:.*", action):
                match = re.match(r"load:.*\['game':.*'(.*)',.*'split':.*'(.*)'\]",
                                 action)
                game, split = match.groups()
                env, goal, obs, game = build_game(game, split)
            elif re.match(r"agent:.*", action):
                match = re.match("agent:.*'(.*)'", action)
                agent_name = match.groups()[0]
                # replace the current agent with the executing agent.
                agent_goal = "score = 10000"
                if agent_name == "human":
                    agent = HumanAgent(agent_goal, device=0)
                else:
                    agent = MetalearnedAgent(agent_goal, device=0, path=f"ttm/data/{agent_name}/")
                rollout.agent = {"name": agent.name, "engine": agent.engine}
            else:
                obs, score, done, infos = env.step(action)
                scores.append(score)
                print(scores)

            pprint.pprint(rollouts)

            if isinstance(env, FrotzEnv):
                print(obs)
            else:
                env.render()
            actions += 1
            print(f"Learning: {rollout.learning()['joint']}")
            if rollout.learning()['joint'] < 0.85: # early exit from failing rollouts in the wrong part of the data
                # distribution.
                  break
        rollouts.append(rollout)

    most_recent_game = rollouts[-1]
    fitness = most_recent_game.fitness()
    learning = most_recent_game.learning()
    print(f"Train epochs: {train_epochs}")
    print(f"Agent fitness: {fitness}")
    print(f"Agent learning: {learning}")
    if True: #policy != "" and (fitness > 0 or game == "zork1.z5"):
        most_recent_game["trajectory"].append([{"obs": "data collection ended", "summary": "",
                                                "expectation":"",
                                                "update": "", "next_update": ""}, goal, f"data_collection_end: {inspect.getsource(data.data_filter)}"])
        print("Saving agent trajectory")
        return list(agent.write_rollouts(rollouts, game, policy)) + [fitness, learning, actions]
    else:
        return "", "", fitness, learning, actions

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="", help="Policy used as a suffix for the filename")
parser.add_argument("--meta_policy", default="baseline", help="Policy used for the metalearning agent")
parser.add_argument("--game", default="cooking_level_2", help="String for game name")
parser.add_argument("--split", default="train", help="one of train, valid or test")
parser.add_argument("--seed", default=1000, help="Random seed")
parser.add_argument("--max_actions", type=int, default=3, help="Max actions")
parser.add_argument("--train_epochs", type=int, default=1, help="Max train epochs")
args = parser.parse_args()
rollout_path = f"ttm/data/{args.policy}/grounding_data.pkl"

max_train_epochs = args.train_epochs
train_epochs = 0
fitness = 0

agent_goal = "score = 10000"
agent = SystemAgent(agent_goal, device=0)
agent.args = args

# Meta-learning agent.
# Currently uses a fixed policy to achieve its objective.
fitnesses = []
learnings = []
joint_learnings = []
lengths = []

while train_epochs < max_train_epochs: #and fitness < 1:
    rollout_txt_path, rollout_pickle_path, fitness, learning, length = run_rollouts(agent, args.policy,
                                                                                      seed=args.seed,
                                                         max_actions=args.max_actions)
    fitnesses.append(fitness)
    learnings.append(learning)
    lengths.append(length)
    joint_learnings.append(learning["joint"])
    #rollouts = data.read_rollouts(rollout_pickle_path)

    # metalearning loop for creating new skills.
    # action, compressed_rollout = agent.cognitive_dissonance(rollouts)
    # trajectory = compressed_rollout["trajectory"]
    # new_agent = None
    # while action != "predict":
    #     if re.match(r".*new_question_policy.*", action):
    #         new_question_agent = MetalearnedAgent(metalearn_goal, path="ttm/data/new_question_policy/")
    #         trajectory[-1][-1] = "new_question_policy"  # compressed version of the trace of the action.
    #         new_question, response, formatted_query = new_question_agent.predict_rollout(compressed_rollout)
    #         compressed_rollout["trajectory"].append([new_question, metalearn_goal, "new_prefix_policy"])
    #     elif re.match(r".*new_prefix_policy.*", action):
    #         new_prefix_agent = MetalearnedAgent(metalearn_goal, path="ttm/data/new_prefix_policy/")
    #         trajectory[-1][-1] = "new_prefix_policy"  # compressed version of the trace of the action.
    #         new_prefix, response, formatted_query = new_prefix_agent.predict_rollout(compressed_rollout)
    #         new_agent = MetalearnedAgent(metalearn_goal, path=None)
    #         full_prefix = new_prefix.split("\n") + ["New example:", f"{{rollout_action_str}} action: [ query: "
    #                                                                 f"{new_question}] ", "state: ["]
    #         # TBD: let the agent generate length and a regex parser for itself.
    #         new_agent.load_agent(full_prefix, str(compressed_rollout), [], new_question, 100)
    #         new_agent.save()
    #         compressed_rollout["trajectory"].append([f"new_prefix: {new_agent.name}", metalearn_goal, "train_grounding"])
    #         print(compressed_rollout)
    #     elif re.match(r".*train_grounding.*", action):
    #         run_rollouts(agent, new_agent.name, known_policies=["whatcanido", "whatshouldido"] + [new_agent.name],
    #                      new_policy=new_agent.name)
    #         compressed_rollout["trajectory"].append([f"ran rollouts", metalearn_goal, "cognitive_dissonance"])
    #     action = compressed_rollout["trajectory"].actions()[-1]

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
    print(f"fitness = {fitnesses}")
    print(f"fitness = {np.mean(fitnesses)}")
    print(f"std dev fitness = {np.std(fitnesses)}")
    print(f"learning = {learnings}")
    print(f"learning = {np.mean(joint_learnings)}")
    print(f"lengths = {lengths}")
    print(f"lengths = {np.mean(lengths)}")
    print(f"num saved epochs/total epochs: {len([f for f in fitnesses if f > 0])}/{len(fitnesses)}")
    train_epochs += 1