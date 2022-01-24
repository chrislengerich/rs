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

from jericho import FrotzEnv

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

def run_rollouts(agent, policy: str, known_policies= ["whatcanido", "whatshouldido"], new_policy: str="",
                 seed: int=994, max_actions=3):
    """Builds a game and run rollouts in that game"""
    if True: # TextWorld game
        #game = agent.build_treasure_hunter(1)
        # level 1 for cooking:
        level_1 = [1, 1,True,False,"valid"]
        level_2 = [1, 1, True, True, "valid"]
        level_3 = [1, 9, False, False, "train"]
        level_4 = [3, 6, True, True, "train"]
        game = agent.build_cooking(*level_2)
        #game = agent.build_game(7, 4, seed)
        env = setup_game(game)
        goal = get_goal(env)
    else:
        #game = "enchanter.z3"
        #game = "enter.z5"
        game = "gold.z5"
        #game = "dragon.z5"
        #game = "zork1.z5"
        env = FrotzEnv(f"jericho_games/z-machine-games-master/jericho-game-suite/{game}")
        goal = "explore, get points, don't die"
    print(f"goal is '{goal}'")
    max_actions = max_actions  # 3
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
        trajectory.append([{"obs": "", "summary": "", "expectation": "", "update": "", "next_update": ""}, goal, \
                                                                                                            "start"])
        scores = []
        rollout = Rollout(trajectory, goal, scores, agent={"name": agent.name, "engine": agent.engine})
        while (actions < max_actions): # or (agent.name == 'human' and game != "zork1.z5")):
            # metalearn_rollout = Rollout(agent.learning_trajectory, metalearn_goal, [])
            #print(rollout)
            state = {"obs": obs}
            trajectory.append([state, goal, "blank", copy.deepcopy(env)])
            if new_policy != "" and actions == 0:
                metalearn_action = new_policy
            else:
                metalearn_action, dict_update, formatted_query = agent.predict_rollout(rollout)
                # carry through the summary.
                if 'summary' in dict_update and dict_update['summary'] == '' and len(trajectory) > 1:
                    dict_update['summary'] = trajectory[-2][0]['summary']
            state.update(dict_update)

            if done:
                break

            teacher_forcing = True
            if teacher_forcing: # using teacher forcing to rewrite the action.
                aux_agent = MetalearnedAgent(agent_goal, device=0, path=f"ttm/data/obs_summary_t_to_expectation_action/")
                old_metalearn_action = metalearn_action
                metalearn_action, dict_update, formatted_query = aux_agent.predict_rollout(rollout, value=True)
                print(f"ACTION_UPDATE >>>> {old_metalearn_action} -> {metalearn_action}")
                state.update(dict_update)

                # rollout_copy = copy.deepcopy(rollout)
                # rollout_copy["trajectory"] = rollout.hindsight_trajectory(rollout_copy["trajectory"])
                # rollout_copy["trajectory"][-1][0]["next_update"] = rollout_copy["trajectory"][-1][0]["expectation"]
                # old_metalearn_action = metalearn_action
                # metalearn_action, _, formatted_query = aux_agent.predict_rollout(rollout_copy)
                # print(f"ACTION_UPDATE >>>> {old_metalearn_action} -> {metalearn_action}")
                # rollout["trajectory"][-1][0]["action"] = metalearn_action

            # given updates, summaries, expectations and actions, predict the current update + summary + expectations (
            # training the WM) -> this is already done by the default model, no need to retrain.

            # given updates, summaries, expectations and actions + current turn's update + summary + expectations,
            # predict the next turn's update + action -> this needs to be trained.
            # Write out data with next_turn's update.

            # At inference time, substitute the model expectations for the next turn's update.
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
                #print(infos)
                scores.append(score)
                print(scores)
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
        print("Saving agent trajectory")
        return list(agent.write_rollouts(rollouts, game, policy)) + [fitness, learning, actions]
    else:
        return "", "", fitness, learning, actions

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="", help="Policy used as a suffix for the filename")
parser.add_argument("--meta_policy", default="baseline", help="Policy used for the metalearning agent")
parser.add_argument("--seed", default=1000, help="Random seed")
parser.add_argument("--max_actions", type=int, default=3, help="Max actions")
parser.add_argument("--train_epochs", type=int, default=1, help="Max train epochs")
args = parser.parse_args()
rollout_path = f"ttm/data/{args.policy}/grounding_data.pkl"

agent_goal = "score = 10000"
metalearn_prefix = "metalearn: "
metalearn_goal = metalearn_prefix + agent_goal

if args.meta_policy == "human":
    agent = HumanAgent(agent_goal, device=0)
else:
    agent = MetalearnedAgent(agent_goal, device=0, path=f"ttm/data/{args.meta_policy}/")

max_train_epochs = args.train_epochs
train_epochs = 0
fitness = 0

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