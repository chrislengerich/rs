import argparse
import copy
import random
import re
import subprocess
from shlex import split as shsplit

import gym
import numpy as np
import textworld.gym
import time
import pickle
import pprint
import inspect

from jericho import FrotzEnv

from agent import TransformerAgent, HumanAgent, MetalearnedAgent, SystemAgent, Agent
from trajectory import Trajectory, Rollout, Goal, Batch
from typing import List

import pdb, sys

from ttm import data
from ttm.data import write_rollouts_finetune


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

def build_game(game: str, split: str, agent):
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



def run_rollouts(policy: str, args):
    """Given a budget of |max_rollouts| and |max_actions| (cumulative total over all rollouts), allow an agent (starting,
    but not necessarily ending, as |agent|) to take actions to sample data, resample data, hindsight label
    data and train on the hindsight labeled data). This is analogous to RL training, but with a stronger emphasis on
    forward prediction and intermittent, rapid retraining on hindsight labeled data for intrinsically motivated
    hypothesis testing and generation.
    """
    max_actions = args.max_actions
    max_rollouts = args.max_rollouts
    rollouts = []

    while len(rollouts) < max_rollouts:
        agent = SystemAgent("", device=0)
        agent.args = args

        trajectory = Trajectory()
        goal = Goal("get a high batch_fitness")
        score, actions, done = 0, 0, False
        trajectory.append([{"obs": "", "summary": "", "expectation": "", "update": "", "next_update": ""}, goal, \
                                                                                                            "start"])
        obs = "start"
        scores = []
        rollout = Rollout(trajectory, goal, scores, agent={"name": "not set" , "engine": "not set"}, args=args)
        while (actions < max_actions):
            state = {"obs": obs}
            trajectory.append([state, goal, "blank"])
            action, dict_update, formatted_query = agent.predict_rollout(rollout)

            if 'summary' in dict_update and dict_update['summary'] == '' and len(trajectory) > 3:
                dict_update['summary'] = trajectory[-2][0]['summary']
            state.update(dict_update)

            if done:
                break

            # key structure:
            # auxiliary dict structure:
            #  id, epoch -> Epoch
            #     Epoch:
                    # { | teacher, student_train, student_test | -> env}
                    #  | teacher, student_train, student_test | -> List[Rollout]

            #  data visualizer:
            #     given an id, calculate relevant teacher, student, test fitnesses.

            # At inference time, substitute the model expectations for the next turn's update.
            # All system commands have the following format: <command>:
            # Training labels (key,value associations) are of the form:
            #   training_set_id (incremented by one with each new span).
            #   start training_set
            #   label training_set
            #   SysController ensures that:

            #   1 epoch of meta-training:

            #   First data filters:
            #       Includes all data before it.
            #   Second pass filter:
            #       Include high-quality + contrastive data, taking all the lessons from data2vec.
            #
            #   Teacher pass -> creates a batch of trajectories to finetune on.
            #      -> load train env [ done ]
            #      -> teacher plays through, adds hindsight annotations and trains occasionally. [ done ]
            #      -> filter: prior teacher and student batches. [ done ]
            #      -> eval train env fitness + hindsight labeling [ done ]
            #      -> retrain on env fitness [ done ]

            #   Student (train env):
            #      -> load train env
            #      -> machine plays through, adding hindsight annotations + fine-tuning itself on the fly.
            #      -> filter: prior teacher and student batches, including itself.
            #      -> eval train env fitness + hindsight labeling

            #   Student (test env):
            #       -> load test env
            #       -> machine plays through, adding hindsight annotations + fine-tuning itself.
            #       -> eval train env fitness + hindsight labeling

            # Reports:
            #   1. Epoch.
            #   2. Teacher fitness (train env).
            #   3. Student fitness (train env).
            #   4. Student fitness (test env).

            # Improvements at the meta-epoch level:
            # Add labeled student + teacher experiences from the train env to the aggregated experiences.
            # Or, possibly just add labeled student experiences.

            # structures:
            #   1. Epoch:
            #     2. Teacher rollouts + fitness (train env).
            #     3. Student rollouts + fitness (train env).
            #     4. Student rollouts + fitness (test env).

            # one epoch:

                # Human commands:
                #  inputs: human x train x epoch x eval_retrain x run_id
                #  valid data: train x preceding epochs
                #  hindsight label, retrain (iteratively) until ended (no batch labels)
                #  generate batch hindsight labels, push out to monitor data structure

                #  retrain with new human batch + last epoch's agent batch -> new_agent_name
                #  push new agent name + data keys to monitor structure such that we can
                #  recreate training data + training process uniquely.

                #  new_agent_name x train x epoch
                #  valid data: train x preceding epochs
                #  hindsight label the batch, push to monitor

                #  new_agent_name x test x epoch
                #  valid data: test x same epoch
                #  hindsight label the batch, push to monitor

                #  increment epoch

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
                env, goal, obs, game = build_game(game, split, agent)
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
            elif re.match(r"sample_data:.*", action):
                sample_arg_string = re.match(r"sample_data: (.*)", action).groups[0].strip()
                sample_args = parser.parse_args(shsplit(sample_arg_string))
                write_rollouts_finetune(sample_args.pickle_path, sample_args.finetune_path, sample_args.format, args)
                obs = f"sampled:"
            elif re.match(r".*finetune:.*", action):
                testing = False
                if False: # rollout.agent["name"] != "human" or testing: # for testing
                    sample_arg_string = shsplit(SystemAgent("").write_finetune(args))
                    #sample_args = parser.parse_args(shsplit(sample_arg_string))
                    output = subprocess.check_output(sample_arg_string)
                    print(output.decode("utf-8"))
                    argstring = shsplit(SystemAgent("").train_command(policy))
                    try:
                        output = subprocess.check_output(argstring)
                        print(output.decode("utf-8"))
                        model_name = re.match(SystemAgent.model_name_regex, str(output)).groups()[0]
                        assert model_name
                    except Exception as e:
                        import pdb
                        pdb.set_trace()
                    obs = f"finetuned: {model_name}"
                    agent.update_engine(model_name, policy)
                    agent.register_agent(policy, model_name, f"ttm/data/{policy}/grounding_data.jsonl")
            elif re.match(r"register:.*", action): # used for testing only
                model_name = re.match("register: (.*)", action).groups()[0]
                # assumes the most recent grounding data file is the output file.
                agent.register_agent(policy, model_name, f"ttm/data/{policy}/grounding_data.jsonl")
                agent.update_engine(model_name)
                obs = f"registered: {model_name}"
            elif re.match(r"fitness:.*", action):  # used for testing only
                fitness_data = Batch.fitness(rollouts)
                print(f"fitness = {fitness_data['fitness']}")
                print(f"mean fitness = {fitness_data['mean_fitness']}")
                print(f"std dev fitness = {fitness_data['std_fitness']}")
                print(f"learning = {fitness_data['learning']}")
                print(f"mean learning = {fitness_data['mean_learning']}")
                print(f"lengths = {fitness_data['length']}")
                print(f"lengths = {np.mean(fitness_data['length'])}")
                obs = f"fitness: {fitness_data}"
            elif re.match(r"end:", action): # end the rollout early
                break
            else:
                obs, score, done, infos = env.step(action)
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
        print(f"Agent rollout fitness: {rollout.fitness()}")
        print(f"Agent rollout learning: {rollout.learning()}")
        print(f"Batch fitness: {Batch.fitness(rollouts)}")
        print(f"Args: {args}")
        print("Saving agent trajectory")
        txt_path, pickle_path = agent.write_rollouts([rollout], game, policy, args)
    return txt_path, pickle_path


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="", help="Policy used as a suffix for the filename")
parser.add_argument("--meta_policy", default="baseline", help="Policy used for the metalearning agent")
parser.add_argument("--env", default="cooking_level_2", help="String for high-level environment name")
parser.add_argument("--split", default="train", help="one of train, valid or test")
parser.add_argument("--run_id", default=0, help="id of the training run")
parser.add_argument("--epoch", default=-1, help="epoch counter for training runs")
parser.add_argument("--seed", default=1000, help="Random seed")
parser.add_argument("--max_actions", type=int, default=3, help="Max actions")
parser.add_argument("--max_rollouts", type=int, default=1, help="Max rollouts within the batch")
parser.add_argument("--max_train_epochs", type=int, default=1, help="Max train epochs")
parser.add_argument("--partition", type=str, default="teacher", help="shortname (one of teacher, student_train, "
                                                                     "or student_test)")
parser.add_argument("--filter", type=str)
args = parser.parse_args()

target_epochs = range(0, args.max_train_epochs) if args.epoch == -1 else [args.epoch]
for epoch in target_epochs:
    args.epoch = epoch
    rollout_txt_path, rollout_pickle_path = run_rollouts(args.policy, args)