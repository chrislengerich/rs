import copy
import os.path
import pickle
import json
import pprint
import re
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from ttm.trajectory import Batch
from typing import List
import seaborn as sns


def write_rollouts_text(rollouts_filepath='rollouts.pkl', filename='rollouts.txt', format='model_inference_str'):
    """
    Write rollouts to a text file, optionally compressing them for training.
    """
    rollouts_dict = read_rollouts(rollouts_filepath)
    print(f"format: {format}")
    raise Exception("Deprecated")

    with open(filename, 'w') as f:
      # temporary hack to evaluate learning to predict-as-a-service.
      # rollouts_dict = {"1": rollouts_dict}
      for game, rollouts in rollouts_dict.items():
        for r in rollouts:
          #print(r)
          if r.fitness() < 1:
            continue
          trajs = r.hindsight_trajectories()
          #print(len(trajs))
          #print(filename)
          for t in trajs:
            if format == "imagination_action_str":
              line = t.imagination_action_str()
            elif format == "model_inference_str":
              if t.states()[0] == "":  # old format for state as a string.
                continue
              model_inference = t.model_inference_str()
              line = model_inference[0] + " " + model_inference[1]
            else:
              line = str(t)
            #print(line)
            f.write(line)
            f.write('\n')

def format_hindsight_value(hindsight_value: float) -> str:
  """Tokenize numerical hindsight value."""
  if hindsight_value == -1:
    return ""
  if hindsight_value < 0.2:
    return "low"
  elif hindsight_value < 0.8:
    return "middle"
  else:
    return "high"

def format_batch_fitness_value(batch_fitness_value: float) -> str:
  """Tokenize numerical hindsight value."""
  if batch_fitness_value == -1:
    return ""
  if batch_fitness_value < 1:
    return "low"
  elif batch_fitness_value < 3:
    return "middle"
  else:
    return "high"

def data_filter(agent_name, game, r, allowed_agent_names=List[str], allowed_splits=List[str], allowed_epochs=[]):
  is_allowed_split = False
  for split in allowed_splits:
    is_allowed_split = is_allowed_split or bool(re.match(f".*{split}.*", game))

  is_allowed_name = False
  for rollout_agent in allowed_agent_names:
    is_allowed_name = is_allowed_name or bool(re.match(f".*{rollout_agent}.*", agent_name))

  if r.args is None:
    return False
  is_allowed_epoch = int(r.args.epoch) in allowed_epochs

  print("Filter values")
  print(is_allowed_split, is_allowed_name, is_allowed_epoch)

  return (is_allowed_epoch and is_allowed_name and is_allowed_split and any(["hindsight_summary" in ri for ri in r[
    "trajectory"].states()]))

def label_summaries(rollouts_filepath, run_id: int, epoch: int, env:str="cooking_level_2", labeled=False,
                    partitions=["student_train"]):
  """Given summaries from |epoch|, |run_id|, |env| and |partition|, find the unmatched rollouts"""
  rollouts_dict = filter_rollouts(env, rollouts_filepath, run_id, partitions, epochs=[epoch])

  rollouts_list = []
  for r in rollouts_dict.values():
    rollouts_list.extend(r)

  with open(rollouts_filepath, "rb") as f:
    full_rollouts = pickle.load(f)

  for e in [epoch]:
      for key, val in rollouts_dict.items():
        print(key)
        for r in val:
          print(r.agent)
          new_labels = False
          for s in r["trajectory"]:
            print(s[0]['obs'])
            print(s[2])
            print(s[0].get('hindsight_summary', ''))
            print(s[0].get('hindsight_length', ''))
            print(s[0].get('value', ''))
            if "hindsight_summary" in s[0] and (s[0]["hindsight_summary"] != "" or labeled):
              while "hindsight_accurate" not in s[0]:
                try:
                  s[0]["hindsight_accurate"] = int(input("Summary accurate? "))
                  new_labels = True
                except ValueError:
                  continue
          full_rollouts[key] = val
          print(Batch.accuracy(val))
          if new_labels:
            with open(rollouts_filepath, "wb") as f:
              pickle.dump(full_rollouts, f)
              print("Saved new labels")

  return rollouts_filepath + ".labeled"


def label_novel_expectation(rollouts_filepath, run_id: int, epoch: int, env:str="cooking_level_2", labeled=False,
                    partitions=["student_train"]):
  """Given summaries from |epoch|, |run_id|, |env| and |partition|, find the unmatched rollouts"""
  rollouts_dict = filter_rollouts(env, rollouts_filepath, run_id, partitions, epochs=[epoch])

  rollouts_list = []
  for r in rollouts_dict.values():
    rollouts_list.extend(r)

  with open(rollouts_filepath, "rb") as f:
    full_rollouts = pickle.load(f)

  for e in [epoch]:
      for key, val in rollouts_dict.items():
        print(key)
        for r in val:
          print(r.agent)
          new_labels = False
          for s in r["trajectory"]:
            print(s[0]['obs'])
            print(s[0].get('hindsight_expectation', ''))
            if "hindsight_expectation" in s[0] and (s[0]["hindsight_expectation"] != "" or labeled):
              while "novel_hindsight_expectation" not in s[0]:
                try:
                  s[0]["novel_hindsight_expectation"] = input("Novel hindsight expectation? ")
                  new_labels = True
                except ValueError:
                  continue
          full_rollouts[key] = val
          if new_labels:
            with open(rollouts_filepath + ".new_expectations", "wb") as f:
              pickle.dump(full_rollouts, f)
              print("Saved new labels")

  return rollouts_filepath + ".labeled"


def print_performance(rollouts_filepath, run_id: int, epoch: int=None, env:str="cooking_level_2"):
  partitions = ["teacher", "student_train", "student_test"]
  rollouts_dict = filter_rollouts(env, rollouts_filepath, run_id, partitions)
  rollouts_list = []
  for r in rollouts_dict.values():
    rollouts_list.extend(r)
  epochs = max_epoch(epoch, rollouts_list, run_id)

  fitnesses = []
  for e in epochs:
    epoch_fitness = {}
    for p in partitions:
      selected_rollouts = []
      for r in rollouts_list:
          if (r.args) and int(r.args.run_id) == run_id and int(r.args.epoch) == e and r.args.partition == p:
            print("appending")
            selected_rollouts.append(r)
      fitness = Batch.fitness(selected_rollouts)
      epoch_fitness[p] = fitness
    epoch_fitness["epoch"] = e
    fitnesses.append(epoch_fitness)

  plot_fitness(fitnesses, env)
  return fitnesses

def filter_rollouts(env, rollouts_filepath, run_id, partitions=["student_train"], epochs=[]):
  rollouts = read_rollouts(rollouts_filepath)
  # filter out only the rollouts where the run_id matches.

  # TBD: do this more efficiently with Pandas, etc.
  filtered_rollouts = {}
  for key, val in rollouts.items():
    print(key)
    partition_match = False
    for p in partitions:
      if re.match(f".*partition='{p}'.*", key):
        partition_match = True

    epoch_match = False
    for e in epochs:
      if re.match(f".*epoch='{e}'.*", key):
        epoch_match = True
    if re.match(f".*run_id='{run_id}'.*", key) and re.match(f".*env='{env}'.*", key) and partition_match and epoch_match:
      filtered_rollouts[key] = val
  return filtered_rollouts

def max_epoch(epoch, rollouts_list, run_id):
  # return a list of epochs given a run_id
  if epoch is None:
    epochs = range(0, max([int(r.args.epoch) for r in rollouts_list if int(r.args.run_id) == int(run_id)]) + 1)
  else:
    epochs = [epoch]
  return epochs


def plot_fitness(fitnesses, env:str="cooking_level_2"):
  partition = "student_train"
  x = []
  y = []
  for f in fitnesses:
    num_fitnesses = len(f[partition]["fitness"])
    y.extend(f[partition]["fitness"])
    x.extend(num_fitnesses * [f["epoch"]])

  # plots x1 using seaborn with lines and estimated variances.
  sns.set(style="whitegrid")
  sns.lineplot(x=x, y=y, err_style="band")
  plt.legend(labels=[partition], title="Partition")
  plt.title("Fitness vs. Epoch")
  plt.xlabel("Epoch")
  plt.ylabel(f"Fitness on {env}")
  plt.show()

def partition_filter(current_args, rollout_args):
  """Return True if the rollout is valid training data."""
  if not current_args: # used for off-policy partitioning.
    return True
  if current_args.partition == "teacher" or current_args.partition == "student_train":
    return rollout_args.run_id == current_args.run_id and rollout_args.epoch <= current_args.epoch and \
           rollout_args.partition != "student_test"
  elif current_args.partition == "student_test":
    # no historical test data + no data from the current epoch's train env.
    return (rollout_args.run_id == current_args.run_id and rollout_args.epoch < current_args.epoch and
            rollout_args.partition != "student_test") or (rollout_args.run_id == current_args.run_id and
                                                          rollout_args.epoch == current_args.epoch and
                                                          rollout_args.partition != "student_train")

def get_args(rollouts_filepath: str, run_id: int, epoch: int, partition: str):
  for key, val in read_rollouts(rollouts_filepath).items():
    for r in val:
      if r.args and int(r.args.run_id) == int(run_id) and int(r.args.epoch) == int(epoch) and r.args.partition == \
          partition:
        return r.args
  else:
    raise Exception(f"Could not find rollout with run_id={run_id}, epoch={epoch}, partition={partition}")

# heuristic - right now we are interested learning dynamics via expectations.
def is_useful(trajectory):
  for t in [trajectory[-1]]:
    # if we see any useful hindsight_information.
    if t[0].get('hindsight_expectation', '') != '':
      return True
  else:
    return False

def write_rollouts_finetune(rollouts_filepath='rollouts.pkl', finetune_filepath='rollouts.txt',
                            format='model_inference_str', current_args=None, hindsight_fitness_current_batch=True,
                            allowed_agent_names=["human"], allowed_splits=["train"], allowed_epochs=[]):
  rollouts_dict = read_rollouts(rollouts_filepath)

  with open(finetune_filepath, 'w') as f:
    total_rollouts = 0
    total_examples = 0
    human_examples = 0
    agent_examples = 0
    rollouts_per_game = {}
    for key, rollouts in rollouts_dict.items():
      print("")
      print(f"{key} - {len(rollouts)} - {len(rollouts[0]['trajectory'])}")

      # build the training batch of rollouts based on the data filter.
      selected_rollouts = []
      for r in rollouts:
        agent_name = r.agent.get('name', '')
        if re.match('.*cooking.*', key):
          print(r.fitness())
          print(r.learning())
          print(r.agent['name'])
        if not data_filter(agent_name, key, r, allowed_agent_names, allowed_splits, allowed_epochs) or not \
            partition_filter(current_args, r.args):
          continue
        print("writing")
        selected_rollouts.append(r)

      batches = {}
      for r in selected_rollouts:
        key = str(r.args.epoch) + " " + r.args.partition
        batches.setdefault(key, []).append(r)

      for key, selected_rollouts in batches.items():
        if (str(current_args.epoch) + " " + current_args.partition) == key and not \
            hindsight_fitness_current_batch:
          batch_fitness = -1 # not set
        else:
          batch_fitness = Batch.fitness(selected_rollouts)['mean_fitness']

        for r in selected_rollouts:
          agent_name = r.agent.get('name', '')
          total_rollouts +=1
          trajs = r.hindsight_trajectories(format)
          for t in trajs:
            if t[0][0] == '':
              continue
            if format == "model_inference_str":
              prompt, completion = t.model_inference_str()
            elif format == "model_expectation_inference_str":
              prompt, completion = t.model_expectation_inference_str()
            elif format == "model_action_inference_str":
              prompt, completion = t.model_action_inference_str()
            elif format == "imitation_inference_str":
              prompt, completion = t.imitation_inference_str()
              prompts = [prompt]
              completions = [completion]
            elif format == "expected_observation_update":
              prompt, completion = t.expected_observation_key("update")
            elif format == "expected_observation_summary":
              prompt, completion = t.expected_observation_key("summary")
            elif format == "obs_summary_t_to_expectation_action":
              prompt, completion = t.obs_summary_t_to_expectation_action_str(str(r.fitness()))
            elif format == "hindsight_expectation_str":
              hindsight_value = t.states()[-1].get("hindsight_value", -1)
              hindsight_value = format_hindsight_value(hindsight_value)
              batch_fitness_str = format_batch_fitness_value(batch_fitness)
              prompts, completions = [], []
              prompt, completion = t.hindsight_expectation_str(hindsight_value, batch_fitness_str)

              if (r.agent["name"] == "human" and (r.args.env == "cooking_level_2" or r.args.env == "zork1.z5"))\
                  or (is_useful(t) and r.args.env == "zork1.z5" and r.agent["engine"] ==
                      "curie:ft-personal-2022-02-01-05-23-34"):
                prompts.append(prompt)
                completions.append(completion)
              # prompt, completion = t.hindsight_labeling_str(hindsight_value, batch_fitness_str)
              # assert prompt in prompts
              # if completion not in completions:
              #   prompts.append(prompt)
              #   completions.append(completion)
            elif format == "expected_observation":
              prompt, completion = t.expected_observation()
            else:
              raise Exception(f"Unknown format: {format}")

            total_examples += 1
            if agent_name == "human" or agent_name == "":
              human_examples += 1
            else:
              agent_examples += 1
            if re.match('.*cooking.*', key):
              game_title = "cooking"
            else:
              game_title = key
            current_count = rollouts_per_game.setdefault(game_title, 0)
            rollouts_per_game[game_title] = current_count + 1
            for (prompt, completion) in zip(prompts, completions):
              j = {"prompt": prompt, "completion": " " + completion}
              f.write(json.dumps(j))
              f.write('\n')
    print(rollouts_per_game)
    print(f"total_rollouts: {total_rollouts}")
    print(f"total_examples: {total_examples}")
    print(f"total_human: {human_examples}")
    print(f"total_agent: {agent_examples}")


def write_reward_policy(rollouts_filepath='rollouts.pkl', finetune_filepath='rollouts.txt',
                            format='model_inference_str'):
  rollouts_dict = read_rollouts(rollouts_filepath)
  rollouts_by_fitness = []

  with open(finetune_filepath, 'w') as f:
    total_rollouts = 0
    total_examples = 0
    human_examples = 0
    agent_examples = 0
    rollouts_per_game = {}
    for game, rollouts in rollouts_dict.items():
      print("")
      print(f"{game} - {len(rollouts)} - {len(rollouts[0]['trajectory'])}")
      for r in rollouts:
        agent_name = r.agent.get('name', '')
        if re.match('.*cooking.*', game):
          print(r.fitness())
          print(r.learning())
          print(r.agent['name'])

        if not ((re.match(
            ".*train.*",game))):
          continue
        print("writing")

        total_rollouts +=1
        trajs = r.hindsight_trajectories(format)
        for t in trajs:
          if t[0][0] == '':
            print("continuing")
            print(t)
            print("")
            continue
          if format == "model_inference_str":
            prompt, completion = t.model_inference_str()
          elif format == "model_expectation_inference_str":
            prompt, completion = t.model_expectation_inference_str()
          elif format == "model_action_inference_str":
            prompt, completion = t.model_action_inference_str()
          elif format == "imitation_inference_str":
            prompt, completion = t.imitation_inference_str()
          elif format == "expected_observation_update":
            prompt, completion = t.expected_observation_key("update")
          elif format == "expected_observation_summary":
            prompt, completion = t.expected_observation_key("summary")
          elif format == "expected_observation":
            prompt, completion = t.expected_observation()
          else:
            raise Exception(f"Unknown format: {format}")
          total_examples += 1
          if agent_name == "human" or agent_name == "":
            human_examples += 1
          else:
            agent_examples += 1
          if re.match('.*cooking.*', game):
            game_title = "cooking"
          else:
            game_title = game
          current_count = rollouts_per_game.setdefault(game_title, 0)
          rollouts_per_game[game_title] = current_count + 1
          j = {"prompt": prompt, "completion": " " + completion, "fitness": r.fitness(), "agent": r.agent['name']}
          rollouts_by_fitness.append(j)

    sorted_rollouts = sorted(rollouts_by_fitness, key=lambda x: x['fitness'], reverse=True)
    print(sorted_rollouts[:5])
    print(sorted_rollouts[-5:])
    top_rollouts = sorted_rollouts[:300]
    print("top rollouts")
    stats(top_rollouts)
    bottom_rollouts = sorted_rollouts[-300:]
    print("bottom rollouts")
    stats(bottom_rollouts)

    for j in top_rollouts:
      j_copy = copy.deepcopy(j)
      j_copy['prompt'] += j_copy['completion']
      j_copy["completion"] = str(j_copy['fitness'])
      del j_copy['fitness']
      del j_copy['agent']
      f.write(json.dumps(j_copy))
      f.write('\n')

    for j in bottom_rollouts:
      j_copy = copy.deepcopy(j)
      j_copy['prompt'] += j_copy['completion']
      j_copy["completion"] = str(j_copy['fitness'])
      del j_copy['fitness']
      del j_copy['agent']
      f.write(json.dumps(j_copy))
      f.write('\n')

    print(rollouts_per_game)
    print(f"total_rollouts: {total_rollouts}")
    print(f"total_examples: {total_examples}")
    print(f"total_human: {human_examples}")
    print(f"total_agent: {agent_examples}")

def stats(rollouts):
  c = Counter([r['agent'] for r in rollouts])
  print(c.most_common())

def read_rollouts(rollouts_filepath: str):
  """Reads rollouts from the pickle file."""
  if os.path.exists(rollouts_filepath):
    with open(rollouts_filepath, 'rb') as f:
      return pickle.load(f)
  else:
    return {}





