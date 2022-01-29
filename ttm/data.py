import copy
import os.path
import pickle
import json
import re
from collections import Counter

from ttm.trajectory import Batch
from typing import List


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

def data_filter(agent_name, game, r):
  return ((agent_name == "human") and ((re.match(
      ".*train.*", game))) and any(["hindsight_summary" in ri for ri in r["trajectory"].states()]))

def print_performance(rollouts_filepath, run_id: int, epoch: int=None):
  partitions = ["teacher", "student_train", "student_test"]
  rollouts = read_rollouts(rollouts_filepath)

  # filter out only the rollouts where the run_id matches.
  rollouts_list = []
  for key, val in rollouts.items():
    print(val[0].args)
    if re.match(f".*run_id='{run_id}'.*", key):
      rollouts_list.extend(val)

  fitnesses = []
  if epoch is None:
    epochs = range(0, max([r.epoch for r in rollouts if r.args.run_id == run_id])+1)
  else:
    epochs = [epoch]

  for e in epochs:
    epoch_fitness = {}
    for p in partitions:
      selected_rollouts = []
      for r in rollouts_list:
          if int(r.args.run_id) == run_id and int(r.args.epoch) == e and r.args.partition == p:
            print("appending")
            selected_rollouts.append(r)
          fitness = Batch.fitness(selected_rollouts)
          epoch_fitness[p] = fitness
    epoch_fitness["epoch"] = e
    fitnesses.append(epoch_fitness)
  return fitnesses

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
      if r.args.run_id == run_id and r.args.epoch == epoch and r.args.partition == partition:
        return r.args
  else:
    raise Exception(f"Could not find rollout with run_id={run_id}, epoch={epoch}, partition={partition}")

def write_rollouts_finetune(rollouts_filepath='rollouts.pkl', finetune_filepath='rollouts.txt',
                            format='model_inference_str', current_args=None, hindsight_fitness_current_batch=True):
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
        if not data_filter(agent_name, key, r) or not partition_filter(current_args, r.args):
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
              prompts.append(prompt)
              completions.append(completion)
              prompt, completion = t.hindsight_labeling_str(hindsight_value, batch_fitness_str)
              assert prompt in prompts
              if completion not in completions:
                prompts.append(prompt)
                completions.append(completion)
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





