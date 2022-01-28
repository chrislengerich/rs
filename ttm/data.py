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

def data_filter(agent_name, game, r):
  return ((agent_name == "human") and ((re.match(
      ".*train.*", game))) and any(["hindsight_summary" in ri for ri in r["trajectory"].states()]))

def write_rollouts_finetune(rollouts_filepath='rollouts.pkl', finetune_filepath='rollouts.txt',
                            format='model_inference_str'):
  rollouts_dict = read_rollouts(rollouts_filepath)

  with open(finetune_filepath, 'w') as f:
    total_rollouts = 0
    total_examples = 0
    human_examples = 0
    agent_examples = 0
    rollouts_per_game = {}
    for game, data in rollouts_dict.items():
      if isinstance(data, Batch):
        rollouts = data.rollouts
        batch_fitness = data.fitness()
      elif isinstance(data, List): # list format
        rollouts = data
        batch_fitness = ""
      else:
        raise Exception("Unknown format")

      print("")
      print(f"{game} - {len(rollouts)} - {len(rollouts[0]['trajectory'])}")
      for r in rollouts:
        agent_name = r.agent.get('name', '')
        if re.match('.*cooking.*', game):
          print(r.fitness())
          print(r.learning())
          print(r.agent['name'])

        # check if we're using the new data format for hindsight trajectories.

        # for i, ri in enumerate(r["trajectory"].actions()):
        #   if re.match(".*restore.*", ri):
        #     is_hindsight = any(["hindsight_summary" in ri for ri in r["trajectory"].states()])

        # or re.match(".*enter.*", game) or re.match(".*enchanter.*", game)
        # re.match(".*zork.*", game) or re.match(".*dragon.*", game) or
        # or r.fitness() >= 2

        # original variant - want to restore this.
        # if not ((agent_name == "human") and ((re.match(
        #     ".*train.*",game))) and is_hindsight):
        if not data_filter(agent_name, game, r):
        # if (r.fitness() < 1 and not re.match(".*zork.*", game))  or re.match(".*964.*", game) or re.match(".*90["
        #                                                                                                   "0-3].*",\
        #     game) or (agent_name != "human" and agent_name != "" and not ((r.learning()['joint'] > 0.70) and
        #            re.match('.*cooking.*', game) and
        #            r.fitness() >= 1)) or \
        #     re.match(".*valid.*",game) or re.match(".*test.*", game) or not re.match(".*cooking.*", game) or (
        #     agent_name != "human"):
          continue
        print("writing")

        # TODO: Only use human trajectories here.

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
            prompt, completion = t.hindsight_expectation_str(hindsight_value)
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





