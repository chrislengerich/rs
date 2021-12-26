import os.path
import pickle
import json
import re


def write_rollouts_text(rollouts_filepath='rollouts.pkl', filename='rollouts.txt', format='model_inference_str'):
    """
    Write rollouts to a text file, optionally compressing them for training.
    """
    rollouts_dict = read_rollouts(rollouts_filepath)
    print(f"format: {format}")

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
              if r["trajectory"].states()[0] == "":  # old format for state as a string.
                continue
              model_inference = t.model_inference_str()
              line = model_inference[0] + " " + model_inference[1]
            else:
              line = str(t)
            #print(line)
            f.write(line)
            f.write('\n')

def write_rollouts_finetune(rollouts_filepath='rollouts.pkl', finetune_filepath='rollouts.txt',
                            format='model_inference_str'):
  rollouts_dict = read_rollouts(rollouts_filepath)

  with open(finetune_filepath, 'w') as f:
    total_rollouts = 0
    total_examples = 0
    for game, rollouts in rollouts_dict.items():
      print(f"{game} - {len(rollouts)} - {len(rollouts[0]['trajectory'])}")
      for r in rollouts:
        agent_name = r.agent.get('name', '')
        if r.fitness() < 1  or re.match(".*964.*", game) or re.match(".*90[0-3].*",game) or (agent_name != "human"
                                                                                             and agent_name != ""):
          continue
        total_rollouts +=1
        trajs = r.hindsight_trajectories()
        for t in trajs:
          if t[0][0] == '':
            print("continuing")
            print(t)
            print("")
            continue
          if format == "model_inference_str":
            prompt, completion = t.model_inference_str()
          elif format == "imitation_inference_str":
            prompt, completion = t.imitation_inference_str()
          else:
            raise Exception(f"Unknown format: {format}")
          total_examples += 1
          j = {"prompt": prompt, "completion": " " + completion}
          f.write(json.dumps(j))
          f.write('\n')
    print(f"total_rollouts: {total_rollouts}")
    print(f"total_examples: {total_examples}")

def read_rollouts(rollouts_filepath: str):
  """Reads rollouts from the pickle file."""
  if os.path.exists(rollouts_filepath):
    with open(rollouts_filepath, 'rb') as f:
      return pickle.load(f)
  else:
    return {}





