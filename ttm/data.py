import os.path
import pickle
import json

def write_rollouts_text(rollouts_filepath='rollouts.pkl', filename='rollouts.txt', format='baseline'):
    """
    Write rollouts to a text file, optionally compressing them for training.
    """
    rollouts_dict = read_rollouts(rollouts_filepath)

    with open(filename, 'w') as f:
      # temporary hack to evaluate learning to predict-as-a-service.
      # rollouts_dict = {"1": rollouts_dict}
      for game, rollouts in rollouts_dict.items():
        for r in rollouts:
          print(r)
          trajs = r.hindsight_trajectories()
          print(len(trajs))
          print(filename)
          for t in trajs:
            if format == "imagination_action_str":
              line = t.imagination_action_str()
            else:
              line = str(t)
            print(line)
            f.write(line)
            f.write('\n')

def write_rollouts_finetune(rollouts_filepath='rollouts.pkl', finetune_filepath='rollouts.txt', format='baseline'):
  rollouts_dict = read_rollouts(rollouts_filepath)

  with open(finetune_filepath, 'w') as f:
    for game, rollouts in rollouts_dict.items():
      for r in rollouts:
        print(r)
        trajs = r.hindsight_trajectories()
        print(len(trajs))
        print(finetune_filepath)
        for t in trajs:
          if format == "imagination_action_str":
            line = t.imagination_action_str()
          else:
            line = str(t)
          print(line)
          if t[0][0] == '':
            continue
          prompt, completion = t.model_inference_str()
          j = {"prompt": prompt, "completion": " " + completion}
          f.write(json.dumps(j))
          f.write('\n')

def read_rollouts(rollouts_filepath: str):
  """Reads rollouts from the pickle file."""
  if os.path.exists(rollouts_filepath):
    with open(rollouts_filepath, 'rb') as f:
      return pickle.load(f)
  else:
    return {}





