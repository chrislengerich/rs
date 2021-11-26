import pickle

def write_rollouts_text(rollouts_filepath='rollouts.pkl', filename='rollouts.txt'):
    """
    Write rollouts to a text file, optionally compressing them for training.
    """
    rollouts_dict = read_rollouts(rollouts_filepath)
    with open(filename, 'w') as f:
      for game, rollouts in rollouts_dict.items():
        for r in rollouts:
          print(r)
          trajs = r.hindsight_trajectories()
          print(len(trajs))
          print(filename)
          for t in trajs:
            line = str(t)
            print(line)
            f.write(line)
            f.write('\n')

def read_rollouts(rollouts_filepath: str):
  """Reads rollouts from the pickle file."""
  with open(rollouts_filepath, 'rb') as f:
    rollouts_dict = pickle.load(f)
  return rollouts_dict





