import pickle

def write_rollouts_text(rollouts_filepath='rollouts.pkl', filename='rollouts.txt'):
    """
    Write rollouts to a text file.
    """
    with open(rollouts_filepath, 'rb') as f:
      rollouts = pickle.load(f)
    with open(filename, 'w') as f:
      for r in rollouts:
        trajs = r.hindsight_trajectories()
        for t in trajs:
          line = str(t)
          f.write(line)
          f.write('\n')



