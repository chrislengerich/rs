import pickle

with open('rollouts.pkl', 'rb') as f:
    rollouts = pickle.load(f)

with open('rollouts.txt', 'w') as f:
  for r in rollouts:
    trajs = r.hindsight_trajectories()
    for t in trajs:
      line = str(t)
      f.write(line)
      f.write('\n')


