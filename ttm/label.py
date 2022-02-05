# add human evaluation labels to determine correctness of a trajectory.

import argparse
import pprint

from ttm.data import label_summaries

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Label steps within a trajectory with human labels.")
  parser.add_argument("--pickle_path", type=str, default=None, help="Path to pickled rollouts")
  parser.add_argument("--run_id", type=int, default=None, help="Format of text file")
  parser.add_argument("--epoch", type=int, default=None, help="Format of text file")
  parser.add_argument("--env", type=str, default="cooking_level_2", help="Format of text file")
  args = parser.parse_args()
  pprint.pprint(label_summaries(args.pickle_path, args.run_id, args.epoch, args.env))