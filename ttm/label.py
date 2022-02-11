# add human evaluation labels to determine correctness of a trajectory.

import argparse
import pprint

from ttm.data import label_summaries, label_novel_expectation

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Label steps within a trajectory with human labels.")
  parser.add_argument("--pickle_path", type=str, default=None, help="Path to pickled rollouts")
  parser.add_argument("--run_id", type=int, default=None, help="Format of text file")
  parser.add_argument("--epoch", type=int, default=None, help="Format of text file")
  parser.add_argument("--env", type=str, default="cooking_level_2", help="Format of text file")
  parser.add_argument("--partition", type=str, default=None, help="Partitions (comma-separated)")
  parser.add_argument("--type", type=str, default="'summary' or 'novel_expectation'", help="Path to label file")

  args = parser.parse_args()
  partitions = args.partition.split(",")
  if args.type == 'summary':
    pprint.pprint(label_summaries(args.pickle_path, args.run_id, args.epoch, args.env, partitions))
  elif args.type == 'novel_expectation':
    pprint.pprint(label_novel_expectation(args.pickle_path, args.run_id, args.epoch, args.env, partitions))
  else:
    raise ValueError("Invalid type: {}".format(args.type))