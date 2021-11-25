import argparse

from ttm.data import write_rollouts_text

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Write rollouts pkl file to text")
  parser.add_argument("--pickle_path", type=str, default=None, help="Path to pickled rollouts")
  parser.add_argument("--file_path", type=str, default=None, help="Path to file containing rollouts")
  args = parser.parse_args()
  print(write_rollouts_text(args.pickle_path, args.file_path))