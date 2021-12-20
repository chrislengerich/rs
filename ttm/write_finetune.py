import argparse

from ttm.data import write_rollouts_finetune

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Write rollouts pkl file to text")
  parser.add_argument("--pickle_path", type=str, default=None, help="Path to pickled rollouts")
  parser.add_argument("--format", type=str, default="imagination_action_str", help="Format of text file")
  parser.add_argument("--finetune_path", type=str, default=None, help="Path to file for fine-tuning data")
  args = parser.parse_args()
  print(write_rollouts_finetune(args.pickle_path, args.finetune_path, args.format))