import argparse

from ttm.data import write_rollouts_finetune, get_args

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Write rollouts pkl file to text")
  parser.add_argument("--pickle_path", type=str, default=None, help="Path to pickled rollouts")
  parser.add_argument("--format", type=str, default="model_inference_str", help="Format of text file")
  parser.add_argument("--run_id", type=int, default=0, help="Run id")
  parser.add_argument("--epoch", type=int, default=0, help="Training epoch")
  parser.add_argument("--partition", type=str, default="teacher", help="One of teacher, student_train, student_test")
  parser.add_argument("--hindsight_fitness_current_batch", type=str, default="1", help="1 or 0. Label current epoch "
                                                                                       "with hindsight labels")
  parser.add_argument("--allowed_splits", type=str, default="train", help="Allowed splits (train, valid, test), "
                                                                          "comma-separated")
  parser.add_argument("--allowed_agent_names", type=str, default="human", help="agent or human, comma-separated")
  parser.add_argument("--finetune_path", type=str, default=None, help="Path to file for fine-tuning data")
  args = parser.parse_args()

  args.hindsight_fitness_current_batch = bool(int(args.hindsight_fitness_current_batch))
  print(args.hindsight_fitness_current_batch)
  args.allowed_agent_names = args.allowed_agent_names.split(",")
  args.allowed_splits = args.allowed_splits.split(",")
  epoch_args = get_args(args.pickle_path, args.run_id, args.epoch, args.partition)
  print(write_rollouts_finetune(args.pickle_path, args.finetune_path, args.format,
                                current_args=epoch_args,
                                hindsight_fitness_current_batch=args.hindsight_fitness_current_batch,
                                allowed_splits=args.allowed_splits,
                                allowed_agent_names=args.allowed_agent_names))