import argparse
from trace import pickle

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Read agent registry data")
  parser.add_argument("--path", type=str, default=None, help="Path to pickled registry")

  args = parser.parse_args()

  with open(args.path, 'rb') as f:
    reg = pickle.load(f)
    print(reg.keys())
    import pdb
    pdb.set_trace()