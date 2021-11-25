import argparse
import random
import re
import subprocess

import gym  # weirdly, this seems to make the cuda import work correctly.
import textworld.gym
import time
import pickle

from agent import MetalearnedAgent
from trajectory import Rollout, Trajectory

def query(path:str, prompt: str=None, goal: str=None, state: str=None):
  agent = MetalearnedAgent(agent_goal="score=10000", path=path)
  print(agent)
  if prompt:
    return agent.predict(prompt)
  elif goal and state:
    return agent.predict_state(goal, state)
  else: # use the built-in prompt
    print("predicting rollout")
    return agent.predict_rollout(Rollout(Trajectory(), "", []))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test the GPT3 Agent")
  parser.add_argument("--prompt", type=str, default=None, help="Prompt to use for testing")
  parser.add_argument("--goal", type=str, default=None, help="Goal to use for testing")
  parser.add_argument("--state", type=str, default=None, help="State to use for testing")
  parser.add_argument("--path", type=str, default="ttm/data/new_question_policy/", help="path to agent data files")
  args = parser.parse_args()
  print(query(args.path, args.prompt, args.goal, args.state))