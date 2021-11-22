import argparse
import random
import re
import subprocess

import gym  # weirdly, this seems to make the cuda import work correctly.
import textworld.gym
import time
import pickle

from agent import agent_registry

def query(agent:str, prompt: str=None, goal: str=None, state: str=None):
  agent = agent_registry[agent]()
  print(agent)
  if prompt:
    return agent.predict(prompt)
  elif goal and state:
    return agent.predict_state(goal, state)
  else:
    raise Exception("Must provide either prompt or goal and state")



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Test the GPT3 Agent")
  parser.add_argument("--prompt", type=str, default=None, help="Prompt to use for testing")
  parser.add_argument("--goal", type=str, default=None, help="Goal to use for testing")
  parser.add_argument("--state", type=str, default=None, help="State to use for testing")
  parser.add_argument("--agent", type=str, default="whatcanido", help="agent classname")
  args = parser.parse_args()
  print(query(args.agent, args.prompt, args.goal, args.state))