import os
import re

import openai

from ttm.data import read_rollouts
from ttm.trajectory import Rollout
from typing import List

class Search:

  def search(self, rollout: Rollout, documents: List[Rollout]):
    """Dispatches a query to the OpenAI search service."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    filter_word = "lantern" # rollout["trajectory"].step_model_inference_str(len(rollout["trajectory"][-1]))
    query = "Am I carrying a lantern?"

    documents_str = []
    for d in [documents[3]]:
      if d["trajectory"][0][0] == '' or d["trajectory"].goals()[0] != rollout["trajectory"].goals()[0]:
        print("continuing")
        continue

      new_traj = d.hindsight_trajectory_inference(d["trajectory"])
      for t in new_traj:
        documents_str.append('state: ' + str(t[0]) + ' action: ' + str(t[2]))
        # for i in range(len(d["trajectory"])):
        # hindsight_string = new_rollout.hindsight_expectation_str()
        # documents_str.append(hindsight_string[0] + hindsight_string[1])

    # TODO: this search functionality is a learned metric based on utility to the problem, not similarity.
    documents_str = [d for d in documents_str if re.match(f".*{filter_word}.*", d, flags=re.DOTALL)]
    for d in documents_str:
      print("\n")
      print(d)
    documents_str = documents_str[:30]

    print("query>>>")
    print(query)
    response = openai.Engine("davinci").search(
      documents=documents_str,
      query=query
    )
    print("response>>>")
    response = response["data"]
    for rank in response:
      rank["document_text"] = documents_str[rank["document"]]
    response = sorted(response, key = lambda x: x["score"], reverse=True)
    return response

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--rollouts_path", type=str, required=True)
  args = parser.parse_args()

  search = Search()
  rollouts = read_rollouts(args.rollouts_path)
  rollouts_list = []
  for key, val in rollouts.items():
    if re.match(".*zork1.z5.*", key):
      rollouts_list.extend(val)
  query_rollout = rollouts_list[3]
  print(search.search(query_rollout, rollouts_list))
