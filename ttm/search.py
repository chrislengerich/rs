import os
import re

import openai

from ttm.data import read_rollouts
from ttm.trajectory import Rollout
from typing import List

class Search:

  def search(self, rollout: Rollout, documents: List[Rollout], query: str, question: str, use_search_results=False):
    """Dispatches a query to the OpenAI search service for question after pre-filtering rollouts on query."""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    documents_str = []
    documents_map = []
    for d in documents:
      if d["trajectory"][0][0] == '' or d["trajectory"].goals()[0] != rollout["trajectory"].goals()[0]:
        print("continuing")
        continue

      new_traj = d.hindsight_trajectory_inference(d["trajectory"])

      # build out the context windows to search on.
      for i,t in enumerate(new_traj):

        # show search +/- 2 context lines for the search.
        candidate_str = ""
        for j in range(max(0, i-3), i+1):
          candidate_str += 'state: ' + str(new_traj[j][0]['obs']).strip() + '\naction: ' + str(new_traj[j][2]) + '\n\n'

        # TODO: this search functionality is a learned metric based on utility to the problem, not similarity.
        if (not re.match(".*question_data.*", str(t[0])) and not use_search_results) and (re.match(f".*{query}.*",
                                                                                               candidate_str,
                                                                                               flags=re.DOTALL)):
          t[0]["step"] = i
          documents_str.append(candidate_str)
          documents_map.append(t)
    documents_str = documents_str[:30]

    print("query>>>")
    print(question)
    response = openai.Engine("davinci").search(
      documents=documents_str,
      query=question
    )
    print("response>>>")
    response = response["data"]
    for rank in response:
      rank["full_document"] = documents_map[rank["document"]]
      rank["document_str"] = documents_str[rank["document"]]
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
  print(search.search(query_rollout, rollouts_list, "lantern", "am I carrying a lantern?"))
