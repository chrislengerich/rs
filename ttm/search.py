import os

import openai

from ttm.data import read_rollouts
from ttm.trajectory import Rollout
from typing import List

class Search:

  def search(self, rollout: Rollout, documents: List[Rollout]):
    """Dispatches a query to the OpenAI search service."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    query = rollout["trajectory"].step_model_inference_str(len(rollout["trajectory"][-1]))
    documents_str = []
    for d in documents:
      if d["trajectory"][0][0] == '': # hack to accomodate the old data format.
        print("continuing")
        continue
      for i in range(len(d["trajectory"])):
        documents_str.append(d["trajectory"].step_model_inference_str(i))
    documents_str = documents_str[:200]
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
  for rollout in rollouts.values():
    rollouts_list.extend(rollout)
  query_rollout = rollouts_list[-1]
  print(search.search(query_rollout, rollouts_list))
