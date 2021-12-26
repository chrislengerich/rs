import gc
import os
import pickle
import pprint

import subprocess
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel, AdamW, \
    DataCollatorForLanguageModeling, AutoModelForCausalLM, LineByLineTextDataset
from transformers import pipeline, set_seed
from trajectory import Trajectory, Rollout
from torch.utils.data import DataLoader, Dataset
import torch
import re
import random
from transformers import get_scheduler
from typing import List
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
from trajectory import Rollout
from transformers import Trainer, TrainingArguments
import openai
import readline
import json

from ttm import data

class Agent:

    learning_trajectory = []
    rollouts = []
    motivating_contexts: List[str] = None
    name: str = None
    prefix: List[str] = None
    parse_regex: str = "(.*)"

    def __init__(self, agent_goal: str, device=0):
        self.agent_goal = agent_goal

    def predict(self, prompt: str, rollout: Rollout):
        pass

    def load_inference(self, path: str):
        pass

    def load_model(self, path: str):
        pass

    def train(self, output_dir: str, train_path: str = 'rollouts.txt', eval_path: str = 'rollouts.txt'):
        pass

    def append_state(self, action):
        self.learning_trajectory.append(["", self.agent_goal, action])

    def reset_state(self, goal, scores):
        self.rollouts.append(Rollout(self.learning_trajectory, goal, scores))
        self.learning_trajectory = []

    def build_game(self, world_size: float, quest_length: int = 1, seed: int = None) -> str:
        """Builds a text-world game at a specific difficulty level, returning the game name."""
        self.append_state(f"build_game: world_size: {world_size} seed: {seed}")
        if not seed:
            seed = random.randint(0, 100000)
        args_hash = "only_last_goal"
        name = f"grounding_game_{world_size}_{seed}_{quest_length}_{args_hash}.z8"
        subprocess.check_output(
            ["tw-make", "custom", "--world-size", str(world_size), "--nb-objects", "4", "--quest-length",
             str(quest_length),
             "--seed",
             str(seed), "--output", f"tw_games/{name}", "--only-last-action"])
        return name

    def build_treasure_hunter(self, level: int = 1) -> str:
        """Builds a text-world game at a specific difficulty level, returning the game name."""
        self.append_state(f"build_treasure_hunter: level: {level}")
        level = 20
        name = f"grounding_game_treasure_hunter_{level}.z8"
        subprocess.check_output(
            ["tw-make", "tw-treasure_hunter", "--level", str(level), "--output", f"tw_games/{name}", "--force"])
        return name

    def write_rollouts(self, rollouts: List[Rollout], game: str, policy: str):
        # write the rollouts data out to pickled versions and flat files for training.
        pickle_path = f"ttm/data/{policy}/grounding_data.pkl"
        txt_path = f"ttm/data/{policy}/grounding_data"
        self.append_state(f"write_rollouts: {txt_path}")
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                old_rollouts = pickle.load(f)
        else:
            old_rollouts = {}
        old_rollouts.setdefault(game, []).extend(rollouts)
        with open(pickle_path, "wb") as f:
            pickle.dump(old_rollouts, f)
        data.write_rollouts_text(pickle_path, txt_path)
        return txt_path, pickle_path

    def get_metalearning_action(self, agent, obs, rollout):
        agent.load_inference("ttm/gpt2-metalearn")
        action = agent.predict(obs, rollout)[0].strip()
        print(action)
        self.append_state(action)
        return action

class GPT3Agent(Agent):

    engine = "curie:ft-personal-2021-12-20-18-02-30"
    engine = "curie:ft-personal-2021-12-20-21-51-56" # 0.34c
    engine = "curie:ft-personal-2021-12-20-22-29-54" # 0.29c
    engine = "curie:ft-personal-2021-12-21-00-09-10" # 0.24c - changed the order of the update steps.
    engine = "curie:ft-personal-2021-12-21-03-51-15" # 0.26c
    engine = "curie:ft-personal-2021-12-21-04-43-50" # 0.34c
    engine = "curie:ft-personal-2021-12-21-05-50-39" # 0.45c - Added 18 data points from environmental selection (starting from 66)
    engine = "curie:ft-personal-2021-12-22-06-34-44" # 0.29c - Imitation learning baseline using training data from ^.
    engine = "curie:ft-personal-2021-12-22-08-45-47" # 0.56c
    engine = "curie:ft-personal-2021-12-22-09-45-32" # 0.28c
    engine = "curie:ft-personal-2021-12-22-22-58-10" # 0.34c
    engine = "curie:ft-personal-2021-12-22-23-34-12" # 0.40c
    engine = "curie:ft-personal-2021-12-23-00-17-19" # 0.45c
    engine = "curie:ft-personal-2021-12-23-03-10-44" # 0.91c - k = 8 sliding window of observations, 131 examples.
    engine = "curie:ft-personal-2021-12-23-03-58-03" # 0.53c - k = 8 imitation learned variant, 131 examples.
    engine = "curie:ft-personal-2021-12-23-23-52-22" # 2.18c - bootstrapped with about 2x the data from
    engine = "curie:ft-personal-2021-12-24-00-11-46" # 2.10c - same as above ^, minus the test data.
    engine = "curie:ft-personal-2021-12-24-04-58-46" # 1.12c - 156 examples, starting to intentionally ask questions.
    engine = "curie:ft-personal-2021-12-24-06-29-10" # 1.24c - 165 examples.
    engine = "curie:ft-personal-2021-12-24-19-53-54" # 1.36c - 181 examples, based on partial exploration towards a
    engine = "curie:ft-personal-2021-12-24-20-54-46" # 1.55c - 199 examples (50 with error conditions).
    engine = "curie:ft-personal-2021-12-25-17-54-43" # 1.57c - 207 examples (50 + 60 with error correction).
    engine = "curie:ft-first-cap-2021-12-25-18-57-54" # 2.67c - 348 examples.
    engine = "curie:ft-first-cap-2021-12-25-21-30-24" # 1.93c - 231 examples.

    # goal.
    # machine-learned variants.
    #engine = "davinci-instruct-beta"

    def __init__(self, agent_goal: str, device=0, path: str="ttm/data/whatcanido"):
        self.path = path
        if path:
            for param in ["prefix", "motivating_examples", "grounding_data", "prefix_ft"]:
                with open(os.path.join(path, param), "r") as f:
                    self.__dict__[param] = [l.strip() for l in f.readlines()]
            for param in ["name", "parse_regex", "engine"]:
                with open(os.path.join(path, param), "r") as f:
                    self.__dict__[param] = [l.strip().replace('\\\\', '\\') for l in f.readlines()][0]
            for param in ["length"]:
                with open(os.path.join(path, param), "r") as f:
                    self.__dict__[param] = int(f.readlines()[0].strip())
            for param in ["keys"]:
                with open(os.path.join(path, param), "r") as f:
                    self.__dict__[param] = json.loads(f.readlines()[0].strip())
        super().__init__(agent_goal, device)

    def build_name(self, name:str):
        return re.sub('[^a-z]', '', name.lower())

    def load_agent(self, prefix: List[str], motivating_examples: List[str], grounding_data: List[str], name: str,
        length: int, parse_regex:str = "([^]]*)\].*"):
        self.prefix = prefix
        self.motivating_examples = motivating_examples
        self.grounding_data = grounding_data
        self.name = "new_" + self.build_name(name)
        self.parse_regex = parse_regex
        self.length = length
        self.path = f"ttm/data/{self.name}"

    def predict(self, prompt: str):
        """Dispatches a query to GPT-3."""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        #engine = "curie:ft-personal-2021-12-20-02-09-16"
        #engine = "curie:ft-personal-2021-12-20-04-07-04"
        #engine = "curie:ft-personal-2021-12-20-05-42-41"
        #engine = "curie:ft-personal-2021-12-20-15-11-50"
        #engine="curie:ft-personal-2021-12-20-16-07-06"
        #engine="curie:ft-personal-2021-12-20-16-56-02"
        #engine="davinci:ft-personal-2021-12-20-17-25-15"
        engine="curie:ft-personal-2021-12-23-19-48-16"
        print("PROMPT>>>>")
        print(prompt)
        max_tokens = self.length # 100 # 500
        print(self.engine)
        if re.match(".*ft-.*", self.engine):
            response = openai.Completion.create(model=self.engine, prompt=prompt, max_tokens=max_tokens,
                                            temperature=1, stop="\n")
        else:
            response = openai.Completion.create(engine=self.engine, prompt=prompt, max_tokens=max_tokens,
                                                temperature=1, stop="\n")
        print("RESPONSE>>>>")
        print(response)
        return response

    def save(self, path: str=None) -> str:
        if path == None:
            path = self.path
        if not os.path.exists(path):
            os.mkdir(path)
        for param in ["prefix", "motivating_examples", "grounding_data"]:
            with open(os.path.join(path, param), "w") as f:
                f.writelines(self.__dict__[param])
        for param in ["name", "parse_regex", "length"]:
            with open(os.path.join(path, param), "w") as f:
                f.write(str(self.__dict__[param]))
        return path

    def __str__(self):
        return pprint.pformat(self.__dict__)

class MetalearnedAgent(GPT3Agent):

    def parse(self, response: str, keys: List[str]) -> str:
        response_str = response["choices"][0]["text"]
        response_str = response_str.replace("\n", " ")
        match = re.match(self.parse_regex, response_str)
        results = dict(zip(keys, len(keys) * [""]))
        if match:
            for i in range(len(keys)):
                results[keys[i]] = match.group(i+1)
            return results
        else:
            # if len(response_str) < 200:
            #     results["action"] = response_str
            # else:
            results["action"] = ""
            return results

    def predict_rollout(self, rollout: Rollout):
        # whatshouldido format - temporary hack to get this to work. tbd: write this as a flat file and get Codex to
        # generate an inference loop.
        rollout_state_str = rollout["trajectory"].state_inference_str()
        rollout_action_str = rollout["trajectory"].action_inference_str()
        model_inference_str, _ = rollout["trajectory"].model_inference_str()
        imagination_action_inference_str = rollout["trajectory"].imagination_action_inference_str()
        imitation_inference_str, _ = rollout["trajectory"].imitation_inference_str()
        agent_type = "finetuned"
        if agent_type == "finetuned":
            prefix = self.prefix_ft
        else:
            prefix = self.prefix
        if self.path == "ttm/data/whatshouldido/":
            state0 = rollout["trajectory"].states()[-2] if len(rollout["trajectory"]) > 1 else ""
            action0 = rollout["trajectory"].actions()[-2] if len(rollout["trajectory"]) > 1 else ""
            formatted_query = "\n".join(prefix).format(goal=rollout["goal"], state1=rollout["trajectory"].states()[-1],
                                                            state0=state0, action0=action0)
        elif self.path == "ttm/data/agent/":
            formatted_query = "\n".join(prefix).format(rollout=rollout_action_str)
        elif self.path == "ttm/data/new_question_policy/":
            formatted_query = "\n".join(prefix).format(rollout=rollout_action_str)
        elif self.path == "ttm/data/new_prefix_policy/":
            question = rollout["trajectory"].states()[-1]
            formatted_query = "\n".join(prefix).format(rollout=rollout_action_str, question=question)
        else:
            print(f"Unknown path for agent data {self.path}")
            formatted_query = "\n".join(prefix).format(
                imagination_action_inference_str=imagination_action_inference_str, rollout_state_str=rollout_state_str,
                rollout_action_str=rollout_action_str, model_inference_str=model_inference_str,
                imitation_inference_str=imitation_inference_str)
        response = self.predict(formatted_query)
        parsed = self.parse(response, keys=self.keys)
        action = parsed["action"]
        action = action[:min(50, len(action))]
        del parsed["action"]
        return action, parsed, formatted_query

    def compress(self, rollouts: dict):
        # Learned policy for compression.
        # stub method to be replaced by a learned policy.
        return list(rollouts.values())[0]

    def cognitive_dissonance(self, rollouts: dict):
        # Returns a signal corresponding to cognitive dissonance of the past trajectory, which triggers the synthesis
        # of a new learning behavior which is parameterized by a name and a handful of examples (can be thought of as
        # changing weights (slow learning) or training data (fast change).
        # This is a hard-coded version of a function that we would like to learn.

        # want: write out a compressed version of the problem, parameterized by the current set of rollouts.
        rollout = list(rollouts.values())[0][0]
        if rollout["scores"][-1] < 1000: # TBD -> rollouts[0][-1][1]:
            action = "new_question_policy"
        else:
            action = "predict"
        #rollout = self.compress(rollouts)
        return action, rollout

    def predict_state(self, goal: str, state: str):
        formatted_query = "\n".join(self.prefix).format(goal=goal, state=state)
        print(formatted_query)
        response = self.predict(formatted_query)
        return self.parse(response), response, formatted_query

agent_registry = {
    "metalearned": MetalearnedAgent,
    "gpt3": GPT3Agent
}

class TransformerAgent(MetalearnedAgent):
    
    def __init__(self, agent_goal:str, device=-1):
        self.device = torch.device('cpu') if device == -1 else torch.device('cuda')
        self.device_no = device
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.agent_goal = agent_goal
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False,
        )

    def load_model(self, path: str="gpt2"):
        self.generator = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(path)

    def load_inference(self, path: str="gpt2"):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        self.generator = pipeline('text-generation', model=path, tokenizer='openai-gpt', device=self.device_no)

    def parse_action(self, generated_text: str, prefix: str):
        split_text = generated_text[len(prefix):]
        return split_text.split(']')[0].strip()
        
    def predict_sequence(self, prompt: str):
        return self.generator(prompt)[0]["generated_text"]

    def predict(self, prompt: str, rollout: Rollout):
        prompt = str(rollout["trajectory"]) + f'state: [{prompt}] action: [ '
        if True:
            prediction = self.predict_rollout(rollout)
            action = self.parse_action(prediction, prompt)
        else:
            prediction = GPT3Agent().predict(prompt)
            # TODO: separate these.
            action = WhatCanIDo().parse(prediction)
        return action, prediction

    def train(self, output_dir: str, train_path: str = 'rollouts.txt', eval_path: str = 'rollouts.txt'):
        self.append_state(f"train: train_path: {train_path} eval_path: {eval_path} output_dir: {output_dir}")
        train_dataset = LineByLineTextDataset(self.tokenizer, train_path, block_size=512)
        eval_dataset = LineByLineTextDataset(self.tokenizer, eval_path, block_size=512)

        training_args = TrainingArguments(
            output_dir=output_dir,  # The output directory
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=3,  # number of training epochs
            per_device_train_batch_size=5, #32,  # batch size for training
            per_device_eval_batch_size=5, #64,  # batch size for evaluation
            eval_steps=400,  # Number of update steps between two evaluations.
            save_steps=800,  # after # steps model is saved
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
        trainer.save_model()
    
class HumanAgent(Agent):
    name = "human"
    engine = ""
        
    def predict(self, prompt: str, rollout: Rollout):
        print(prompt)
        return input("action> "), "unused_prediction"

    def rlinput(self, prompt, prefill=''):
        readline.set_startup_hook(lambda: readline.insert_text(prefill))
        try:
            return input(prompt)  # or raw_input in Python 2
        finally:
            readline.set_startup_hook()

    # returns metalearn action, full action and query.
    def predict_rollout(self, rollout: Rollout):
        try:
            update = input("update:")
        except ValueError:
            update = 0
        prefill_summary = rollout["trajectory"].states()[-2]["summary"] if len(rollout["trajectory"].states()) >= 2 \
            else ""
        summary = self.rlinput("summary:", prefill_summary)
        expectation = input("expectation:")
        state = {"summary": summary, "expectation": expectation, "update": update}
        action = input("action: ")
        return action, state, ""



        

