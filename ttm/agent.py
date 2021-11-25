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

from ttm import data

class Agent:

    learning_trajectory = []
    rollouts = []

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

    def build_game(self, world_size: float, seed: int = None) -> str:
        """Builds a text-world game at a specific difficulty level, returning the game name."""
        self.append_state(f"build_game: world_size: {world_size} seed: {seed}")
        if not seed:
            seed = random.randint(0, 100000)
        name = f"grounding_game_{world_size}_{seed}.z8"
        subprocess.check_output(
            ["tw-make", "custom", "--world-size", str(world_size), "--nb-objects", "4", "--quest-length", "1", "--seed",
             str(seed), "--output", f"tw_games/{name}"])
        return name

    def write_rollouts(self, rollouts: List[Rollout], game: str, policy: str):
        # write the rollouts data out to pickled versions and flat files for training.
        pickle_path = f"ttm/data/rollouts_{game}_{policy}.pkl"
        txt_path = f"ttm/data/rollouts_{game}_{policy}.txt"
        self.append_state(f"write_rollouts: {txt_path}")
        with open(pickle_path, "wb") as f:
            pickle.dump(rollouts, f)
        data.write_rollouts_text(pickle_path, txt_path)
        return txt_path

    def get_metalearning_action(self, agent, obs, rollout):
        agent.load_inference("ttm/gpt2-metalearn")
        action = agent.predict(obs, rollout)[0].strip()
        print(action)
        self.append_state(action)
        return action

class GPT3Agent(Agent):

    motivating_contexts: List[str] = None
    name: str = None
    prefix: List[str] = None
    parse_regex: str = None

    def __init__(self, agent_goal: str, device=0, path: str="ttm/data/whatcanido"):
        for param in ["prefix", "motivating_examples", "grounding_data"]:
            with open(os.path.join(path, param), "r") as f:
                self.__dict__[param] = [l.strip() for l in f.readlines()]
        for param in ["name", "parse_regex"]:
            with open(os.path.join(path, param), "r") as f:
                self.__dict__[param] = [l.strip().replace('\\\\', '\\') for l in f.readlines()][0]
        super().__init__(agent_goal, device)

    def predict(self, prompt: str):
        """Dispatches a query to GPT-3."""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print("PROMPT>>>>")
        print(prompt)
        response = openai.Completion.create(engine="davinci-instruct-beta", prompt=prompt, max_tokens=500)
        print("RESPONSE>>>>")
        print(response)
        return response

    def __str__(self):
        return pprint.pformat(self.__dict__)

class WhatCanIDo(GPT3Agent):

    def parse(self, response: str):
        response_str = response["choices"][0]["text"]
        response_str = response_str.replace("\n", " ")
        match = re.match(self.parse_regex, response_str)
        if match:
            return match.group(1)
        else:
            if len(response_str) < 200:
                return response_str
            else:
                return ""

    def predict_rollout(self, rollout: Rollout):
        formatted_query = "\n".join(self.prefix).format(goal=rollout.goal, state=rollout["trajectory"].state()[-1])
        response = self.predict(formatted_query)
        return self.parse(response), response, formatted_query

    def predict_state(self, goal: str, state: str):
        formatted_query = "\n".join(self.prefix).format(goal=goal, state=state)
        print(formatted_query)
        response = self.predict(formatted_query)
        return self.parse(response), response, formatted_query

agent_registry = {
    "whatcanido": WhatCanIDo,
    "gpt3": GPT3Agent
}

class TransformerAgent(Agent):
    
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
            prediction = self.predict_sequence(prompt)
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
        
    def predict(self, prompt: str, rollout: Rollout):
        print(prompt)
        return input("> "), "unused_prediction"


        

