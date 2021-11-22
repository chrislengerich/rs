import gc
import os
import pickle

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
    def predict(self, prompt: str, rollout: Rollout):
        pass

class GPT3Agent(Agent):

    def predict(self, prompt: str):
        """Dispatches a query to GPT-3."""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        print(prompt)
        response = openai.Completion.create(engine="davinci-instruct-beta", prompt=prompt, max_tokens=500)
        print(response)
        return response

class WhatCanIDo(GPT3Agent):

    name = "whatcanido"

    prefix = """
    New example:
    Goal [ Close the trunk ]
    State [ You've just walked into a spare room. The room is well lit. You smell an interesting smell, 
    and follow it to a trunk. The trunk is empty! This is the worst thing that could possibly happen, ever! Were you looking for a workbench? Because look over there, it's a workbench. The workbench is normal. But there isn't a thing on it. ] Action [ query: what can I do? ]
    Answer [ You can open and close the trunk. You can look at the workbench. ]
            
    New example:
    Goal [ Get the camera ]
    State [ You're in a room with a window. There's a camera on the window. ] Action [ query: what can I do? ]
    Answer [ You can get the camera ]
           
    New example:
    Goal [ {goal} ]
    State [ {state} ]
    Answer ["""

    def parse(self, response: str, parse_regex='([^]]*)\].*'):
        response_str = response["choices"][0]["text"]
        response_str = response_str.replace("\n", " ")
        match = re.match(parse_regex, response_str)
        if match:
            return match.group(1)
        else:
            if len(response_str) < 40:
                return response_str
            else:
                return None

    def predict_rollout(self, rollout: Rollout):
        formatted_query = self.prefix.format(goal=rollout.goal, state=rollout["trajectory"].state()[-1])
        response = self.predict(formatted_query)
        return self.parse(response), response, formatted_query

    def predict_state(self, goal: str, state: str):
        formatted_query = self.prefix.format(goal=goal, state=state)
        print(formatted_query)
        response = self.predict(formatted_query)
        return self.parse(response), response, formatted_query

agent_registry = {
    "whatcanido": WhatCanIDo,
    "gpt3": GPT3Agent
}

class TransformerAgent(Agent):

    learning_trajectory = []
    rollouts = []
    
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

    def reset_state(self, goal, scores):
        self.rollouts.append(Rollout(self.learning_trajectory, goal, scores))
        self.learning_trajectory = []

    def parse_action(self, generated_text: str, prefix: str):
        split_text = generated_text[len(prefix):]
        return split_text.split(']')[0].strip()
        
    def predict_sequence(self, prompt: str):
        return self.generator(prompt)[0]["generated_text"]

    def append_state(self, action):
        self.learning_trajectory.append(["", self.agent_goal, action])
    
    def predict(self, prompt: str, rollout: Rollout):
        self.append_state("predict_rollout")
        prompt = str(rollout["trajectory"]) + f'state: [{prompt}] action: [ '
        if False:
            prediction = self.predict_sequence(prompt)
            action = self.parse_action(prediction, prompt)
        else:
            prediction = GPT3Agent().predict(prompt)
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

    def write_rollouts(self, rollouts: List[Rollout], game: str):
        # write the rollouts data out to pickled versions and flat files for training.
        pickle_path = f"ttm/data/rollouts_{game}.pkl"
        txt_path = f"ttm/data/rollouts_{game}.txt"
        self.append_state(f"write_rollouts: {txt_path}")
        with open(pickle_path, "wb") as f:
            pickle.dump(rollouts, f)
        data.write_rollouts_text(pickle_path, txt_path)
        return txt_path
    
class HumanAgent(Agent):
        
    def predict(self, prompt: str, rollout: Rollout):
        print(prompt)
        return input("> "), "unused_prediction"
        

