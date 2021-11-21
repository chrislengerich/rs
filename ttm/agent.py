import gc
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

from ttm import data


class Agent:
    def predict(prompt: str, rollout: Rollout):
        pass

class TransformerAgent(Agent):

    learning_trajectories = Trajectory()
    
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

    def reset_state(self):
        self.learning_trajectories = Trajectory()

    def parse_action(self, generated_text: str, prefix: str):
        split_text = generated_text[len(prefix):]
        return split_text.split(']')[0].strip()
        
    def predict_sequence(self, prompt: str):
        return self.generator(prompt)[0]["generated_text"]

    def append_state(self, action):
        self.learning_trajectories.append(["", self.agent_goal, action])
    
    def predict(self, prompt: str, rollout: Rollout):
        self.append_state("predict_rollout")
        prompt = str(rollout["trajectory"]) + f'state: [{prompt}] action: [ '
        prediction = self.predict_sequence(prompt)
        action = self.parse_action(prediction, prompt)
        return action, prediction
                
    def train(self, output_dir: str, train_path: str = 'rollouts.txt', eval_path: str = 'rollouts.txt', ):
        self.append_state(f"train: train_path: {train_path} eval_path: {eval_path} output_dir: {output_dir}")
        train_dataset = LineByLineTextDataset(self.tokenizer, train_path, block_size=512)
        eval_dataset = LineByLineTextDataset(self.tokenizer, eval_path, block_size=512)

        training_args = TrainingArguments(
            output_dir=output_dir,  # The output directory
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=3,  # number of training epochs
            per_device_train_batch_size=32,  # batch size for training
            per_device_eval_batch_size=64,  # batch size for evaluation
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
        

