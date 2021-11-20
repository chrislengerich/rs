from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel, AdamW
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

class Agent:
    def predict(prompt: str, rollout: Rollout):
        pass

x = Agent()
    
class SequenceDataset(Dataset):
    
    def __init__(self, sequences: List[str], agent):
        self.sequences = sequences
        self.agent = agent
        
    def tokenize_sequence(self, sequence: str):
        inputs = self.agent.tokenizer(sequence, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'][:,-self.agent.max_prompt_length:]
        inputs['labels'] = inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask'][:,-self.agent.max_prompt_length:]
        tokens = self.agent.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        # print("INPUT>>> " + self.agent.tokenizer.convert_tokens_to_string(tokens))
        return inputs
    
    def __len__(self): 
        return len(self.sequences)
        
    def __getitem__(self, loc: int):
        return self.tokenize_sequence(self.sequences[loc])
    
def init_models(device=-1):
    return OpenAIGPTTokenizer.from_pretrained('openai-gpt'), pipeline('text-generation', model='gpt2', device=device)

class TransformerAgent(Agent):
    
    def __init__(self, tokenizer, generator, device=-1):
        self.device = torch.device('cpu') if device == -1 else torch.device('cuda')
        self.tokenizer = tokenizer
        self.generator = generator
        self.model = self.generator.model # OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.max_prompt_length = 400
        self.max_output_length = 10
        self.return_sequences = 4
        set_seed(42)
        
        # fixed training configuration.
        self.training_epochs = 1
        #modules = [module for module in self.model.modules() if not isinstance(module, torch.nn.Sequential)]
        #fine_tuned_params = list(list(modules[0].modules())[0].modules())[-1].parameters()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-6)
        
    def predict_sequence(self, prompt: str):
        self.generator.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'][:,-self.max_prompt_length:]
        inputs['attention_mask'] = inputs['attention_mask'][:,-self.max_prompt_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        prompt_clipped = self.tokenizer.convert_tokens_to_string(tokens)
        print(f"prompt clipped: {prompt_clipped}")
        print(f"prompt ids: {inputs['input_ids']}")
        results = self.generator(prompt_clipped, max_length=512, num_return_sequences=self.return_sequences)
        results = [r['generated_text'][len(prompt_clipped):len(prompt_clipped) + 40] for r in results]
        print(results)
        choice = random.randint(0,self.return_sequences - 1)
        print(choice)
        text = results[choice]
        text = " ".join(text.split()[:self.max_output_length])
        text = text.split(".")[0]
        text = text.split("]")[0]
        print(f"action: {text}")
        return text
    
    def predict_rollout(self, prompt: str, rollout: Rollout):
        prompt = str(rollout["trajectory"]) + f'state: [{prompt}] goal: [{rollout["goal"]}] action: [ '
        return self.predict_sequence(prompt)
    
    def rollout_dataloader(self, rollouts: List[Rollout]):
        sequences = [str(rollout.hindsight_trajectory()) for rollout in rollouts]
        return self.dataloader(sequences)
    
    def sequence_dataloader(self, sequences: List[str]):
        dataset = SequenceDataset(sequences, self)
        return DataLoader(dataset, shuffle=False, batch_size=1)
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader):
        """Run fine-tuning, given two dataloaders which output sequences."""
        model = self.generator.model
        num_training_steps = self.training_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=10,
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(self.training_epochs):
            for i, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if i == 0:
                    print("BATCH>>>" + str(batch))
                self.optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                if i == 0:
                    print(f"Train loss: {loss}")
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                self.optimizer.zero_grad()
                print(f"Eval loss: {loss}")
                
    def trainer(self, train_dataloader: DataLoader, eval_dataloader: DataLoader): 
        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned", #The output directory
            overwrite_output_dir=True, #overwrite the content of the output directory
            num_train_epochs=10, # number of training epochs
            per_device_train_batch_size=1, # batch size for training
            per_device_eval_batch_size=1,  # batch size for evaluation
            eval_steps = 1, # Number of update steps between two evaluations.
            save_steps=10000, # after # steps model is saved 
            warmup_steps=100, #500,# number of warmup steps for learning rate scheduler
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=self.generator.model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=eval_dataloader.dataset,
        )
        trainer.train()

    def train_rollout(self, train_rollouts: List[Rollout], eval_rollouts: List[Rollout]):
        """Run a short fine-tuning of the model on a rollout batch."""
        train_dataloader = self.rollout_dataloader(train_rollouts)
        eval_dataloader = self.rollout_dataloader(eval_rollouts)
        self.train(train_dataloader, eval_dataloader)
                
    def train_sequence(self, train_sequences: List[str], eval_sequences: List[str]):
        """Run fine-tuning on sequences."""
        train_dataloader = self.sequence_dataloader(train_sequences)
        eval_dataloader = self.sequence_dataloader(eval_sequences)
        #self.trainer(train_dataloader, eval_dataloader)
        self.train(train_dataloader, eval_dataloader)
    
class HumanAgent(Agent):
        
    def predict(self, prompt: str, rollout: Rollout):
        print(prompt)
        return input("> ")
        

