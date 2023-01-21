# rs
Research repo. This is generally WIP code, feel free to ping me for questions.

```
# ttm (Transformer Turing Machine)
conda env create -name <your_env_name> python=3.7
conda activate <your_env_name>
pip install -r requirements.txt



# get an OpenAI beta key from beta.openai.com and add to your ~/.bashrc
# we use OpenAI's GPT-3 fine-tuning for an experimental high-capacity model.
export OPENAI_API_KEY=
export OPENAI_ORGANIZATION=

# download and install Z-machine games
wget https://github.com/BYU-PCCL/z-machine-games/archive/master.zip
gunzip master.zip

# Policies are grouped by folders in the ttm/data subdir
mkdir ttm/data/<your_policy_name>

# play Zork as a human and write the trajectories to the ttm/data/<your_policy_name> subdir
PYTHONPATH=. python ttm/play.py --policy=<your_policy_name> --meta_policy=human --run_id=0 --epoch=0 --env=zork1.z5 --max_rollouts=1 --max_actions=10

# write out the fine-tuning data
PYTHONPATH=. python ttm/write_finetune.py --pickle_path=ttm/data/<your_policy_name>/grounding_data.pkl --finetune_path=ttm/data/<your_policy_name>/grounding_data.jsonl --run_id=0 --format=hindsight_expectation_str --allowed_splits=train --allowed_agent_names=human --partition=teacher --epoch=0

# train the model
openai api fine_tunes.create -t ttm/data/<your_policy_name>/grounding_data.jsonl

# use the model
PYTHONPATH=. python ttm/play.py --policy=<your_policy_name> --meta_policy=<your_policy_name> --run_id=0 --epoch=0 --env=zork1.z5 --max_rollouts=1 --max_actions=10 --partition=student_train

rollouts are partitioned (by order of hierarchy):
* run_id (0,1,2,3, ...). Experiment id for the run.
* epoch (0,1,2,3, ...). Epoch identifier for the run. Loosely corresponds to a single fine-tuning run.
* partition (teacher (human agent), student_train (machine agent, same envs as teacher), student_test (machine agent, unseen envs). This is just a high-level tag which adds metadata for filtering to rollouts and depends on the env and split tags being set correctly during data collection.
* env (zork1.z5, cooking_level_2, etc.) Cooking has level_1,...,level_4 corresponding to levels from https://arxiv.org/pdf/2002.09127.pdf
* split (the cooking env has splits of train/valid/test, for other environments this is a NOOP and we split by environment)
```

