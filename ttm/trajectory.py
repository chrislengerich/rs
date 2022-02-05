import re
import datetime

from typing import Optional
from typing import List
import copy
import numpy as np

class Goal:
    def __init__(self, goal: str): 
        self.goal = goal
    
    def __str__(self): 
        return str(self.goal)

class Trajectory(list):
    
    def states(self):
        return [i[0] for i in self]

    def goals(self):
        return [i[1] for i in self]
    
    def actions(self): 
        return [i[2] for i in self]

    def action_inference_str(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.strip_state(str(state))
            string_repr += f"step {i} state: [{state}] action: ["
            if i < len(self) - 1:
                string_repr += f"{action}]\n"
        return string_repr

    def imagination_action_inference_str(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.strip_state(str(state))
            string_repr += f"step {i} state: [{state}] next_state: ["
            if i < len(self) - 1:
                next_state = self.strip_state(str(self[i + 1][0]))
                string_repr += f"{next_state}] action: [ {action} ]\n"
        return string_repr

    def strip_state(self, state: str):
        homogenized_whitespace = re.sub("[\n\t ]+", " ", state)
        return re.sub("[{}]", "", homogenized_whitespace)

    def imagination_action_str(self):
        """Imagine the next state, and act accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            if i < len(self) - 1:
                state = self.strip_state(str(state))
                next_state = self.strip_state(str(self[i + 1][0]))
                string_repr += f"step {i} state: [{state}] next_state: [{next_state}] action: [{action}]\n"
        return string_repr

    def state_inference_str(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            string_repr += f"step {i} state: ["
            if i < len(self) - 1:
                string_repr += f"{state}] action: [{action}]\n"
        return string_repr

    def dict_to_str(self, dict, causal_order=["update", "summary", "expectation", "next_update"]):
        string_repr = ""
        for i,c in enumerate(causal_order):
            if c in dict:
                repr = re.sub("[\'\"\[\]]", "", str(dict[c]))
                string_repr += f"'{c}': '{repr}', "
                if i == len(causal_order) - 1:
                    string_repr = string_repr[:-2]
        return string_repr

    def model_inference_str(self):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=[
                "summary", "next_obs"]))
            if i == len(self) - 2:
                state_others_context = self.strip_state(self.dict_to_str(state_others, causal_order=[
                    "summary"]))
            else:
                state_others_context = "" #self.strip_state(self.dict_to_str(state_others, causal_order=["next_obs"]))
            #state_others_pred = ""
            state_prefix = f"step {i} " # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def obs_summary_t_to_expectation_action_str(self, fitness_str: str = "4"):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=[
                "next_obs"]))
            if i == len(self) - 1:
                state_others_context = f"fitness: '{fitness_str}' " + self.strip_state(self.dict_to_str(
                    state_others, causal_order=[
                    "summary"]))
            else:
                state_others_context = ""  # self.strip_state(self.dict_to_str(state_others, causal_order=["next_obs"]))
            # state_others_pred = ""
            state_prefix = f"step {i} "  # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                string_repr += f"{state_others_context}"
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def hindsight_expectation_str(self, value_str: str = "4", batch_fitness_str=""):
        """Predict what will happen in the future and how valuable it will be, using labels from the future."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            if 'invisible' in state:
                continue

            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            # for key in ["hindsight_summary", "hindsight_length", "value"]:
            #     state_others[key] = ""
            if "hindsight_expectation" not in state_others:
                state_others["hindsight_expectation"] = ""
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=[
                "hindsight_expectation", "hindsight_summary", "hindsight_length", "value"]))
            if i == len(self) - 1:
                state_others_context = f"fitness: '{value_str}' batch_fitness: '{batch_fitness_str}'"
            else:
                state_others_context = self.strip_state(self.dict_to_str(state_others, causal_order=[
                    "hindsight_expectation"]))
            # state_others_pred = ""
            state_prefix = f"step {i} "  # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                string_repr += f"{state_others_context}"
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def hindsight_expectation_str_old(self, value_str: str = "4", batch_fitness_str=""):
        """Predict what will happen in the future and how valuable it will be, using labels from the future."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            if 'invisible' in state:
                continue

            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            for key in ["hindsight_summary", "hindsight_length", "value"]:
                state_others[key] = ""
            if "hindsight_expectation" not in state_others:
                state_others["hindsight_expectation"] = ""
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=[
                "hindsight_expectation", "hindsight_summary", "hindsight_length", "value"]))
            if i == len(self) - 1:
                state_others_context = f"fitness: '{value_str}' batch_fitness: '{batch_fitness_str}'"
            else:
                state_others_context = ""  # self.strip_state(self.dict_to_str(state_others, causal_order=["next_obs"]))
            # state_others_pred = ""
            state_prefix = f"step {i} "  # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                string_repr += f"{state_others_context}"
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def hindsight_labeling_str(self, value_str: str = "4", batch_fitness_str=""):
        """Emit labels from the past."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others["hindsight_expectation"] = ""
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=[
                "hindsight_expectation", "hindsight_summary", "hindsight_length", "value"]))
            if i == len(self) - 1:
                state_others_context = f"fitness: '{value_str}' batch_fitness: '{batch_fitness_str}'"
            else:
                state_others_context = ""  # self.strip_state(self.dict_to_str(state_others, causal_order=["next_obs"]))
            # state_others_pred = ""
            state_prefix = f"step {i} "  # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                string_repr += f"{state_others_context}"
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def expected_observation_key(self, key:str):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=["next_obs",
                                                                                              f"next_{key}"]))
            state_others_context = ""
            #state_others_pred = ""
            state_prefix = f"step {i} " # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def expected_observation(self):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others_pred = self.strip_state(self.dict_to_str(state_others, causal_order=["next_obs"]))
            state_others_context = ""
            #state_others_context = self.strip_state(self.dict_to_str(state_others, causal_order=["update", "summary"]))
            #state_others_pred = ""
            state_prefix = f"step {i} " # unused for now.
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others_context}] action: [ {action} ]\n"
            else:
                completion_str = f" {state_others_pred}] action: [ {action} ]\n"
        return string_repr, completion_str

    def model_expectation_inference_str(self):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others = self.strip_state(self.dict_to_str(state_others, causal_order=["update", "summary", "expectation", "next_update"]))
            string_repr += f"step {i} state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others}] action: [ {action} ]\n"
            else:
                completion_str = f"{state_others}] action: [ {action} ]\n"
        return string_repr, completion_str


    def model_action_inference_str(self):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = self.trim_commas(state)
            if i < len(self) - 6:
                continue
            state_obs = self.strip_state(
                self.dict_to_str(state, causal_order=["obs", "summary", "next_update"]))
            string_repr += f"step {i} state: [{state_obs}] "
            if i < len(self) - 1:
                string_repr += f"action: [ {action} ]\n"
            else:
                completion_str = f"action: [ {action} ]\n"
        return string_repr, completion_str

    def trim_commas(self, state):
        for k in state.keys():
            state[k] = str(state[k])
            #state[k] = str(state[k]).replace(",", " ")
            state[k] = re.sub("[\s][\s]*", " ", state[k])
        return state

    def step_model_inference_str(self, target_i: int):
        """Return a model inference string for the step |target_i|"""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others = self.strip_state(self.dict_to_str(state_others))
            if i == target_i:
                return string_repr + f"step {i} state: [{state_obs},{state_others}] action: [ {action} ]\n"

    def imitation_inference_str(self):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            if i < len(self) - 6:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            string_repr += f"state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"action: [ {action} ]\n"
            else:
                completion_str = f"action: [ {action} ]\n"
        return string_repr, completion_str

    def __str__(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state_str = self.strip_state(str(state))
            string_repr += f"step {i} state: [{state_str}] action: [{action.strip()}]\n"
        return string_repr

class Rollout(dict):

    agent = {}
    timestamp = None
    args = None
    
    def __init__(self, trajectory: Trajectory, goal: Goal, scores: List[int], agent:dict = {}, args=None):
        self.agent = agent
        self.timestamp = datetime.datetime.now()
        self.args = args # context that was passed in to create the rollout
        return self.update({"trajectory": trajectory, "goal": goal, "scores": scores})
    
    def hindsight_goal(self, trajectory: Trajectory):
        state = trajectory.states()[-1]
        state = re.sub("[\n\t]", "", str(state)).split(".")[0]
        return state

    def restore(self, env, offset: int):
        # returns the environment and state context at that time.
        env.reset()
        for a in self["trajectory"].actions()[:offset]:
            env.step(a)
        new_traj = Trajectory()
        new_traj.extend(self["trajectory"][:offset])

        # fork a new rollout from the current rollout, copying the context.
        new_rollout = copy.deepcopy(self)
        new_rollout.timestamp = datetime.datetime.now()
        new_rollout["trajectory"] = new_traj
        new_rollout["scores"] = new_rollout["scores"][:offset]
        return env, new_rollout

    def fitness(self):
        if re.match(".*cooking.*", self.args.env):
            return self["scores"][-1]
        else: # text adventure style
            return sum(self["scores"])

    def unique_ratio(self, dataset):
        return len(set(dataset)) / len(dataset)

    def learning(self):
        action_diversity = self.unique_ratio(self["trajectory"].actions())
        obs_diversity = self.unique_ratio([i["obs"] for i in self["trajectory"].states()])
        obs = [i["obs"] for i in self["trajectory"].states()]
        blank_ratio = len([i for i in self["trajectory"].actions() if i == ""]) / len(self["trajectory"].actions())
        return {"action_diversity": action_diversity, "obs_diversity": obs_diversity, "blank_ratio": blank_ratio,
                "obs_diversity": \
            obs_diversity, "joint": np.mean([action_diversity, obs_diversity, (1- blank_ratio)])}

    def hindsight_trajectory_inference(self, trajectory: Trajectory):
        for s in trajectory:
            s[0]['hindsight_data'] = []

        # label the past.
        for i in range(len(trajectory) - 1, 0, -1):
            if trajectory.states()[i].get('hindsight_summary', '') != '':
                try:
                    length = int(trajectory[i][0]['hindsight_length'])
                except Exception as e:
                    print(e)
                    length = 1
                for j in range(i - length, i):
                    hindsight_data = {
                        'hindsight_summary': trajectory[i][0]['hindsight_summary'],
                        'hindsight_length': length,
                        'hindsight_value': trajectory[i][0]['value']
                    }
                    trajectory[j][0].setdefault(
                        'hindsight_data', []).append(hindsight_data)

        new_traj = copy.deepcopy(trajectory)
        i = len(new_traj)
        for s in new_traj:
            s[0]['hindsight_expectation'] = ''
        for j in range(len(new_traj) - 1):
            if 'hindsight_data' in new_traj[j][0]:
                # summarize and flatten hindsight trajectories which would have been visible up to time t-1.
                visible_summaries = []
                for d in new_traj[j][0]['hindsight_data']:
                    if j + d['hindsight_length'] < i - 1:
                        visible_summaries.append(d)
                new_traj[j][0]['hindsight_expectation'] = " || ".join([d['hindsight_summary'] for d in
                                                                       visible_summaries])
                new_traj[j][0]['hindsight_value'] = np.mean([float(d['hindsight_value']) for d in
                                                                 visible_summaries if isinstance(d['hindsight_value'], int)])

        return new_traj

    def accurate_hindsight(self, state):
        accurate = 'hindsight_accurate' in state and state['hindsight_accurate'] == 1
        return accurate

    def hindsight_trajectory(self, trajectory: Trajectory, unused_format: str):
        trajectory = Trajectory([t for t in trajectory if not 'invisible' in t[0]])
        trajectory.append(({'obs': 'end game'}, trajectory[-1][1], "sequence_end"))

        blank_expectations = (self.agent["name"] != "human")
        if blank_expectations:
            for t in trajectory:
                if not self.accurate_hindsight(t[0]):
                    t[0]['hindsight_summary'] = ''
                    t[0]['hindsight_length'] = ''
                    t[0]['value'] = ''
                t[0]['hindsight_data'] = []
                t[0]['hindsight_expectation'] = ''

        num_obs = 1
        for i in range(len(trajectory) - num_obs):
            # prior: obs
            # current data = update
            composite_obs = " || ".join([trajectory[i + j][0]['obs'] for j in range(1, num_obs + 1)])
            trajectory[i][0]['next_obs'] = composite_obs
            if 'update' in trajectory[i+1][0]:
                composite_obs = " || ".join([str(trajectory[i + j][0]['update']) for j in range(1, num_obs + 1)])
                trajectory[i][0]['next_update'] = composite_obs # add a key for
            if 'summary' in trajectory[i+1][0]:
                composite_obs = " || ".join([trajectory[i + j][0]['summary'] for j in range(1, num_obs + 1)])
                trajectory[i][0]['next_summary'] = composite_obs # add a key for
            if 'expectation' in trajectory[i + 1][0]:
                composite_obs = " || ".join([trajectory[i + j][0]['expectation'] for j in range(1, num_obs + 1)])
                trajectory[i][0]['next_expectation'] = composite_obs  # add a key for

        # label with dynamic-range hindsight trajectories.
        for i in range(len(trajectory)-1,0,-1):

            if trajectory.states()[i].get('hindsight_summary', '') != '':
                try:
                    length = int(trajectory[i][0]['hindsight_length'])
                    for j in range(i-length, i):
                        hindsight_data = {
                            'hindsight_summary': trajectory[i][0]['hindsight_summary'],
                            'hindsight_length': length,
                            'hindsight_value': trajectory[i][0]['value']
                        }
                        trajectory[j][0].setdefault(
                            'hindsight_data', []).append(hindsight_data)
                except Exception as e:
                    print(e)
                    continue

        new_trajectory = copy.deepcopy(trajectory)

        return new_trajectory[:-num_obs]
    
    def hindsight_trajectories(self, format: str =""):
        trajectories = []
        if isinstance(self["trajectory"].states()[0], str):
            return trajectories
        self["trajectory"] = self.hindsight_trajectory(self["trajectory"], format)
        for i in range(2,len(self["trajectory"])+1):
            if len(self["trajectory"]) >= 2:
                new_traj = Trajectory(copy.deepcopy(self["trajectory"][:i]))

                for j in range(len(new_traj) - 1):
                    if 'hindsight_data' in new_traj[j][0]:
                        # summarize and flatten hindsight trajectories which would have been visible up to time t-1.
                        # TODO: also perform this labeling during inference.

                        visible_summaries = []
                        for d in new_traj[j][0]['hindsight_data']:
                            if j + d['hindsight_length'] < i - 1:
                                visible_summaries.append(d)
                        new_traj[j][0]['hindsight_expectation'] = " || ".join([d['hindsight_summary'] for d in
                                                                           visible_summaries])
                        try:
                            new_traj[j][0]['hindsight_value'] = np.mean([float(d['hindsight_value']) for d in
                                                                         visible_summaries])
                        except Exception as e:

                            print(e)
                            new_traj[j][0]['hindsight_value'] = 0.0

                # hindsight label the current state to better predict what will happen in the future.
                current_state_foresight_data = new_traj[-1][0].get('hindsight_data', [])
                new_traj[-1][0]['hindsight_expectation'] = " || ".join([d['hindsight_summary'] for d in current_state_foresight_data])
                try:
                    new_traj[-1][0]['hindsight_value'] = np.mean([float(d['hindsight_value']) for d in
                                                                  current_state_foresight_data])
                except Exception as e:
                    print(e)
                    new_traj[-1][0]['hindsight_value'] = 0.0

                trajectories.append(new_traj)
        return trajectories

    def run_id(self):
        return self.args.run_id

    def epoch_index(self):
        return self.args.epoch_index

class Batch:

    @classmethod
    def accuracy(self, rollouts):
        """Calculate the accuracy of the hindsight summaries of a batch of rollouts."""
        labeled = []
        state_count = 0
        for r in rollouts:
            for t in r["trajectory"]:
                if 'hindsight_summary' in t[0] and t[0]['hindsight_summary'] != '' and 'hindsight_accurate' in t[0]:
                    labeled.append(t)
                state_count += 1
        accurate = len([l for l in labeled if l[0]['hindsight_accurate'] == 1])
        stats = {
            "accurate_count": accurate,
            "labeled_count": len(labeled),
            "labeled": labeled,
            "total": state_count,
            "accuracy": accurate / float(len(labeled)),
            "fraction_labeled": len(labeled) / float(state_count)
        }
        return stats

    @classmethod
    def fitness(self, rollouts):
        """Calculate fitness over the batch of rollouts."""
        fitness = [r.fitness() for r in rollouts]
        num_finetunes = len([s for r in rollouts for s in r["trajectory"] if re.match(".*finetune.*", s[2])])
        num_summaries = len([s for r in rollouts for s in r["trajectory"] if s[0].get('hindsight_summary',
                                                                                      '').strip() != ''])
        num_expectations = len([s for r in rollouts for s in r["trajectory"] if s[0].get('hindsight_expectation',
                                                                                '') != ''])
        if len(rollouts) > 0:
            print(rollouts[0].args)
        for r in rollouts:
            for s in r["trajectory"]:
                if s[0].get('hindsight_expectation', '') != '':
                    print(s)
        for r in rollouts:
            for s in r["trajectory"]:
                if s[0].get('hindsight_summary', '') != '':
                    print(s)

        learning = [r.learning()["joint"] for r in rollouts]
        length = [len(r["trajectory"]) for r in rollouts]
        return {"mean_fitness": np.mean(fitness), "std_fitness": np.std(fitness), "fitness": fitness,
                "mean_learning": np.mean(learning), "std_learning": np.std(learning), "learning": learning,
                "length": length,
                "num_finetunes": num_finetunes,
                "num_summaries": num_summaries,
                "num_expectations": num_expectations}
    
    
