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

    def dict_to_str(self, dict):
        causal_order = ["update", "summary", "expectation"]
        string_repr = ""
        for i,c in enumerate(causal_order):
            if c in dict:
                repr = re.sub("[\'\"]", "", str(dict[c]))
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
            state_others = self.strip_state(self.dict_to_str(state_others))
            string_repr += f"step {i} state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others}] action: [ {action} ]\n"
            else:
                completion_str = f" {state_others}] action: [ {action} ]\n"
        return string_repr, completion_str

    def trim_commas(self, state):
        for k in state.keys():
            state[k] = str(state[k]).replace(",", " ")
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
            # TODO(experiment with a summary version).
            if i < len(self) - 8:
                continue
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            string_repr += f"step {i} state: [{state_obs},"
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
    
    def __init__(self, trajectory: Trajectory, goal: Goal, scores: List[int], agent:dict = {}):
        self.agent = agent
        self.timestamp = datetime.datetime.now()
        return self.update({"trajectory": trajectory, "goal": goal, "scores": scores})
    
    def hindsight_goal(self, trajectory: Trajectory):
        state = trajectory.states()[-1]
        state = re.sub("[\n\t]", "", str(state)).split(".")[0]
        return state

    def fitness(self):
        return self["scores"][-1]

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
    
    def hindsight_trajectories(self):
        trajectories = []
        for i in range(2,len(self["trajectory"])+1):
            #print(f"i={i}")
            if len(self["trajectory"]) >= 2:
                #print(f"i={i}")
                new_traj = Trajectory(copy.deepcopy(self["trajectory"][:i]))
                # hindsight_goal = self.hindsight_goal(new_traj)
                # for i in range(len(new_traj)):
                #     new_traj[i][1] = hindsight_goal
                trajectories.append(new_traj)
        #print(len(trajectories))
        return trajectories
    
    
