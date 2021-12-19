import re

from typing import Optional
from typing import List
import copy

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

    def model_inference_str(self):
        """Expect the next state and act to explore accordingly."""
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state_obs = dict([item for item in list(state.items()) if item[0] == 'obs'])
            state_obs = self.strip_state(str(state_obs))
            state_others = dict([item for item in list(state.items()) if item[0] != 'obs'])
            state_others = self.strip_state(str(state_others))
            string_repr += f"step {i} state: [{state_obs},"
            if i < len(self) - 1:
                string_repr += f"{state_others}] action: [ {action} ]\n"
        return string_repr
    
    def __str__(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state_str = self.strip_state(str(state))
            string_repr += f"step {i} state: [{state_str}] action: [{action.strip()}]\n"
        return string_repr

class Rollout(dict):
    
    def __init__(self, trajectory: Trajectory, goal: Goal, scores: List[int]):
        return self.update({"trajectory": trajectory, "goal": goal, "scores": scores})
    
    def hindsight_goal(self, trajectory: Trajectory):
        state = trajectory.states()[-1]
        state = re.sub("[\n\t]", "", str(state)).split(".")[0]
        return state

    def fitness(self):
        return self["scores"][-1]
    
    def hindsight_trajectories(self):
        trajectories = []
        for i in range(2,len(self["trajectory"])+1):
            print(f"i={i}")
            if len(self["trajectory"]) >= 2:
                print(f"i={i}")
                new_traj = Trajectory(copy.deepcopy(self["trajectory"][:i]))
                # hindsight_goal = self.hindsight_goal(new_traj)
                # for i in range(len(new_traj)):
                #     new_traj[i][1] = hindsight_goal
                trajectories.append(new_traj)
        print(len(trajectories))
        return trajectories
    
    
