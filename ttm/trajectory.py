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
            state = re.sub("[\n\t ]+", " ", state)
            string_repr += f"step {i} state: [{state}] action: ["
        return string_repr

    def state_inference_str(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            string_repr += f"step {i} state: ["
        return string_repr
    
    def __str__(self):
        goal = str(self.goals()[0])
        string_repr = f"goal: [{goal}]\n"
        for i, (state, _, action) in enumerate(self):
            state = re.sub("[\n\t ]+", " ", state)
            string_repr += f"step {i} state: [{state}] action: [{action.strip()}]\n"
        return string_repr

class Rollout(dict):
    
    def __init__(self, trajectory: Trajectory, goal: Goal, scores: List[int]):
        return self.update({"trajectory": trajectory, "goal": goal, "scores": scores})
    
    def hindsight_goal(self, trajectory: Trajectory):
        state = trajectory.states()[-1]
        state = re.sub("[\n\t]", "", state).split(".")[0]
        return state
    
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
    
    
