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
    
    def __str__(self): 
        string_repr = ""
        for (state, goal, action) in self:
            string_repr += f"state: [{state}] goal: [{goal}] action: [{action}]\n"
        return string_repr

class Rollout(dict):
    
    def __init__(self, trajectory: Trajectory, goal: Goal, scores: List[int]):
        return self.update({"trajectory": trajectory, "goal": goal, "scores": scores})
    
    def hindsight_goal(self):
        return self["trajectory"].states()[-1]
    
    def hindsight_trajectory(self):
        new_traj = copy.deepcopy(self["trajectory"])
        hindsight_goal = self.hindsight_goal()
        for i in range(len(new_traj)):
            new_traj[i][1] = hindsight_goal
        return new_traj
    
    
