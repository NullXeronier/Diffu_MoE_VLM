"""
MC-Planner Selector Module
Goal selection and horizon planning without Java dependencies
"""

import random
from typing import List, Dict, Any, Optional
import numpy as np


class Selector:
    """
    Goal selector for MC-Planner
    Implements various selection strategies for goal prioritization
    """
    
    def __init__(self, strategy: str = "priority"):
        self.strategy = strategy
        self.goal_priorities = self._initialize_priorities()
        self.selection_history = []
        
    def _initialize_priorities(self) -> Dict[str, float]:
        """Initialize goal priorities"""
        return {
            # Basic resources (high priority)
            "mine_wood": 1.0,
            "mine_stone": 0.9,
            "mine_cobblestone": 0.9,
            
            # Tools (medium-high priority)
            "obtain_wooden_pickaxe": 0.8,
            "obtain_stone_pickaxe": 0.7,
            "obtain_iron_pickaxe": 0.6,
            
            # Advanced materials (medium priority)
            "mine_iron_ore": 0.5,
            "obtain_iron_ingot": 0.5,
            "mine_coal": 0.4,
            
            # Crafted items (lower priority)
            "obtain_wooden_slab": 0.3,
            "obtain_stone_stairs": 0.3,
            "obtain_painting": 0.2,
            
            # Rare materials (variable priority)
            "mine_diamond": 0.1,
        }
    
    def check_precondition(self, goal: str, inventory: Dict[str, int] = None) -> bool:
        """
        Check if preconditions for a goal are met
        """
        if inventory is None:
            return True  # No inventory check
        
        # Define preconditions for common goals
        preconditions = {
            "mine_cobblestone": {"wooden_pickaxe": 1},
            "mine_iron_ore": {"stone_pickaxe": 1},
            "mine_diamond": {"iron_pickaxe": 1},
            "obtain_wooden_slab": {"wooden_planks": 3},
            "obtain_stone_stairs": {"cobblestone": 6},
            "obtain_painting": {"stick": 8, "wool": 1},
            "obtain_wooden_pickaxe": {"wooden_planks": 3, "stick": 2},
            "obtain_stone_pickaxe": {"cobblestone": 3, "stick": 2},
            "obtain_iron_pickaxe": {"iron_ingot": 3, "stick": 2},
        }
        
        goal_preconditions = preconditions.get(goal, {})
        
        for item, required_amount in goal_preconditions.items():
            if inventory.get(item, 0) < required_amount:
                return False
        
        return True
    
    def generate_candidate_goal_list(self, goal_list: List[str], 
                                   inventory: Dict[str, int] = None,
                                   max_candidates: int = 5) -> List[str]:
        """
        Generate candidate goals from the full goal list
        """
        if not goal_list:
            return []
        
        candidates = []
        
        # Filter goals based on preconditions
        for goal in goal_list:
            if self.check_precondition(goal, inventory):
                candidates.append(goal)
        
        # If no goals meet preconditions, include some basic goals
        if not candidates:
            basic_goals = ["mine_wood", "mine_stone", "obtain_wooden_pickaxe"]
            candidates = [goal for goal in basic_goals if goal in goal_list]
        
        # Limit candidates
        return candidates[:max_candidates]
    
    def horizon_select(self, candidate_goal_list: List[str], 
                      inventory: Dict[str, int] = None,
                      context: Dict[str, Any] = None) -> str:
        """
        Select the best goal from candidates using the configured strategy
        """
        if not candidate_goal_list:
            return ""
        
        if len(candidate_goal_list) == 1:
            selected = candidate_goal_list[0]
        else:
            if self.strategy == "priority":
                selected = self._priority_select(candidate_goal_list)
            elif self.strategy == "random":
                selected = self._random_select(candidate_goal_list)
            elif self.strategy == "round_robin":
                selected = self._round_robin_select(candidate_goal_list)
            elif self.strategy == "dependency":
                selected = self._dependency_select(candidate_goal_list, inventory)
            else:
                # Default to priority
                selected = self._priority_select(candidate_goal_list)
        
        # Record selection
        self.selection_history.append(selected)
        
        return selected
    
    def _priority_select(self, candidates: List[str]) -> str:
        """Select based on predefined priorities"""
        best_goal = candidates[0]
        best_priority = self.goal_priorities.get(best_goal, 0.0)
        
        for goal in candidates[1:]:
            priority = self.goal_priorities.get(goal, 0.0)
            if priority > best_priority:
                best_goal = goal
                best_priority = priority
        
        return best_goal
    
    def _random_select(self, candidates: List[str]) -> str:
        """Random selection"""
        return random.choice(candidates)
    
    def _round_robin_select(self, candidates: List[str]) -> str:
        """Round-robin selection to ensure diversity"""
        if not self.selection_history:
            return candidates[0]
        
        # Find candidates not recently selected
        recent_selections = set(self.selection_history[-len(candidates):])
        unselected = [goal for goal in candidates if goal not in recent_selections]
        
        if unselected:
            return unselected[0]
        else:
            return candidates[0]
    
    def _dependency_select(self, candidates: List[str], 
                          inventory: Dict[str, int] = None) -> str:
        """Select based on dependency analysis"""
        if inventory is None:
            return self._priority_select(candidates)
        
        # Score goals based on how many dependencies they fulfill
        dependency_scores = {}
        
        for goal in candidates:
            score = 0
            
            # Check what this goal enables
            if "pickaxe" in goal:
                score += 2  # Tools enable mining
            if "mine" in goal:
                score += 1  # Mining provides materials
            if "wood" in goal:
                score += 1.5  # Wood is fundamental
            
            # Bonus for goals that use current inventory
            if goal.startswith("obtain_"):
                # Check if we have materials for crafting
                if "wooden" in goal and inventory.get("wood", 0) > 0:
                    score += 1
                if "stone" in goal and inventory.get("cobblestone", 0) > 0:
                    score += 1
            
            dependency_scores[goal] = score
        
        # Select highest scoring goal
        best_goal = max(candidates, key=lambda g: dependency_scores.get(g, 0))
        return best_goal
    
    def set_strategy(self, strategy: str):
        """Change selection strategy"""
        valid_strategies = ["priority", "random", "round_robin", "dependency"]
        if strategy in valid_strategies:
            self.strategy = strategy
        else:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
    
    def update_priorities(self, new_priorities: Dict[str, float]):
        """Update goal priorities"""
        self.goal_priorities.update(new_priorities)
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about goal selection"""
        if not self.selection_history:
            return {"total_selections": 0}
        
        from collections import Counter
        selection_counts = Counter(self.selection_history)
        
        return {
            "total_selections": len(self.selection_history),
            "unique_goals": len(selection_counts),
            "most_common": selection_counts.most_common(5),
            "recent_selections": self.selection_history[-10:],
        }
    
    def reset(self):
        """Reset selector state"""
        self.selection_history = []


class HorizonSelector(Selector):
    """
    Advanced selector with horizon planning capabilities
    """
    
    def __init__(self, horizon: int = 3, strategy: str = "priority"):
        super().__init__(strategy)
        self.horizon = horizon
        self.planned_sequence = []
        
    def horizon_select(self, candidate_goal_list: List[str], 
                      inventory: Dict[str, int] = None,
                      context: Dict[str, Any] = None) -> str:
        """
        Select goal considering multi-step horizon
        """
        if not candidate_goal_list:
            return ""
        
        # If we have a planned sequence, continue with it
        if self.planned_sequence:
            next_goal = self.planned_sequence.pop(0)
            if next_goal in candidate_goal_list:
                self.selection_history.append(next_goal)
                return next_goal
            else:
                # Clear invalid sequence
                self.planned_sequence = []
        
        # Plan new sequence
        self.planned_sequence = self._plan_horizon_sequence(
            candidate_goal_list, inventory, self.horizon
        )
        
        if self.planned_sequence:
            selected = self.planned_sequence.pop(0)
            self.selection_history.append(selected)
            return selected
        else:
            # Fallback to single-step selection
            return super().horizon_select(candidate_goal_list, inventory, context)
    
    def _plan_horizon_sequence(self, candidates: List[str], 
                              inventory: Dict[str, int], 
                              horizon: int) -> List[str]:
        """
        Plan a sequence of goals for the given horizon
        """
        if horizon <= 1 or not candidates:
            return candidates[:1]
        
        # Simple horizon planning: prioritize dependency chains
        sequence = []
        remaining_candidates = candidates.copy()
        current_inventory = inventory.copy() if inventory else {}
        
        for step in range(min(horizon, len(candidates))):
            if not remaining_candidates:
                break
            
            # Select best goal for current state
            best_goal = self._select_best_for_state(remaining_candidates, current_inventory)
            sequence.append(best_goal)
            remaining_candidates.remove(best_goal)
            
            # Simulate inventory update
            current_inventory = self._simulate_goal_completion(best_goal, current_inventory)
        
        return sequence
    
    def _select_best_for_state(self, candidates: List[str], 
                              inventory: Dict[str, int]) -> str:
        """Select best goal for current inventory state"""
        # Use dependency-based selection
        return self._dependency_select(candidates, inventory)
    
    def _simulate_goal_completion(self, goal: str, 
                                 inventory: Dict[str, int]) -> Dict[str, int]:
        """Simulate inventory changes after goal completion"""
        new_inventory = inventory.copy()
        
        # Simple simulation of goal outcomes
        if goal == "mine_wood":
            new_inventory["wood"] = new_inventory.get("wood", 0) + 4
        elif goal == "mine_cobblestone":
            new_inventory["cobblestone"] = new_inventory.get("cobblestone", 0) + 3
        elif goal == "obtain_wooden_pickaxe":
            new_inventory["wooden_pickaxe"] = new_inventory.get("wooden_pickaxe", 0) + 1
            # Consume materials
            new_inventory["wooden_planks"] = max(0, new_inventory.get("wooden_planks", 0) - 3)
            new_inventory["stick"] = max(0, new_inventory.get("stick", 0) - 2)
        
        return new_inventory
