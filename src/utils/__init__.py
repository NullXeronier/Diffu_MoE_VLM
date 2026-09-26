"""
Utility functions for MC-Planner migration
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import cv2
from pathlib import Path


def resize_image_numpy(img: np.ndarray, target_resolution: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize image to target resolution"""
    return cv2.resize(img, dsize=target_resolution, interpolation=cv2.INTER_LINEAR)


def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize action values to expected ranges"""
    normalized = action.copy()
    
    # Ensure camera values are in [-1, 1] range
    if 'camera' in normalized:
        camera = np.array(normalized['camera'])
        camera = np.clip(camera, -1.0, 1.0)
        normalized['camera'] = camera
    
    # Ensure discrete actions are integers
    discrete_actions = ['movement', 'use', 'attack', 'jump']
    for key in discrete_actions:
        if key in normalized:
            normalized[key] = int(normalized[key])
    
    return normalized


def calculate_inventory_hash(inventory: Dict[str, int]) -> str:
    """Calculate hash of inventory state for caching"""
    import hashlib
    
    # Sort items for consistent hashing
    sorted_items = sorted(inventory.items())
    inventory_str = str(sorted_items)
    
    return hashlib.md5(inventory_str.encode()).hexdigest()


def format_inventory_description(inventory: Dict[str, int]) -> str:
    """Format inventory for human-readable description"""
    if not inventory:
        return "Empty inventory"
    
    items = []
    for item, count in sorted(inventory.items()):
        if count > 0:
            if count == 1:
                items.append(f"{count} {item}")
            else:
                items.append(f"{count} {item}s")
    
    if not items:
        return "Empty inventory"
    
    return ", ".join(items)


def check_goal_completion(goal: str, inventory: Dict[str, int]) -> bool:
    """Check if a goal is completed based on inventory"""
    # Extract target item from goal
    if goal.startswith("obtain_"):
        target_item = goal.replace("obtain_", "")
    elif goal.startswith("mine_"):
        target_item = goal.replace("mine_", "")
    else:
        target_item = goal
    
    return inventory.get(target_item, 0) > 0


def estimate_goal_difficulty(goal: str, inventory: Dict[str, int]) -> float:
    """Estimate difficulty of achieving a goal given current inventory"""
    # Basic difficulty scoring
    base_difficulties = {
        'mine_wood': 0.1,
        'obtain_wooden_planks': 0.1,
        'obtain_stick': 0.2,
        'obtain_wooden_pickaxe': 0.3,
        'mine_cobblestone': 0.4,
        'obtain_stone_pickaxe': 0.5,
        'mine_iron_ore': 0.6,
        'obtain_iron_ingot': 0.7,
        'obtain_iron_pickaxe': 0.8,
        'mine_diamond': 0.9,
        'obtain_wooden_slab': 0.3,
        'obtain_stone_stairs': 0.4,
        'obtain_painting': 0.8,
    }
    
    base_difficulty = base_difficulties.get(goal, 0.5)
    
    # Adjust based on available resources
    if goal.startswith("obtain_"):
        target_item = goal.replace("obtain_", "")
        
        # Check if we have required materials
        requirements = {
            'wooden_slab': {'wooden_planks': 3},
            'stone_stairs': {'cobblestone': 6},
            'painting': {'stick': 8, 'wool': 1},
            'wooden_pickaxe': {'wooden_planks': 3, 'stick': 2},
            'stone_pickaxe': {'cobblestone': 3, 'stick': 2},
            'iron_pickaxe': {'iron_ingot': 3, 'stick': 2},
        }
        
        if target_item in requirements:
            required_items = requirements[target_item]
            missing_items = 0
            total_items = len(required_items)
            
            for item, required_amount in required_items.items():
                if inventory.get(item, 0) < required_amount:
                    missing_items += 1
            
            # Reduce difficulty if we have required materials
            availability_factor = (total_items - missing_items) / total_items
            base_difficulty *= (1 - availability_factor * 0.5)
    
    return base_difficulty


def create_action_sequence(action_type: str, **kwargs) -> List[Dict[str, Any]]:
    """Create a sequence of actions for common tasks"""
    actions = []
    
    if action_type == "move_forward":
        times = kwargs.get('times', 5)
        for _ in range(times):
            actions.append({
                'movement': 0,  # forward
                'camera': np.array([0.0, 0.0]),
                'use': 0,
                'attack': 0,
                'jump': 0
            })
    
    elif action_type == "mine":
        times = kwargs.get('times', 20)
        for _ in range(times):
            actions.append({
                'movement': 0,
                'camera': np.array([0.0, 0.0]),
                'use': 0,
                'attack': 1,  # mine/attack
                'jump': 0
            })
    
    elif action_type == "use_item":
        actions.append({
            'movement': 0,
            'camera': np.array([0.0, 0.0]),
            'use': 1,  # use/interact
            'attack': 0,
            'jump': 0
        })
    
    elif action_type == "look_around":
        # Look in different directions
        directions = [
            np.array([0.3, 0.0]),   # right
            np.array([0.0, 0.3]),   # up
            np.array([-0.3, 0.0]),  # left
            np.array([0.0, -0.3]),  # down
            np.array([0.0, 0.0])    # center
        ]
        for direction in directions:
            actions.append({
                'movement': 0,
                'camera': direction,
                'use': 0,
                'attack': 0,
                'jump': 0
            })
    
    elif action_type == "no_op":
        times = kwargs.get('times', 1)
        for _ in range(times):
            actions.append({
                'movement': 0,
                'camera': np.array([0.0, 0.0]),
                'use': 0,
                'attack': 0,
                'jump': 0
            })
    
    return actions


def save_experiment_results(results: Dict[str, Any], output_path: str):
    """Save experiment results to file"""
    import json
    from datetime import datetime
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    results['experiment_info'] = {
        'framework': 'MC-Planner Migration',
        'environment': 'Gymnasium',
        'version': '0.1.0'
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration"""
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
