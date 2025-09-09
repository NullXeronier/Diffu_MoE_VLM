"""
Simple model implementations for MC-Planner migration
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class SimpleNetwork:
    """
    Simple network implementation without heavy dependencies
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, output_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize simple weights (normally distributed)
        self.weights_1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
        self.bias_1 = np.zeros(hidden_dim)
        self.weights_2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
        self.bias_2 = np.zeros(output_dim)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        # First layer
        hidden = np.dot(x, self.weights_1) + self.bias_1
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Second layer
        output = np.dot(hidden, self.weights_2) + self.bias_2
        
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make prediction"""
        return self.forward(x)


class GoalEmbedding:
    """
    Simple goal embedding without heavy ML dependencies
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.goal_embeddings = {}
        
    def encode_goal(self, goal: str) -> np.ndarray:
        """Encode goal as embedding vector"""
        if goal in self.goal_embeddings:
            return self.goal_embeddings[goal]
        
        # Simple hash-based embedding
        embedding = self._hash_to_embedding(goal)
        self.goal_embeddings[goal] = embedding
        
        return embedding
    
    def _hash_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding using simple hashing"""
        # Use hash function to generate consistent embedding
        hash_value = hash(text)
        
        # Convert to vector
        np.random.seed(abs(hash_value) % (2**31))
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def similarity(self, goal1: str, goal2: str) -> float:
        """Calculate similarity between two goals"""
        emb1 = self.encode_goal(goal1)
        emb2 = self.encode_goal(goal2)
        
        # Cosine similarity
        return np.dot(emb1, emb2)


class SimpleController:
    """
    Simple controller for action selection
    """
    
    def __init__(self):
        self.action_history = []
        self.goal_progress = {}
        
    def select_action(self, observation: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Select action based on observation and goal"""
        # Simple rule-based action selection
        
        if "mine" in goal.lower():
            return self._mining_action()
        elif "craft" in goal.lower() or "obtain" in goal.lower():
            return self._crafting_action()
        else:
            return self._exploration_action()
    
    def _mining_action(self) -> Dict[str, Any]:
        """Generate mining action"""
        return {
            'movement': 0,
            'camera': np.array([0.0, 0.0]),
            'use': 0,
            'attack': 1,  # mine
            'jump': 0
        }
    
    def _crafting_action(self) -> Dict[str, Any]:
        """Generate crafting action"""
        return {
            'movement': 0,
            'camera': np.array([0.0, 0.0]),
            'use': 1,  # use/craft
            'attack': 0,
            'jump': 0
        }
    
    def _exploration_action(self) -> Dict[str, Any]:
        """Generate exploration action"""
        # Random movement for exploration
        movement = np.random.choice([0, 1, 2, 3])  # forward, back, left, right
        camera = np.random.normal(0, 0.1, 2)
        camera = np.clip(camera, -0.5, 0.5)
        
        return {
            'movement': movement,
            'camera': camera,
            'use': 0,
            'attack': 0,
            'jump': 0
        }
    
    def update_progress(self, goal: str, success: bool):
        """Update goal progress"""
        if goal not in self.goal_progress:
            self.goal_progress[goal] = {'attempts': 0, 'successes': 0}
        
        self.goal_progress[goal]['attempts'] += 1
        if success:
            self.goal_progress[goal]['successes'] += 1
    
    def get_success_rate(self, goal: str) -> float:
        """Get success rate for goal"""
        if goal not in self.goal_progress:
            return 0.0
        
        progress = self.goal_progress[goal]
        if progress['attempts'] == 0:
            return 0.0
        
        return progress['successes'] / progress['attempts']


class ImageProcessor:
    """
    Simple image processing without heavy dependencies
    """
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
    
    def process_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process observation image"""
        if 'rgb' not in obs:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        image = obs['rgb']
        
        # Resize image if needed
        if image.shape[:2] != self.target_size:
            import cv2
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract simple features from image"""
        # Simple color histogram features
        features = []
        
        # Color channel means
        for channel in range(3):
            features.append(np.mean(image[:, :, channel]))
            features.append(np.std(image[:, :, channel]))
        
        # Simple edge detection (gradient magnitude)
        gray = np.mean(image, axis=2)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(edge_magnitude))
        features.append(np.std(edge_magnitude))
        
        return np.array(features)


def create_backbone(backbone_name: str = "simple") -> SimpleNetwork:
    """Create model backbone"""
    if backbone_name == "simple":
        return SimpleNetwork()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def resize_image(image: np.ndarray, target_resolution: Tuple[int, int]) -> np.ndarray:
    """Resize image to target resolution"""
    import cv2
    return cv2.resize(image, target_resolution, interpolation=cv2.INTER_LINEAR)
