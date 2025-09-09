"""
MC-Planner Planning Module
Migrated to support local LLM and remove OpenAI dependencies
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path


class Planner:
    """
    Planning module using local LLM instead of OpenAI API
    """
    
    def __init__(self):
        self.dialogue = ''
        self.logging_dialogue = ''
        
        # Load configuration
        self.data_dir = Path(__file__).parent / "data"
        self.goal_lib = self.load_goal_lib()
        self.supported_objects = self.get_supported_objects(self.goal_lib)
        
        # Local LLM configuration
        self.llm_api_base = os.getenv("LLM_API_BASE", "http://localhost:9999/v1")
        self.llm_model = os.getenv("LLM_MODEL", "meta-llama/Llama-3.2-1B")
        self.llm_api_key = os.getenv("LLM_API_KEY", "DUMMY")
        
    def reset(self):
        """Reset planner state"""
        self.dialogue = ''
        self.logging_dialogue = ''

    def check_llm_server_status(self) -> bool:
        """Check if LLM server is running"""
        try:
            response = requests.get(
                f"{self.llm_api_base}/v1/models",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def load_goal_lib(self) -> Dict:
        """Load goal library"""
        goal_lib_file = self.data_dir / "goal_lib.json"
        if goal_lib_file.exists():
            with open(goal_lib_file, 'r') as f:
                return json.load(f)
        else:
            # Default goal library
            return {
                "obtain_wooden_slab": {
                    "type": "craft",
                    "requirements": ["wooden_planks"],
                    "description": "Craft wooden slab from wooden planks"
                },
                "obtain_stone_stairs": {
                    "type": "craft", 
                    "requirements": ["cobblestone"],
                    "description": "Craft stone stairs from cobblestone"
                },
                "obtain_painting": {
                    "type": "craft",
                    "requirements": ["stick", "wool"],
                    "description": "Craft painting from sticks and wool"
                },
                "mine_cobblestone": {
                    "type": "mine",
                    "requirements": ["wooden_pickaxe"],
                    "description": "Mine cobblestone with pickaxe"
                },
                "mine_iron_ore": {
                    "type": "mine",
                    "requirements": ["stone_pickaxe"], 
                    "description": "Mine iron ore with stone pickaxe"
                },
                "mine_diamond": {
                    "type": "mine",
                    "requirements": ["iron_pickaxe"],
                    "description": "Mine diamond with iron pickaxe"
                }
            }
    
    def get_supported_objects(self, goal_lib: Dict) -> Dict:
        """Get supported objects from goal library"""
        supported_objs = {}
        for key, value in goal_lib.items():
            obj_name = key.replace("obtain_", "").replace("mine_", "")
            supported_objs[obj_name] = value.get("type", "unknown")
        return supported_objs
    
    def load_initial_planning_prompt(self, group: str) -> str:
        """Load initial planning prompt"""
        task_prompt_file = self.data_dir / "task_prompt.txt"
        if task_prompt_file.exists():
            with open(task_prompt_file, 'r') as f:
                context = f.read()
        else:
            context = self.get_default_task_prompt()
        
        replan_prompt_file = self.data_dir / "deps_prompt.txt"
        if replan_prompt_file.exists():
            with open(replan_prompt_file, 'r') as f:
                context += "\n" + f.read()
        else:
            context += "\n" + self.get_default_replan_prompt()
        
        return context
    
    def get_default_task_prompt(self) -> str:
        """Get default task prompt"""
        return """You are an expert Minecraft player tasked with planning actions to achieve specific goals.

Given a task, break it down into a sequence of achievable sub-goals. Consider:
1. What items are needed
2. What tools are required  
3. The order of operations
4. Dependencies between goals

Respond with a clear, step-by-step plan.

Example task: "How to obtain wooden slab?"
Plan:
1. Find trees
2. Mine wood
3. Craft wooden planks
4. Craft wooden slab

"""
    
    def get_default_replan_prompt(self) -> str:
        """Get default replan prompt"""
        return """When replanning, consider the current inventory state and adjust the plan accordingly.
Focus on the most efficient path to achieve the goal given current resources.

"""
    
    def load_parser_prompt(self) -> str:
        """Load parser prompt"""
        parse_prompt_file = self.data_dir / "parse_prompt.txt"
        if parse_prompt_file.exists():
            with open(parse_prompt_file, 'r') as f:
                return f.read()
        else:
            return "Parse the following text and extract actionable goals:"
    
    # def _post_completion(self, prompt_text: str, temperature: float = 0.0, 
    #                     max_tokens: int = 512, stop: Optional[List[str]] = None) -> str:
    #     """Post completion request to local LLM"""
    #     try:
    #         response = requests.post(
    #             f"{self.llm_api_base}/completions",
    #             headers={
    #                 "Authorization": f"Bearer {self.llm_api_key}",
    #                 "Content-Type": "application/json"
    #             },
    #             json={
    #                 "model": self.llm_model,
    #                 "prompt": prompt_text,
    #                 "temperature": temperature,
    #                 "max_tokens": max_tokens,
    #                 "stop": stop
    #             },
    #             timeout=30
    #         )
            
    #         if response.status_code == 200:
    #             result = response.json()
    #             return result.get("choices", [{}])[0].get("text", "").strip()
    #         else:
    #             print(f"[Warning] LLM API error: {response.status_code}")
    #             return self._fallback_response(prompt_text)
                
    #     except Exception as e:
    #         print(f"[Warning] LLM API connection failed: {e}")
    #         return self._fallback_response(prompt_text)

    """
    def _post_completion(self, prompt_text: str, temperature: float = 0.0, 
                        max_tokens: int = 512, stop: Optional[List[str]] = None) -> str:
        #Post completion request to local LLM with improved error handling
        
        # 재시도 로직 추가
        max_retries = 3
        base_timeout = 60  # 타임아웃 증가
        
        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (attempt + 1)  # 점진적 타임아웃 증가
                
                response = requests.post(
                    f"{self.llm_api_base}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.llm_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt_text
                            }
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stop": stop,
                        "do_sample": True if temperature > 0 else False,
                        "stream": False
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    print(f"[Warning] LLM API error: {response.status_code} - {response.text}")
                    if attempt == max_retries - 1:
                        return self._fallback_response(prompt_text)
                    
            except requests.exceptions.Timeout:
                print(f"[Warning] LLM API timeout on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    return self._fallback_response(prompt_text)
                    
            except Exception as e:
                print(f"[Warning] LLM API connection failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._fallback_response(prompt_text)
        
        return self._fallback_response(prompt_text)
    """
    def _post_completion(self, prompt_text: str, temperature: float = 0.0, 
                    max_tokens: int = 512, stop: Optional[List[str]] = None) -> str:
        """Post completion request to local LLM"""
        try:
            response = requests.post(
                f"{self.llm_api_base}/chat/completions",  # Changed from /completions
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_model,
                    "messages": [  # Changed to messages format
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop,
                    "do_sample": True if temperature > 0 else False,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Changed to access message content instead of text
                return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                print(f"[Warning] LLM API error: {response.status_code} - {response.text}")
                return self._fallback_response(prompt_text)
                
        except Exception as e:
            print(f"[Warning] LLM API connection failed: {e}")
            return self._fallback_response(prompt_text)

    
    def _fallback_response(self, prompt_text: str) -> str:
        """Fallback response when LLM is unavailable"""
        # Simple rule-based fallback
        if "wooden_slab" in prompt_text.lower():
            return "1. Find trees\\n2. Mine wood\\n3. Craft wooden planks\\n4. Craft wooden slab"
        elif "stone_stairs" in prompt_text.lower():
            return "1. Find stone\\n2. Mine cobblestone\\n3. Craft stone stairs"
        elif "painting" in prompt_text.lower():
            return "1. Find trees\\n2. Mine wood\\n3. Craft wooden planks\\n4. Craft sticks\\n5. Find sheep\\n6. Get wool\\n7. Craft painting"
        else:
            return "1. Assess current situation\\n2. Gather required materials\\n3. Craft or mine target item"
    
    def query_llm(self, prompt_text: str) -> str:
        """Query local LLM"""
        return self._post_completion(prompt_text)
    
    def online_parser(self, text: str) -> List[str]:
        """Parse text to extract goals"""
        parser_prompt = self.load_parser_prompt()
        full_prompt = f"{parser_prompt}\\n\\nText: {text}\\n\\nGoals:"
        
        response = self.query_llm(full_prompt)
        
        # Simple parsing - extract lines that look like goals
        goals = []
        for line in response.split('\\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or 
                        any(word in line.lower() for word in ['obtain', 'mine', 'craft', 'get'])):
                # Clean up the line
                goal = line.lstrip('- *').strip()
                if goal:
                    goals.append(goal)
        
        return goals[:5]  # Limit to 5 goals
    
    def check_object(self, obj: str) -> bool:
        """Check if object is supported"""
        return obj in self.supported_objects
    
    def generate_goal_list(self, plan: str) -> List[str]:
        """Generate goal list from plan"""
        goals = self.online_parser(plan)
        
        # Filter and normalize goals
        filtered_goals = []
        for goal in goals:
            # Normalize goal format
            goal_lower = goal.lower()
            
            # Map common patterns to standard format
            if "wood" in goal_lower and "slab" in goal_lower:
                filtered_goals.append("obtain_wooden_slab")
            elif "stone" in goal_lower and "stairs" in goal_lower:
                filtered_goals.append("obtain_stone_stairs")
            elif "painting" in goal_lower:
                filtered_goals.append("obtain_painting")
            elif "cobblestone" in goal_lower:
                filtered_goals.append("mine_cobblestone")
            elif "iron" in goal_lower and "ore" in goal_lower:
                filtered_goals.append("mine_iron_ore")
            elif "diamond" in goal_lower:
                filtered_goals.append("mine_diamond")
            else:
                # Keep original goal if it looks valid
                if any(keyword in goal_lower for keyword in ['obtain', 'mine', 'craft']):
                    filtered_goals.append(goal)
        
        return filtered_goals
    
    def initial_planning(self, group: str, task_question: str) -> str:
        """Generate initial plan for task"""
        prompt = self.load_initial_planning_prompt(group)
        full_prompt = f"{prompt}\\n\\nTask: {task_question}\\nPlan:"
        
        plan = self.query_llm(full_prompt)
        
        # Update dialogue for context
        self.dialogue += f"Task: {task_question}\\nPlan: {plan}\\n\\n"
        self.logging_dialogue += f"Task: {task_question}\\nPlan: {plan}\\n\\n"
        
        return plan
    
    def generate_inventory_description(self, inventory: Dict[str, int]) -> str:
        """Generate inventory description"""
        if not inventory:
            return "The inventory is empty."
        
        items = []
        for item, count in inventory.items():
            if count > 1:
                items.append(f"{count} {item}s")
            else:
                items.append(f"1 {item}")
        
        return f"Current inventory: {', '.join(items)}."
    
    def generate_success_description(self, step: str) -> str:
        """Generate success description"""
        return f"Successfully completed: {step}"
    
    def generate_failure_description(self, step: str) -> str:
        """Generate failure description"""
        return f"Failed to complete: {step}"
    
    def generate_explanation(self) -> str:
        """Generate explanation of current plan"""
        if not self.dialogue:
            return "No plan generated yet."
        
        explanation_prompt = f"Explain the following plan:\\n\\n{self.dialogue}\\n\\nExplanation:"
        return self.query_llm(explanation_prompt)
    
    def replan(self, task_question: str, inventory_desc: str = "") -> str:
        """Replan based on current state"""
        context = f"Previous dialogue:\\n{self.dialogue}\\n"
        if inventory_desc:
            context += f"Current situation: {inventory_desc}\\n"
        
        replan_prompt = f"""Given the current situation, create a new plan for: {task_question}

{context}

New plan:"""
        
        new_plan = self.query_llm(replan_prompt)
        
        # Update dialogue
        self.dialogue += f"Replan for: {task_question}\\nNew plan: {new_plan}\\n\\n"
        self.logging_dialogue += f"Replan for: {task_question}\\nNew plan: {new_plan}\\n\\n"
        
        return new_plan
