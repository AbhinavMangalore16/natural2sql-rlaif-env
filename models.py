from typing import List, Optional
from openenv.core.env_server import Action, Observation, State

class SqlAction(Action):
    """User SQL query."""
    query: str

class SqlObservation(Observation):
    prompt: str
    last_execution_result: str
    remaining_attempts: int

class SqlState(State):
    difficulty: str = "medium"
    target_answer: str = ""
    max_attempts: int = 5