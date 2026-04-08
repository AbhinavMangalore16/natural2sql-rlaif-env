from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SqlAction, SqlObservation, SqlState

class SqlEnvClient(EnvClient[SqlAction, SqlObservation, SqlState]):
    """Client for interacting with the SQL environment server."""
    def _step_payload(self, action: SqlAction) -> dict:
        """Convert a SqlAction into the payload format expected by the server."""
        return {"query": action.query}

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse the server's response payload into a StepResult containing a SqlObservation."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=SqlObservation(
                done=payload.get("done", False),
                reward=payload.get("reward"),
                prompt=obs_data.get("prompt", ""),
                last_execution_result=obs_data.get("last_execution_result", ""),
                remaining_attempts=obs_data.get("remaining_attempts", 0)
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SqlState:
        """Parse the server's response payload into a SqlState containing episode information."""
        return SqlState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "medium"),
            target_answer=payload.get("target_answer", ""),
            max_attempts=payload.get("max_attempts", 5)
        )