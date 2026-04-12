import asyncio
import os
import textwrap
from typing import List, Optional

# MANDATORY: Use the sync OpenAI client as per example
from openai import OpenAI

from client import SqlEnvClient
from models import SqlAction

# --- CONFIGURATION (Following Example Strictly) ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME","gpt-4.1-mini")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # Optional for your setup

BENCHMARK = "natural2sql"
MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.80  

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert SQL engineer. Convert the user's question into a valid SQLite query.
    Rules:
    1. Use the provided schema.
    2. Output ONLY the query inside square brackets, e.g., [SELECT * FROM table;].
    3. If you receive an error message, fix your previous query.
    """
).strip()

# --- STDOUT FORMAT (Mandatory) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    safe_score = min(max(score, 0.010), 0.990)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={safe_score:.3f} rewards={rewards_str}", flush=True)

# --- LLM LOGIC ---

def get_model_message(client: OpenAI, prompt: str, schema: str, feedback: str) -> str:
    user_prompt = f"Schema: {schema}\nTask: {prompt}\nLast Feedback: {feedback}\nGenerate SQL:"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SELECT 1;" # Fallback

# --- MAIN EXECUTION ---

async def main() -> None:
    if not API_KEY:
        raise ValueError("MANDATORY: HF_TOKEN/API_KEY must be set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Connect to your WebSocket environment
    env_url = os.getenv("ENV_URL", "ws://localhost:8000")
    env = SqlEnvClient(base_url=env_url)

    # We loop through tasks to ensure at least 3 are completed
    for difficulty in ["easy", "medium", "hard", "super_hard"]:
        rewards: List[float] = []
        steps_taken = 0
        success = False
        schema_hint = "customers(id, name, email), orders(id, customer_id, total_amount), order_items(id, order_id, product_name)"

        log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

        try:
            result = await env.reset(difficulty=difficulty)
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                sql_query = get_model_message(client, obs.prompt, schema_hint, obs.last_execution_result)
                
                result = await env.step(SqlAction(query=sql_query))
                obs = result.observation
                
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=sql_query, reward=reward, done=result.done, error=obs.last_execution_result)

                if result.done:
                    if reward >= SUCCESS_THRESHOLD:
                        success = True
                    break

            final_score = max(rewards) if rewards else 0.01
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

        except Exception as e:
            print(f"[DEBUG] Error during {difficulty}: {e}", flush=True)
            log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())