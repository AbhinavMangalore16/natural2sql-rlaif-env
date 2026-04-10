import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from client import SqlEnvClient
from models import SqlAction

# 1. & 2. STRICT INITIALIZATION (Required by Validator)
# We use os.environ[] directly to ensure the script fails-fast if keys are missing.
try:
    API_BASE_URL = os.environ["API_BASE_URL"]
    API_KEY = os.environ["API_KEY"]
except KeyError as e:
    print(f"[CRITICAL] Missing required environment variable: {e}")
    raise

# MODEL_NAME can have a fallback as per the checklist
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "natural2sql"
MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.8 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert SQL engineer. Convert the user's question into a valid SQLite query.
    Rules:
    1. Use the provided schema.
    2. Output ONLY the query inside square brackets, e.g., [SELECT * FROM table;].
    3. If you receive an error message, fix your previous query.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    action_clean = action.replace("\n", " ").strip()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_sql_query(client: OpenAI, prompt: str, schema: str, feedback: str) -> str:
    user_prompt = f"Schema: {schema}\nTask: {prompt}\nLast Feedback: {feedback}\nGenerate SQL:"
    try:
        # 3. LLM CALL (Uses the OpenAI client initialized above)
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
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        raise e

async def main() -> None:
    # Initialize client once
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    env_url = os.getenv("ENV_URL", "ws://localhost:8000")
    env = SqlEnvClient(base_url=env_url)

    for difficulty in ["easy", "medium", "hard", "super_hard"]:
        rewards: List[float] = []
        steps_taken = 0
        success = False
        schema_hint = "customers(id, name, email), orders(id, customer_id, total_amount, status), order_items(id, order_id, product_name, quantity, price)"

        log_start(task=difficulty, env=BENCHMARK, model=MODEL_NAME)

        try:
            # PROPER ASYNC HANDLING: Use 'await' instead of 'resolve'
            result = await env.reset(difficulty=difficulty)
            obs = result.observation
            last_feedback = obs.last_execution_result

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                
                sql_query = get_sql_query(client, obs.prompt, schema_hint, last_feedback)
                
                # Step the environment
                result = await env.step(SqlAction(query=sql_query))
                
                obs = result.observation
                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step
                last_feedback = obs.last_execution_result
                
                log_step(step=step, action=sql_query, reward=reward, done=result.done, error=last_feedback)

                if result.done and reward >= SUCCESS_THRESHOLD:
                    success = True
                    break
            
            final_score = max(rewards) if rewards else 0.0
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

        except Exception as e:
            print(f"[ERROR] Task {difficulty} failed: {e}", flush=True)
            log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

if __name__ == "__main__":
    # Start the single event loop
    asyncio.run(main())