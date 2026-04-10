import os
import asyncio
from openai import AsyncOpenAI
from client import SqlEnvClient
from models import SqlAction

client = AsyncOpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

async def main():
    """Baseline agent that uses a simple prompt to generate SQL queries based on the environment's feedback."""
    env = SqlEnvClient(base_url="ws://localhost:8000")
    
    print("--- Starting Baseline Evaluation ---")
    result = await env.reset(difficulty="easy")
    obs = result.observation
    print(f"Task: {obs.prompt}")
    
    # Give the agent 5 attempts to get the right answer
    for turn in range(5):
        if result.done:
            break
            
        print(f"\n--- Turn {turn + 1} ---")
        print(f"Environment Feedback: {obs.last_execution_result}")
        
        system_prompt = "You are an expert SQL engineer. Convert the question into valid SQLite. Output ONLY the query inside square brackets, e.g., [SELECT * FROM table;]. Do not explain."
        user_prompt = f"Schema: customers(id, name, email), orders(id, customer_id, total_amount, status, created_at), order_items(id, order_id, product_name, quantity, price). Task: {obs.prompt}\nFeedback from previous attempt: {obs.last_execution_result}"
        
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=100
            )
            llm_reply = response.choices[0].message.content.strip()
            print(f"LLM Generated: {llm_reply}")

            result = await env.step(SqlAction(query=llm_reply))
            obs = result.observation
            print(f"\nFinal Environment Feedback: {obs.last_execution_result}")
            print(f"Reward: {result.reward}")
            
        except Exception as e:
            print(f"LLM Error: {e}")
            break

    print(f"\nEpisode Complete. Final Reward: {result.reward}")

if __name__ == "__main__":
    asyncio.run(main())