---
title: Natural2SQL: RL-Driven Data Engineering Bench
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - rlaif
  - reinforcement-learning
  - sql
---

# Natural2SQL Benchmark

A multi-turn reinforcement learning environment designed to train LLM agents in professional SQL generation, schema pathfinding, and self-correction.

## Innovation Highlights
* **Safety Guardrails:** Implements a strict penalty (-1.0) for destructive SQL commands (`DROP`, `DELETE`, `TRUNCATE`), simulating a production-safe environment.
* **Efficiency Grader:** Discourages `SELECT *` by providing an "Efficiency Bonus" for specific column selection, encouraging professional coding standards.
* **Algorithmic Feedback:** Provides raw SQLite compiler errors back to the agent, enabling multi-turn trajectory correction.

## Environment Specification
-   **Observation Space:** Pydantic model containing `prompt`, `last_execution_result` (actual DB feedback), and `remaining_attempts`.
-   **Action Space:** Pydantic model containing a single `query` string.
-   **Reward Function:** -   `1.0`: Success (Correct data + Efficient query).
    -   `0.8`: Success (Correct data but used `SELECT *`).
    -   `0.3`: Logic/Schema mismatch (Valid SQL, wrong data).
    -   `-0.3`: Syntax Error.
    -   `-1.0`: Safety violation or exhaustion of attempts.

## Task Descriptions
1.  **Easy:** Single table filter (Customer ID lookup).
2.  **Medium:** Aggregate calculation (Total revenue from completed orders).
3.  **Hard:** Multi-table join with `DISTINCT` constraints.
4.  **Super Hard:** Three-table relational pathfinding (Customer -> Order -> Item).

## Baseline Performance
| Model | Avg. Success Rate | Avg. Steps |
| :--- | :--- | :--- |
| **Qwen/Qwen2.5-72B-Instruct** | 100% | 1.2 |

## Local Setup & Testing
1. **Build:** `docker build -t natural2sql .`
2. **Run Server:** `docker run -p 8000:7860 natural2sql`
3. **Evaluate:** `python inference.py`