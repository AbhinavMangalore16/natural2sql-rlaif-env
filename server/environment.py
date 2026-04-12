import sqlite3
import uuid
import re
from openenv.core.env_server import Environment
from models import SqlAction, SqlObservation, SqlState

class SqlEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 5

    def __init__(self):
        """Initializes the SQL environment with an in-memory SQLite database and sets up the schema and seed data."""
        self._state = SqlState()
        self._conn = None
        self._cursor = None
        self._expected_answer = []
        self._current_prompt = ""

    def _initialize_schema(self):
        """Creates the necessary tables for the SQL environment."""
        self._cursor.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        self._cursor.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, total_amount REAL, status TEXT, created_at TEXT, FOREIGN KEY(customer_id) REFERENCES customers(id))")
        self._cursor.execute("CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER, product_name TEXT, quantity INTEGER, price REAL, FOREIGN KEY(order_id) REFERENCES orders(id))")

    def _seed_data(self):
        """Populates the database with sample data for customers, orders, and order items."""
        customers = [("Aarav Sharma", "aarav@ex.com"), ("Fatima Khan", "fatima@ex.com"), ("Arjun Nair", "arjun@ex.com"), ("Imran Ali", "imran@ex.com"), ("Ananya Iyer", "ananya@ex.com"), ("Simran Kaur", "simran@ex.com"), ("Joseph Thomas", "joseph@ex.com"), ("Priya Verma", "priya@ex.com")]
        self._cursor.executemany("INSERT INTO customers (name, email) VALUES (?, ?)", customers)
        orders = [(1, 250.0, "completed", "2024-01-01"), (2, 120.0, "completed", "2024-01-02"), (3, 300.0, "pending", "2024-01-03"), (4, 180.0, "completed", "2024-01-04"), (5, 220.0, "completed", "2024-01-05")]
        self._cursor.executemany("INSERT INTO orders (customer_id, total_amount, status, created_at) VALUES (?, ?, ?, ?)", orders)
        order_items = [(1, "Laptop", 1, 250.0), (2, "Mouse", 2, 60.0), (3, "Keyboard", 1, 300.0), (4, "Headphones", 1, 180.0), (5, "Smartphone", 1, 220.0)]
        self._cursor.executemany("INSERT INTO order_items (order_id, product_name, quantity, price) VALUES (?, ?, ?, ?)", order_items)
        self._conn.commit()

    def reset(self, seed=None, episode_id=None, difficulty="medium", **kwargs) -> SqlObservation:
        """Resets the environment to its initial state, setting up a new in-memory database and defining the prompt and expected answer based on the specified difficulty level."""
        self._conn = sqlite3.connect(":memory:")
        self._cursor = self._conn.cursor()
        self._initialize_schema()
        self._seed_data()

        # ✅ Schema context (added)
        schema_context = """
    Database Schema:
    - customers(id, name, email)
    - orders(id, customer_id, total_amount, status, created_at)
    - order_items(id, order_id, product_name, quantity, price)
    """

        if difficulty == "easy":
            task = "Which customer placed order with id = 1?"
            self._expected_answer = [("Aarav Sharma",)]
        elif difficulty == "medium":
            task = "What is the total revenue from completed orders?"
            self._expected_answer = [(770.0,)] 
        elif difficulty == "hard":
            task = "How many distinct customers have placed at least one order?"
            self._expected_answer = [(5,)]
        else: 
            task = "List the names of all customers who purchased a 'Laptop'."
            self._expected_answer = [("Aarav Sharma",)]

        # ✅ Combined prompt (schema + task)
        self._current_prompt = f"""
    You are an SQL agent.

    {schema_context}

    Task:
    {task}

    Return only a valid SQL query.
    """

        self._state = SqlState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty,
            target_answer=str(self._expected_answer),
            max_attempts=self.MAX_ATTEMPTS
        )

        return SqlObservation(
            done=False,
            reward=None,
            prompt=self._current_prompt,
            last_execution_result="Waiting for first query...",
            remaining_attempts=self.MAX_ATTEMPTS
        )
    def step(self, action: SqlAction, timeout_s=None, **kwargs) -> SqlObservation:
        self._state.step_count += 1
        query_upper = action.query.strip().upper()
        
        # 1. Define Difficulty-Based Reward Mapping
        diff = self._state.difficulty
        if diff == "easy":
            success_r, failure_r = 0.75, 0.20
        elif diff == "medium":
            success_r, failure_r = 0.60, 0.15
        else: # hard and super_hard
            success_r, failure_r = 0.35, 0.10
        
        safety_r = 0.01

        # 2. Safety Check
        blocked_patterns = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "UNION", "CROSS JOIN", "ALTER", "GRANT", "REVOKE", "CREATE"]
        if any(k in query_upper for k in blocked_patterns):
            self._state.step_count = self.MAX_ATTEMPTS
            return SqlObservation(done=True, reward=safety_r, prompt=self._current_prompt, last_execution_result="❌ SAFETY VIOLATION", remaining_attempts=0)

        raw_query = action.query.strip()
        if raw_query.startswith("[") and raw_query.endswith("]"): raw_query = raw_query[1:-1]
        
        remaining = self.MAX_ATTEMPTS - self._state.step_count
        try:
            self._cursor.execute(raw_query)
            result = self._cursor.fetchall()
            
            # Success Path
            if set(result) == set(self._expected_answer):
                # If they used SELECT *, give them slightly less than the max for that difficulty
                is_wildcard = bool(re.search(r'SELECT\s+(\*|.*,\s*\*|\w+\.\*)', query_upper))
                reward = (success_r - 0.05) if is_wildcard else success_r
                
                msg = f"✅ Success! Output: {result}"
                done = True
            
            # Persistence Path
            else:
                done = self._state.step_count >= self.MAX_ATTEMPTS
                if not result:
                    reward, msg = (failure_r + 0.05, "Executed, but returned an empty set.")
                else:
                    reward = failure_r + 0.10
                    msg = "Executed, but output does not match expected."
                
                # Final failure after max attempts
                if done: reward = failure_r

        except sqlite3.Error as e:
            done = self._state.step_count >= self.MAX_ATTEMPTS
            # Syntax errors get the lowest non-safety reward
            reward = 0.05 if not done else 0.02
            msg = f"SQL Syntax Error: {str(e)}"

        return SqlObservation(done=done, reward=reward, prompt=self._current_prompt, last_execution_result=msg, remaining_attempts=remaining)

    @property
    def state(self) -> SqlState: return self._state