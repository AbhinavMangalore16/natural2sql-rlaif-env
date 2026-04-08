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

        if difficulty == "easy":
            self._current_prompt = "Which customer placed order with id = 1?"
            self._expected_answer = [("Aarav Sharma",)]
        elif difficulty == "medium":
            self._current_prompt = "What is the total revenue from completed orders?"
            self._expected_answer = [(770.0,)] 
        elif difficulty == "hard":
            self._current_prompt = "How many distinct customers have placed at least one order?"
            self._expected_answer = [(5,)]
        else: 
            self._current_prompt = "List the names of all customers who purchased a 'Laptop'."
            self._expected_answer = [("Aarav Sharma",)]

        self._state = SqlState(episode_id=episode_id or str(uuid.uuid4()), step_count=0, difficulty=difficulty, target_answer=str(self._expected_answer), max_attempts=self.MAX_ATTEMPTS)
        return SqlObservation(done=False, reward=None, prompt=self._current_prompt, last_execution_result="Waiting for first query...", remaining_attempts=self.MAX_ATTEMPTS)

    def step(self, action: SqlAction, timeout_s=None, **kwargs) -> SqlObservation:
        self._state.step_count += 1
        query_upper = action.query.strip().upper()
        blocked_patterns = [
            "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT",  # Destructive
            "UNION",                                           # Unauthorized Extraction/Leakage
            "CROSS JOIN",                                      # Cartesian Product (Resource Exhaustion)
            "ALTER", "GRANT", "REVOKE", "CREATE"               # Schema/Permission manipulation
        ]
        if any(k in query_upper for k in blocked_patterns):
            self._state.step_count = self.MAX_ATTEMPTS
            return SqlObservation(done=True, reward=-1.0, prompt=self._current_prompt, last_execution_result="❌ CRITICAL SAFETY VIOLATION: Destructive commands are prohibited.", remaining_attempts=0)

        raw_query = action.query.strip()
        if raw_query.startswith("[") and raw_query.endswith("]"): raw_query = raw_query[1:-1]
        
        remaining = self.MAX_ATTEMPTS - self._state.step_count
        try:
            self._cursor.execute(raw_query)
            result = self._cursor.fetchall()

            if result == self._expected_answer:
                reward, msg = (0.8, f"Success! Output: {result}. (Tip: Avoid SELECT *, specify columns.)") if "*" in query_upper else (1.0, f"✅ Perfect! Output: {result}.")
                done = True
            else:
                done = self._state.step_count >= self.MAX_ATTEMPTS
                if not result:
                    reward, msg = (0.1, "Executed, but returned an empty set. Check your WHERE clause filters.")
                else:
                    reward = 0.3
                    try:
                        actual_val, expected_val = result[0][0], self._expected_answer[0][0]
                        diff_msg = "HIGHER" if actual_val > expected_val else "LOWER"
                        msg = f"Logic Error: Your result ({actual_val}) is {diff_msg} than expected. Review your aggregation filters."
                    except:
                        msg = f"Executed, but output {result} does not match the expected answer."
                if done: reward = -1.0
        except sqlite3.Error as e:
            done = self._state.step_count >= self.MAX_ATTEMPTS
            reward, msg = (-0.3 if not done else -1.0, f"SQL Syntax Error: {str(e)}")

        return SqlObservation(done=done, reward=reward, prompt=self._current_prompt, last_execution_result=msg, remaining_attempts=remaining)

    @property
    def state(self) -> SqlState: return self._state