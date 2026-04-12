# Use a slim Python image for efficiency
FROM python:3.11-slim

# Install system dependencies for SQLite
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy all project files
COPY . .

# Install dependencies directly into the system environment
RUN uv pip install --system .

# Expose the port Hugging Face expects
EXPOSE 8000

# Run the Uvicorn server
# Using server.app:app because of your project structure
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]