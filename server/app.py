import uvicorn
from openenv.core.env_server import create_fastapi_app
from .environment import SqlEnvironment
from models import SqlAction, SqlObservation


app = create_fastapi_app(SqlEnvironment, SqlAction, SqlObservation)

def main():
    """Entry point for openenv validate and project scripts"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()