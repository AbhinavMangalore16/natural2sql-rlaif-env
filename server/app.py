import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core.env_server import EnvServer

# Import your local modules
from .environment import SqlEnvironment
from models import SqlAction, SqlObservation

# 1. Create a fresh FastAPI instance
app = FastAPI(title="Natural2SQL Sandbox")

# 2. Add the Landing Page for the Hugging Face Space
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Natural2SQL - Sandbox</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-200 flex items-center justify-center min-h-screen">
        <div class="bg-slate-900 p-10 rounded-3xl shadow-2xl border border-slate-800 text-center max-w-lg w-full">
            
            <h1 class="text-4xl font-black bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent mb-2">
                Natural2SQL
            </h1>
            <p class="text-slate-400 font-medium mb-6 uppercase tracking-wider text-sm">
                RL-AIF Training Environment
            </p>
            
            <div class="inline-flex items-center gap-2 bg-green-500/10 border border-green-500/20 text-green-400 px-4 py-2 rounded-full text-sm font-bold mb-8 shadow-[0_0_15px_rgba(74,222,128,0.1)]">
                <span class="relative flex h-3 w-3">
                  <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                System Online & Healthy
            </div>

            <div class="bg-slate-950 p-5 rounded-xl text-left border border-slate-800">
                <p class="text-slate-500 mb-3 uppercase tracking-widest text-[10px] font-bold">Active WebSocket Endpoint</p>
                <code class="text-blue-300 text-sm break-all">
                    wss://abhinavm16104-natural2sql-rlaif-env.hf.space
                </code>
            </div>
            
            <p class="text-slate-500 text-xs mt-6">
                Waiting for agent connections on port 8000...
            </p>
        </div>
    </body>
    </html>
    """

# 3. Mount the OpenEnv endpoints (/reset, /step, /ws) AFTER the UI
env_server = EnvServer(SqlEnvironment())
app.include_router(env_server.router)

def main():
    """Entry point for openenv validate and project scripts"""
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()