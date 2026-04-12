import uvicorn
import os
import sys
from fastapi.responses import HTMLResponse

# 1. PATH FIX: Ensure the app can see models.py in the root
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

# 2. Import the known working OpenEnv function
from openenv.core.env_server import create_fastapi_app

# 3. Import your local modules
from environment import SqlEnvironment
from models import SqlAction, SqlObservation

# 4. Initialize the FastAPI app using the official wrapper
app = create_fastapi_app(SqlEnvironment, SqlAction, SqlObservation)

# 5. Attach the Health Check Landing Page directly to the generated app
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Natural2SQL Sandbox</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                background: radial-gradient(circle at 20% 20%, #1e293b, #020617);
            }
        </style>
    </head>

    <body class="text-slate-200 min-h-screen flex items-center justify-center">

        <div class="relative w-full max-w-4xl p-8">

            <!-- Glow Background -->
            <div class="absolute inset-0 blur-3xl opacity-20 bg-gradient-to-r from-blue-500 to-indigo-600"></div>

            <!-- Main Card -->
            <div class="relative bg-slate-900/70 backdrop-blur-xl border border-slate-800 rounded-3xl shadow-2xl p-10">

                <!-- Header -->
                <div class="text-center mb-8">
                    <h1 class="text-5xl font-extrabold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">
                        Natural2SQL
                    </h1>
                    <p class="text-slate-400 mt-2 tracking-widest uppercase text-xs">
                        RL-AIF Training Sandbox
                    </p>
                </div>

                <!-- Status -->
                <div class="flex justify-center mb-10">
                    <div class="flex items-center gap-3 bg-green-500/10 border border-green-500/20 text-green-400 px-5 py-2 rounded-full font-semibold shadow-lg">
                        <span class="relative flex h-3 w-3">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                        </span>
                        System Online
                    </div>
                </div>

                <!-- Info Grid -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">

                    <div class="bg-slate-800/60 p-6 rounded-xl border border-slate-700 hover:border-blue-500 transition">
                        <h3 class="text-lg font-bold text-blue-400">Environment</h3>
                        <p class="text-slate-400 mt-2 text-sm">SQL RL Sandbox</p>
                    </div>

                    <div class="bg-slate-800/60 p-6 rounded-xl border border-slate-700 hover:border-indigo-500 transition">
                        <h3 class="text-lg font-bold text-indigo-400">Mode</h3>
                        <p class="text-slate-400 mt-2 text-sm">Multi-Turn Training</p>
                    </div>

                    <div class="bg-slate-800/60 p-6 rounded-xl border border-slate-700 hover:border-purple-500 transition">
                        <h3 class="text-lg font-bold text-purple-400">Port</h3>
                        <p class="text-slate-400 mt-2 text-sm">8000</p>
                    </div>

                </div>

                <!-- Divider -->
                <div class="my-8 border-t border-slate-800"></div>

                <!-- Logs Section -->
                <div class="bg-black/40 rounded-xl p-4 border border-slate-800 font-mono text-xs text-green-400 h-32 overflow-hidden">
                    <p>> Initializing environment...</p>
                    <p>> Loading schema...</p>
                    <p>> Reward system active</p>
                    <p class="animate-pulse">> Waiting for agent connection...</p>
                </div>

            </div>

            <!-- Footer -->
            <p class="text-center text-slate-600 text-xs mt-6">
                Natural2SQL • Reinforcement Learning Environment
            </p>

        </div>

    </body>
    </html>
    """

def main():
    # Note: Running directly on host 0.0.0.0 and port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()