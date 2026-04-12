import uvicorn
import os
import sys
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# 1. PATH FIX: Ensure the app can see models.py in the root
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from openenv.core.env_server import create_fastapi_app
from environment import SqlEnvironment
from models import SqlAction, SqlObservation

LIVE_METRICS = {
    "total_steps": 0,
    "last_query": "Waiting for agent payload...",
    "last_reward": 0.00,
    "status": "Idling ⏳",
    "difficulty": "UNKNOWN"
}
AGENT_IS_RUNNING = False

class TrackedSqlEnvironment(SqlEnvironment):
    def reset(self, difficulty="medium", **kwargs):
        LIVE_METRICS["status"] = "Resetting DB 🔄"
        LIVE_METRICS["difficulty"] = difficulty.upper()
        return super().reset(difficulty=difficulty, **kwargs)

    def step(self, action: SqlAction, **kwargs) -> SqlObservation:
        obs = super().step(action, **kwargs)
        LIVE_METRICS["total_steps"] += 1
        LIVE_METRICS["last_query"] = action.query
        LIVE_METRICS["last_reward"] = obs.reward or 0.0
        LIVE_METRICS["status"] = "Agent Executing ⚡"
        LIVE_METRICS["difficulty"] = self._state.difficulty.upper()
        return obs

app = create_fastapi_app(TrackedSqlEnvironment, SqlAction, SqlObservation)

@app.post("/trigger-inference")
async def trigger_inference():
    global AGENT_IS_RUNNING
    
    if AGENT_IS_RUNNING:
        return {"status": "ignored", "message": "Agent is already running."}
    
    AGENT_IS_RUNNING = True
    LIVE_METRICS["status"] = "Initializing Agent 🚀"
    
    async def run_script():
        global AGENT_IS_RUNNING
        try:
            env = os.environ.copy()
            env["ENV_URL"] = "ws://localhost:8000" 
            
            process = await asyncio.create_subprocess_shell(
                "python inference.py",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        except Exception as e:
            LIVE_METRICS["status"] = f"Error: {str(e)}"
        finally:
            AGENT_IS_RUNNING = False
            LIVE_METRICS["status"] = "Agent Finished ✅"

    asyncio.create_task(run_script())
    return {"status": "started", "message": "Inference triggered."}

@app.get("/metrics")
async def get_metrics():
    return {**LIVE_METRICS, "is_running": AGENT_IS_RUNNING}

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Natural2SQL | Mission Control</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { background: radial-gradient(circle at 20% 20%, #1e293b, #020617); }
            .glass { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(16px); }
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
            .cursor::after { content: '█'; animation: blink 1s step-start infinite; color: #4ade80; }
            @keyframes blink { 50% { opacity: 0; } }
        </style>
    </head>

    <body class="text-slate-200 min-h-screen flex items-center justify-center p-4">
        <div class="relative w-full max-w-5xl">
            <div class="absolute inset-0 blur-3xl opacity-20 bg-gradient-to-r from-cyan-500 to-blue-600"></div>

            <div class="glass relative border border-slate-700/50 rounded-3xl shadow-2xl p-8 md:p-12 overflow-hidden">
                
                <div class="flex flex-col md:flex-row justify-between items-center mb-8 gap-6">
                    <div>
                        <h1 class="text-4xl md:text-5xl font-black bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
                            Natural2SQL
                        </h1>
                        <p class="text-slate-400 mt-2 tracking-widest uppercase text-xs font-bold">RL-AIF Mission Control</p>
                    </div>
                    <div class="flex items-center gap-3 bg-green-500/10 border border-green-500/30 text-green-400 px-5 py-2.5 rounded-full font-bold shadow-[0_0_20px_rgba(74,222,128,0.15)]">
                        <span class="relative flex h-3 w-3">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                        </span>
                        <span id="ui-status">System Online</span>
                    </div>
                </div>

                <button id="launch-btn" onclick="triggerAgent()" class="w-full mb-8 bg-blue-600 hover:bg-blue-500 text-white font-black py-4 rounded-2xl shadow-[0_0_20px_rgba(37,99,235,0.3)] transition active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed">
                    🚀 DEPLOY AUTONOMOUS AGENT
                </button>

                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <div class="bg-slate-900/80 p-5 rounded-2xl border border-slate-700/50">
                        <p class="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">Total Steps</p>
                        <p class="text-3xl font-mono text-cyan-400" id="ui-steps">0</p>
                    </div>
                    <div class="bg-slate-900/80 p-5 rounded-2xl border border-slate-700/50">
                        <p class="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">Latest Reward</p>
                        <p class="text-3xl font-mono text-yellow-400" id="ui-reward">0.00</p>
                    </div>
                    <div class="bg-slate-900/80 p-5 rounded-2xl border border-slate-700/50">
                        <p class="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">Latest Difficulty</p>
                        <p class="text-2xl font-mono text-pink-400" id="ui-difficulty">UNKNOWN</p>
                    </div>
                    <div class="bg-slate-900/80 p-5 rounded-2xl border border-slate-700/50 flex flex-col justify-center">
                        <p class="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">Environment</p>
                        <p class="text-sm font-mono text-slate-300 truncate">Hugging Face Container</p>
                    </div>
                </div>

                <div class="bg-[#050505] rounded-2xl p-5 border border-slate-800 shadow-inner h-64 overflow-y-auto font-mono text-xs md:text-sm" id="terminal">
                    <div class="text-slate-500 mb-2">>> Booting Natural2SQL Sandbox...</div>
                    <div class="text-slate-500 mb-2">>> Port 8000 Bound. WebSockets Ready.</div>
                    <div class="text-slate-500 mb-4">>> Awaiting manual deployment...</div>
                    <div id="term-content"></div>
                    <div class="cursor mt-1"></div>
                </div>
            </div>
        </div>

        <script>
            const termContent = document.getElementById('term-content');
            const launchBtn = document.getElementById('launch-btn');
            let lastSteps = 0;

            async function triggerAgent() {
                try {
                    await fetch('/trigger-inference', { method: 'POST' });
                } catch (e) {
                    console.error("Trigger failed", e);
                }
            }

            async function typeLine(text, colorClass) {
                const line = document.createElement('div');
                line.className = `mb-1 ${colorClass}`;
                termContent.appendChild(line);
                
                for (let i = 0; i < text.length; i++) {
                    line.textContent += text.charAt(i);
                    const terminalObj = document.getElementById('terminal');
                    terminalObj.scrollTop = terminalObj.scrollHeight;
                    await new Promise(r => setTimeout(r, 5)); 
                }
            }

            setInterval(async () => {
                try {
                    const res = await fetch('/metrics');
                    const data = await res.json();
                    
                    document.getElementById('ui-steps').innerText = data.total_steps;
                    document.getElementById('ui-reward').innerText = data.last_reward.toFixed(2);
                    document.getElementById('ui-status').innerText = data.status;
                    
                    const diffEl = document.getElementById('ui-difficulty');
                    diffEl.innerText = data.difficulty;
                    diffEl.className = "text-2xl font-mono " +
                        (data.difficulty === "EASY" ? "text-green-400" :
                        data.difficulty === "MEDIUM" ? "text-yellow-400" :
                        data.difficulty === "HARD" ? "text-orange-400" :
                        data.difficulty === "SUPER_HARD" ? "text-red-400" :
                        "text-pink-400");

                    if (data.is_running) {
                        launchBtn.disabled = true;
                        launchBtn.innerText = "⚡ AGENT IS ACTIVE...";
                        launchBtn.classList.replace("bg-blue-600", "bg-slate-700");
                    } else {
                        launchBtn.disabled = false;
                        launchBtn.innerText = "🚀 DEPLOY AUTONOMOUS AGENT";
                        launchBtn.classList.replace("bg-slate-700", "bg-blue-600");
                    }

                    if (data.total_steps > lastSteps) {
                        lastSteps = data.total_steps;
                        await typeLine(`[ACTION TRIGGERED] Step ${data.total_steps}`, "text-blue-400 font-bold");
                        await typeLine(`Query Executed: ${data.last_query}`, "text-slate-300");
                        await typeLine(`Reward Granted: ${data.last_reward.toFixed(2)}`, "text-yellow-400");
                        await typeLine(`----------------------------------------`, "text-slate-700");
                    }
                } catch (e) {}
            }, 1000);
        </script>
    </body>
    </html>
    """

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()