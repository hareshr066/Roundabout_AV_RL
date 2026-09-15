"""
web_server.py
====================================================================
Interactive Web Dashboard & Real-Time Simulation Server
for Roundabout AV RL Research Platform.

Features:
  - WebSocket streaming of real-time SUMO / PPO agent simulation (2D coordinates, speeds, headings, TTC, rewards)
  - REST APIs for simulation control (start, pause, step, reset, speed, model switch, traffic presets)
  - Launch native SUMO-GUI window on-demand from browser
  - Model benchmarking, ablation data, and system diagnostics APIs
  - Static file server for modern HTML5/CSS/JS frontend
====================================================================
"""

import os
import sys
import json
import time
import glob
import math
import shutil
import asyncio
import threading
import subprocess
import traceback
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tornado.web
import tornado.ioloop
import tornado.websocket
import tornado.httpserver
from stable_baselines3 import PPO
from env.roundabout_env import RoundaboutEnv

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder converting numpy types and inf/nan to standard JSON values."""
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return 999.0
            return val
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_json_dumps(obj):
    return json.dumps(obj, cls=NumpyEncoder)


# Global simulation state manager
class SimulationManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.paused = False
        self.sim_thread = None
        self.stop_requested = False
        self.step_delay = 0.05  # seconds per step (~20 FPS)
        
        # Config params — auto-detect best available model
        self.active_model_name = self._find_best_model()
        self.active_model_path = os.path.join(_ROOT, "results", "models", self.active_model_name)
        self.model = None
        self.preset_name = "MEDIUM"
        self.hdv_ratio = 0.60
        self.spawn_distance = 80.0
        self.max_steps = 800
        self.traffic_density = "medium"
        self.deterministic = True
        
        # Active environment & metrics
        self.env = None
        self.current_obs = None
        self.current_step = 0
        self.current_episode = 0
        self.last_accel = 0.0
        self.total_reward = 0.0
        self.prev_speed = 0.0
        self.episode_ttcs = []
        self.min_ttc = float('inf')
        self.episode_history = []
        self.last_event = None
        
        # Connected WebSocket clients
        self.ws_clients = set()
        
        # Cache models list
        self.load_model(self.active_model_path)

    def _find_best_model(self):
        """Auto-detect the best available model in priority order."""
        models_dir = os.path.join(_ROOT, "results", "models")
        priority_models = [
            "final_best_agent.zip",
            "best_model.zip",
            "agent_spatial_curriculum_30k.zip",
            "ablation_v4_shaping.zip",
        ]
        for name in priority_models:
            if os.path.exists(os.path.join(models_dir, name)):
                print(f"[SimulationManager] Auto-detected best model: {name}")
                return name
        # Fallback: any .zip file
        zips = glob.glob(os.path.join(models_dir, "*.zip"))
        if zips:
            best = os.path.basename(sorted(zips, key=os.path.getmtime, reverse=True)[0])
            print(f"[SimulationManager] Fallback model: {best}")
            return best
        return "agent_spatial_curriculum_30k.zip"

    def load_model(self, model_path):
        with self.lock:
            try:
                if os.path.exists(model_path):
                    self.model = PPO.load(model_path)
                    self.active_model_path = model_path
                    self.active_model_name = os.path.basename(model_path)
                    print(f"[SimulationManager] Loaded model: {self.active_model_name}")
                    return True
                else:
                    print(f"[SimulationManager] Model not found: {model_path}")
                    return False
            except Exception as e:
                print(f"[SimulationManager] Error loading model {model_path}: {e}")
                return False

    def list_available_models(self):
        models_dir = os.path.join(_ROOT, "results", "models")
        results = []
        if os.path.exists(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".zip"):
                    path = os.path.join(models_dir, f)
                    mtime = os.path.getmtime(path)
                    size = os.path.getsize(path)
                    results.append({
                        "filename": f,
                        "path": path,
                        "size_bytes": size,
                        "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime)),
                        "is_active": (f == self.active_model_name)
                    })
        return results

    def add_client(self, client):
        self.ws_clients.add(client)

    def remove_client(self, client):
        self.ws_clients.discard(client)

    def broadcast(self, message_dict):
        if not self.ws_clients:
            return
        try:
            msg = safe_json_dumps(message_dict)
        except Exception as e:
            print(f"[Broadcast Error]: {e}")
            return
        
        loop = getattr(self, "main_loop", None) or tornado.ioloop.IOLoop.current()
        for client in list(self.ws_clients):
            try:
                loop.add_callback(client.write_message, msg)
            except Exception:
                pass

    def start_simulation(self, config=None):
        if config:
            if "model" in config:
                m_path = os.path.join(_ROOT, "results", "models", config["model"])
                if os.path.exists(m_path):
                    self.load_model(m_path)
            if "preset" in config:
                self.preset_name = config["preset"].upper()
                self.traffic_density = config["preset"].lower()
            if "hdv_ratio" in config:
                self.hdv_ratio = float(config["hdv_ratio"])
            if "spawn_distance" in config:
                self.spawn_distance = float(config["spawn_distance"])
            if "speed" in config:
                # speed is a multiplier (0.1x to 5x)
                mult = max(0.1, min(5.0, float(config["speed"])))
                self.step_delay = (0.1 / mult)

        self.stop_simulation()
        self.stop_requested = False
        self.paused = False
        self.running = True
        self.sim_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.sim_thread.start()
        return {"status": "started", "model": self.active_model_name, "preset": self.preset_name}

    def pause_simulation(self):
        self.paused = not self.paused
        return {"paused": self.paused}

    def stop_simulation(self):
        self.stop_requested = True
        self.running = False
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=2.0)
        self._cleanup_env()
        return {"status": "stopped"}

    def _cleanup_env(self):
        if self.env:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _extract_simulation_frame(self, step_info=None, reward=0.0):
        if not self.env or not hasattr(self.env, "sim") or not self.env.sim.conn:
            return None

        traci_conn = self.env.sim.conn
        ego_id = self.env.ego_id

        # Query all vehicles currently in the simulation
        vehicles = []
        ego_data = None

        try:
            veh_ids = traci_conn.vehicle.getIDList()
        except Exception:
            return None

        for vid in veh_ids:
            try:
                x, y = traci_conn.vehicle.getPosition(vid)
                angle = traci_conn.vehicle.getAngle(vid)  # 0 is north, clockwise
                speed = traci_conn.vehicle.getSpeed(vid)
                vtype = traci_conn.vehicle.getTypeID(vid)
                lane = traci_conn.vehicle.getLaneID(vid)
                
                is_ego = (vid == ego_id)
                # Classify type
                if is_ego:
                    category = "ego"
                elif "av" in vtype.lower():
                    category = "av"
                else:
                    category = "hdv"

                veh_obj = {
                    "id": vid,
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "angle": round(float(angle), 1),
                    "speed": round(float(speed), 2),
                    "speed_kmh": round(float(speed * 3.6), 1),
                    "type": category,
                    "lane": lane,
                    "is_ego": is_ego
                }
                vehicles.append(veh_obj)
                if is_ego:
                    ego_data = veh_obj
            except Exception:
                continue

        # Telemetry & Observations
        obs = self.current_obs
        speed_val = float(obs[0]) if obs is not None else 0.0
        dist_to_entry = float(obs[1]) if obs is not None else 0.0
        nearest_circ_dist = float(obs[2]) if obs is not None else 50.0
        nearest_circ_speed = float(obs[3]) if obs is not None else 0.0
        gap_size = float(obs[4]) if obs is not None else 50.0
        current_hdv_ratio = float(obs[5]) if obs is not None else self.hdv_ratio

        # TTC
        ttc = self.env._get_ttc_after_merge()
        if ttc < float('inf'):
            ttc_val = round(min(ttc, 10.0), 2)
            self.episode_ttcs.append(ttc_val)
            self.min_ttc = min(self.min_ttc, ttc_val)
        else:
            ttc_val = 999.0

        # Acceleration
        inst_accel = round((speed_val - self.prev_speed) / self.env.dt, 2)
        self.prev_speed = speed_val

        # Merge state
        if self.env.entered_circulating:
            merge_state = "CIRCULATING"
        elif self.env.reached_merge_zone:
            merge_state = "MERGE_ZONE"
        else:
            merge_state = "APPROACH"

        avg_ttc = round(float(np.mean(self.episode_ttcs)), 2) if self.episode_ttcs else 999.0
        min_ttc_val = round(self.min_ttc, 2) if self.min_ttc < float('inf') else 999.0

        frame = {
            "type": "telemetry",
            "step": self.current_step,
            "sim_time": round(self.current_step * self.env.dt, 2),
            "episode": self.current_episode,
            "ego": ego_data,
            "vehicles": vehicles,
            "telemetry": {
                "speed": round(speed_val, 2),
                "speed_kmh": round(speed_val * 3.6, 1),
                "accel": inst_accel,
                "dist_to_entry": round(dist_to_entry, 2),
                "nearest_circ_dist": round(nearest_circ_dist, 2),
                "nearest_circ_speed": round(nearest_circ_speed, 2),
                "gap_size": round(gap_size, 2),
                "hdv_ratio": round(current_hdv_ratio, 2),
                "ttc": ttc_val,
                "avg_ttc": avg_ttc,
                "min_ttc": min_ttc_val,
                "merge_state": merge_state,
                "step_reward": round(float(reward), 3),
                "total_reward": round(float(self.total_reward), 2),
                "event": self.last_event
            }
        }
        self.last_event = None
        return frame

    def _run_loop(self):
        config_file = os.path.join(_ROOT, "sumo_network", "roundabout.sumocfg")

        while self.running and not self.stop_requested:
            self.current_episode += 1
            self.current_step = 0
            self.total_reward = 0.0
            self.prev_speed = 0.0
            self.episode_ttcs = []
            self.min_ttc = float('inf')
            self.last_accel = 0.0
            self.last_event = None

            print(f"[SimulationManager] Starting Episode {self.current_episode} [{self.preset_name}]...")
            self._cleanup_env()

            try:
                self.env = RoundaboutEnv(
                    config_file=config_file,
                    gui=False,  # Headless SUMO for ultra-smooth WebSocket streaming to HTML5 canvas
                    max_steps=self.max_steps,
                    fixed_hdv_ratio=self.hdv_ratio,
                    fixed_spawn_distance=self.spawn_distance,
                    use_spatial_curriculum=False,
                    traffic_density=self.traffic_density,
                    label=f"web_sim_ep{self.current_episode}_{int(time.time())}"
                )
                self.current_obs, info = self.env.reset()
            except Exception as e:
                print(f"[SimulationManager] Failed to start SUMO for episode: {e}")
                traceback.print_exc()
                time.sleep(2.0)
                continue

            done = False
            time_to_merge_val = "N/A"
            merge_event_sent = False
            circ_event_sent = False

            # Broadcast initial frame
            init_frame = self._extract_simulation_frame(reward=0.0)
            if init_frame:
                self.broadcast(init_frame)

            while not done and not self.stop_requested:
                if self.paused:
                    time.sleep(0.1)
                    continue

                # Policy action prediction
                if self.model is not None:
                    action, _ = self.model.predict(self.current_obs, deterministic=self.deterministic)
                else:
                    # Fallback heuristic acceleration
                    action = np.array([0.5], dtype=np.float32)

                try:
                    obs, reward, terminated, truncated, step_info = self.env.step(action)
                    self.current_obs = obs
                    self.current_step += 1
                    self.total_reward += float(reward)
                    done = terminated or truncated

                    # Check notable events
                    if step_info.get("reached_merge_zone") and not merge_event_sent:
                        merge_event_sent = True
                        self.last_event = {
                            "name": "MERGE_ZONE",
                            "message": "Ego entered Merge Zone -- Evaluating circulating traffic gap"
                        }

                    if step_info.get("success") and not circ_event_sent:
                        circ_event_sent = True
                        time_to_merge_val = step_info.get("time_to_merge", self.current_step * self.env.dt)
                        self.last_event = {
                            "name": "MERGE_SUCCESS",
                            "message": f"Merge Successful into Circulating Ring (T={time_to_merge_val:.2f}s)"
                        }

                    if step_info.get("collision"):
                        self.last_event = {
                            "name": "COLLISION",
                            "message": "Collision Detected with Circulating Traffic!"
                        }

                    # Broadcast telemetry frame to web clients
                    frame = self._extract_simulation_frame(step_info, reward)
                    if frame:
                        self.broadcast(frame)

                    time.sleep(self.step_delay)

                except Exception as e:
                    print(f"[SimulationManager] Step exception: {e}")
                    traceback.print_exc()
                    break

            if self.stop_requested:
                break

            # Episode outcome
            outcome = step_info.get("termination_reason", "timeout").upper() if 'step_info' in locals() else "TIMEOUT"
            avg_ttc = float(np.mean(self.episode_ttcs)) if self.episode_ttcs else 999.0
            min_ttc_val = self.min_ttc if self.min_ttc < float('inf') else 999.0

            ep_summary = {
                "episode": self.current_episode,
                "model": self.active_model_name,
                "preset": self.preset_name,
                "hdv_ratio": self.hdv_ratio,
                "outcome": outcome,
                "steps": self.current_step,
                "duration_s": round(self.current_step * self.env.dt, 2),
                "time_to_merge": round(time_to_merge_val, 2) if isinstance(time_to_merge_val, float) else "N/A",
                "avg_ttc": round(avg_ttc, 2) if avg_ttc < 900 else "N/A",
                "min_ttc": round(min_ttc_val, 2) if min_ttc_val < 900 else "N/A",
                "total_reward": round(self.total_reward, 2)
            }
            self.episode_history.insert(0, ep_summary)
            if len(self.episode_history) > 50:
                self.episode_history.pop()

            self.broadcast({
                "type": "episode_end",
                "summary": ep_summary
            })

            # Pause briefly between episodes
            time.sleep(1.5)

        self._cleanup_env()
        self.running = False
        print("[SimulationManager] Simulation loop exited.")


sim_mgr = SimulationManager()


# ----------------------------------------------------------------------
# TORNADO HTTP & WEBSOCKET HANDLERS
# ----------------------------------------------------------------------

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(os.path.join(_ROOT, "web", "index.html"))


class TelemetryWebSocket(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        sim_mgr.add_client(self)
        # Send initial status
        try:
            self.write_message(safe_json_dumps({
                "type": "connected",
                "models": sim_mgr.list_available_models(),
                "active_model": sim_mgr.active_model_name,
                "preset": sim_mgr.preset_name,
                "hdv_ratio": float(sim_mgr.hdv_ratio),
                "running": sim_mgr.running,
                "paused": sim_mgr.paused,
                "history": sim_mgr.episode_history
            }))
        except Exception as e:
            print(f"[WebSocket] Error sending connected message: {e}")
            traceback.print_exc()

    def on_close(self):
        sim_mgr.remove_client(self)

    def on_message(self, message):
        try:
            data = json.loads(message)
            action = data.get("action")
            if action == "start":
                sim_mgr.start_simulation(data.get("config"))
            elif action == "pause":
                sim_mgr.pause_simulation()
            elif action == "stop":
                sim_mgr.stop_simulation()
            elif action == "set_speed":
                mult = max(0.1, min(5.0, float(data.get("speed", 1.0))))
                sim_mgr.step_delay = (0.1 / mult)
            elif action == "load_model":
                model_file = data.get("model")
                if model_file:
                    m_path = os.path.join(_ROOT, "results", "models", model_file)
                    sim_mgr.load_model(m_path)
                    sim_mgr.broadcast({"type": "model_changed", "active_model": sim_mgr.active_model_name})
        except Exception as e:
            print(f"[WebSocket] Message handler error: {e}")


class StatusApiHandler(tornado.web.RequestHandler):
    def get(self):
        # Gather system info
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A (CPU Mode)"
        
        sumo_home = os.environ.get("SUMO_HOME", "C:\\Program Files (x86)\\Eclipse\\Sumo")
        sumo_bin = shutil.which("sumo") or os.path.join(sumo_home, "bin", "sumo.exe")
        
        status_data = {
            "status": "online",
            "cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "sumo_home": sumo_home,
            "sumo_binary": sumo_bin,
            "sumo_found": os.path.exists(sumo_bin) if sumo_bin else False,
            "active_model": sim_mgr.active_model_name,
            "running": sim_mgr.running,
            "paused": sim_mgr.paused,
            "models_count": len(sim_mgr.list_available_models())
        }
        self.set_header("Content-Type", "application/json")
        self.write(safe_json_dumps(status_data))


class ModelsApiHandler(tornado.web.RequestHandler):
    def get(self):
        models = sim_mgr.list_available_models()
        self.set_header("Content-Type", "application/json")
        self.write(safe_json_dumps({
            "active_model": sim_mgr.active_model_name,
            "models": models
        }))

    def post(self):
        try:
            body = json.loads(self.request.body)
            model_name = body.get("model")
            if model_name:
                m_path = os.path.join(_ROOT, "results", "models", model_name)
                success = sim_mgr.load_model(m_path)
                self.set_header("Content-Type", "application/json")
                self.write(safe_json_dumps({"success": success, "active_model": sim_mgr.active_model_name}))
                return
        except Exception as e:
            self.set_status(400)
            self.write(safe_json_dumps({"error": str(e)}))


class SimControlApiHandler(tornado.web.RequestHandler):
    def post(self, action):
        try:
            body = json.loads(self.request.body) if self.request.body else {}
        except Exception:
            body = {}

        if action == "start":
            res = sim_mgr.start_simulation(body)
        elif action == "pause":
            res = sim_mgr.pause_simulation()
        elif action == "stop":
            res = sim_mgr.stop_simulation()
        else:
            self.set_status(400)
            res = {"error": f"Unknown action {action}"}

        self.set_header("Content-Type", "application/json")
        self.write(safe_json_dumps(res))


class LaunchSumoGuiApiHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            body = json.loads(self.request.body) if self.request.body else {}
        except Exception:
            body = {}

        model = body.get("model")
        if not model or model == "Loading..." or not os.path.exists(os.path.join(_ROOT, "results", "models", model)):
            model = sim_mgr._find_best_model()

        preset = body.get("preset", "MEDIUM")
        speed = body.get("speed", 0.25)
        episodes = body.get("episodes", 3)
        hdv_ratio = body.get("hdv_ratio", None)

        model_path = os.path.join(_ROOT, "results", "models", model)
        if not os.path.exists(model_path):
            # Fallback search
            model_path = os.path.join(_ROOT, "results", "models", sim_mgr._find_best_model())

        cmd = [
            sys.executable,
            os.path.join(_ROOT, "evaluation", "demo_mode.py"),
            "--model", model_path,
            "--preset", str(preset),
            "--episodes", str(episodes),
            "--speed", str(speed)
        ]
        if hdv_ratio is not None:
            cmd.extend(["--hdv-ratio", str(hdv_ratio)])

        # Launch background GUI process
        try:
            creation_flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            proc = subprocess.Popen(cmd, cwd=_ROOT, creationflags=creation_flags)
            self.set_header("Content-Type", "application/json")
            self.write(safe_json_dumps({
                "success": True,
                "message": f"SUMO-GUI Demo launched (PID {proc.pid}) with model {os.path.basename(model_path)}",
                "pid": proc.pid
            }))
        except Exception as e:
            self.set_status(500)
            self.write(safe_json_dumps({"success": False, "error": str(e)}))


class ResultsDataApiHandler(tornado.web.RequestHandler):
    def get(self):
        data = {
            "hdv_penetration": [],
            "ablation": []
        }

        hdv_csv = os.path.join(_ROOT, "results", "study_hdv_penetration.csv")
        if os.path.exists(hdv_csv):
            import csv
            with open(hdv_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data["hdv_penetration"] = list(reader)

        ablation_csv = os.path.join(_ROOT, "results", "ablation_study_results.csv")
        if os.path.exists(ablation_csv):
            import csv
            with open(ablation_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data["ablation"] = list(reader)

        self.set_header("Content-Type", "application/json")
        self.write(safe_json_dumps(data))

class FiguresApiHandler(tornado.web.RequestHandler):
    """API endpoint listing all generated publication figures."""
    def get(self):
        figures_dir = os.path.join(_ROOT, "results", "figures")
        figures = []
        if os.path.exists(figures_dir):
            for f in sorted(os.listdir(figures_dir)):
                if f.endswith(".png"):
                    fpath = os.path.join(figures_dir, f)
                    figures.append({
                        "name": f,
                        "url": f"/results_figures/{f}",
                        "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                        "title": f.replace("_", " ").replace(".png", "").title()
                    })
        self.set_header("Content-Type", "application/json")
        self.write(safe_json_dumps({"figures": figures, "count": len(figures)}))


def make_app():
    web_dir = os.path.join(_ROOT, "web")
    results_dir = os.path.join(_ROOT, "results")

    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/ws/telemetry", TelemetryWebSocket),
        (r"/api/status", StatusApiHandler),
        (r"/api/models", ModelsApiHandler),
        (r"/api/sim/(start|pause|stop)", SimControlApiHandler),
        (r"/api/launch-sumo-gui", LaunchSumoGuiApiHandler),
        (r"/api/results-data", ResultsDataApiHandler),
        (r"/api/figures", FiguresApiHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": web_dir}),
        (r"/results_media/(.*)", tornado.web.StaticFileHandler, {"path": results_dir}),
        (r"/results_figures/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(results_dir, "figures")}),
    ],
    template_path=web_dir,
    static_path=web_dir,
    debug=False
    )


def start_server(port=8080):
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    
    # Try preferred port, fallback if bound
    for p in [port, 8000, 8888, 5000, 3000]:
        try:
            server.listen(p)
            print("=" * 75)
            print(f"  ROUNDABOUT AV RL RESEARCH PLATFORM -- WEB DASHBOARD RUNNING")
            print("=" * 75)
            print(f"  Local URL:        http://localhost:{p}")
            print(f"  Network URL:      http://127.0.0.1:{p}")
            print(f"  WebSocket Stream: ws://localhost:{p}/ws/telemetry")
            print("=" * 75 + "\n")
            
            # Store main IOLoop for cross-thread broadcasts from simulation thread
            sim_mgr.main_loop = tornado.ioloop.IOLoop.current()
            
            # Start automatic simulation on launch
            sim_mgr.start_simulation()
            
            tornado.ioloop.IOLoop.current().start()
            break
        except OSError:
            print(f"Port {p} in use, trying next...")
            continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Roundabout RL Web App Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()
    start_server(args.port)
