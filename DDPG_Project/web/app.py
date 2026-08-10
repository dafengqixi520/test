"""Flask应用 — PER-DDPG云-雾-边卸载系统"""
import sys, os, json, threading, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np

from config import AppConfig, EnvConfig, DDPGConfig
from algorithm.environment.cloud_fog_edge_env import CloudFogEdgeEnv
from algorithm.per_ddpg.ddpg_agent import PERDDPGAgent
from algorithm.runner.runner import PERDDPGRunner

TRAINING_OUTPUT = (
    Path(__file__).resolve().parents[1] / "output" / "training" / "latest_training.json"
)


def _load_saved_training():
    if not TRAINING_OUTPUT.exists():
        return None
    try:
        with TRAINING_OUTPUT.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        logger.exception("Failed to load saved training history")
        return None


def _save_training_snapshot(runner):
    payload = {
        "provenance": "locally generated PER-DDPG training history",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "progress": runner.get_progress(),
        "history": runner.get_history(),
        "metrics": sim_state["metrics"],
    }
    TRAINING_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = TRAINING_OUTPUT.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(TRAINING_OUTPUT)
    sim_state["saved_training"] = payload


sim_state = {
    "running": False, "paused": False, "config": None,
    "env": None, "agent": None, "runner": None, "thread": None,
    "mode": "train", "speed": 0.05,
    "metrics": [], "current_detail": None,
    "saved_training": _load_saved_training(),
}


def create_app(config=None):
    if config is None:
        config = AppConfig()
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'per-ddpg-edge'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    sim_state["config"] = config
    sim_state["speed"] = config.step_delay

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/config', methods=['GET'])
    def get_config():
        c = config
        return jsonify({
            "edge_num": c.env.edge_device_num, "fog_num": c.env.fog_node_num,
            "cloud_cpu": c.env.cloud_cpu, "fog_cpu": c.env.fog_cpu, "edge_cpu": c.env.edge_cpu,
            "max_episodes": c.env.max_episodes, "mode": sim_state["mode"],
            "speed": sim_state["speed"],
        })

    @app.route('/api/config', methods=['POST'])
    def update_config():
        data = request.get_json() or {}
        if 'speed' in data:
            sim_state["speed"] = float(data['speed'])
        if 'mode' in data:
            sim_state["mode"] = data['mode']
        if 'max_episodes' in data:
            config.env.max_episodes = max(1, min(int(data['max_episodes']), 5000))
        return jsonify({"status": "ok"})

    def _init_sim():
        c = sim_state["config"]
        env = CloudFogEdgeEnv(c.env)
        agent = PERDDPGAgent(env.get_state_dim(), env.get_action_dim(), c.ddpg, c.env)
        runner = PERDDPGRunner(env, agent, c.env, c.ddpg)
        sim_state["env"] = env
        sim_state["agent"] = agent
        sim_state["runner"] = runner
        sim_state["metrics"] = []
        runner.start_episode()

    @app.route('/api/simulation/start', methods=['POST'])
    def start_sim():
        _init_sim()
        sim_state["running"] = True
        sim_state["paused"] = False
        t = threading.Thread(target=_sim_loop, args=(socketio,), daemon=True)
        t.start()
        sim_state["thread"] = t
        return jsonify({"status": "started"})

    @app.route('/api/simulation/pause', methods=['POST'])
    def pause_sim():
        sim_state["paused"] = True
        return jsonify({"status": "paused"})

    @app.route('/api/simulation/resume', methods=['POST'])
    def resume_sim():
        sim_state["paused"] = False
        return jsonify({"status": "resumed"})

    @app.route('/api/simulation/step', methods=['POST'])
    def step_sim():
        if not sim_state["runner"]:
            _init_sim()
        explore = sim_state["mode"] == "train"
        detail = sim_state["runner"].run_step(explore=explore)
        if detail and detail.get("done"):
            _save_training_snapshot(sim_state["runner"])
            sim_state["runner"].start_episode()
        return jsonify(detail or {"error": "episode done"})

    @app.route('/api/simulation/reset', methods=['POST'])
    def reset_sim():
        sim_state["running"] = False
        sim_state["paused"] = False
        _init_sim()
        return jsonify({"status": "reset"})

    @app.route('/api/simulation/state', methods=['GET'])
    def get_state():
        if sim_state["runner"] and sim_state["runner"].env:
            snap = sim_state["runner"].env.get_state_snapshot()
            saved = sim_state.get("saved_training")
            use_saved = sim_state["runner"].total_episodes == 0 and saved
            prog = saved["progress"] if use_saved else sim_state["runner"].get_progress()
            snap["progress"] = prog
            snap["history"] = saved["history"] if use_saved else sim_state["runner"].get_history()
            snap["history_provenance"] = saved.get("provenance") if use_saved else "current process"
            return jsonify(snap)
        saved = sim_state.get("saved_training")
        return jsonify({
            "nodes": [], "dag": None, "list_seq": [],
            "progress": saved.get("progress", {}) if saved else {},
            "history": saved.get("history", {"rewards": [], "makespans": []}) if saved else {"rewards": [], "makespans": []},
            "history_provenance": saved.get("provenance") if saved else None,
        })

    @app.route('/api/simulation/metrics', methods=['GET'])
    def get_metrics():
        if sim_state["metrics"]:
            return jsonify(sim_state["metrics"])
        saved = sim_state.get("saved_training")
        return jsonify(saved.get("metrics", []) if saved else [])

    @app.route('/api/experiments/latest', methods=['GET'])
    def latest_experiment():
        output_dir = Path(__file__).resolve().parents[1] / "output" / "experiments"
        files = sorted(
            output_dir.glob("comparison_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return jsonify({"error": "no local experiment output"}), 404
        with files[0].open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["file"] = str(files[0])
        return jsonify(payload)

    def _sim_loop(socketio):
        runner = sim_state["runner"]
        while sim_state["running"]:
            if sim_state["paused"]:
                time.sleep(0.1)
                continue
            try:
                explore = sim_state["mode"] == "train"
                detail = runner.run_step(explore=explore)
                if detail is None:
                    break
                sim_state["current_detail"] = detail
                sim_state["metrics"].append({
                    "step": detail["step"],
                    "total_steps": detail["total_steps"],
                    "reward": detail["reward"],
                    "episode_reward": detail["episode_reward"],
                    "eta": detail["eta"],
                    "done": detail["done"],
                    "train": detail["train_info"],
                })
                if len(sim_state["metrics"]) > 500:
                    sim_state["metrics"] = sim_state["metrics"][-500:]

                socketio.emit('sim_step', {
                    "detail": detail,
                    "snapshot": runner.env.get_state_snapshot(),
                    "progress": runner.get_progress(),
                })

                if detail.get("done"):
                    _save_training_snapshot(runner)
                    socketio.emit('episode_end', {
                        "episode_reward": detail["episode_reward"],
                        "progress": runner.get_progress(),
                        "history": runner.get_history(),
                    })
                    if runner.total_episodes >= runner.env_config.max_episodes:
                        sim_state["running"] = False
                        socketio.emit('training_complete', {
                            "progress": runner.get_progress(),
                            "history": runner.get_history(),
                        })
                        break
                    time.sleep(min(1.0, max(0.02, sim_state["speed"] * 4)))
                    runner.start_episode()

                time.sleep(sim_state["speed"])
            except Exception as e:
                logger.error(f"Sim error: {e}")
                traceback.print_exc()
                sim_state["running"] = False
                socketio.emit('sim_error', {"message": str(e)})
                break

    return app, socketio
