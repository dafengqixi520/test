"""前端按钮单元测试 — 使用内置unittest，覆盖所有API端点、状态转换和边界情况"""
import sys, os, json, time, threading, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['NO_PROXY'] = '*'

import requests
from web.app import create_app
from config import AppConfig

BASE = "http://127.0.0.1:5002"

def setUpModule():
    """模块级setup: 启动测试服务器"""
    config = AppConfig()
    config.port = 5002
    config.debug = False
    config.use_llm = False
    app, socketio = create_app(config)
    t = threading.Thread(target=lambda: socketio.run(
        app, host="127.0.0.1", port=5002, debug=False, use_reloader=False), daemon=True)
    t.start()
    time.sleep(2)

def tearDownModule():
    requests.post(f"{BASE}/api/simulation/reset")


class TestPageLoad(unittest.TestCase):
    """首页加载、配置获取、场景信息"""

    def test_index_page_loads(self):
        r = requests.get(f"{BASE}/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("LeDRL", r.text)

    def test_config_endpoint(self):
        r = requests.get(f"{BASE}/api/config")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["edge_node_num"], 10)
        self.assertIn("topology_type", j)
        self.assertIn("scenario", j)

    def test_scenario_endpoint(self):
        r = requests.get(f"{BASE}/api/scenario")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["node_count"], 10)
        self.assertEqual(len(j["task_types"]), 5)

    def test_topology_endpoint_initial(self):
        r = requests.get(f"{BASE}/api/topology")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertIn("nodes", j)
        self.assertIn("edges", j)


class TestControlButtons(unittest.TestCase):
    """开始、暂停、继续、重置按钮"""

    def setUp(self):
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)

    def test_start_button(self):
        r = requests.post(f"{BASE}/api/simulation/start")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "started")

    def test_pause_button(self):
        requests.post(f"{BASE}/api/simulation/start")
        time.sleep(0.5)
        r = requests.post(f"{BASE}/api/simulation/pause")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "paused")

    def test_pause_stops_progress(self):
        requests.post(f"{BASE}/api/simulation/start")
        time.sleep(1)
        requests.post(f"{BASE}/api/simulation/pause")
        time.sleep(1.5)  # 确保后台线程已完成当前步并检测到暂停标志
        r = requests.get(f"{BASE}/api/simulation/state")
        step_before = r.json().get("step", 0)
        time.sleep(2)
        r2 = requests.get(f"{BASE}/api/simulation/state")
        step_after = r2.json().get("step", 0)
        self.assertEqual(step_after, step_before,
                        f"Pause failed: step {step_before} -> {step_after}")

    def test_resume_button(self):
        requests.post(f"{BASE}/api/simulation/start")
        time.sleep(0.3)
        requests.post(f"{BASE}/api/simulation/pause")
        r = requests.post(f"{BASE}/api/simulation/resume")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "resumed")

    def test_reset_button(self):
        requests.post(f"{BASE}/api/simulation/start")
        time.sleep(0.5)
        r = requests.post(f"{BASE}/api/simulation/reset")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "reset")

    def test_state_after_reset(self):
        requests.post(f"{BASE}/api/simulation/start")
        time.sleep(0.3)
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)
        r = requests.get(f"{BASE}/api/simulation/state")
        self.assertEqual(r.json().get("step", 0), 0)
        self.assertEqual(len(r.json().get("tasks", [])), 0)

    def test_double_start_no_crash(self):
        r1 = requests.post(f"{BASE}/api/simulation/start")
        r2 = requests.post(f"{BASE}/api/simulation/start")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

    def test_pause_when_idle_no_crash(self):
        r = requests.post(f"{BASE}/api/simulation/pause")
        self.assertEqual(r.status_code, 200)

    def test_resume_when_idle_no_crash(self):
        r = requests.post(f"{BASE}/api/simulation/resume")
        self.assertEqual(r.status_code, 200)


class TestStepButton(unittest.TestCase):
    """单步执行按钮"""

    def setUp(self):
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)

    def test_step_when_idle(self):
        r = requests.post(f"{BASE}/api/simulation/step")
        self.assertEqual(r.status_code, 200)
        detail = r.json()
        self.assertIn("step", detail)
        self.assertIn("actions", detail)
        self.assertEqual(len(detail["actions"]), 10)

    def test_step_increments(self):
        r1 = requests.post(f"{BASE}/api/simulation/step")
        s1 = r1.json()["step"]
        r2 = requests.post(f"{BASE}/api/simulation/step")
        s2 = r2.json()["step"]
        self.assertEqual(s2, s1 + 1, f"Step {s2} != {s1} + 1")

    def test_step_has_env_state(self):
        r = requests.post(f"{BASE}/api/simulation/step")
        detail = r.json()
        self.assertIn("env_state", detail)
        self.assertIn("nodes", detail["env_state"])

    def test_step_has_llm_details(self):
        r = requests.post(f"{BASE}/api/simulation/step")
        detail = r.json()
        self.assertIn("llm_details", detail)

    def test_step_has_lambda(self):
        r = requests.post(f"{BASE}/api/simulation/step")
        detail = r.json()
        self.assertIn("lambda_llm", detail)
        self.assertGreaterEqual(detail["lambda_llm"], 0.0)
        self.assertLessEqual(detail["lambda_llm"], 1.0)

    def test_step_has_reward(self):
        r = requests.post(f"{BASE}/api/simulation/step")
        detail = r.json()
        self.assertIn("reward", detail)
        self.assertIn("episode_return", detail)

    def test_step_has_step_stats(self):
        r = requests.post(f"{BASE}/api/simulation/step")
        detail = r.json()
        self.assertIn("step_stats", detail)
        stats = detail["step_stats"]
        for k in ["finished", "success", "failed", "dropped"]:
            self.assertIn(k, stats, f"step_stats missing '{k}'")

    def test_multiple_steps(self):
        for i in range(5):
            r = requests.post(f"{BASE}/api/simulation/step", timeout=30)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["step"], i + 1)


class TestConfigUpdates(unittest.TestCase):
    """速度滑块、LLM开关"""

    def test_update_step_delay(self):
        r = requests.post(f"{BASE}/api/config", json={"step_delay": 0.5})
        self.assertEqual(r.status_code, 200)
        r2 = requests.get(f"{BASE}/api/config")
        self.assertEqual(r2.json()["step_delay"], 0.5)

    def test_update_speed_min(self):
        r = requests.post(f"{BASE}/api/config", json={"step_delay": 0.01})
        self.assertEqual(r.status_code, 200)

    def test_update_speed_max(self):
        r = requests.post(f"{BASE}/api/config", json={"step_delay": 1.0})
        self.assertEqual(r.status_code, 200)

    def test_toggle_llm_on(self):
        r = requests.post(f"{BASE}/api/config", json={"use_llm": True})
        self.assertEqual(r.status_code, 200)

    def test_toggle_llm_off(self):
        r = requests.post(f"{BASE}/api/config", json={"use_llm": False})
        self.assertEqual(r.status_code, 200)

    def test_invalid_key_ignored(self):
        r = requests.post(f"{BASE}/api/config", json={"nonexistent": 999})
        self.assertEqual(r.status_code, 200)

    def test_empty_body(self):
        r = requests.post(f"{BASE}/api/config", json={})
        self.assertEqual(r.status_code, 200)


class TestStateQueries(unittest.TestCase):
    """状态查询接口"""

    def setUp(self):
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)
        requests.post(f"{BASE}/api/simulation/step")
        time.sleep(0.3)

    def test_state_schema(self):
        r = requests.get(f"{BASE}/api/simulation/state")
        j = r.json()
        for key in ["step", "nodes", "tasks", "topology", "env_stats"]:
            self.assertIn(key, j, f"Missing key: {key}")

    def test_node_schema(self):
        r = requests.get(f"{BASE}/api/simulation/state")
        for node in r.json()["nodes"]:
            for key in ["id", "name", "type", "cpu_cores", "is_active",
                        "queue_usage", "success_rate", "neighbors"]:
                self.assertIn(key, node, f"Node missing key: {key}")

    def test_topology_schema(self):
        r = requests.get(f"{BASE}/api/simulation/state")
        topo = r.json()["topology"]
        self.assertIn("nodes", topo)
        self.assertIn("edges", topo)
        self.assertEqual(len(topo["nodes"]), 10)

    def test_task_schema(self):
        r = requests.get(f"{BASE}/api/simulation/state")
        tasks = r.json().get("tasks", [])
        for task in tasks:
            for key in ["task_id", "status", "task_size", "hop_count", "path"]:
                self.assertIn(key, task, f"Task missing key: {key}")

    def test_metrics_endpoint(self):
        r = requests.get(f"{BASE}/api/simulation/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_metrics_schema(self):
        r = requests.get(f"{BASE}/api/simulation/metrics")
        metrics = r.json()
        if len(metrics) > 0:
            m = metrics[0]
            for key in ["step", "reward", "success", "failed",
                        "lambda_llm", "cumulative_success_rate"]:
                self.assertIn(key, m, f"Metrics missing: {key}")

    def test_topology_endpoint_after_step(self):
        r = requests.get(f"{BASE}/api/topology")
        j = r.json()
        self.assertEqual(len(j["nodes"]), 10)
        self.assertGreater(len(j["edges"]), 0)


class TestLLMLogs(unittest.TestCase):
    """LLM日志查询"""

    def test_llm_logs_empty(self):
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)
        r = requests.get(f"{BASE}/api/llm/logs")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_llm_logs_with_limit(self):
        r = requests.get(f"{BASE}/api/llm/logs?limit=5")
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(len(r.json()), 5)

    def test_llm_stats_schema(self):
        r = requests.get(f"{BASE}/api/llm/stats")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        for key in ["total_llm_requests", "successful_llm_responses",
                    "total_reflections"]:
            self.assertIn(key, j, f"Missing key: {key}")


class TestFrontendRendering(unittest.TestCase):
    """HTML模板渲染: 按钮、Tab、图表"""

    @classmethod
    def setUpClass(cls):
        r = requests.get(f"{BASE}/")
        cls.html = r.text

    def test_has_start_button(self):
        self.assertIn("btnStart", self.html)

    def test_has_pause_button(self):
        self.assertIn("btnPause", self.html)

    def test_has_resume_button(self):
        self.assertIn("btnResume", self.html)

    def test_has_step_button(self):
        self.assertIn("btnStep", self.html)

    def test_has_reset_button(self):
        self.assertIn("btnReset", self.html)

    def test_has_5_tabs(self):
        tabs = ["dashboard", "topology", "taskflow", "nodes", "llmlogs"]
        for tab in tabs:
            self.assertIn(tab.lower(), self.html.lower(),
                         f"Tab '{tab}' not found")

    def test_has_charts(self):
        for cid in ["chartSuccessRate", "chartTaskFlow", "chartReturn", "chartLambda"]:
            self.assertIn(cid, self.html, f"Chart '{cid}' not found")

    def test_has_llm_toggle(self):
        self.assertIn("llmToggle", self.html)

    def test_has_speed_slider(self):
        self.assertIn("speedSlider", self.html)

    def test_has_metric_cards(self):
        for mid in ["metFinished", "metSuccessRate", "metDecTime", "metLambda"]:
            self.assertIn(mid, self.html, f"Metric card '{mid}' not found")

    def test_has_websocket(self):
        self.assertIn("socket.io", self.html.lower())

    @classmethod
    def tearDownClass(cls):
        cls.html = None


class TestConcurrency(unittest.TestCase):
    """并发 & 竞态测试"""

    def setUp(self):
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)

    def test_start_and_immediate_pause(self):
        r1 = requests.post(f"{BASE}/api/simulation/start")
        r2 = requests.post(f"{BASE}/api/simulation/pause")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

    def test_start_and_immediate_reset(self):
        requests.post(f"{BASE}/api/simulation/start")
        r = requests.post(f"{BASE}/api/simulation/reset")
        self.assertEqual(r.status_code, 200)

    def test_rapid_steps(self):
        for i in range(5):
            r = requests.post(f"{BASE}/api/simulation/step", timeout=30)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["step"], i + 1)

    def test_simultaneous_starts(self):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(requests.post, f"{BASE}/api/simulation/start")
                      for _ in range(3)]
            results = [f.result() for f in futures]
        for r in results:
            self.assertIn(r.status_code, [200, 500])


class TestEdgeCases(unittest.TestCase):
    """边界情况"""

    def test_step_then_reset_then_step(self):
        """单步 -> 重置 -> 单步: 步数应重置"""
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)
        r1 = requests.post(f"{BASE}/api/simulation/step")
        self.assertEqual(r1.json()["step"], 1)
        requests.post(f"{BASE}/api/simulation/reset")
        time.sleep(0.3)
        r2 = requests.post(f"{BASE}/api/simulation/step")
        self.assertEqual(r2.json()["step"], 1)

    def test_pause_resume_pause(self):
        """暂停 -> 继续 -> 暂停"""
        requests.post(f"{BASE}/api/simulation/start")
        time.sleep(0.5)
        r = requests.post(f"{BASE}/api/simulation/pause")
        self.assertEqual(r.status_code, 200)
        r = requests.post(f"{BASE}/api/simulation/resume")
        self.assertEqual(r.status_code, 200)
        r = requests.post(f"{BASE}/api/simulation/pause")
        self.assertEqual(r.status_code, 200)

    def test_config_persistence(self):
        """配置修改后持久化"""
        requests.post(f"{BASE}/api/config", json={"step_delay": 0.77})
        r = requests.get(f"{BASE}/api/config")
        self.assertEqual(r.json()["step_delay"], 0.77)
        requests.post(f"{BASE}/api/config", json={"step_delay": 0.1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
