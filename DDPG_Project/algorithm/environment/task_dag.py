"""DAG依赖任务模型"""
import numpy as np


INDUSTRIAL_TASK_NAMES = (
    "Vibration windowing",
    "Thermal preprocessing",
    "Acoustic feature extraction",
    "Surface defect inference",
    "Motor health diagnosis",
    "Multi-sensor fusion",
    "Quality drift prediction",
    "Maintenance risk scoring",
    "Adaptive process tuning",
    "Alarm aggregation",
    "Production quality report",
    "Control command validation",
)


class SubTask:
    """单个子任务，论文中的 I_i = {C_i, ρ_i, f_i^min}"""

    def __init__(self, task_id, data_size_kb, compute_density, min_cpu=0.1, name=None):
        self.id = task_id                           # 任务编号
        base_name = INDUSTRIAL_TASK_NAMES[task_id % len(INDUSTRIAL_TASK_NAMES)]
        self.name = name or base_name
        self.data_size = data_size_kb * 1024 * 8    # 数据量 (bits)，C_i
        self.compute_density = compute_density       # 计算密度 Cycle/MB，ρ_i
        self.min_cpu = min_cpu                       # 最小所需计算资源 (GHz)，f_i^min
        self.total_cycles = (data_size_kb / 1024) * compute_density * 1e6  # 总CPU周期

        # 运行时状态
        self.remaining_cycles = self.total_cycles
        self.assigned_node = None     # 被分配到的节点ID
        self.allocated_cpu = 0.0      # 分配到的计算资源(GHz)
        self.start_time = None        # 开始执行时刻
        self.finish_time = None       # 完成时刻
        self.ready_time = 0.0         # 最早就绪时刻（RT_i，等前置任务完成+数据传输）
        self.input_transfer_time = 0.0
        self.dependency_transfer_time = 0.0
        self.wait_time = 0.0
        self.exec_time = 0.0
        self.status = "pending"       # pending/ready/executing/completed

        # 依赖关系
        self.predecessors = []        # 直接前驱任务ID列表
        self.successors = []          # 直接后继任务ID列表

        # 算法3.1计算值
        self.rank = 0.0               # 静态优先级
        self.urgency = 0.0            # 动态紧迫性
        self.schedule_priority = 0.0  # 卸载调度优先级 W_i

    @property
    def is_ready(self):
        """所有前驱任务是否已完成"""
        return self.status == "ready"

    @property
    def edge_density(self):
        possible = self.N * (self.N - 1) // 2
        return len(self.edges) / possible if possible else 0.0
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "data_size_kb": self.data_size / (1024 * 8),
            "compute_density": self.compute_density,
            "min_cpu": self.min_cpu,
            "status": self.status,
            "assigned_node": self.assigned_node,
            "allocated_cpu": round(self.allocated_cpu, 3),
            "remaining_cycles_pct": round(
                self.remaining_cycles / max(1, self.total_cycles) * 100, 1),
            "ready_time": round(self.ready_time, 3),
            "start_time": None if self.start_time is None else round(self.start_time, 4),
            "finish_time": None if self.finish_time is None else round(self.finish_time, 4),
            "input_transfer_time": round(self.input_transfer_time, 4),
            "dependency_transfer_time": round(self.dependency_transfer_time, 4),
            "wait_time": round(self.wait_time, 4),
            "exec_time": round(self.exec_time, 4),
            "predecessors": self.predecessors,
            "successors": self.successors,
            "rank": round(self.rank, 2),
            "urgency": round(self.urgency, 4),
            "schedule_priority": round(self.schedule_priority, 4),
        }


class DAGTask:
    """DAG聚合任务 I*，包含N个有依赖关系的子任务"""

    def __init__(self, env_config, rng=None, tx_rates=None):
        self.config = env_config
        self.rng = rng or np.random.RandomState()
        self.tx_rates = tx_rates
        self.N = env_config.edge_device_num
        self.subtasks = []
        self.edges = []  # (from_id, to_id, data_amount)
        self.adj_matrix = None  # 邻接矩阵
        self._build_dag()
        self._compute_avg_times()

    def _build_dag(self):
        """生成随机DAG"""
        # 创建N个子任务
        for i in range(self.N):
            size = self.rng.uniform(*self.config.task_size_range)
            density = self.rng.uniform(*self.config.task_density_range)
            task = SubTask(
                i, size, density,
                min_cpu=self.config.edge_cpu * 0.1,
                name=INDUSTRIAL_TASK_NAMES[i % len(INDUSTRIAL_TASK_NAMES)],
            )
            self.subtasks.append(task)

        # 从全部 i < j 候选边中按目标密度无放回抽样，天然保证无环。
        self.adj_matrix = np.zeros((self.N, self.N), dtype=int)
        candidates = [(i, j) for i in range(self.N) for j in range(i + 1, self.N)]
        target_count = round(self.config.dag_edge_prob * len(candidates))
        if target_count:
            selected = self.rng.choice(
                len(candidates), size=target_count, replace=False
            )
            for candidate_index in np.atleast_1d(selected):
                i, j = candidates[int(candidate_index)]
                self.adj_matrix[i][j] = 1
                data_amount = self.rng.uniform(0.1, 0.5) * self.subtasks[i].data_size
                self.edges.append((i, j, data_amount))
        # 更新前后驱关系
        for i in range(self.N):
            self.subtasks[i].predecessors = [
                j for j in range(self.N) if self.adj_matrix[j][i] == 1]
            self.subtasks[i].successors = [
                j for j in range(self.N) if self.adj_matrix[i][j] == 1]

        # 找到入口和出口任务
        self.entry_tasks = [i for i in range(self.N)
                           if len(self.subtasks[i].predecessors) == 0]
        self.exit_tasks = [i for i in range(self.N)
                          if len(self.subtasks[i].successors) == 0]

    def _compute_avg_times(self):
        """计算每个子任务在本地/雾/云的平均处理时间（用于rank计算）"""
        for task in self.subtasks:
            cycles = task.total_cycles
            # 本地执行时延
            task.T_local = cycles / (self.config.edge_cpu * 1e9)
            # 雾节点执行时延（含传输）
            fog_rate = self._nearest_fog_rate(task.id)
            tx_time = task.data_size / fog_rate if fog_rate > 0 else 0
            exec_time = cycles / (self.config.fog_cpu * 1e9)
            task.T_fog = tx_time + exec_time
            # 云节点执行时延（含传输）
            tx_to_cloud = (task.data_size / fog_rate +
                          task.data_size / self.config.wan_rate +
                          self.config.wan_delay)
            exec_cloud = cycles / (self.config.cloud_cpu * 1e9)
            task.T_cloud = tx_to_cloud + exec_cloud
            # 平均处理时延 (公式3.18)
            task.T_avg = (task.T_local + task.T_fog + task.T_cloud) / 3.0

    def _avg_transmission_rate(self):
        """估算平均传输速率（香农公式3.3）"""
        avg_dist = np.mean(self.config.device_fog_distance)
        h = avg_dist ** (-self.config.path_loss_exponent)
        rate = self.config.uplink_bandwidth * np.log2(
            1 + h * self.config.edge_tx_power / self.config.noise_power)
        return max(rate, 1e6)

    def _nearest_fog_rate(self, task_id):
        """Equation 3.18 uses the nearest/lowest-cost fog node."""
        if self.tx_rates is not None and len(self.tx_rates) > task_id:
            return max(float(np.max(self.tx_rates[task_id])), 1e6)
        return self._avg_transmission_rate()

    def get_edge_data(self, from_id, to_id):
        """获取从前驱任务到后继任务的中间数据量"""
        for f, t, d in self.edges:
            if f == from_id and t == to_id:
                return d
        return 0

    @property
    def edge_density(self):
        possible = self.N * (self.N - 1) // 2
        return len(self.edges) / possible if possible else 0.0
    def to_dict(self):
        return {
            "N": self.N,
            "edges": [(f, t, round(d / 1024, 1)) for f, t, d in self.edges],
            "entry_tasks": self.entry_tasks,
            "exit_tasks": self.exit_tasks,
            "subtasks": [t.to_dict() for t in self.subtasks],
        }
