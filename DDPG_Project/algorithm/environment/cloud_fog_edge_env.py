"""云-雾-边三层计算网络环境 — 论文第三章"""
import numpy as np
from algorithm.environment.compute_node import ComputeNode
from algorithm.environment.task_dag import DAGTask


class CloudFogEdgeEnv:
    """
    云-雾-边协同计算环境
    - 1个云节点 (F0), M个雾节点 (F1..FM), N个边缘设备 (D1..DN)
    - DAG依赖型子任务
    - OFDMA无线通信 + 光纤骨干网
    """

    def __init__(self, env_config):
        env_config.validate()
        self.config = env_config
        self.N = env_config.edge_device_num   # 边缘设备数 = 子任务数
        self.M = env_config.fog_node_num       # 雾节点数
        self.rng = np.random.RandomState(env_config.seed)

        # 创建所有计算节点
        self.nodes = {}  # key: "cloud_0", "fog_1", "edge_0"...
        self.all_node_ids = []

        # 云节点
        cloud = ComputeNode("cloud_0", "cloud", env_config.cloud_cpu, env_config)
        self.nodes["cloud_0"] = cloud
        self.all_node_ids.append("cloud_0")
        self.cloud = cloud

        # 雾节点
        self.fog_nodes = []
        for i in range(self.M):
            fid = f"fog_{i+1}"
            fog = ComputeNode(fid, "fog", env_config.fog_cpu, env_config)
            self.nodes[fid] = fog
            self.all_node_ids.append(fid)
            self.fog_nodes.append(fog)

        # 边缘设备
        self.edge_devices = []
        for i in range(self.N):
            eid = f"edge_{i}"
            edge = ComputeNode(eid, "edge", env_config.edge_cpu, env_config)
            self.nodes[eid] = edge
            self.all_node_ids.append(eid)
            self.edge_devices.append(edge)

        # 设备到雾节点的距离
        self.device_fog_distances = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                self.device_fog_distances[i][j] = self.rng.uniform(
                    *env_config.device_fog_distance)

        # 计算传输速率矩阵 [edge_idx][fog_idx] (公式3.3)
        self.tx_rates = np.zeros((self.N, self.M))
        for i in range(self.N):
            for j in range(self.M):
                d = self.device_fog_distances[i][j]
                h = d ** (-env_config.path_loss_exponent)  # 公式3.4
                self.tx_rates[i][j] = env_config.uplink_bandwidth * np.log2(
                    1 + h * env_config.edge_tx_power / env_config.noise_power)

        # 当前DAG任务
        self.dag = None
        self.list_seq = []    # 卸载调度优先级序列（算法3.1输出）
        self.current_task_idx = 0  # 当前处理到序列中第几个任务
        self.global_time = 0.0
        self.episode_done = False
        self.episode_index = 0

        # 已完成任务追踪
        self.completed_tasks = set()
        self.task_finish_times = {}  # task_id -> finish_time
        self.last_step_info = None

    def reset(self):
        """重置环境，生成新的DAG任务"""
        self.rng = np.random.RandomState(self.config.seed + self.episode_index)
        self.episode_index += 1
        self.dag = DAGTask(self.config, rng=self.rng, tx_rates=self.tx_rates)
        self.list_seq = self._decompose_dag()  # 算法3.1
        self.current_task_idx = 0
        self.global_time = 0.0
        self.episode_done = False
        self.completed_tasks = set()
        self.task_finish_times = {}
        self.last_step_info = None

        # 重置所有节点
        for node in self.nodes.values():
            node.reset()

        return self._get_state()

    def _decompose_dag(self):
        """算法3.1: 基于优先级和紧迫性的DAG分解，返回List_seq"""
        tasks = self.dag.subtasks

        # Step 1: 计算静态优先级 rank_i (公式3.19)
        # 沿后继方向递归计算，必须覆盖所有任务。
        def compute_rank(task_id, visited=None):
            if visited is None:
                visited = set()
            if task_id in visited:
                return tasks[task_id].rank
            visited.add(task_id)
            task = tasks[task_id]
            succ_ranks = [compute_rank(s, visited.copy()) for s in task.successors]
            max_succ = max(succ_ranks) if succ_ranks else 0
            task.rank = task.T_avg + max_succ  # 公式3.19
            return task.rank

        for task in tasks:
            compute_rank(task.id)

        rank_max = max(t.rank for t in tasks)
        for t in tasks:
            t.rank_norm = t.rank / rank_max if rank_max > 0 else 0

        # Step 2-3: 生成调度序列
        T_now = 0.0
        deadline = self.config.task_deadline
        omega = self.config.weight_factor_omega

        # 入度为0的任务入就绪队列
        ready_set = set(self.dag.entry_tasks)
        completed = set()
        list_seq = []

        while ready_set:
            # 计算紧迫性 (公式3.20)
            urgencies = {}
            insufficient = []
            for tid in ready_set:
                t = tasks[tid]
                slack = (deadline - T_now) - t.rank
                if slack > 1e-9:
                    urgencies[tid] = 1.0 / slack
                else:
                    insufficient.append(tid)

            # 公式3.21：时间预算不足的任务赋予当前队列最高紧迫性。
            highest_finite = max(urgencies.values(), default=1e9)
            for tid in insufficient:
                urgencies[tid] = highest_finite
            for tid, urgency in urgencies.items():
                tasks[tid].urgency = urgency

            urgency_max = max(urgencies.values()) if urgencies else 1.0

            # 计算调度优先级 W_i (公式3.22)
            best_tid = None
            best_W = -float('inf')
            for tid in ready_set:
                t = tasks[tid]
                urg_norm = t.urgency / urgency_max if urgency_max > 0 else 0
                W = omega * t.rank_norm + (1 - omega) * urg_norm  # 公式3.22
                t.schedule_priority = W
                if W > best_W:
                    best_W = W
                    best_tid = tid

            # 加入序列
            list_seq.append(best_tid)
            completed.add(best_tid)
            ready_set.remove(best_tid)

            # 推进虚拟时间
            T_now += tasks[best_tid].T_avg

            # 解锁后继任务
            for succ_id in tasks[best_tid].successors:
                if all(p in completed for p in tasks[succ_id].predecessors):
                    ready_set.add(succ_id)

        return list_seq

    def _get_state(self):
        """构建当前状态 s_t = {I_t, CR_t, R_t} (公式3.23)"""
        if self.current_task_idx >= len(self.list_seq):
            return None

        tid = self.list_seq[self.current_task_idx]
        task = self.dag.subtasks[tid]

        # 动作选择前目标节点未知，RT_i先取所有前驱完成时刻；
        # 目标相关的中间结果传输时间在step中按公式3.8补入。
        if task.predecessors:
            task.ready_time = max(
                self.task_finish_times.get(pid, 0.0)
                for pid in task.predecessors
            )
        else:
            task.ready_time = 0.0

        # I_t: 当前任务信息
        I_t = np.array([
            task.data_size / 1e6,          # 归一化数据量
            task.compute_density / 2000,   # 归一化计算密度
            task.min_cpu / self.config.cloud_cpu,   # 归一化最小CPU
            task.ready_time / self.config.task_deadline,  # 归一化就绪时间
        ], dtype=np.float32)

        # CR_t: 所有节点的剩余资源
        CR_t = []
        for nid in self.all_node_ids:
            node = self.nodes[nid]
            CR_t.append(node.available_cpu_at(task.ready_time) / node.cpu_total)
        CR_t = np.array(CR_t, dtype=np.float32)

        # R_t: 当前边缘设备的上行传输速率
        edge_idx = tid  # 子任务i由边缘设备i产生
        R_t = np.zeros(self.M, dtype=np.float32)
        for j in range(self.M):
            R_t[j] = self.tx_rates[edge_idx][j] / 1e8  # 归一化

        # 拼接状态
        state = np.concatenate([I_t, CR_t, R_t]).astype(np.float32)
        return state

    def _validated_action(self, action):
        """校验并规范化论文动作 a_t={A_t,F_t}。"""
        validated = np.asarray(action, dtype=np.float32).reshape(-1)
        if validated.size != self.get_action_dim():
            raise ValueError(
                f"action length must be {self.get_action_dim()}, got {validated.size}"
            )
        if not np.all(np.isfinite(validated)):
            raise ValueError("action values must be finite")

        node_probs = np.clip(validated[:-1], 0.0, None)
        probability_sum = float(node_probs.sum())
        if probability_sum <= 1e-12:
            node_probs.fill(1.0 / len(node_probs))
        else:
            node_probs /= probability_sum
        resource_ratio = float(np.clip(validated[-1], 0.01, 1.0))
        return np.concatenate([
            node_probs, np.array([resource_ratio], dtype=np.float32)
        ])
    def step(self, action):
        """
        执行动作 a_t = {A_t, F_t} (公式3.24)
        A_t: 目标节点索引 (离散→连续化: softmax后继概率最大的节点)
        F_t: 资源分配比例 (0, 1]
        返回: (next_state, reward, done)
        """
        if self.current_task_idx >= len(self.list_seq):
            return None, 0.0, True

        action = self._validated_action(action)
        tid = self.list_seq[self.current_task_idx]
        task = self.dag.subtasks[tid]

        # 解析论文动作：云 + M个雾节点 + 当前任务本地节点。
        feasible_node_ids = self.get_feasible_node_ids(tid)
        node_idx_probs = action[:-1]
        resource_ratio = np.clip(action[-1], 0.01, 1.0)  # F_t ∈ (0,1]
        target_node_idx = int(np.argmax(node_idx_probs))
        target_nid = feasible_node_ids[target_node_idx]
        target_node = self.nodes[target_nid]

        # 公式3.1/3.8：所有前驱完成且中间结果传输完成后才可执行。
        input_tx = self._task_upload_time(task, target_nid)
        dependency_ready = 0.0
        dependency_tx = 0.0
        for pred_id in task.predecessors:
            predecessor = self.dag.subtasks[pred_id]
            edge_data = self.dag.get_edge_data(pred_id, tid)
            transfer = self._inter_node_transfer(predecessor, target_nid, edge_data)
            dependency_tx = max(dependency_tx, transfer)
            dependency_ready = max(
                dependency_ready, predecessor.finish_time + transfer
            )
        arrival_time = max(input_tx, dependency_ready)

        # 公式3.16/3.17：寻找满足最小CPU及并发容量约束的执行区间。
        reservation = target_node.schedule(task, arrival_time, resource_ratio)
        allocated = reservation["cpu"]
        start_time = reservation["start"]
        finish_time = reservation["finish"]
        wait_time = max(0.0, start_time - arrival_time)
        task.allocated_cpu = allocated
        task.assigned_node = target_nid
        task.status = "executing"
        task.ready_time = arrival_time
        task.start_time = start_time
        task.finish_time = finish_time
        task.input_transfer_time = input_tx
        task.dependency_transfer_time = dependency_tx
        task.wait_time = wait_time
        task.exec_time = finish_time - start_time

        # 更新全局时间和完成记录
        self.global_time = max(self.global_time, finish_time)
        self.task_finish_times[tid] = finish_time
        self.completed_tasks.add(tid)
        task.status = "completed"

        # 推进到下一任务
        self.current_task_idx += 1

        # 计算奖励: R = -max(FT_i) (公式3.25)
        max_ft = max(self.task_finish_times.values()) if self.task_finish_times else 0.0
        reward = -max_ft
        self.last_step_info = {
            "task_id": int(tid),
            "node_id": target_nid,
            "node_type": target_node.type,
            "resource_ratio": float(resource_ratio),
            "requested_resource_ratio": float(resource_ratio),
            "requested_cpu": float(reservation["requested_cpu"]),
            "allocated_cpu": float(allocated),
            "effective_resource_ratio": float(reservation["effective_ratio"]),
            "ready_time": float(arrival_time),
            "start_time": float(start_time),
            "finish_time": float(finish_time),
            "input_transfer_time": float(input_tx),
            "dependency_transfer_time": float(dependency_tx),
            "wait_time": float(wait_time),
            "exec_time": float(finish_time - start_time),
        }

        # 判断是否完成所有任务
        done = self.current_task_idx >= len(self.list_seq)

        next_state = self._get_state() if not done else np.zeros(
            self.get_state_dim(), dtype=np.float32
        )

        return next_state, reward, done

    def _task_upload_time(self, task, target_nid):
        """Equations 3.5-3.6 for the task's input data."""
        target = self.nodes[target_nid]
        edge_idx = task.id % self.N
        if target.type == "edge":
            return 0.0
        if target.type == "fog":
            fog_idx = int(target_nid.split("_")[1]) - 1
            rate = max(float(self.tx_rates[edge_idx][fog_idx]), 1e-9)
            return task.data_size / rate
        nearest_rate = max(float(np.max(self.tx_rates[edge_idx])), 1e-9)
        return (task.data_size / nearest_rate +
                task.data_size / self.config.wan_rate + self.config.wan_delay)

    def _inter_node_transfer(self, predecessor, target_nid, data_bits):
        """Transfer a predecessor's intermediate result to the selected node."""
        source_nid = predecessor.assigned_node
        if not source_nid or source_nid == target_nid or data_bits <= 0:
            return 0.0
        source = self.nodes[source_nid]
        target = self.nodes[target_nid]
        edge_idx = predecessor.id % self.N
        nearest_rate = max(float(np.max(self.tx_rates[edge_idx])), 1e-9)

        if source.type == "edge" and target.type == "fog":
            fog_idx = int(target_nid.split("_")[1]) - 1
            rate = max(float(self.tx_rates[edge_idx][fog_idx]), 1e-9)
            return data_bits / rate
        if source.type == "edge" and target.type == "cloud":
            return (data_bits / nearest_rate + data_bits / self.config.wan_rate +
                    self.config.wan_delay)
        if {source.type, target.type} == {"fog", "cloud"}:
            return data_bits / self.config.wan_rate + self.config.wan_delay
        if source.type == "fog" and target.type == "fog":
            return data_bits / self.config.wan_rate
        return data_bits / nearest_rate

    def get_feasible_node_ids(self, task_id=None):
        """返回论文动作A_t允许选择的云、雾和当前任务本地节点。"""
        if task_id is None:
            if self.current_task_idx >= len(self.list_seq):
                raise RuntimeError("No current task is available")
            task_id = self.list_seq[self.current_task_idx]
        return ["cloud_0", *[node.id for node in self.fog_nodes], f"edge_{task_id}"]

    def get_state_dim(self):
        """状态空间维度: I_t(4) + CR_t(1+M+N) + R_t(M)。"""
        return 4 + len(self.all_node_ids) + self.M

    def get_action_dim(self):
        """动作空间维度: 可行节点选择(M+2) + 资源分配(1)。"""
        return self.M + 3

    def get_node_count(self):
        return len(self.all_node_ids)

    def get_state_snapshot(self):
        """获取当前环境快照（前端用）"""
        nodes_state = [n.get_state(self.global_time) for n in self.nodes.values()]
        dag_data = self.dag.to_dict() if self.dag else None
        return {
            "nodes": nodes_state,
            "dag": dag_data,
            "list_seq": [int(t) for t in self.list_seq],
            "current_idx": self.current_task_idx,
            "global_time": round(self.global_time, 3),
            "completed": len(self.completed_tasks),
            "total_tasks": self.N,
            "task_finish_times": {str(k): round(v, 4)
                                 for k, v in self.task_finish_times.items()},
        }
