"""计算节点 — 云节点 / 雾节点 / 边缘设备"""
import numpy as np


class ComputeNode:
    """通用计算节点：云/雾/边缘设备"""

    def __init__(self, node_id, node_type, cpu_capacity, env_config):
        self.id = node_id
        self.type = node_type        # 'cloud' / 'fog' / 'edge'
        self.cpu_total = cpu_capacity  # GHz
        self.cpu_available = cpu_capacity  # 当前可用
        self.env_config = env_config
        self.task_queue = []          # 等待执行的任务列表
        self.current_task = None      # 正在执行的任务
        self.reservations = []        # 已调度任务的CPU时间区间

        # 统计
        self.total_tasks_processed = 0
        self.total_exec_time = 0.0

    @property
    def is_busy(self):
        return self.current_task is not None

    @property
    def queue_length(self):
        return len(self.task_queue)

    def allocate_resource(self, ratio):
        """分配ratio比例的计算资源给新任务（0 < ratio <= 1）"""
        allocated = self.cpu_available * ratio
        self.cpu_available -= allocated
        return allocated

    def release_resource(self, allocated):
        """释放已分配的计算资源"""
        self.cpu_available = min(self.cpu_total, self.cpu_available + allocated)

    def available_cpu_at(self, timestamp):
        used = sum(
            item["cpu"] for item in self.reservations
            if item["start"] <= timestamp < item["finish"]
        )
        return max(0.0, self.cpu_total - used)

    def _minimum_available(self, start, finish):
        points = [start]
        points.extend(
            item["start"] for item in self.reservations
            if start < item["start"] < finish
        )
        return min(self.available_cpu_at(point) for point in points)

    def schedule(self, task, earliest_start, resource_ratio):
        """Reserve a feasible CPU interval under constraints 3.16 and 3.17."""
        candidate = max(0.0, float(earliest_start))
        ratio = float(np.clip(resource_ratio, 0.01, 1.0))
        if self.cpu_total + 1e-12 < task.min_cpu:
            raise RuntimeError(
                f"Node {self.id} physical CPU {self.cpu_total} cannot satisfy "
                f"minimum CPU {task.min_cpu}"
            )

        for _ in range(max(16, len(self.reservations) * 4 + 8)):
            available = self.available_cpu_at(candidate)
            if available + 1e-12 < task.min_cpu:
                endings = [r["finish"] for r in self.reservations if r["finish"] > candidate]
                if not endings:
                    raise RuntimeError(
                        f"Node {self.id} cannot satisfy minimum CPU {task.min_cpu}"
                    )
                candidate = min(endings)
                continue

            requested_cpu = available * ratio
            allocated = min(available, max(task.min_cpu, requested_cpu))
            for _refine in range(8):
                duration = task.total_cycles / (allocated * 1e9)
                finish = candidate + duration
                feasible = self._minimum_available(candidate, finish)
                new_allocated = min(allocated, feasible)
                if new_allocated + 1e-12 < task.min_cpu:
                    break
                if abs(new_allocated - allocated) < 1e-12:
                    reservation = {
                        "task_id": task.id,
                        "start": candidate,
                        "finish": finish,
                        "cpu": allocated,
                        "available_at_start": available,
                        "requested_cpu": requested_cpu,
                        "effective_ratio": allocated / available,
                    }
                    self.reservations.append(reservation)
                    self.reservations.sort(key=lambda item: (item["start"], item["finish"]))
                    self.total_tasks_processed += 1
                    self.total_exec_time += duration
                    self.cpu_available = self.available_cpu_at(candidate)
                    return reservation
                allocated = new_allocated

            conflicts = [r["finish"] for r in self.reservations if r["finish"] > candidate]
            if not conflicts:
                raise RuntimeError(f"Unable to schedule task {task.id} on {self.id}")
            candidate = min(conflicts)

        raise RuntimeError(f"Scheduling search did not converge for task {task.id}")

    def submit_task(self, task, allocated_cpu):
        """提交任务到此节点执行"""
        task.assigned_node = self.id
        task.allocated_cpu = allocated_cpu
        if self.current_task is None:
            self.current_task = task
        else:
            self.task_queue.append(task)

    def step(self, dt=0.01):
        """时间步进，执行当前任务dt秒"""
        finished_tasks = []
        if self.current_task is not None:
            self.current_task.remaining_cycles -= self.current_task.allocated_cpu * dt * 1e9
            if self.current_task.remaining_cycles <= 0:
                self.current_task.remaining_cycles = 0
                finished_tasks.append(self.current_task)
                self.release_resource(self.current_task.allocated_cpu)
                self.total_tasks_processed += 1
                # 从队列取下一个任务
                if self.task_queue:
                    self.current_task = self.task_queue.pop(0)
                else:
                    self.current_task = None
        return finished_tasks

    def reset(self):
        self.cpu_available = self.cpu_total
        self.task_queue = []
        self.current_task = None
        self.reservations = []
        self.total_tasks_processed = 0
        self.total_exec_time = 0.0

    def get_state(self, current_time=0.0):
        available = self.available_cpu_at(current_time)
        active = [
            item for item in self.reservations
            if item["start"] <= current_time < item["finish"]
        ]
        queued = [item for item in self.reservations if item["start"] > current_time]
        return {
            "id": self.id,
            "type": self.type,
            "cpu_total": self.cpu_total,
            "cpu_available": available,
            "cpu_usage": 1 - available / self.cpu_total,
            "queue_length": len(queued),
            "is_busy": bool(active),
            "total_processed": self.total_tasks_processed,
            "reservations": [dict(item) for item in self.reservations],
        }
