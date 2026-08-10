# 论文一致性核心严谨性改造实施计划

> **执行要求：** 使用 `superpowers:executing-plans` 在当前会话逐项执行。每项生产代码修改必须先写测试、确认测试按预期失败，再进行最小实现。

**目标：** 在不增加论文外状态、奖励和学习机制的前提下，修正 DAG 密度、动作空间、资源审计、模型兼容和实验统计问题。

**架构：** 保留算法 3.1 与算法 3.2 两阶段结构。环境新增当前任务可行节点映射，Actor 动作缩减为 `M+3`；DAG 生成按目标边数采样；实验层在单种子实现之上增加消融组合与多种子汇总。

**技术栈：** Python 3.8、NumPy、PyTorch、`unittest`、Flask、JSON、CSV。

## 全局约束

- 状态严格保持 `s_t={I_t,CR_t,R_t}`，不增加论文外字段。
- 奖励严格保持 `R_t=-max(FT_i)`。
- 动作为 `M+2` 个节点概率和一个资源比例，总维度 `M+3`。
- 保留现有 Web 路由和 `resource_ratio` 字段。
- 旧检查点允许失效，新模型结构版本固定为 `2`。
- Smoke 实验只验证流程，不声明复现论文性能排序。

---

### 任务 1：按目标边密度生成 DAG

**文件：**
- 修改：`config.py`
- 修改：`algorithm/environment/task_dag.py`
- 测试：`tests/test_paper_alignment.py`

**接口：**
- `EnvConfig.validate() -> None`
- `DAGTask._build_dag() -> None`
- `DAGTask.edge_density -> float`

- [ ] **步骤 1：编写失败测试**

```python
def test_dag_edge_count_matches_configured_density(self):
    config = self.make_config()
    config.env.edge_device_num = 8
    config.env.dag_edge_prob = 0.25
    env = CloudFogEdgeEnv(config.env)
    env.reset()
    possible = 8 * 7 // 2
    self.assertEqual(len(env.dag.edges), round(0.25 * possible))

def test_sparse_dag_can_have_multiple_entries(self):
    config = self.make_config()
    config.env.edge_device_num = 8
    config.env.dag_edge_prob = 0.0
    env = CloudFogEdgeEnv(config.env)
    env.reset()
    self.assertEqual(env.dag.entry_tasks, list(range(8)))

def test_invalid_dag_density_is_rejected(self):
    config = self.make_config()
    config.env.dag_edge_prob = 1.1
    with self.assertRaisesRegex(ValueError, "dag_edge_prob"):
        CloudFogEdgeEnv(config.env)
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_alignment.PaperAlignmentTests.test_dag_edge_count_matches_configured_density tests.test_paper_alignment.PaperAlignmentTests.test_sparse_dag_can_have_multiple_entries tests.test_paper_alignment.PaperAlignmentTests.test_invalid_dag_density_is_rejected -v`

预期：旧生成器强制补边，前两项断言失败；配置尚未校验，第三项失败。

- [ ] **步骤 3：最小实现**

```python
def validate(self):
    if self.edge_device_num < 1:
        raise ValueError("edge_device_num must be at least 1")
    if not 0.0 <= self.dag_edge_prob <= 1.0:
        raise ValueError("dag_edge_prob must be in [0, 1]")
```

在 `_build_dag` 中构造全部 `(i,j), i<j`，使用环境 RNG 无放回抽取 `round(p*N*(N-1)/2)` 条边，并删除强制补父节点逻辑。增加只读 `edge_density` 属性。

- [ ] **步骤 4：运行任务测试与原有论文测试**

运行：`python -m unittest tests.test_paper_alignment -v`

预期：全部通过。

### 任务 2：恢复论文状态并实现紧凑动作空间

**文件：**
- 修改：`algorithm/environment/cloud_fog_edge_env.py`
- 修改：`experiments/paper_experiments.py`
- 测试：`tests/test_paper_alignment.py`

**接口：**
- `CloudFogEdgeEnv.get_feasible_node_ids(task_id=None) -> list[str]`
- `CloudFogEdgeEnv.get_state_dim() -> int`
- `CloudFogEdgeEnv.get_action_dim() -> int`

- [ ] **步骤 1：编写失败测试**

```python
def test_state_matches_equation_323_dimensions(self):
    env = CloudFogEdgeEnv(self.make_config().env)
    state = env.reset()
    self.assertEqual(len(state), 4 + len(env.all_node_ids) + env.M)

def test_action_contains_only_cloud_fog_and_current_local(self):
    env = CloudFogEdgeEnv(self.make_config().env)
    env.reset()
    task_id = env.list_seq[0]
    self.assertEqual(
        env.get_feasible_node_ids(),
        ["cloud_0", "fog_1", "fog_2", "fog_3", f"edge_{task_id}"],
    )
    self.assertEqual(env.get_action_dim(), env.M + 3)
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_alignment.PaperAlignmentTests.test_state_matches_equation_323_dimensions tests.test_paper_alignment.PaperAlignmentTests.test_action_contains_only_cloud_fog_and_current_local -v`

预期：旧状态多一个 WAN 速率，旧动作包含所有边缘节点。

- [ ] **步骤 3：最小实现**

```python
def get_feasible_node_ids(self, task_id=None):
    if task_id is None:
        task_id = self.list_seq[self.current_task_idx]
    return ["cloud_0", *[node.id for node in self.fog_nodes], f"edge_{task_id}"]

def get_action_dim(self):
    return self.M + 3
```

`R_t` 只保留 M 个雾上行速率；`step` 直接在可行列表上执行 `argmax`，删除全节点掩码。实验 `action_vector` 改为通过 `get_feasible_node_ids()` 查找动作索引。

- [ ] **步骤 4：运行论文测试**

运行：`python -m unittest tests.test_paper_alignment -v`

预期：全部通过。

### 任务 3：动作校验和资源分配审计

**文件：**
- 修改：`algorithm/environment/cloud_fog_edge_env.py`
- 修改：`algorithm/environment/compute_node.py`
- 测试：`tests/test_paper_alignment.py`

**接口：**
- `CloudFogEdgeEnv._validated_action(action) -> np.ndarray`
- `ComputeNode.schedule(...) -> dict`

- [ ] **步骤 1：编写失败测试**

```python
def test_invalid_action_shape_and_nan_are_rejected(self):
    env = CloudFogEdgeEnv(self.make_config().env)
    env.reset()
    with self.assertRaisesRegex(ValueError, "action length"):
        env.step(np.zeros(env.get_action_dim() + 1))
    bad = np.zeros(env.get_action_dim())
    bad[0] = np.nan
    with self.assertRaisesRegex(ValueError, "finite"):
        env.step(bad)

def test_step_reports_requested_and_effective_resource(self):
    env = CloudFogEdgeEnv(self.make_config().env)
    env.reset()
    action = np.zeros(env.get_action_dim(), dtype=np.float32)
    action[-2] = 1.0
    action[-1] = 0.01
    env.step(action)
    info = env.last_step_info
    for key in ["requested_resource_ratio", "requested_cpu", "allocated_cpu", "effective_resource_ratio"]:
        self.assertIn(key, info)
    self.assertEqual(info["resource_ratio"], info["requested_resource_ratio"])
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_alignment.PaperAlignmentTests.test_invalid_action_shape_and_nan_are_rejected tests.test_paper_alignment.PaperAlignmentTests.test_step_reports_requested_and_effective_resource -v`

预期：旧环境不校验动作，且审计字段缺失。

- [ ] **步骤 3：最小实现**

动作转换为一维浮点数组，校验长度和有限性；概率和为 0 时替换为均匀分布，否则归一化。`ComputeNode.schedule` 的 reservation 增加 `available_at_start`、`requested_cpu` 和 `effective_ratio`。环境将这些值复制到 `last_step_info`，并保留兼容字段 `resource_ratio`。

- [ ] **步骤 4：运行资源与论文测试**

运行：`python -m unittest tests.test_paper_alignment -v`

预期：全部通过。

### 任务 4：探索动作稳定性与模型结构版本

**文件：**
- 修改：`algorithm/per_ddpg/ddpg_agent.py`
- 测试：`tests/test_paper_alignment.py`

**接口：**
- `MODEL_SCHEMA_VERSION = 2`
- `PERDDPGAgent.save(path) -> None`
- `PERDDPGAgent.load(path) -> None`

- [ ] **步骤 1：编写失败测试**

```python
def test_zero_node_probabilities_fall_back_to_uniform(self):
    config = self.make_config()
    env = CloudFogEdgeEnv(config.env)
    agent = PERDDPGAgent(env.get_state_dim(), env.get_action_dim(), config.ddpg, config.env)
    action = np.zeros(env.get_action_dim(), dtype=np.float32)
    normalized = agent._normalize_exploratory_action(action)
    np.testing.assert_allclose(normalized[:-1], 1.0 / (env.M + 2))

def test_checkpoint_metadata_rejects_incompatible_schema(self):
    with tempfile.TemporaryDirectory() as path:
        agent.save(path)
        metadata_path = os.path.join(path, "metadata.json")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        metadata["schema_version"] = 1
        Path(metadata_path).write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema"):
            agent.load(path)
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_alignment.PaperAlignmentTests.test_zero_node_probabilities_fall_back_to_uniform tests.test_paper_alignment.PaperAlignmentTests.test_checkpoint_metadata_rejects_incompatible_schema -v`

预期：归一化辅助方法和元数据文件均不存在。

- [ ] **步骤 3：最小实现**

增加 `_normalize_exploratory_action`；保存 `metadata.json`，字段为 `schema_version/state_dim/action_dim/fog_node_num/task_num`；加载权重前逐项校验并给出中文可理解错误。

- [ ] **步骤 4：运行论文和 PER-DDPG 测试**

运行：`python -m unittest tests.test_paper_alignment -v`

预期：全部通过。

### 任务 5：实现 PER 与 LD-Noise 独立消融

**文件：**
- 修改：`experiments/paper_experiments.py`
- 创建：`tests/test_paper_experiments.py`

**接口：**
- `train_ddpg(config, episodes, prioritized, adaptive_noise, seed)`
- `compare_algorithms(...) -> dict`

- [ ] **步骤 1：编写失败测试**

```python
def test_ddpg_ablation_matrix(self):
    expected = {
        "DDPG": (False, False),
        "DDPG+PER": (True, False),
        "DDPG+LD": (False, True),
        "PER-DDPG": (True, True),
    }
    self.assertEqual(ddpg_variants(), expected)
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_experiments.PaperExperimentTests.test_ddpg_ablation_matrix -v`

预期：`ddpg_variants` 尚不存在。

- [ ] **步骤 3：最小实现**

增加返回上述固定映射的 `ddpg_variants()`。`prioritized=False` 时设置 `alpha=0,beta=0`；`adaptive_noise=False` 时使用 `FixedGaussianNoise`。比较实验遍历四种组合，并继续加入 FLE、RO、DQN。

- [ ] **步骤 4：运行实验单元测试**

运行：`python -m unittest tests.test_paper_experiments -v`

预期：通过。

### 任务 6：多随机种子汇总和边密度审计

**文件：**
- 修改：`experiments/paper_experiments.py`
- 测试：`tests/test_paper_experiments.py`

**接口：**
- `parse_seeds(seed, seeds_text) -> list[int]`
- `aggregate_seed_results(results_by_seed) -> dict`
- `measured_edge_density(env) -> float`

- [ ] **步骤 1：编写失败测试**

```python
def test_seed_override_and_aggregate_statistics(self):
    self.assertEqual(parse_seeds(7, "42,43"), [7])
    self.assertEqual(parse_seeds(None, "42,43"), [42, 43])
    result = aggregate_seed_results({
        42: {"mean_makespan": 2.0},
        43: {"mean_makespan": 4.0},
    })
    self.assertEqual(result["seed_count"], 2)
    self.assertEqual(result["mean_makespan"], 3.0)
    self.assertEqual(result["std_makespan"], 1.0)
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_experiments.PaperExperimentTests.test_seed_override_and_aggregate_statistics -v`

预期：解析与汇总函数不存在。

- [ ] **步骤 3：最小实现**

CLI 将 `--seed` 默认值改为 `None`，新增 `--seeds` 默认 `42,43,44,45,46`。每个种子独立训练和评估；汇总以各训练种子的评估均值为样本。JSON 保存 `seed_results`，CSV 增加 `seed_count`。边密度实验每个点写入 `configured_density` 与实际生成图的 `measured_density`。

- [ ] **步骤 4：运行实验测试**

运行：`python -m unittest tests.test_paper_experiments -v`

预期：全部通过。

### 任务 7：Web 与文档兼容

**文件：**
- 修改：`web/templates/index.html`
- 修改：`README_paper_alignment.md`
- 测试：`tests/test_paper_alignment.py`

**接口：** 保留现有路由和 `resource_ratio`，页面新增实际比例显示。

- [ ] **步骤 1：编写失败测试**

```python
def test_decision_keeps_resource_ratio_compatibility(self):
    detail = runner.run_step(explore=False)
    decision = detail["decision"]
    self.assertEqual(decision["resource_ratio"], decision["requested_resource_ratio"])
    self.assertIn("effective_resource_ratio", decision)
```

- [ ] **步骤 2：确认测试失败**

运行：`python -m unittest tests.test_paper_alignment.PaperAlignmentTests.test_decision_keeps_resource_ratio_compatibility -v`

预期：新增字段不存在。

- [ ] **步骤 3：最小实现**

页面在任务日志中显示“请求比例/实际比例/实际 GHz”。README 使用中文说明动作维度、旧检查点失效、多种子命令和 smoke 结论边界。

- [ ] **步骤 4：运行论文测试**

运行：`python -m unittest tests.test_paper_alignment -v`

预期：全部通过。

### 任务 8：全量验证与两种子 Smoke

**文件：**
- 验证：全部已修改文件
- 输出：`output/experiments/`

- [ ] **步骤 1：运行核心测试**

运行：`python -m unittest tests.test_paper_alignment tests.test_paper_experiments -v`

预期：全部通过，无异常和非有限数值。

- [ ] **步骤 2：运行现有 PER-DDPG 测试**

运行：`python tests/test_per_ddpg.py`

预期：环境、动作、训练循环和前端快照检查通过。

- [ ] **步骤 3：运行两种子 Smoke**

运行：`python experiments/paper_experiments.py --experiment comparison --profile smoke --seeds 42,43`

预期：生成 JSON 与 CSV；包含 FLE、RO、DQN、DDPG、DDPG+PER、DDPG+LD、PER-DDPG，每个学习算法包含两个种子结果。

- [ ] **步骤 4：检查输出结构**

确认 provenance 为本地生成、所有 makespan 有限、`seed_count=2`，且不对算法性能排序作验收要求。
