# 论文第3章实现与复现实验说明

## 两个前端入口

- `python run.py` -> `http://127.0.0.1:5000`
  - 真实 PyTorch Actor/Critic、目标网络、PER、LD-Noise 和梯度更新。
  - 用于论文算法验证和实验。
- `python industrial_iot_per_ddpg_demo.py` -> `http://127.0.0.1:8765`
  - 固定工业物联网场景的轻量可视化演示。
  - 节点策略是启发式近似，不应作为 PER-DDPG 训练或论文性能复现证据。

## 依赖

```powershell
python -m pip install -r requirements.txt
```

核心依赖为 Python 3、NumPy、PyTorch、Flask、Flask-SocketIO 和 NetworkX。
CUDA/GPU 是可选项；CPU 可以运行测试和 smoke 实验，完整 500 回合参数扫描
建议使用 GPU 或预留较长运行时间。

## 验证

```powershell
python -m unittest tests.test_paper_alignment -v
```

该测试覆盖：

- 算法3.1静态优先级与拓扑合法性；
- 最近雾节点平均处理时延；
- 公式3.16/3.17最小CPU和并发容量约束；
- 经验池超过 Batch Size=64 后真实反向传播；
- Actor 参数变化及 SumTree 优先级有限性。

## 本轮严谨性修正

- 状态严格保持论文公式 `s_t={I_t,CR_t,R_t}`；
- 动作只包含云、全部雾节点和当前任务本地节点，总维度为 `M+3`；
- DAG 按目标边密度精确抽样，允许多个入口任务；
- 资源审计同时记录请求比例、实际比例和实际分配 CPU；
- 新检查点结构版本为 2，旧检查点需要重新训练；
- 对比实验增加 `DDPG+PER` 与 `DDPG+LD`，分别验证 PER 和 LD-Noise 的贡献；
- 多种子统计以独立训练种子的评估均值为样本。
## 本地复现实验

快速链路验证：

```powershell
python experiments/paper_experiments.py --experiment comparison --profile smoke --seeds 42,43
```

论文参数规模运行：

```powershell
python experiments/paper_experiments.py --experiment comparison --profile full --seed 42
python experiments/paper_experiments.py --experiment task-count --profile full --seed 42
python experiments/paper_experiments.py --experiment data-size --profile full --seed 42
python experiments/paper_experiments.py --experiment fog-count --profile full --seed 42
python experiments/paper_experiments.py --experiment edge-density --profile full --seed 42
python experiments/paper_experiments.py --experiment learning-rate --profile full --seed 42
python experiments/paper_experiments.py --experiment batch-size --profile full --seed 42
python experiments/paper_experiments.py --experiment gamma --profile full --seed 42
```

输出写入 `output/experiments/`，同时生成 JSON 和 CSV。未指定 `--seed` 时，
默认使用 `42,43,44,45,46` 五个独立训练种子；显式指定 `--seed` 时执行单种子
调试。结果均标注为本项目本地生成数据，不能冒充论文作者的原始实验数据。
Smoke 模式只验证代码链路和输出结构，不作为论文性能结论的复现证据。

## 无法独立保证的内容

论文没有提供原始 DAG 样本、全部随机种子、训练检查点和图3.4-3.11的原始
数据表，因此无法逐点恢复论文曲线或保证得到完全相同的百分比。项目可以按
论文公式和参数范围重新运行实验，但结论必须基于重新生成的数据。
