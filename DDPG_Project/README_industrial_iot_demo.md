# 工业物联网 PER-DDPG 调度演示

这个演示根据论文第 3 章的算法思想实现：

- 算法 3.1：根据 DAG 依赖、`rank_i`、`urgency_i` 和 `W_i` 生成 `List_seq`。
- 算法 3.2：按 `List_seq` 顺序执行云-雾-边卸载决策，输出节点选择 `A_t` 和连续资源比例 `F_t`。
- PER：根据 TD Error 保存高价值经验样本，并在前端展示优先级最高的经验。
- LD-Noise：训练早期探索噪声大，后期线性衰减。

## 运行

```powershell
python industrial_iot_per_ddpg_demo.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

如果只想验证算法输出：

```powershell
python industrial_iot_per_ddpg_demo.py --once
```

## 工业物联网场景

场景是一条工业产线，从设备采集到质量检测、故障诊断、工艺调优和告警归档：

- 边缘层：PLC edge controller、Vision edge box、Robot arm gateway。
- 雾层：Line-A fog server、Quality fog server。
- 云层：Factory cloud。
- DAG 子任务：振动信号窗口化、热像预处理、声学异常特征、表面缺陷推理、多传感融合、质量漂移预测、维修风险评分、工艺参数调优、告警归档等。

## 前端能看到什么

- `List_seq` 调度序列。
- DAG 依赖图。
- 每个任务的执行层、执行节点、CPU 资源比例、开始/完成时间。
- 传输、等待、执行三类时延。
- 动作概率 `A_t`，体现智能体为什么选某个节点。
- Makespan 收敛曲线和 LD-Noise 衰减曲线。
- PER 高优先级经验样本。

## 说明

仓库中原有 `algorithm/` 和 `web/` 是 Flask + Torch 版本骨架；当前环境缺少 `flask`、`flask_socketio`、`torch`，所以新增的 `industrial_iot_per_ddpg_demo.py` 使用标准库 HTTP 服务和 NumPy，实现一个能立即运行和展示算法结果的落地版。
